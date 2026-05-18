"""Model-guided mutant ranking with PLM-RankReg ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_embeddings
from .training import load_checkpoint_config, load_plm_checkpoint, predict_plm_model


def list_checkpoints(models_dir: str | Path, pattern: str = "*.pt", max_n: int | None = None) -> list[Path]:
    """Return checkpoint paths sorted by file name."""
    files = sorted(Path(models_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {models_dir} matching '{pattern}'.")
    return files[:max_n] if max_n is not None else files


def _load_names_from_excel(
    names_excel: str | Path | None,
    fallback_names: np.ndarray | None,
    n_candidates: int,
) -> np.ndarray:
    """Read candidate names from an Excel file or from the embedding files."""
    if names_excel:
        df = pd.read_excel(names_excel)
        names = df["name"].astype(str).values if "name" in df.columns else df.iloc[:, 0].astype(str).values
    elif fallback_names is not None:
        names = fallback_names.astype(str)
    else:
        raise ValueError("Candidate names are required. Store key 'n' in embeddings or provide --names_excel.")
    if len(names) != n_candidates:
        raise ValueError(f"Names length ({len(names)}) does not match embeddings ({n_candidates}).")
    return names


def predict_ensemble(
    embeddings_folder: str | Path,
    models_dir: str | Path,
    *,
    ckpt_pattern: str = "*.pt",
    ensemble_size: int = 100,
    model_type: str = "mlp",
    batch_size: int = 4096,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return raw ensemble predictions, prediction mean, prediction std, and candidate names."""
    X, _, names = load_embeddings(embeddings_folder, return_names=True, require_y=False)
    X = np.asarray(X, dtype=np.float32)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpts = list_checkpoints(models_dir, pattern=ckpt_pattern, max_n=ensemble_size)

    all_preds = []
    for ckpt_path in ckpts:
        model = load_plm_checkpoint(ckpt_path, input_dim=X.shape[1], model_type=model_type, device=device)
        preds = predict_plm_model(model, X, batch_size=batch_size, device=device)
        config = load_checkpoint_config(ckpt_path)
        if config.get("standardize_y", False):
            y_mean = float(config.get("y_mean_train", 0.0))
            y_std = float(config.get("y_std_train", 1.0))
            preds = preds * y_std + y_mean
        if len(preds) != len(X):
            raise RuntimeError(f"Prediction length mismatch for {ckpt_path.name}: {len(preds)} vs {len(X)}")
        all_preds.append(preds)
    all_preds = np.stack(all_preds, axis=0)
    return all_preds, all_preds.mean(axis=0), all_preds.std(axis=0, ddof=0), names


def rank_by_ucb(
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    *,
    lambda_sigma: float = 1.0,
    top_n: int | None = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank candidates by UCB = prediction_mean + lambda_sigma * prediction_std."""
    prediction_mean = np.asarray(prediction_mean, dtype=float)
    prediction_std = np.asarray(prediction_std, dtype=float)
    ucb_score = prediction_mean + float(lambda_sigma) * prediction_std
    order = np.argsort(-ucb_score)
    if top_n is not None and top_n > 0:
        order = order[: min(int(top_n), len(order))]
    return order.astype(int), ucb_score


def guide_mutations(
    embeddings_folder: str | Path,
    models_dir: str | Path,
    *,
    names_excel: str | Path | None = None,
    output_dir: str | Path = "results/reap_iteration",
    ensemble_size: int = 100,
    ckpt_pattern: str = "*.pt",
    lambda_sigma: float = 1.0,
    top_n: int = 96,
    batch_size: int = 4096,
    model_type: str = "mlp",
    excel_out: str = "selected_candidates.xlsx",
    device: str | torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict candidate activities, rank by UCB, and export next-round candidates."""
    all_preds, prediction_mean, prediction_std, names_from_emb = predict_ensemble(
        embeddings_folder,
        models_dir,
        ckpt_pattern=ckpt_pattern,
        ensemble_size=ensemble_size,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
    )
    names = _load_names_from_excel(names_excel, names_from_emb, len(prediction_mean))
    selected_idx, ucb_score = rank_by_ucb(
        prediction_mean,
        prediction_std,
        lambda_sigma=lambda_sigma,
        top_n=top_n,
    )

    unsorted_df = pd.DataFrame(
        {
            "name": names,
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "lambda_sigma": float(lambda_sigma),
            "ucb_score": ucb_score,
        }
    )
    all_df = unsorted_df.sort_values("ucb_score", ascending=False, kind="mergesort").reset_index(drop=True)
    all_df.insert(0, "ucb_rank", np.arange(1, len(all_df) + 1))

    selected_df = unsorted_df.iloc[selected_idx].copy().reset_index(drop=True)
    selected_df.insert(1, "selection_rank", np.arange(1, len(selected_df) + 1))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "all_candidate_predictions.csv", index=False)
    selected_df.to_excel(out_dir / excel_out, index=False)
    np.save(out_dir / "ensemble_predictions.npy", all_preds)
    return selected_df, all_df
