"""Training and evaluation utilities for PLM-RankReg and baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .data import set_seed
from .losses import get_loss_function_baseline, rank_reg_loss
from .models import PLM_RankReg


def evaluate_mse_spearman(true_labels, predictions) -> tuple[float, float]:
    """Return MSE and Spearman correlation, with NaN Spearman mapped to 0."""
    true_arr = np.asarray(true_labels).reshape(-1)
    pred_arr = np.asarray(predictions).reshape(-1)
    mse = float(mean_squared_error(true_arr, pred_arr))
    spearman, _ = spearmanr(true_arr, pred_arr)
    if np.isnan(spearman):
        spearman = 0.0
    return mse, float(spearman)


def _resolve_device(seed: int, device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    set_seed(seed)
    if resolved.type == "cuda" and resolved.index is not None:
        torch.cuda.set_device(resolved.index)
    return resolved


def _prepare_data_loaders(X_train, y_train, X_val, y_val, batch_size: int, device: torch.device):
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    pin_memory = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, int(X_train_t.shape[1])


def predict_plm_model(model: torch.nn.Module, X, *, batch_size: int = 4096, device=None) -> np.ndarray:
    """Batch inference for a PLM-RankReg model."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(X_t, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            preds.append(model(batch).squeeze(-1).detach().cpu())
    return torch.cat(preds).numpy()


def train_plm_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int,
    seed: int,
    save_path: str | None,
    model_type: str,
    loss_name: str = "RankReg",
    alpha: float = 0.5,
    margin: float = 0.1,
    rank_num_samples: int | None = None,
    patience: int = 10,
    batch_size: int = 1024,
    lr: float = 3e-4,
    wd: float = 1e-5,
    device: str | torch.device | None = None,
    checkpoint_metadata: dict | None = None,
):
    """Train one PLM regression head and optionally save the best checkpoint."""
    device = _resolve_device(seed, device)
    train_loader, val_loader, input_dim = _prepare_data_loaders(X_train, y_train, X_val, y_val, batch_size, device)

    model = PLM_RankReg(input_dim, model_type=model_type, dropout=0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=min(1e-5, lr),
        max_lr=lr,
        cycle_momentum=False,
    )

    loss_name_lower = str(loss_name).lower()
    criterion = None if loss_name_lower == "rankreg" else get_loss_function_baseline(loss_name)

    best_spearman = -float("inf")
    best_mse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(int(epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            if loss_name_lower == "rankreg":
                loss = rank_reg_loss(outputs, labels, alpha=alpha, margin=margin, num_samples=rank_num_samples)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        preds, targs = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                preds.append(outputs.squeeze(-1).detach().cpu())
                targs.append(labels.detach().cpu())
        pred_arr = torch.cat(preds).numpy()
        targ_arr = torch.cat(targs).numpy()
        val_mse, val_spearman = evaluate_mse_spearman(targ_arr, pred_arr)

        improved = (val_spearman > best_spearman + 1e-12) or (
            abs(val_spearman - best_spearman) <= 1e-12 and val_mse < best_mse - 1e-12
        )
        if improved:
            best_spearman = val_spearman
            best_mse = val_mse
            patience_counter = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                config = {
                    "loss_name": loss_name,
                    "alpha": alpha,
                    "margin": margin,
                    "rank_num_samples": rank_num_samples,
                    "lr": lr,
                    "wd": wd,
                    "batch_size": batch_size,
                    "model_type": model_type,
                    "seed": seed,
                    "input_dim": input_dim,
                }
                if checkpoint_metadata:
                    config.update(checkpoint_metadata)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "spearman": best_spearman,
                        "mse": best_mse,
                        "config": config,
                    },
                    save_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return float(best_spearman), float(best_mse), save_path, model


def train_plm_rankreg(*args, **kwargs):
    """Train with RankReg loss; wrapper kept for script readability."""
    kwargs["loss_name"] = "RankReg"
    return train_plm_model(*args, **kwargs)


def train_evolvepro(X_train, y_train, X_test, y_test, seed: int = 42, **rf_kwargs):
    """Random-forest baseline on fixed PLM embeddings, following the EvolvePro-style setup."""
    params = {"n_estimators": 100, "random_state": seed, "n_jobs": -1}
    params.update(rf_kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse, spearman = evaluate_mse_spearman(y_test, preds)
    return float(spearman), float(mse), model


def load_plm_checkpoint(checkpoint_path: str | Path, input_dim: int, model_type: str = "mlp", device=None) -> PLM_RankReg:
    """Load a saved PLM-RankReg checkpoint for inference."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = PLM_RankReg(input_dim=input_dim, model_type=config.get("model_type", model_type), dropout=0).to(device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_checkpoint_config(checkpoint_path: str | Path) -> dict:
    """Return checkpoint configuration metadata when available."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
