#!/usr/bin/env python3
"""Few-shot evaluation of RankReg and baselines on mutation-effect datasets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reap.data import load_embeddings, load_requested_datasets, parse_int_list, parse_str_list, standardize_by_train
from reap.training import evaluate_mse_spearman, load_plm_checkpoint, predict_plm_model, train_evolvepro, train_plm_model


def discover_datasets(embedding_folder: str, requested=None):
    if requested:
        candidates = requested
    else:
        candidates = sorted(p.name for p in Path(embedding_folder).iterdir() if p.is_dir())
    out = []
    for name in candidates:
        if (Path(embedding_folder) / name).is_dir():
            out.append(name)
        else:
            print(f"Warning: missing embedding folder for {name}")
    return out


def make_few_shot_split(names, train_size: int, seed: int):
    names = np.asarray(names).astype(str)
    single_idx = np.where(np.asarray(["-" not in x for x in names]))[0]
    rng = np.random.RandomState(seed)
    selected = rng.permutation(single_idx)[: 2 * train_size]
    train_idx = selected[:train_size]
    val_idx = selected[train_size:]
    used = set(selected.tolist())
    test_idx = np.asarray([i for i in range(len(names)) if i not in used], dtype=int)
    return train_idx, val_idx, test_idx


def run_one_dataset(dataset_name: str, args):
    X, y, names = load_embeddings(Path(args.embedding_folder) / dataset_name, return_names=True, require_y=True)
    n_total = len(y)
    n_single = int(np.sum(["-" not in str(x) for x in names]))
    n_multi = n_total - n_single
    rows = []
    if n_total < args.min_total_samples:
        return [{"dataset": dataset_name, "status": "SKIP", "error": f"n_total={n_total} < {args.min_total_samples}"}]

    for train_size in parse_int_list(args.train_sizes):
        if n_single < 2 * train_size:
            rows.append({"dataset": dataset_name, "train_size": train_size, "status": "SKIP", "error": f"n_single={n_single} < {2 * train_size}"})
            continue
        for seed in parse_int_list(args.seeds):
            train_idx, val_idx, test_idx = make_few_shot_split(names, train_size, seed)
            X_train, y_train_raw = X[train_idx], y[train_idx]
            X_val, y_val_raw = X[val_idx], y[val_idx]
            X_test, y_test_raw = X[test_idx], y[test_idx]
            test_names = names[test_idx].astype(str)
            for loss_name in parse_str_list(args.losses):
                start = time.perf_counter()
                try:
                    y_train, y_val, y_test = y_train_raw.copy(), y_val_raw.copy(), y_test_raw.copy()
                    y_mean = float(np.mean(y_train))
                    y_std = float(np.std(y_train))
                    if args.standardize_y:
                        (y_train, y_val, y_test), y_mean, y_std = standardize_by_train(y_train, y_val, y_test)

                    if loss_name.lower() == "evolvepro":
                        spearman, mse, _ = train_evolvepro(X_train, y_train, X_test, y_test, seed=seed)
                        val_spearman = np.nan
                        val_mse = np.nan
                        model_path = ""
                    else:
                        model_dir = Path(args.model_dir if args.save_models else args.output_dir) / "models" / dataset_name / f"train_{train_size}" / f"seed_{seed}"
                        model_dir.mkdir(parents=True, exist_ok=True)
                        model_path = model_dir / f"{loss_name}.pt"
                        val_spearman, val_mse, _, _ = train_plm_model(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            epochs=args.epochs,
                            seed=seed,
                            save_path=str(model_path),
                            model_type=args.model_type,
                            loss_name=loss_name,
                            alpha=args.alpha,
                            margin=args.margin,
                            rank_num_samples=args.rank_num_samples,
                            patience=args.patience,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            wd=args.wd,
                            device=args.device,
                        )
                        model = load_plm_checkpoint(model_path, input_dim=X.shape[1], model_type=args.model_type, device=args.device or None)
                        preds = predict_plm_model(model, X_test, batch_size=args.batch_size, device=args.device or None)
                        mse, spearman = evaluate_mse_spearman(y_test, preds)
                        if not args.save_models and model_path.exists():
                            model_path.unlink()
                            model_path = ""

                    rows.append({
                        "dataset": dataset_name,
                        "train_size": train_size,
                        "seed": seed,
                        "loss": loss_name,
                        "spearman": spearman,
                        "mse": mse,
                        "val_spearman": val_spearman,
                        "val_mse": val_mse,
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "n_test": int(len(test_idx)),
                        "n_test_single": int(np.sum(["-" not in x for x in test_names])),
                        "n_test_multi": int(np.sum(["-" in x for x in test_names])),
                        "n_total": int(n_total),
                        "n_single": int(n_single),
                        "n_multi": int(n_multi),
                        "standardize_y": bool(args.standardize_y),
                        "y_mean_train": y_mean,
                        "y_std_train": y_std,
                        "runtime_sec": time.perf_counter() - start,
                        "model_path": str(model_path),
                        "status": "OK",
                        "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "dataset": dataset_name,
                        "train_size": train_size,
                        "seed": seed,
                        "loss": loss_name,
                        "spearman": np.nan,
                        "mse": np.nan,
                        "status": "ERROR",
                        "error": str(exc),
                        "runtime_sec": time.perf_counter() - start,
                    })
    return rows


def summarize(raw_df: pd.DataFrame, out: Path):
    ok = raw_df[raw_df["status"] == "OK"].copy()
    if ok.empty:
        return
    by_dataset = ok.groupby(["dataset", "train_size", "loss"]).agg(
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        runtime_sec_mean=("runtime_sec", "mean"),
        n_runs=("spearman", "count"),
    ).reset_index()
    overall = by_dataset.groupby(["train_size", "loss"]).agg(
        spearman_mean=("spearman_mean", "mean"),
        spearman_std_across_datasets=("spearman_mean", "std"),
        mse_mean=("mse_mean", "mean"),
        mse_std_across_datasets=("mse_mean", "std"),
        n_datasets=("dataset", "nunique"),
    ).reset_index()
    by_dataset.to_csv(out / "few_shot_summary_by_dataset.csv", index=False)
    overall.to_csv(out / "few_shot_summary_overall.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run few-shot comparison on mutation-effect datasets.")
    parser.add_argument("--embedding_folder", default="data/embeddings")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--dataset_list", default="")
    parser.add_argument("--losses", default="RankReg,MSE,Huber,L1,EvolvePro")
    parser.add_argument("--train_sizes", default="50,100,200,400")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--min_total_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--rank_num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="results/few_shot_comparison")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--model_dir", default="results/few_shot_comparison/models")
    parser.add_argument("--device", default="")
    parser.add_argument("--standardize_y", action="store_true", default=True)
    parser.add_argument("--no_standardize_y", action="store_false", dest="standardize_y")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    requested = load_requested_datasets(args.datasets, args.dataset_list)
    datasets = discover_datasets(args.embedding_folder, requested or None)
    all_rows = []
    for dataset in datasets:
        print(f"Running {dataset}")
        all_rows.extend(run_one_dataset(dataset, args))
        pd.DataFrame(all_rows).to_csv(out / "few_shot_raw.csv", index=False)
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(out / "few_shot_raw.csv", index=False)
    summarize(raw_df, out)
    with open(out / "few_shot_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4)
    print(f"Saved few-shot results to {out}")


if __name__ == "__main__":
    main()
