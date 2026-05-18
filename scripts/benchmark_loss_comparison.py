#!/usr/bin/env python3
"""Compare RankReg with pointwise losses on DMS datasets using official CV folds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reap.data import load_embeddings, load_requested_datasets, parse_int_list, parse_str_list, standardize_by_train
from reap.training import train_evolvepro, train_plm_model


def discover_datasets(embedding_folder: str, cv_folder: str, requested=None):
    requested = requested or sorted(p.name for p in Path(embedding_folder).iterdir() if p.is_dir())
    datasets = []
    missing = []
    for name in requested:
        emb_path = Path(embedding_folder) / name
        cv_path = Path(cv_folder) / f"{name}.csv"
        if emb_path.is_dir() and cv_path.exists():
            datasets.append(name)
        else:
            missing.append({"dataset": name, "embedding_exists": emb_path.is_dir(), "cv_exists": cv_path.exists()})
    if missing:
        print("Warning: some datasets were skipped:")
        for row in missing:
            print(row)
    return datasets


def run_one_dataset(dataset_name: str, args) -> list[dict]:
    emb_folder = Path(args.embedding_folder) / dataset_name
    cv_file = Path(args.cv_folder) / f"{dataset_name}.csv"
    X, y = load_embeddings(emb_folder, require_y=True)
    cv_df = pd.read_csv(cv_file)
    if args.cv_split not in cv_df.columns:
        raise KeyError(f"{args.cv_split} not found in {cv_file}; available columns: {list(cv_df.columns)}")
    cv_indices = cv_df[args.cv_split].values
    if len(X) != len(cv_indices):
        raise ValueError(f"Length mismatch for {dataset_name}: embeddings={len(X)} cv={len(cv_indices)}")

    folds = sorted(pd.unique(cv_indices)) if args.folds == "auto" else parse_int_list(args.folds)
    losses = parse_str_list(args.losses)
    seeds = parse_int_list(args.seeds)
    rows = []

    for fold in folds:
        train_mask = cv_indices != fold
        test_mask = cv_indices == fold
        X_train, y_train_raw = X[train_mask], y[train_mask]
        X_test, y_test_raw = X[test_mask], y[test_mask]
        if len(y_train_raw) == 0 or len(y_test_raw) == 0:
            rows.append({"dataset": dataset_name, "fold": fold, "status": "SKIP", "error": "empty split"})
            continue

        for seed in seeds:
            for loss_name in losses:
                start = time.perf_counter()
                try:
                    y_train, y_test = y_train_raw.copy(), y_test_raw.copy()
                    y_mean = float(np.mean(y_train))
                    y_std = float(np.std(y_train))
                    if args.standardize_y:
                        (y_train, y_test), y_mean, y_std = standardize_by_train(y_train, y_test)

                    if loss_name.lower() == "evolvepro":
                        spearman, mse, _ = train_evolvepro(X_train, y_train, X_test, y_test, seed=seed)
                    else:
                        model_path = None
                        if args.save_models:
                            model_dir = Path(args.model_dir) / dataset_name / f"fold_{fold}" / f"seed_{seed}"
                            model_dir.mkdir(parents=True, exist_ok=True)
                            model_path = str(model_dir / f"{loss_name}.pt")
                        spearman, mse, _, _ = train_plm_model(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_test,
                            y_val=y_test,
                            epochs=args.epochs,
                            seed=seed,
                            save_path=model_path,
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
                    rows.append({
                        "dataset": dataset_name,
                        "fold": fold,
                        "seed": seed,
                        "loss": loss_name,
                        "spearman": spearman,
                        "mse": mse,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "standardize_y": bool(args.standardize_y),
                        "y_mean_train": y_mean,
                        "y_std_train": y_std,
                        "runtime_sec": time.perf_counter() - start,
                        "status": "OK",
                        "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "dataset": dataset_name,
                        "fold": fold,
                        "seed": seed,
                        "loss": loss_name,
                        "spearman": np.nan,
                        "mse": np.nan,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "standardize_y": bool(args.standardize_y),
                        "runtime_sec": time.perf_counter() - start,
                        "status": "ERROR",
                        "error": str(exc),
                    })
    return rows


def summarize(raw_df: pd.DataFrame, output_dir: Path) -> None:
    ok = raw_df[raw_df["status"] == "OK"].copy()
    if ok.empty:
        print("No successful runs to summarize.")
        return
    by_dataset = ok.groupby(["dataset", "loss"]).agg(
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        runtime_sec_mean=("runtime_sec", "mean"),
        n_runs=("spearman", "count"),
    ).reset_index()
    overall = by_dataset.groupby("loss").agg(
        spearman_mean=("spearman_mean", "mean"),
        spearman_std_across_datasets=("spearman_mean", "std"),
        mse_mean=("mse_mean", "mean"),
        mse_std_across_datasets=("mse_mean", "std"),
        runtime_sec_mean=("runtime_sec_mean", "mean"),
        n_datasets=("dataset", "nunique"),
    ).reset_index()
    by_dataset.to_csv(output_dir / "loss_comparison_summary_by_dataset.csv", index=False)
    overall.to_csv(output_dir / "loss_comparison_summary_overall.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RankReg against baseline losses.")
    parser.add_argument("--embedding_folder", default="data/embeddings")
    parser.add_argument("--cv_folder", default="data/cv_folds")
    parser.add_argument("--cv_split", default="fold_random_5")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names. Empty means discover all.")
    parser.add_argument("--dataset_list", default="", help="Text file with one dataset name per line.")
    parser.add_argument("--losses", default="RankReg,MSE,Huber,L1")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Use 'auto' to infer folds from cv_split.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--rank_num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="results/loss_comparison")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--model_dir", default="results/loss_comparison/models")
    parser.add_argument("--device", default="", help="Device such as cuda:0 or cpu. Empty chooses automatically.")
    parser.add_argument("--standardize_y", action="store_true", default=True)
    parser.add_argument("--no_standardize_y", action="store_false", dest="standardize_y")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = load_requested_datasets(args.datasets, args.dataset_list)
    datasets = discover_datasets(args.embedding_folder, args.cv_folder, requested or None)
    if not datasets:
        raise ValueError("No matched datasets were found.")
    all_rows = []
    for dataset in datasets:
        print(f"Running {dataset}")
        all_rows.extend(run_one_dataset(dataset, args))
        pd.DataFrame(all_rows).to_csv(output_dir / "loss_comparison_raw.csv", index=False)
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(output_dir / "loss_comparison_raw.csv", index=False)
    summarize(raw_df, output_dir)
    with open(output_dir / "benchmark_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4)
    print(f"Saved benchmark results to {output_dir}")


if __name__ == "__main__":
    main()
