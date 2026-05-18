#!/usr/bin/env python3
"""Train an ensemble of PLM-RankReg models for REAP candidate scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from reap.data import load_embeddings, parse_int_list, set_seed, standardize_by_train
from reap.training import train_plm_rankreg


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PLM-RankReg ensemble.")
    parser.add_argument("--embeddings_folder", required=True, help="Folder containing labeled training batch_*.npz files.")
    parser.add_argument("--output_dir", default="checkpoints/ensemble", help="Output directory for model checkpoints.")
    parser.add_argument("--seeds", default="42,715,1388,2061,2734,3407,4080,4753,5426,6099", help="Comma-separated model seeds.")
    parser.add_argument("--ensemble_size", type=int, default=10, help="Maximum number of models to save.")
    parser.add_argument("--min_val_spearman", type=float, default=None, help="Optional threshold for saving models.")
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--device", default="")
    parser.add_argument("--standardize_y", action="store_true", default=True)
    parser.add_argument("--no_standardize_y", action="store_false", dest="standardize_y")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_embeddings(args.embeddings_folder, require_y=True)
    seeds = parse_int_list(args.seeds)
    saved = 0
    rows = []

    for seed in seeds:
        if saved >= args.ensemble_size:
            break
        set_seed(seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=seed, shuffle=True)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if args.standardize_y:
            (y_train, y_val), y_mean, y_std = standardize_by_train(y_train, y_val)

        tmp_ckpt = out / f"candidate_seed_{seed}.pt"
        spearman, mse, _, _ = train_plm_rankreg(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            seed=seed,
            save_path=str(tmp_ckpt),
            model_type=args.model_type,
            alpha=args.alpha,
            margin=args.margin,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            device=args.device,
            checkpoint_metadata={
                "standardize_y": bool(args.standardize_y),
                "y_mean_train": float(y_mean),
                "y_std_train": float(y_std),
            },
        )
        keep = args.min_val_spearman is None or spearman >= args.min_val_spearman
        final_path = ""
        if keep:
            saved += 1
            final = out / f"ensemble_{saved:03d}_seed_{seed}.pt"
            tmp_ckpt.replace(final)
            final_path = str(final)
        elif tmp_ckpt.exists():
            tmp_ckpt.unlink()
        rows.append({"seed": seed, "val_spearman": spearman, "val_mse": mse, "saved": keep, "path": final_path, "y_mean_train": y_mean, "y_std_train": y_std})
        print(rows[-1])

    with open(out / "ensemble_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=4)
    print(f"Saved {saved} ensemble checkpoints to {out}")


if __name__ == "__main__":
    main()
