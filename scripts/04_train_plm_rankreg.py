#!/usr/bin/env python3
"""Train a PLM-RankReg model on fixed protein embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from reap.data import load_embeddings, set_seed, standardize_by_train
from reap.training import train_plm_rankreg


def parse_args():
    parser = argparse.ArgumentParser(description="Train PLM-RankReg on precomputed embeddings.")
    parser.add_argument("--embeddings_folder", required=True, help="Folder containing labeled batch_*.npz files.")
    parser.add_argument("--output_dir", default="checkpoints/plm_rankreg", help="Directory for checkpoints and metrics.")
    parser.add_argument("--model_type", default="mlp", help="Prediction head: mlp, cnn, or light_attention.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--device", default="", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--standardize_y", action="store_true", default=True, help="Standardize y using training mean/std.")
    parser.add_argument("--no_standardize_y", action="store_false", dest="standardize_y")
    parser.add_argument("--name_suffix", default="", help="Optional suffix for saved files.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_embeddings(args.embeddings_folder, require_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.seed, shuffle=True)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if args.standardize_y:
        (y_train, y_val), y_mean, y_std = standardize_by_train(y_train, y_val)

    tag_parts = [f"head_{args.model_type}", f"seed_{args.seed}"]
    if args.name_suffix:
        tag_parts.append(args.name_suffix)
    tag = "_".join(tag_parts)
    ckpt_path = output_dir / f"plm_rankreg_{tag}.pt"

    best_spearman, best_mse, ckpt_path_str, _ = train_plm_rankreg(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        seed=args.seed,
        save_path=str(ckpt_path),
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

    metrics = {
        "best_spearman": best_spearman,
        "best_mse": best_mse,
        "checkpoint_path": ckpt_path_str,
        "standardize_y": bool(args.standardize_y),
        "y_mean_train": y_mean,
        "y_std_train": y_std,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }
    with open(output_dir / f"metrics_{tag}.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)
    with open(output_dir / f"config_{tag}.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
