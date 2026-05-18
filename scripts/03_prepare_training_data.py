#!/usr/bin/env python3
"""Align experimental labels to precomputed embeddings and save training batches."""

import argparse

from reap.data import align_embeddings_to_table, save_embedding_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare labeled training embeddings from an assay table.")
    parser.add_argument("--emb_dir", required=True, help="Embedding folder containing candidate batch_*.npz files with names.")
    parser.add_argument("--table", required=True, help="CSV/XLSX table with name and yield columns.")
    parser.add_argument("--out_dir", required=True, help="Output folder for labeled training batch_*.npz files.")
    parser.add_argument("--name_col", default="name", help="Variant-name column in the table.")
    parser.add_argument("--label_col", default="yield", help="Label/activity column in the table.")
    parser.add_argument("--batch_size", type=int, default=0, help="Rows per output batch; 0 means one batch.")
    parser.add_argument("--batch_idx_start", type=int, default=0, help="Starting batch index.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip rows whose embeddings are missing.")
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, names, missing = align_embeddings_to_table(
        args.emb_dir,
        args.table,
        name_col=args.name_col,
        label_col=args.label_col,
        skip_missing=args.skip_missing,
    )
    save_embedding_batches(
        X,
        y,
        names,
        args.out_dir,
        batch_size=(None if args.batch_size <= 0 else args.batch_size),
        batch_idx_start=args.batch_idx_start,
    )
    print(f"Saved {len(names)} labeled variants to {args.out_dir}")
    if missing:
        print(f"Skipped {len(missing)} missing embeddings. First few: {missing[:5]}")


if __name__ == "__main__":
    main()
