#!/usr/bin/env python3
"""Use a trained ensemble to rank mutants for the next REAP round."""

import argparse

from reap.selection import guide_mutations


def parse_args():
    parser = argparse.ArgumentParser(description="Rank candidate mutants with an ensemble of PLM-RankReg checkpoints.")
    parser.add_argument("--models_dir", required=True, help="Directory with trained checkpoints.")
    parser.add_argument("--embeddings_folder", required=True, help="Unlabeled candidate embedding folder.")
    parser.add_argument("--names_excel", default=None, help="Optional Excel file with candidate names.")
    parser.add_argument("--output_dir", default="results/reap_iteration", help="Directory to save predictions and selected candidates.")
    parser.add_argument("--ensemble_size", type=int, default=100)
    parser.add_argument("--ckpt_pattern", default="*.pt")
    parser.add_argument("--lambda_sigma", type=float, default=1.0, help="UCB score = prediction_mean + lambda_sigma * prediction_std.")
    parser.add_argument("--top_n", type=int, default=96, help="Number of top-ranked candidates to export.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--model_type", default="mlp")
    parser.add_argument("--excel_out", default="selected_candidates.xlsx")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_df, _ = guide_mutations(**vars(args))
    print(f"Selected {len(selected_df)} candidates by UCB. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
