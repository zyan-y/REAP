#!/usr/bin/env python3
"""Clean raw replicate assay results and merge cleaned files."""

import argparse

from reap.assay import clean_assay_file, merge_cleaned_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Clean experimental assay results for REAP training.")
    parser.add_argument("--input_excel", required=True, help="Raw Excel file with name and replicate columns.")
    parser.add_argument("--cleaned_dir", required=True, help="Directory to save cleaned and merged files.")
    parser.add_argument("--cleaned_filename", default="cleaned.xlsx", help="Cleaned output filename.")
    parser.add_argument("--merged_filename", default="data.xlsx", help="Merged output filename.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path, _ = clean_assay_file(args.input_excel, args.cleaned_dir, cleaned_filename=args.cleaned_filename)
    merged = merge_cleaned_folder(args.cleaned_dir, merged_filename=args.merged_filename)
    print(f"Saved cleaned file to {out_path}")
    print(f"Saved merged training table to {merged}")


if __name__ == "__main__":
    main()
