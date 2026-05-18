#!/usr/bin/env python3
"""Extract mean-pooled ESM2 embeddings for labeled or unlabeled variant CSV files."""

import argparse

from reap.embeddings import extract_folder_or_file


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ESM2 sequence embeddings.")
    parser.add_argument("--dms_file", default="", help="Single CSV file to embed.")
    parser.add_argument("--dms_folder", default="", help="Folder containing CSV files to embed.")
    parser.add_argument("--embed_folder", default="data/embeddings", help="Directory to save batch_*.npz embeddings.")
    parser.add_argument("--model_name", default="esm2_t33_650M_UR50D", help="ESM model name for torch.hub.")
    parser.add_argument("--repr_layers", type=int, default=-1, help="Representation layer; -1 means final layer.")
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument("--device", default="", help="Single-device mode, e.g. cuda:0 or cpu.")
    parser.add_argument("--gpu", default="0", help="Legacy single GPU id fallback.")
    parser.add_argument("--gpu_ids", default="0,1,2,3", help="Comma-separated GPU ids for folder mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    extract_folder_or_file(**vars(args))


if __name__ == "__main__":
    main()
