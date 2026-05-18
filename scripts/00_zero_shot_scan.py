#!/usr/bin/env python3
"""Run ESM zero-shot site-saturation scanning for a wild-type enzyme."""

import argparse

from reap.zero_shot import load_esm_model, run_site_saturation_scan


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot single-mutant scanning with ESM.")
    parser.add_argument("--fasta", required=True, help="Wild-type FASTA file.")
    parser.add_argument("--output", default="results/zero_shot/site_scan.csv", help="Output CSV/XLSX path.")
    parser.add_argument("--model_name", default="esm2_t33_650M_UR50D", help="ESM model name for torch.hub.")
    parser.add_argument("--device", default="", help="Device, e.g. cuda:0 or cpu. Empty chooses automatically.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, alphabet, batch_converter, device = load_esm_model(args.model_name, args.device)
    df = run_site_saturation_scan(model, alphabet, batch_converter, device, fasta_path=args.fasta, out_path=args.output)
    print(f"Saved {len(df)} zero-shot single-mutant scores to {args.output}")


if __name__ == "__main__":
    main()
