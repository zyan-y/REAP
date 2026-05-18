#!/usr/bin/env python3
"""Design overlapping PCR primers for site-directed mutagenesis."""

from __future__ import annotations

import argparse

from reap.mutation_design import design_mutation_primers, load_fasta_sequence, read_mutations_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design overlapping PCR primers for site-directed mutagenesis.")
    parser.add_argument("--mutations_xlsx", required=True, help="Excel file containing single mutations.")
    parser.add_argument("--fasta", required=True, help="FASTA file containing the template DNA sequence.")
    parser.add_argument("--cds_start", type=int, default=None, help="Zero-based coding-sequence start in the FASTA sequence.")
    parser.add_argument(
        "--idx_pos_na",
        type=int,
        default=None,
        help="Deprecated alias for --cds_start, retained for compatibility.",
    )
    parser.add_argument("--sheet", default=0, help="Excel sheet index or name. Default: 0.")
    parser.add_argument("--codon_pref", default="codon_preference.txt", help="AA:CODON mapping file.")
    parser.add_argument("--save_path", default="design_primers.xlsx", help="Output Excel file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cds_start = args.cds_start if args.cds_start is not None else args.idx_pos_na
    if cds_start is None:
        cds_start = 0

    mutations = read_mutations_xlsx(args.mutations_xlsx, sheet=args.sheet)
    wt_seq = load_fasta_sequence(args.fasta)
    df = design_mutation_primers(
        mutations,
        wt_seq,
        args.save_path,
        codon_pref_file=args.codon_pref,
        cds_start=int(cds_start),
    )
    print(f"Saved {len(df)} primer designs to {args.save_path}")


if __name__ == "__main__":
    main()
