#!/usr/bin/env python3
"""Generate mutant plasmid FASTA files from protein mutation strings."""

from __future__ import annotations

import argparse

from reap.mutation_design import design_plasmid_batch, load_fasta_sequence, read_mutations_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mutant plasmid FASTA files from mutation strings.")
    parser.add_argument("--mutations_xlsx", required=True, help="Excel file containing mutation names.")
    parser.add_argument("--plasmid_fasta", required=True, help="FASTA file containing the wild-type plasmid sequence.")
    parser.add_argument("--output_dir", required=True, help="Directory for mutant plasmid FASTA files.")
    parser.add_argument("--enzyme", default="enzyme", help="Enzyme name used in FASTA headers.")
    parser.add_argument("--cds_start", type=int, required=True, help="Zero-based coding-sequence start in the plasmid.")
    parser.add_argument("--cds_end", type=int, required=True, help="Zero-based exclusive coding-sequence end in the plasmid.")
    parser.add_argument("--sheet", default=0, help="Excel sheet index or name. Default: 0.")
    parser.add_argument("--codon_pref", default="codon_preference.txt", help="AA:CODON mapping file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mutations = read_mutations_xlsx(args.mutations_xlsx, sheet=args.sheet)
    plasmid_seq = load_fasta_sequence(args.plasmid_fasta)
    saved_files = design_plasmid_batch(
        mutations,
        plasmid_seq,
        args.cds_start,
        args.cds_end,
        args.output_dir,
        args.enzyme,
        codon_pref_file=args.codon_pref,
    )
    print(f"Saved {len(saved_files)} mutant plasmid FASTA files to {args.output_dir}")


if __name__ == "__main__":
    main()
