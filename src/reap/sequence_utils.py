"""
Functions for converting between protein sequence strings and 
mutation notations (e.g., "A330Y"). Supports translation of 
variant descriptions into full-length sequences and vice versa.
"""

import os
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold


def excel2fa(excel_file, fa_file):
    """
    Read an Excel file with two columns (name, seq; order-insensitive) and write a multi-record FASTA.
    - The function tolerates column names in any case (e.g., 'Name', 'Seq').
    - Empty rows are ignored; sequences are uppercased and stripped of whitespace/newlines.
    """
    df = pd.read_excel(excel_file, header=0, usecols=[0, 1])
    cols_lower = [str(c).strip().lower() for c in df.columns]
    if "name" in cols_lower and "seq" in cols_lower:
        name_col = df.columns[cols_lower.index("name")]
        seq_col = df.columns[cols_lower.index("seq")]
        df = df[[name_col, seq_col]].rename(columns={name_col: "name", seq_col: "seq"})
    else:
        # Fall back to positional assumption: first column is name, second is seq
        df.columns = ["name", "seq"]

    df = df.dropna(subset=["name", "seq"])

    with open(fa_file, "w") as f:
        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            seq = str(row["seq"]).replace(" ", "").replace("\n", "").upper()
            if not name or not seq:
                continue
            f.write(f">{name}\n{seq}\n")


def seqs_to_short(file, first_wt=True, wt_file=""):
    """
    Convert mutant sequences to short mutation strings relative to a WT sequence.
    - If first_wt=True, the first row's second column is treated as the WT sequence.
    - Otherwise, WT is read from the provided FASTA file.
    - Exports 'muts_seq_short.xlsx' with two columns [name, mutations].
    """
    data = pd.read_excel(file, usecols=[0, 1]).astype(str).values
    if data.shape[0] < 1:
        raise ValueError("No sequences found in the provided Excel file.")

    seq_wt = data[0][1] if first_wt else str(SeqIO.read(wt_file, "fasta").seq)
    seq_wt = str(seq_wt).strip().upper()

    muts_seq = []
    for name, seq_mut in data[1:]:
        name = str(name).strip()
        seq_mut = str(seq_mut).strip().upper()
        if not name or not seq_mut:
            continue
        if len(seq_mut) != len(seq_wt):
            print(f"Length mismatch for {name}: WT={len(seq_wt)}, MUT={len(seq_mut)}; skipping.")
            continue

        muts = []
        for i, (a_mut, a_wt) in enumerate(zip(seq_mut, seq_wt), start=1):
            if a_mut != a_wt:
                muts.append(f"{a_wt}{i}{a_mut}")
        muts_seq.append([name, "-".join(muts)])

    df = pd.DataFrame(muts_seq)
    df.to_excel("muts_seq_short.xlsx", index=False, header=False)


def short_to_seqs(data, seq_wt, save_fa=True, save_fa_each=False, save_excel=False, folder="", delimiter="-"):
    """
    Build mutant sequences from short mutation strings (e.g., 'A123C-G45D').
    - On invalid tokens or coordinate mismatches, the corresponding sequence is left as WT and the entry is still emitted.
    - Exports (depending on flags):
        * 'muts_seqs.xlsx' (name, seq)
        * one FASTA per entry (if save_fa_each=True)
        * a combined 'mutants.fa' (if save_fa=True)
    """
    seq_wt = str(seq_wt).strip().upper()
    muts_seqs = []

    output_folder = Path(folder or ".")
    if save_fa or save_fa_each:
        output_folder.mkdir(parents=True, exist_ok=True)

    for name in data:
        name = str(name).strip()
        if not name:
            continue

        shorts = [s for s in str(name).split(delimiter) if s]
        mut_seq_list = list(seq_wt)
        valid = True

        for short in shorts:
            if len(short) < 3 or not short[1:-1].isdigit():
                print(f'Invalid mutation token "{short}" in {name}')
                valid = False
                break

            wt, pos, mut = short[0].upper(), int(short[1:-1]) - 1, short[-1].upper()
            if pos < 0 or pos >= len(seq_wt):
                print(f"Out-of-bounds position in {name}: {pos + 1}")
                valid = False
                break
            if seq_wt[pos] != wt:
                print(f"Coordinate mismatch for {name}")
                valid = False
                break

            mut_seq_list[pos] = mut

        mut_seq = seq_wt if not valid else "".join(mut_seq_list)
        muts_seqs.append([name, mut_seq])

    if save_excel:
        df = pd.DataFrame(muts_seqs, columns=["name", "seq"])
        df.to_excel("muts_seqs.xlsx", index=False)

    if save_fa_each:
        for name, seq in muts_seqs:
            safe = re.sub(r"[^\w.\-+]+", "_", name)
            with (output_folder / f"{safe}.fa").open("w", encoding="utf-8") as fa:
                fa.write(f">{name}\n{seq}\n")

    if save_fa:
        with (output_folder / "mutants.fa").open("w", encoding="utf-8") as fa:
            for name, seq in muts_seqs:
                fa.write(f">{name}\n{seq}\n")


def generate_all_other_single(seq_wt, seq_muts, save_df):
    """
    Generate all possible single-residue substitutions not already present in `seq_muts`.
    - `seq_muts` should contain mutation strings like 'A5V'.
    - Writes a two-column Excel file [mut, seq] to `save_df`.
    """
    seq_wt = str(seq_wt).strip().upper()
    seq_muts = {str(x).strip().upper() for x in seq_muts if str(x).strip()}
    aa = "ACDEFGHIKLMNPQRSTVWY"

    other_mut_seqs = []
    for pos, wt in enumerate(seq_wt, start=1):
        for mut in aa:
            if mut != wt:
                name = f"{wt}{pos}{mut}"
                if name not in seq_muts:
                    other_mut = seq_wt[: pos - 1] + mut + seq_wt[pos:]
                    other_mut_seqs.append([name, other_mut])

    os.makedirs(os.path.dirname(save_df) or ".", exist_ok=True)
    df = pd.DataFrame(other_mut_seqs, columns=["mut", "seq"])
    df.to_excel(save_df, index=False)


# drop NaN and assign fold index
def get_clean_data(data_file="", fold_idx=False):
    """
    Load [name, seq, yield] from Excel, drop rows with any NaN, and optionally assign a stratified 5-fold index.
    - Stratification is based on 'single' vs 'multi' (presence of '-' in the name).
    - If stratification is not feasible (e.g., only one class or too few samples), a round-robin fallback is used.
    - Writes './data.xlsx' with a standardized header.
    """
    df = pd.read_excel(data_file, header=0, index_col=None, usecols=[0, 1, 2])
    df.dropna(axis=0, how="any", inplace=True)
    df = df.reset_index(drop=True)

    header = ["name", "seq", "yield"]

    if fold_idx:
        # Derive mutation type from the first column (name)
        df["mutation_type"] = df.iloc[:, 0].astype(str).apply(lambda x: "multi" if "-" in x else "single")
        df["fold"] = -1

        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold_number, (_, val_idx) in enumerate(skf.split(df, df["mutation_type"])):
                df.loc[val_idx, "fold"] = fold_number
        except ValueError:
            # Fallback if stratification fails (e.g., only one class or too few samples)
            for i in range(len(df)):
                df.loc[i, "fold"] = i % 5

        df = df.drop("mutation_type", axis=1)
        header.append("fold")

    # Standardize the output header regardless of original column names
    df.to_excel("./data.xlsx", index=False, header=header)

