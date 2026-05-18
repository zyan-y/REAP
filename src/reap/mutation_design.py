"""Primer and plasmid design utilities for mutation construction."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
DNA_RE = re.compile(r"^[ACGT]+$")
TARGET_TM = (56.0, 60.0)
MIN_PRIMER_LEN = 16
MAX_PRIMER_LEN = 30
OVERLAP_LEFT = 10
RIGHT_EXTRA_AFTER_CODON = 7

STANDARD_DNA_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def parse_sheet_name(value):
    """Return an integer sheet index when possible; otherwise return the sheet name."""
    if isinstance(value, int):
        return value
    value = str(value)
    return int(value) if value.isdigit() else value


def read_mutations_xlsx(path: str | Path, sheet=0) -> list[str]:
    """Read mutation names from an Excel sheet.

    Preferred columns are ``name``, ``mutation``, ``mut``, or ``variant``. If
    none is present, the first column is used.
    """
    df = pd.read_excel(path, sheet_name=parse_sheet_name(sheet), header=0)
    if df.empty:
        return []
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for key in ("name", "mutation", "mut", "variant"):
        if key in lowered:
            series = df[lowered[key]]
            break
    else:
        series = df.iloc[:, 0]
    return [str(x).strip().upper() for x in series.dropna().tolist() if str(x).strip()]


def load_fasta_sequence(path: str | Path) -> str:
    """Read a single FASTA record as an uppercase string."""
    return str(SeqIO.read(path, "fasta").seq).upper()


def load_codon_preferences(codon_pref_file: str | Path) -> dict[str, str]:
    """Load an amino-acid-to-codon preference file with lines such as A:GCT."""
    codon_dict: dict[str, str] = {}
    path = Path(codon_pref_file)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise RuntimeError(f"Failed to read codon preferences from '{codon_pref_file}': {exc}") from exc

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid codon preference line {line_no}: {raw_line!r}. Expected AA:CODON.")
        aa, codon = [x.strip().upper() for x in line.split(":", 1)]
        if len(aa) != 1 or aa not in "ACDEFGHIKLMNPQRSTVWY":
            raise ValueError(f"Invalid amino-acid code on line {line_no}: {aa!r}.")
        if len(codon) != 3 or not DNA_RE.match(codon):
            raise ValueError(f"Invalid DNA codon on line {line_no}: {codon!r}.")
        if STANDARD_DNA_CODE.get(codon) != aa:
            raise ValueError(f"Codon {codon!r} on line {line_no} does not encode amino acid {aa!r}.")
        codon_dict[aa] = codon

    if not codon_dict:
        raise ValueError(f"No codon preferences were loaded from {codon_pref_file}.")
    return codon_dict


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """Parse a single mutation string such as A123C using 1-based protein coordinates."""
    match = MUTATION_RE.match(str(mutation).strip().upper())
    if match is None:
        raise ValueError(f"Invalid mutation string: '{mutation}'. Expected format such as A123C.")
    wt_aa, pos_aa, mut_aa = match.groups()
    return wt_aa, int(pos_aa), mut_aa


def split_mutation_name(mutation_name: str) -> list[str]:
    """Split a mutation name into validated single-mutation tokens."""
    tokens = [x for x in str(mutation_name).strip().upper().split("-") if x]
    if not tokens:
        raise ValueError("Empty mutation name.")
    for token in tokens:
        parse_mutation(token)
    return tokens


def safe_filename(name: str) -> str:
    """Return a safe filename stem for a mutation name."""
    return re.sub(r"[^\w.\-+]+", "_", str(name).strip())


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return str(seq).upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


def check_wild_type_codon(seq: str, pos_na: int, wt_aa: str, mutation: str) -> str:
    """Return the WT codon and fail if the coordinate does not match the WT residue."""
    codon = str(seq[pos_na : pos_na + 3]).upper()
    observed_aa = STANDARD_DNA_CODE.get(codon)
    if observed_aa is None:
        warnings.warn(f"Codon for {mutation} contains non-standard bases: {codon}.")
    elif observed_aa != wt_aa:
        raise ValueError(
            f"Coordinate mismatch for {mutation}: codon {codon} at nt {pos_na} encodes {observed_aa}, not {wt_aa}."
        )
    return codon


def estimate_tm(seq: str) -> float:
    """Estimate melting temperature with primer3-py."""
    try:
        import primer3  # type: ignore
    except ImportError as exc:
        raise ImportError("primer3-py is required for primer design. Install it with `pip install primer3-py`.") from exc
    return float(primer3.calc_tm(str(seq)))


def estimate_tm_rc(seq: str) -> float:
    """Estimate Tm after reverse-complementing the input sequence."""
    return estimate_tm(reverse_complement(seq))


def design_one_mutation_primer(
    wt_seq: str,
    mutation: str,
    codon_dict: dict[str, str],
    cds_start: int = 0,
) -> list:
    """Design a primer pair for one amino-acid substitution."""
    if "-" in str(mutation):
        raise ValueError("Primer design expects one single mutation per row, e.g. A123C.")

    wt_seq = str(wt_seq).strip().upper()
    wt_aa, pos_aa, mut_aa = parse_mutation(mutation)
    if mut_aa not in codon_dict:
        raise KeyError(f"Missing codon preference for {mut_aa}.")

    pos_na = int(cds_start) + (pos_aa - 1) * 3
    if pos_na < 0 or pos_na + 3 > len(wt_seq):
        raise ValueError(f"Mutation {mutation} maps outside the provided DNA sequence.")
    check_wild_type_codon(wt_seq, pos_na, wt_aa, mutation)

    left = pos_na - OVERLAP_LEFT
    right = pos_na + 3 + RIGHT_EXTRA_AFTER_CODON
    min_right = pos_na + 3 + MIN_PRIMER_LEN
    if left < 0 or right > len(wt_seq):
        raise ValueError(f"Mutation {mutation} is too close to the sequence boundary for a 20-nt overlap.")
    if pos_na < MIN_PRIMER_LEN or min_right > len(wt_seq):
        raise ValueError(f"Mutation {mutation} is too close to the sequence boundary for primer binding segments.")

    mut_seq = wt_seq[:pos_na] + codon_dict[mut_aa] + wt_seq[pos_na + 3 :]
    primer_overlap = mut_seq[left:right]

    primer_f_len = 20
    primer_f_bind = mut_seq[pos_na + 3 : pos_na + 3 + primer_f_len]
    while estimate_tm(primer_f_bind) < TARGET_TM[0] and primer_f_len < MAX_PRIMER_LEN:
        primer_f_len += 1
        primer_f_bind = mut_seq[pos_na + 3 : pos_na + 3 + primer_f_len]
    while estimate_tm(primer_f_bind) > TARGET_TM[1] and primer_f_len > MIN_PRIMER_LEN:
        primer_f_len -= 1
        primer_f_bind = mut_seq[pos_na + 3 : pos_na + 3 + primer_f_len]

    primer_r_len = 20
    primer_r_bind = mut_seq[pos_na - primer_r_len : pos_na]
    while estimate_tm_rc(primer_r_bind) < TARGET_TM[0] and primer_r_len < MAX_PRIMER_LEN and pos_na - primer_r_len > 0:
        primer_r_len += 1
        primer_r_bind = mut_seq[pos_na - primer_r_len : pos_na]
    while estimate_tm_rc(primer_r_bind) > TARGET_TM[1] and primer_r_len > MIN_PRIMER_LEN:
        primer_r_len -= 1
        primer_r_bind = mut_seq[pos_na - primer_r_len : pos_na]

    primer_f = primer_overlap + primer_f_bind[RIGHT_EXTRA_AFTER_CODON:]
    primer_r = reverse_complement(primer_r_bind + mut_seq[pos_na : pos_na + 3 + RIGHT_EXTRA_AFTER_CODON])

    return [
        mutation,
        pos_na,
        primer_f,
        primer_r,
        primer_f_bind,
        reverse_complement(primer_r_bind),
        estimate_tm(primer_f_bind),
        estimate_tm_rc(primer_r_bind),
    ]


def design_mutation_primers(
    mutations: list[str],
    wt_seq: str,
    save_path: str | Path,
    codon_pref_file: str | Path = "codon_preference.txt",
    cds_start: int = 0,
) -> pd.DataFrame:
    """Design primers for a list of single mutations and save an Excel report."""
    codon_dict = load_codon_preferences(codon_pref_file)
    rows = [design_one_mutation_primer(wt_seq, str(mut).strip().upper(), codon_dict, cds_start) for mut in mutations]
    columns = [
        "mutation",
        "mutation_start_nt",
        "primer_forward",
        "primer_reverse",
        "forward_binding_segment",
        "reverse_binding_segment",
        "tm_forward",
        "tm_reverse",
    ]
    df = pd.DataFrame(rows, columns=columns)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False)
    return df


def design_one_mutant_plasmid(
    coding_seq: str,
    mutation_name: str,
    codon_dict: dict[str, str],
    plasmid_seq: str,
    cds_start: int,
    cds_end: int,
) -> str:
    """Return a full plasmid sequence carrying one single- or multi-mutant design."""
    coding_seq = str(coding_seq).strip().upper()
    plasmid_seq = str(plasmid_seq).strip().upper()
    mutant_coding = coding_seq

    for mutation in split_mutation_name(mutation_name):
        wt_aa, pos_aa, mut_aa = parse_mutation(mutation)
        if mut_aa not in codon_dict:
            raise KeyError(f"Missing codon preference for {mut_aa}.")

        pos_na = (pos_aa - 1) * 3
        if pos_na < 0 or pos_na + 3 > len(mutant_coding):
            raise ValueError(f"Mutation {mutation} maps outside the coding sequence.")
        check_wild_type_codon(coding_seq, pos_na, wt_aa, mutation)
        mutant_coding = mutant_coding[:pos_na] + codon_dict[mut_aa] + mutant_coding[pos_na + 3 :]

    return plasmid_seq[:cds_start] + mutant_coding + plasmid_seq[cds_end:]


def design_plasmid_batch(
    mutations: list[str],
    plasmid_seq: str,
    cds_start: int,
    cds_end: int,
    output_dir: str | Path,
    enzyme: str,
    codon_pref_file: str | Path = "codon_preference.txt",
) -> list[Path]:
    """Generate one FASTA file per requested mutant plasmid."""
    plasmid_seq = str(plasmid_seq).strip().upper()
    if cds_start < 0 or cds_end <= cds_start or cds_end > len(plasmid_seq):
        raise ValueError("Invalid coding-sequence coordinates. Use zero-based half-open coordinates.")
    if (cds_end - cds_start) % 3 != 0:
        raise ValueError("The coding-sequence length must be divisible by 3.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codon_dict = load_codon_preferences(codon_pref_file)
    coding_seq = plasmid_seq[cds_start:cds_end]

    saved_files: list[Path] = []
    for mutation_name in mutations:
        mutation_name = str(mutation_name).strip().upper()
        if not mutation_name:
            continue
        mutant_plasmid = design_one_mutant_plasmid(coding_seq, mutation_name, codon_dict, plasmid_seq, cds_start, cds_end)
        output_file = out_dir / f"{safe_filename(mutation_name)}.fa"
        with output_file.open("w", encoding="utf-8") as handle:
            handle.write(f">plasmid-{enzyme}-{mutation_name}\n{mutant_plasmid}\n")
        saved_files.append(output_file)
    return saved_files
