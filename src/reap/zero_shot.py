"""Zero-shot single-mutant scanning with ESM-style masked language models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

from .esm_utils import load_esm_model

DEFAULT_AA_ORDER = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]


def chunks(items, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_intervals_and_weights(seq_len: int, min_overlap: int = 512, max_len: int = 1022, s: int = 20):
    """Create overlapping windows and simple averaging weights for long sequences."""
    if seq_len <= max_len:
        idx = np.arange(seq_len)
        return [idx], np.ones((1, seq_len)), np.ones((1, seq_len))
    step = max_len - min_overlap
    if step <= 0:
        raise ValueError("max_len must be greater than min_overlap.")
    intervals = []
    start = 0
    while start < seq_len:
        end = min(start + max_len, seq_len)
        intervals.append(np.arange(start, end))
        if end == seq_len:
            break
        start += step
    M = np.zeros((len(intervals), seq_len), dtype=float)
    for i, idx in enumerate(intervals):
        M[i, idx] = 1.0
    denom = M.sum(axis=0)
    denom[denom == 0] = 1.0
    M_norm = M / denom
    return intervals, M, M_norm


# Method follows the ESM zero-shot LLR calculation commonly used for mutation-effect prediction.
def get_wt_llr(input_df, model, alphabet, batch_converter, device=0, silent: bool = False):
    """Compute wild-type log-likelihood ratio maps for input sequences."""
    llrs = []
    input_df_ids = []
    for gname in tqdm(input_df.id.values, disable=silent):
        wt_seq = input_df.loc[input_df.id == gname, "seq"].values[0]
        seq_length = int(input_df.loc[input_df.id == gname, "length"].values[0])

        if seq_length <= 1022:
            dt = [(f"{gname}_WT", wt_seq)]
            _, _, batch_tokens = batch_converter(dt)
            with torch.no_grad():
                results = torch.log_softmax(model(batch_tokens.to(device), return_contacts=False)["logits"], dim=-1)
            logits = results[0, :, :].cpu().numpy()[1:-1, :]
        else:
            intervals, _, M_norm = get_intervals_and_weights(seq_length, min_overlap=512, max_len=1022, s=20)
            dt = [(f"{gname}_WT_{i}", "".join(np.asarray(list(wt_seq))[idx])) for i, idx in enumerate(intervals)]
            logit_parts = []
            for batch in chunks(dt, 20):
                _, _, batch_tokens = batch_converter(batch)
                with torch.no_grad():
                    results = torch.log_softmax(model(batch_tokens.to(device), return_contacts=False)["logits"], dim=-1)
                for i in range(results.shape[0]):
                    logit_parts.append(results[i, :, :].cpu().numpy()[1:-1, :])
            logits = np.zeros((seq_length, len(alphabet.all_toks)))
            for i, idx in enumerate(intervals):
                tmp = np.zeros((seq_length, len(alphabet.all_toks)))
                tmp[idx] = logit_parts[i]
                logits += (tmp.T * M_norm[i, :]).T

        wt_logits = pd.DataFrame(logits, columns=alphabet.all_toks, index=list(wt_seq)).T.iloc[4:24].loc[DEFAULT_AA_ORDER]
        wt_logits.columns = [f"{aa} {i + 1}" for i, aa in enumerate(wt_seq)]
        wt_norm = np.diag(wt_logits.loc[[x.split(" ")[0] for x in wt_logits.columns]])
        llr = wt_logits - wt_norm
        llrs.append(llr)
        input_df_ids.append(gname)
    return input_df_ids, llrs


def build_llr_map(wt_seq: str, model, alphabet, batch_converter, device):
    if not isinstance(wt_seq, str) or len(wt_seq) == 0:
        raise ValueError("wt_seq must be a non-empty string.")
    df_llr = pd.DataFrame([("P1", "protein", wt_seq, len(wt_seq))], columns=["id", "gene", "seq", "length"])
    model.eval()
    with torch.no_grad():
        _, llr_list = get_wt_llr(df_llr, model, alphabet, batch_converter, device=device)
    if not llr_list or not isinstance(llr_list[0], pd.DataFrame):
        raise RuntimeError("LLR calculation did not return a valid DataFrame.")
    return llr_list[0]


def site_saturation_scan(wt_seq: str, llr_map: pd.DataFrame, amino_acids=None) -> pd.DataFrame:
    """Score all non-self single substitutions in mutation notation, e.g. A12G."""
    amino_acids = amino_acids or DEFAULT_AA_ORDER
    rows = []
    for i, wt_aa in enumerate(wt_seq):
        header = f"{wt_aa} {i + 1}"
        if header not in llr_map.columns:
            raise KeyError(f"LLR map missing position header: {header}")
        for aa in amino_acids:
            if aa == wt_aa:
                continue
            rows.append((f"{wt_aa}{i + 1}{aa}", float(llr_map.loc[aa, header])))
    return pd.DataFrame(rows, columns=["mutation", "score"])


def save_scan(df_scan: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".xlsx", ".xls"}:
        df_scan.to_excel(out_path, index=False)
    else:
        df_scan.to_csv(out_path, index=False)


def run_site_saturation_scan(model, alphabet, batch_converter, device, fasta_path="wt.fa", out_path="site_scan.csv", amino_acids=None):
    wt_seq = str(SeqIO.read(fasta_path, "fasta").seq)
    llr_map = build_llr_map(wt_seq, model, alphabet, batch_converter, device)
    df_scan = site_saturation_scan(wt_seq, llr_map, amino_acids=amino_acids)
    save_scan(df_scan, out_path)
    return df_scan
