"""Cleaning utilities for experimental assay results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_assay_file(
    input_excel: str | Path,
    cleaned_dir: str | Path,
    *,
    cleaned_filename: str = "cleaned.xlsx",
    name_col: str = "name",
    replicate_cols: tuple[str, ...] = ("replicate1", "replicate2", "replicate3"),
    output_label_col: str = "yield",
) -> tuple[Path, pd.DataFrame]:
    """Average replicate columns into a single activity/yield label."""
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(input_excel)
    required = [name_col, *replicate_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_excel}: {missing}")

    corr = df[list(replicate_cols)].astype(float).corr(method="pearson")
    print("Pearson correlations between replicates:")
    print(corr.to_string(float_format=lambda x: f"{x:0.4f}"))

    df_out = pd.DataFrame(
        {
            "name": df[name_col].astype(str),
            output_label_col: df[list(replicate_cols)].astype(float).mean(axis=1, skipna=True),
        }
    )
    out_path = cleaned_dir / cleaned_filename
    df_out.to_excel(out_path, index=False, sheet_name="cleaned")
    return out_path, corr


def merge_cleaned_folder(
    cleaned_dir: str | Path,
    *,
    merged_filename: str = "data.xlsx",
    name_col: str = "name",
    label_col: str = "yield",
) -> Path:
    """Merge cleaned assay files into one training table."""
    cleaned_dir = Path(cleaned_dir)
    files = sorted(cleaned_dir.glob("*.xlsx"))
    merged = []
    for file_path in files:
        if file_path.name == merged_filename:
            continue
        try:
            df = pd.read_excel(file_path)
            merged.append(df[[name_col, label_col]])
        except Exception as exc:
            print(f"[Warn] Skipping {file_path}: {exc}")
    if not merged:
        raise RuntimeError(f"No valid cleaned files were found in {cleaned_dir}.")
    out = pd.concat(merged, axis=0, ignore_index=True).dropna(subset=[name_col, label_col])
    out_path = cleaned_dir / merged_filename
    out.to_excel(out_path, index=False, sheet_name="merged")
    return out_path
