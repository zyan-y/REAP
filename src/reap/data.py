"""Data loading, splitting, and small utility functions for REAP."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Keep this utility usable in environments where torch is not installed.
        pass


def parse_int_list(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_str_list(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def get_batch_index(file_name: str) -> int | None:
    match = re.match(r"batch_(\d+)\.npz$", Path(file_name).name)
    return int(match.group(1)) if match else None


def list_batch_files(embeddings_folder: str | os.PathLike) -> list[Path]:
    folder = Path(embeddings_folder)
    batch_files: list[tuple[int, Path]] = []
    for file_path in folder.glob("batch_*.npz"):
        idx = get_batch_index(file_path.name)
        if idx is not None:
            batch_files.append((idx, file_path))
    files = [p for _, p in sorted(batch_files, key=lambda x: x[0])]
    if not files:
        raise FileNotFoundError(f"No batch_*.npz files found in {folder}")
    return files


def _to_str_array(values: Iterable) -> np.ndarray:
    out = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return np.asarray(out, dtype=object)


def load_embeddings(
    embeddings_folder: str | os.PathLike,
    *,
    return_names: bool = False,
    require_y: bool = True,
    deduplicate_by_name: bool = False,
    delete_name: str | None = None,
):
    """Load batch-wise PLM embeddings saved as batch_*.npz.

    Expected keys are X and optionally y and n. Candidate libraries used for
    prediction can omit y by setting require_y=False. When return_names=True,
    the returned tuple includes names as the last element.
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    n_list: list[np.ndarray] = []
    saw_y = False
    saw_names = False

    for file_path in list_batch_files(embeddings_folder):
        data = np.load(file_path, allow_pickle=True)
        if "X" not in data:
            raise KeyError(f"Missing key 'X' in {file_path}")
        X = np.asarray(data["X"])
        X_list.append(X)

        if "y" in data:
            y = np.asarray(data["y"])
            if len(y) != X.shape[0]:
                raise ValueError(f"Length mismatch in {file_path}: len(y)={len(y)} X={X.shape[0]}")
            y_list.append(y)
            saw_y = True
        elif require_y:
            raise KeyError(f"Missing key 'y' in {file_path}; use require_y=False for unlabeled candidates.")

        if "n" in data:
            names = _to_str_array(data["n"])
            if len(names) != X.shape[0]:
                raise ValueError(f"Length mismatch in {file_path}: len(n)={len(names)} X={X.shape[0]}")
            n_list.append(names)
            saw_names = True
        elif return_names:
            raise KeyError(f"Missing key 'n' in {file_path}; names are required by return_names=True.")

    if saw_y and len(y_list) != len(X_list):
        raise ValueError("Embedding batches mix labeled and unlabeled files; keep training and candidate embeddings in separate folders.")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_list, axis=0).astype(np.float32, copy=False) if saw_y else None
    n_all = np.concatenate(n_list, axis=0) if saw_names else None

    if delete_name and n_all is not None:
        keep = n_all != delete_name
        X_all = X_all[keep]
        if y_all is not None:
            y_all = y_all[keep]
        n_all = n_all[keep]

    if deduplicate_by_name:
        if n_all is None:
            raise ValueError("deduplicate_by_name=True requires names stored under key 'n'.")
        by_name = {}
        for idx, name in enumerate(n_all):
            by_name[str(name)] = idx
        keep_idx = np.asarray(list(by_name.values()), dtype=int)
        X_all = X_all[keep_idx]
        if y_all is not None:
            y_all = y_all[keep_idx]
        n_all = n_all[keep_idx]

    if return_names:
        return X_all, y_all, n_all
    if require_y:
        return X_all, y_all
    return X_all, y_all


def load_embeddings_as_dict(embeddings_folder: str | os.PathLike) -> dict[str, np.ndarray]:
    """Load unlabeled embeddings as a name-to-vector dictionary."""
    X, _, names = load_embeddings(embeddings_folder, return_names=True, require_y=False)
    return {str(name): X[i] for i, name in enumerate(names)}


def align_embeddings_to_table(
    embeddings_folder: str | os.PathLike,
    table_file: str | os.PathLike,
    *,
    name_col: str = "name",
    label_col: str = "yield",
    skip_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align embeddings to an experimental table containing name and label columns."""
    table_path = Path(table_file)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path)
    else:
        df = pd.read_csv(table_path)
    if name_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Input table must contain columns '{name_col}' and '{label_col}'.")

    name_to_embedding = load_embeddings_as_dict(embeddings_folder)
    names = df[name_col].astype(str).tolist()
    labels = df[label_col].to_numpy(dtype=np.float32)

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    kept_names: list[str] = []
    missing: list[str] = []

    for name, label in zip(names, labels):
        if name not in name_to_embedding:
            missing.append(name)
            if skip_missing:
                continue
            raise KeyError(f"Missing embedding for '{name}'.")
        X_rows.append(name_to_embedding[name])
        y_rows.append(float(label))
        kept_names.append(name)

    if not X_rows:
        raise RuntimeError("No rows remained after alignment.")
    return np.stack(X_rows, axis=0), np.asarray(y_rows, dtype=np.float32), np.asarray(kept_names, dtype=object), missing


def save_embedding_batches(
    X: np.ndarray,
    y: np.ndarray | None,
    names: Iterable[str],
    out_dir: str | os.PathLike,
    *,
    batch_size: int | None = None,
    batch_idx_start: int = 0,
) -> None:
    """Save aligned embeddings to batch_*.npz files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = np.asarray(list(names), dtype=object)
    if batch_size is None or batch_size <= 0:
        batch_size = len(names)

    batch_idx = batch_idx_start
    for start in range(0, len(names), batch_size):
        end = min(start + batch_size, len(names))
        payload = {"X": X[start:end], "n": names[start:end]}
        if y is not None:
            payload["y"] = y[start:end]
        np.savez(out / f"batch_{batch_idx}.npz", **payload)
        batch_idx += 1


def normalize_dataset_name(name: str) -> str:
    base = os.path.basename(str(name).strip())
    return os.path.splitext(base)[0] if base.endswith(".csv") else base


def load_requested_datasets(datasets_arg: str = "", dataset_list_file: str = "") -> list[str]:
    requested: list[str] = []
    requested.extend(normalize_dataset_name(x) for x in parse_str_list(datasets_arg))
    if dataset_list_file:
        with open(dataset_list_file, "r", encoding="utf-8") as handle:
            requested.extend(normalize_dataset_name(line) for line in handle if line.strip())
    return sorted(set(requested))


def standardize_by_train(*arrays: np.ndarray) -> tuple[list[np.ndarray], float, float]:
    """Standardize arrays using the mean and standard deviation of the first array."""
    if not arrays:
        raise ValueError("At least one array is required.")
    mean = float(np.mean(arrays[0]))
    std = float(np.std(arrays[0]))
    if std < 1e-12:
        std = 1.0
    return [(arr - mean) / std for arr in arrays], mean, std
