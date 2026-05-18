"""ESM2 sequence-level embedding extraction."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty

import numpy as np
import pandas as pd
import torch

from .esm_utils import load_esm_model


def extract_esm2_embedding(batch, model, batch_converter, device, repr_layers: int = -1) -> np.ndarray:
    """Mean-pool token representations into one vector per sequence."""
    _, batch_strs, batch_tokens = batch_converter(batch)
    if repr_layers == -1:
        repr_layers = model.num_layers
    if repr_layers < 1 or repr_layers > model.num_layers:
        raise ValueError(f"repr_layers={repr_layers} is invalid for model with {model.num_layers} layers.")

    model.eval()
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[repr_layers], return_contacts=False)
    token_embeddings = results["representations"][repr_layers]
    sequence_embeddings = torch.stack(
        [token_embeddings[i, 1 : len(seq) + 1].mean(0) for i, seq in enumerate(batch_strs)]
    )
    return sequence_embeddings.cpu().numpy()


def get_batch_embedding(
    save_folder: str | Path,
    csv_file: str | Path,
    batch_size: int,
    *,
    device: str = "",
    model_name: str = "esm2_t33_650M_UR50D",
    repr_layers: int = -1,
    model=None,
    batch_converter=None,
) -> None:
    """Extract embeddings from one CSV file and save batch_*.npz files."""
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_file)
    data = df.iloc[:, : min(3, df.shape[1])].values
    has_label = data.shape[1] > 2

    if model is None or batch_converter is None:
        model, _, batch_converter, device_obj = load_esm_model(model_name, device)
        device = str(device_obj)

    existing = [f for f in os.listdir(save_folder) if f.startswith("batch_") and f.endswith(".npz")]
    batch_idx_start = len(existing)
    batch_num = (len(data) + batch_size - 1) // batch_size

    for idx in range(batch_num):
        save_name = save_folder / f"batch_{batch_idx_start + idx}.npz"
        if save_name.exists():
            continue
        batch_data = data[idx * batch_size : (idx + 1) * batch_size].astype(str)
        if len(batch_data) == 0:
            continue
        sequences = [(row[0], row[1]) for row in batch_data]
        X = extract_esm2_embedding(sequences, model, batch_converter, device, repr_layers)
        names = batch_data[:, 0].astype(str)
        payload = {"X": X, "n": names}
        if has_label:
            payload["y"] = batch_data[:, 2].astype(float)
        np.savez(save_name, **payload)
        print(f"Finished batch {idx + 1}/{batch_num} for {csv_file}", flush=True)

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def parse_gpu_ids(gpu_ids_str: str = "", fallback: str = "0") -> list[int]:
    text = gpu_ids_str if gpu_ids_str else fallback
    gpu_ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No valid GPU ids were provided.")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        invalid = [idx for idx in gpu_ids if idx < 0 or idx >= n_gpu]
        if invalid:
            raise ValueError(f"Invalid GPU ids {invalid}; this machine has {n_gpu} CUDA device(s).")
    return gpu_ids


def worker_process(worker_id, gpu_id, file_queue, remaining_counter, counter_lock, embed_folder, batch_size, model_name, repr_layers):
    """One worker process bound to one GPU."""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {worker_id}] start on {device}", flush=True)
    model, _, batch_converter, _ = load_esm_model(model_name, device)

    while True:
        try:
            csv_file = file_queue.get_nowait()
        except Empty:
            break
        try:
            file_name = os.path.basename(csv_file)
            save_folder = Path(embed_folder) / Path(file_name).stem
            get_batch_embedding(save_folder, csv_file, batch_size, device=device, model_name=model_name, repr_layers=repr_layers, model=model, batch_converter=batch_converter)
            with counter_lock:
                remaining_counter.value -= 1
                print(f"[Worker {worker_id}] finished {file_name}; remaining {remaining_counter.value}", flush=True)
        except Exception as exc:
            with counter_lock:
                remaining_counter.value -= 1
                print(f"[Worker {worker_id}] failed {os.path.basename(csv_file)}: {exc}", flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def extract_folder_or_file(
    *,
    dms_file: str = "",
    dms_folder: str = "",
    embed_folder: str = "embeddings",
    batch_size: int = 256,
    device: str = "",
    gpu_ids: str = "0,1,2,3",
    gpu: str = "0",
    model_name: str = "esm2_t33_650M_UR50D",
    repr_layers: int = -1,
) -> None:
    """CLI-friendly wrapper for single-file or folder embedding extraction."""
    embed_folder = str(embed_folder)
    Path(embed_folder).mkdir(parents=True, exist_ok=True)
    if dms_file:
        files = [dms_file]
    else:
        if not dms_folder:
            raise ValueError("Provide either dms_file or dms_folder.")
        files = sorted(str(p) for p in Path(dms_folder).glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files found in {dms_folder}.")

    if dms_file or device or not torch.cuda.is_available():
        run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        for csv_file in files:
            save_folder = Path(embed_folder) / Path(csv_file).stem
            get_batch_embedding(save_folder, csv_file, batch_size, device=run_device, model_name=model_name, repr_layers=repr_layers)
        return

    ids = parse_gpu_ids(gpu_ids, gpu)
    num_workers = min(len(ids), len(files))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for csv_file in files:
        queue.put(csv_file)
    remaining = ctx.Value("i", len(files))
    lock = ctx.Lock()
    processes = []
    for worker_id, gpu_id in enumerate(ids[:num_workers]):
        p = ctx.Process(target=worker_process, args=(worker_id, gpu_id, queue, remaining, lock, embed_folder, batch_size, model_name, repr_layers))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
