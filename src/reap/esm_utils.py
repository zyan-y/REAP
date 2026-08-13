"""Utilities for loading ESM models used by REAP."""

from __future__ import annotations

import torch


def load_esm_model(model_name: str = "esm2_t33_650M_UR50D", device: str = ""):
    """Load an ESM model, preferring the installed torch.hub.
    """
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)

    model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter(), torch.device(device)
