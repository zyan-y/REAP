"""Utilities for loading ESM models used by REAP."""

from __future__ import annotations

import torch


def load_esm_model(model_name: str = "esm2_t33_650M_UR50D", device: str = ""):
    """Load an ESM model, preferring the installed fair-esm package.

    The PyPI package is named ``fair-esm`` and exposes the Python module
    ``esm``. If it is not installed, this function falls back to torch.hub.
    """
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import esm  # type: ignore

        if not hasattr(esm.pretrained, model_name):
            raise AttributeError(f"esm.pretrained has no model named {model_name!r}.")
        model_fn = getattr(esm.pretrained, model_name)
        model, alphabet = model_fn()
    except Exception:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)

    model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter(), torch.device(device)
