"""Loss functions used by PLM-RankReg and baseline regressors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function_baseline(loss_name: str) -> nn.Module:
    """Return a standard pointwise regression loss by name."""
    name = str(loss_name).lower()
    criterions = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "smoothl1": nn.SmoothL1Loss(),
        "huber": nn.HuberLoss(delta=1.0),
    }
    if name not in criterions:
        raise ValueError(f"Unknown loss '{loss_name}'. Choose from {sorted(criterions)} or RankReg.")
    return criterions[name]


def rank_reg_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.1,
    alpha: float = 0.5,
    num_samples: int | None = None,
    tie_threshold: float = 1e-4,
) -> torch.Tensor:
    """Hybrid ranking-regression loss.

    The pointwise term preserves quantitative accuracy, while the pairwise term
    encourages correctly ordered activity predictions. Pair sampling can be used
    for large batches to reduce memory cost.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    mse_loss = F.mse_loss(preds, targets)
    batch_size = int(preds.size(0))
    if batch_size < 2:
        return mse_loss

    indices = torch.arange(batch_size, device=preds.device)
    rows, cols = torch.combinations(indices, r=2).unbind(1)

    target_diff = targets[rows] - targets[cols]
    abs_target_diff = torch.abs(target_diff)
    scaled_margin = margin * (1.0 + abs_target_diff)

    if num_samples is not None and rows.numel() > num_samples:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive when provided.")
        weight_sum = abs_target_diff.sum()
        if weight_sum > 0:
            prob = abs_target_diff / weight_sum
            sampled = torch.multinomial(prob, num_samples, replacement=False)
        else:
            sampled = torch.randperm(rows.numel(), device=preds.device)[:num_samples]
        rows = rows[sampled]
        cols = cols[sampled]
        target_diff = target_diff[sampled]
        abs_target_diff = abs_target_diff[sampled]
        scaled_margin = scaled_margin[sampled]

    pred_diff = preds[rows] - preds[cols]

    mask_pos = (target_diff > tie_threshold).float()
    mask_neg = (target_diff < -tie_threshold).float()
    mask_tie = (abs_target_diff <= tie_threshold).float()

    loss_pos = F.relu(scaled_margin - pred_diff) * mask_pos * abs_target_diff
    loss_neg = F.relu(scaled_margin + pred_diff) * mask_neg * abs_target_diff
    loss_tie = torch.pow(pred_diff, 2) * mask_tie
    ranking_loss = (loss_pos.sum() + loss_neg.sum() + loss_tie.sum()) / max(rows.numel(), 1)

    return alpha * ranking_loss + (1.0 - alpha) * mse_loss
