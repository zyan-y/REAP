import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_function_baseline(loss_name):
    criterions = {
            'MSE': nn.MSELoss(),
            'SmoothL1': nn.SmoothL1Loss(),
            'Huber': nn.HuberLoss(delta=1.0),
            }
    if loss_name in criterions:
        return criterions[loss_name]
    else:
        raise ValueError(f"Loss function '{loss_name}' is not defined. Choose from {list(criterions.keys())}")


def rank_reg_loss(preds, targets, margin=0.1, alpha=0.5, num_samples=None):
    batch_size = preds.size(0)
    rows, cols = torch.combinations(torch.arange(batch_size), r=2).unbind(1)
    target_diff = targets[rows] - targets[cols]
    abs_target_diff = torch.abs(target_diff)
    scaled_margin = margin * (1 + abs_target_diff)

    if num_samples and len(rows) > num_samples:
        prob = abs_target_diff / abs_target_diff.sum()
        indices = torch.multinomial(prob, num_samples, replacement=False)
        rows, cols = rows[indices], cols[indices]
        target_diff = target_diff[indices]
        abs_target_diff = abs_target_diff[indices]
        scaled_margin = scaled_margin[indices]
    
    pred_diff = preds[rows] - preds[cols]

    mask_pos = (target_diff > 0).float()
    mask_neg = (target_diff < 0).float()
    mask_tie = (abs_target_diff < 1e-4).float()
    
    loss_pos = F.relu(scaled_margin - pred_diff) * mask_pos * abs_target_diff
    loss_neg = F.relu(scaled_margin + pred_diff) * mask_neg * abs_target_diff
    loss_tie = torch.pow(pred_diff, 2) * mask_tie
    
    ranking_loss = (loss_pos.sum() + loss_neg.sum() + loss_tie.sum()) / len(rows)
    mse_loss = F.mse_loss(preds, targets)
    
    return alpha * ranking_loss + (1 - alpha) * mse_loss


