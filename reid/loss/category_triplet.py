"""Single-category batch-hard Triplet on L2-normalized projected features."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CategoryBatchHardTriplet(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        if not math.isfinite(margin) or margin < 0:
            raise ValueError('triplet margin must be finite and non-negative')
        self.margin = float(margin)

    def forward(self, features, targets):
        if features.ndim != 2 or features.shape[0] < 2 or not features.is_floating_point():
            raise ValueError('features must be a floating [B,D] tensor')
        if targets.ndim != 1 or targets.shape[0] != features.shape[0] or targets.dtype != torch.long:
            raise ValueError('targets must be LongTensor [B]')
        if features.device != targets.device:
            raise ValueError('features and targets must be on the same device')
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            if not torch.isfinite(features).all() or (features.norm(dim=1) <= 1e-12).any():
                raise FloatingPointError('triplet features must be finite and nonzero')
            same = targets[:, None].eq(targets[None, :])
            positive = same & ~torch.eye(len(targets), device=targets.device, dtype=torch.bool)
            negative = ~same
            if not positive.any(dim=1).all() or not negative.any(dim=1).all():
                raise ValueError('every triplet anchor needs another same-identity sample and an identity negative')
            normalized = F.normalize(features, dim=1)
            distances = torch.cdist(normalized, normalized, p=2, compute_mode='donot_use_mm_for_euclid_dist')
            hardest_positive = distances.masked_fill(~positive, -torch.inf).max(dim=1).values
            hardest_negative = distances.masked_fill(~negative, torch.inf).min(dim=1).values
            loss = F.relu(hardest_positive - hardest_negative + self.margin).mean()
        return loss, {
            'positive_distance': hardest_positive.detach().mean().item(),
            'negative_distance': hardest_negative.detach().mean().item(),
            'active_triplet_fraction': (hardest_positive - hardest_negative + self.margin > 0).float().mean().item(),
        }
