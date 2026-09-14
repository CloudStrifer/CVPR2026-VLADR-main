"""PGCA recurring-category cosine consistency and stage-fixed weights."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PGCAConsistencyConfig:
    mode: str = 'off'
    lambda_con: float = 1.0
    gamma: float = 1.0
    summary: str = 'ecpm'

    def __post_init__(self):
        from reid.memory.prototype_views import check_summary
        check_summary(self.summary)
        if self.mode not in ('off', 'fixed', 'drift'):
            raise ValueError('PGCA consistency mode must be off, fixed or drift')
        for name in ('lambda_con', 'gamma'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('{} must be finite and nonnegative'.format(name))


def consistency_weight(config, drift=None, recurring=True):
    if drift is not None and (type(drift) not in (int, float) or not math.isfinite(drift) or not 0 <= drift <= 2):
        raise ValueError('prototype drift must be finite and within [0,2]')
    if config.mode == 'off' or not recurring:
        return 0.0
    if config.mode == 'drift' and drift is None:
        raise ValueError('recurring drift mode requires an ECPM drift')
    if config.mode == 'fixed':
        return float(config.lambda_con)
    return float(config.lambda_con * math.exp(-config.gamma * drift))


class FeatureConsistencyLoss(nn.Module):
    """Mean(1-cos(student, stopgrad(teacher))), evaluated in FP32."""

    def forward(self, student, teacher):
        if (student.ndim != 2 or student.shape != teacher.shape or min(student.shape) == 0
                or student.device != teacher.device):
            raise ValueError('student/teacher features must share nonempty [B,D] shape and device')
        with torch.autocast(device_type=student.device.type, enabled=False):
            student, teacher = student.float(), teacher.detach().float()
            for name, feature in (('student', student), ('teacher', teacher)):
                norms = feature.norm(dim=1)
                if not torch.isfinite(feature).all() or not torch.isfinite(norms).all() or (norms <= 1e-8).any():
                    raise FloatingPointError('non-finite or near-zero {} consistency feature'.format(name))
            cosine = F.cosine_similarity(student, teacher, dim=1, eps=1e-8).clamp(-1., 1.)
            return (1. - cosine).mean()
