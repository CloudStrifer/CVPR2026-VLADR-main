from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearningWithoutForgettingLoss(nn.Module):
    """Temperature-scaled logit distillation used by LwF.

    The teacher logits are detached internally so the frozen previous model
    never receives gradients. Inputs are promoted to float32 for numerical
    stability when CLIP runs in mixed precision.
    """

    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = float(temperature)
        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0.0
        ):
            raise ValueError('LwF temperature must be finite and positive')

    def forward(self, student_logits, teacher_logits):
        if student_logits.ndim != 2 or teacher_logits.ndim != 2:
            raise ValueError('LwF logits must both have shape [B, C]')
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                'LwF student/teacher shapes must match, got {} and {}'
                .format(
                    tuple(student_logits.shape),
                    tuple(teacher_logits.shape),
                )
            )
        if student_logits.size(1) == 0:
            return student_logits.sum() * 0.0
        if not torch.isfinite(student_logits).all():
            raise FloatingPointError('LwF student logits are not finite')
        if not torch.isfinite(teacher_logits).all():
            raise FloatingPointError('LwF teacher logits are not finite')

        temperature = self.temperature
        student_log_prob = F.log_softmax(
            student_logits.float() / temperature,
            dim=1,
        )
        teacher_prob = F.softmax(
            teacher_logits.detach().float() / temperature,
            dim=1,
        )
        return F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction='batchmean',
        ) * (temperature ** 2)
