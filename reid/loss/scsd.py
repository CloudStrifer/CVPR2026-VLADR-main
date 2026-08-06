"""Semantic Compatibility-Aware Selective Distillation (SCSD)."""

from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DISTILLATION_MODES = (
    'none',
    'index',
    'semantic-hard',
    'semantic-soft',
    'scsd',
)
KL_DIRECTIONS = ('old_to_new', 'paper')


def attribute_relation_matrix(
    attribute_features,
    temperature=0.07,
    mask_diagonal=False,
):
    """Return row-normalized per-attribute relations with shape [P, B, B]."""

    if attribute_features.ndim != 3:
        raise ValueError(
            'attribute_features must have shape [B, P, D], got {}'.format(
                tuple(attribute_features.shape)
            )
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError('relation temperature must be finite and positive')
    if attribute_features.size(0) == 0 or attribute_features.size(1) == 0:
        raise ValueError('attribute features need a non-empty batch and slots')

    features = F.normalize(attribute_features.float(), dim=-1)
    logits = torch.einsum('bpd,cpd->pbc', features, features)
    logits = logits / temperature
    if mask_diagonal and logits.size(-1) > 1:
        diagonal = torch.eye(
            logits.size(-1),
            device=logits.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    return F.softmax(logits, dim=-1)


def semantic_compatibility(old_text_features, current_text_features):
    """Return frozen-CLIP cosine similarities with shape [P_old, P_new]."""

    if old_text_features.ndim != 2 or current_text_features.ndim != 2:
        raise ValueError('attribute text features must both be 2-D tensors')
    if old_text_features.size(1) != current_text_features.size(1):
        raise ValueError('old/current attribute text dimensions do not match')
    old_norm = F.normalize(old_text_features.float(), dim=-1)
    current_norm = F.normalize(current_text_features.float(), dim=-1)
    return old_norm @ current_norm.t()


def semantic_transfer_matrix(similarity, mode, threshold):
    """Build old-to-current routing weights and per-current confidence."""

    if mode not in DISTILLATION_MODES:
        raise ValueError('unsupported attribute distillation mode {!r}'.format(mode))
    if similarity.ndim != 2:
        raise ValueError('similarity must have shape [P_old, P_new]')
    if not math.isfinite(float(threshold)):
        raise ValueError('semantic threshold must be finite')

    old_count, current_count = similarity.shape
    routing = similarity.new_zeros(old_count, current_count)
    confidence = similarity.new_zeros(current_count)
    if mode == 'none':
        return routing, confidence
    if mode == 'index':
        matched = min(old_count, current_count)
        indices = torch.arange(matched, device=similarity.device)
        routing[indices, indices] = 1.0
        confidence[:matched] = 1.0
        return routing, confidence

    gated = F.relu(similarity - float(threshold))
    best_values, best_indices = gated.max(dim=0)
    valid = best_values > 0
    if mode == 'semantic-hard':
        columns = torch.arange(current_count, device=similarity.device)
        routing[best_indices[valid], columns[valid]] = 1.0
        confidence[valid] = 1.0
        return routing, confidence

    denominators = gated.sum(dim=0)
    routing = gated / denominators.clamp_min(torch.finfo(gated.dtype).eps)
    routing[:, ~valid] = 0.0
    if mode == 'semantic-soft':
        confidence[valid] = 1.0
    elif mode == 'scsd':
        confidence = best_values
    else:
        raise ValueError('mode {!r} does not define semantic routing'.format(mode))
    return routing, confidence


def relation_kl_divergence(
    teacher_relations,
    student_relations,
    direction='old_to_new',
    eps=1e-8,
):
    if direction not in KL_DIRECTIONS:
        raise ValueError('unsupported KL direction {!r}'.format(direction))
    if teacher_relations.shape != student_relations.shape:
        raise ValueError('teacher/student relation shapes do not match')
    teacher = teacher_relations.detach().clamp_min(eps)
    student = student_relations.clamp_min(eps)
    if direction == 'old_to_new':
        divergence = teacher * (teacher.log() - student.log())
    else:
        divergence = student * (student.log() - teacher.log())
    return divergence.sum(dim=-1).mean(dim=-1)


def global_relation_distillation(
    old_global_features,
    current_global_features,
    temperature=0.07,
    kl_direction='old_to_new',
    mask_diagonal=True,
):
    """Preserve the teacher's batch relation in the global CLIP space."""

    if old_global_features.ndim != 2 or current_global_features.ndim != 2:
        raise ValueError('global features must both have shape [B, D]')
    if old_global_features.shape != current_global_features.shape:
        raise ValueError('teacher/student global feature shapes do not match')
    old_relations = attribute_relation_matrix(
        old_global_features.detach().unsqueeze(1),
        temperature=temperature,
        mask_diagonal=mask_diagonal,
    )[0]
    current_relations = attribute_relation_matrix(
        current_global_features.unsqueeze(1),
        temperature=temperature,
        mask_diagonal=mask_diagonal,
    )[0]
    return relation_kl_divergence(
        old_relations,
        current_relations,
        direction=kl_direction,
    )


class SCSDLoss(nn.Module):
    """Relation distillation with semantic routing and confidence gating."""

    def __init__(
        self,
        mode='scsd',
        semantic_threshold=0.70,
        relation_temperature=0.07,
        kl_direction='old_to_new',
        mask_diagonal=True,
        eps=1e-8,
    ):
        super().__init__()
        if mode not in DISTILLATION_MODES:
            raise ValueError('unsupported attribute distillation mode {!r}'.format(mode))
        if kl_direction not in KL_DIRECTIONS:
            raise ValueError('unsupported KL direction {!r}'.format(kl_direction))
        self.mode = mode
        self.semantic_threshold = float(semantic_threshold)
        self.relation_temperature = float(relation_temperature)
        self.kl_direction = kl_direction
        self.mask_diagonal = bool(mask_diagonal)
        self.eps = float(eps)

    def forward(
        self,
        old_attribute_features,
        current_attribute_features,
        old_text_features,
        current_text_features,
    ):
        if old_attribute_features.size(0) != current_attribute_features.size(0):
            raise ValueError('teacher and student must use the same image batch')

        old_relations = attribute_relation_matrix(
            old_attribute_features.detach(),
            temperature=self.relation_temperature,
            mask_diagonal=self.mask_diagonal,
        )
        current_relations = attribute_relation_matrix(
            current_attribute_features,
            temperature=self.relation_temperature,
            mask_diagonal=self.mask_diagonal,
        )
        similarity = semantic_compatibility(
            old_text_features.detach(),
            current_text_features.detach(),
        ).to(current_relations.device)
        routing, confidence = semantic_transfer_matrix(
            similarity,
            self.mode,
            self.semantic_threshold,
        )
        mixed_old = torch.einsum(
            'pq,pij->qij',
            routing,
            old_relations.to(current_relations.device),
        )

        valid = confidence > 0
        if not bool(valid.any()):
            zero = current_relations.sum() * 0.0
            return zero, {
                'effective_attributes': zero.detach(),
                'mean_confidence': zero.detach(),
                'mean_similarity': similarity.mean().detach(),
            }

        per_attribute = relation_kl_divergence(
            mixed_old,
            current_relations,
            direction=self.kl_direction,
            eps=self.eps,
        )
        loss = (
            confidence * per_attribute
        ).sum() / confidence.sum().clamp_min(self.eps)
        return loss, {
            'effective_attributes': valid.sum().detach(),
            'mean_confidence': confidence[valid].mean().detach(),
            'mean_similarity': similarity.mean().detach(),
        }
