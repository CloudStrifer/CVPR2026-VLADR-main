"""Text-conditioned visual attribute pooling for cross-category ReID."""

from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionedAttributePooler(nn.Module):
    """Pool CLIP patch tokens with frozen CLIP attribute embeddings.

    The module intentionally has no trainable parameters. This keeps every
    visual slot tied to its text description even in the first continual
    stage, where no historical teacher and therefore no SCSD supervision is
    available.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.set_temperature(temperature)

    def set_temperature(self, temperature):
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                'attribute pooling temperature must be finite and positive'
            )
        self.temperature = temperature

    def forward(self, patch_features, attribute_text_features):
        if patch_features.ndim != 3:
            raise ValueError(
                'patch_features must have shape [B, N, D], got {}'.format(
                    tuple(patch_features.shape)
                )
            )
        if attribute_text_features.ndim != 2:
            raise ValueError(
                'attribute_text_features must have shape [P, D], got {}'
                .format(tuple(attribute_text_features.shape))
            )
        if patch_features.size(1) == 0:
            raise ValueError('at least one visual patch token is required')
        if patch_features.size(2) != attribute_text_features.size(1):
            raise ValueError(
                'visual/text dimensions do not match: {} versus {}'.format(
                    patch_features.size(2),
                    attribute_text_features.size(1),
                )
            )
        if attribute_text_features.size(0) == 0:
            raise ValueError('at least one attribute text is required')

        patch_norm = F.normalize(patch_features.float(), dim=-1)
        text_norm = F.normalize(
            attribute_text_features.to(
                device=patch_norm.device,
                dtype=patch_norm.dtype,
            ),
            dim=-1,
        )
        attention_logits = torch.einsum(
            'bnd,pd->bpn',
            patch_norm,
            text_norm,
        ) / self.temperature
        attention = F.softmax(attention_logits, dim=-1)
        attributes = torch.einsum(
            'bpn,bnd->bpd',
            attention,
            patch_norm,
        )
        return F.normalize(attributes, dim=-1)
