from collections.abc import Mapping

import torch
import torch.nn.functional as F


ROUTING_ENTRY_KEYS = (
    'semantic_key',
    'visual_prototypes',
    'category_basis',
)


def validate_adapter_routing_bank(routing_bank):
    """Validate the fixed source-domain statistics used by OSAF."""

    if not isinstance(routing_bank, Mapping) or not routing_bank:
        raise ValueError('adapter routing bank must be a non-empty mapping')

    feature_dim = None
    for domain_name, entry in routing_bank.items():
        if not isinstance(domain_name, str) or not domain_name:
            raise ValueError('adapter routing bank has an invalid domain name')
        if not isinstance(entry, Mapping):
            raise ValueError(
                'routing entry for {!r} must be a mapping'.format(
                    domain_name
                )
            )
        missing = [key for key in ROUTING_ENTRY_KEYS if key not in entry]
        if missing:
            raise ValueError(
                'routing entry for {!r} is missing {}'.format(
                    domain_name,
                    ', '.join(missing),
                )
            )

        semantic_key = entry['semantic_key']
        visual_prototypes = entry['visual_prototypes']
        category_basis = entry['category_basis']
        if semantic_key.ndim != 1:
            raise ValueError(
                'semantic key for {!r} must be 1-D, got {}'.format(
                    domain_name,
                    tuple(semantic_key.shape),
                )
            )
        if visual_prototypes.ndim != 2 or not visual_prototypes.size(0):
            raise ValueError(
                'visual prototypes for {!r} must have non-empty shape '
                '[N, D], got {}'.format(
                    domain_name,
                    tuple(visual_prototypes.shape),
                )
            )
        if category_basis.ndim != 2 or not category_basis.size(1):
            raise ValueError(
                'category basis for {!r} must have non-empty shape [D, R], '
                'got {}'.format(
                    domain_name,
                    tuple(category_basis.shape),
                )
            )

        current_dim = int(semantic_key.numel())
        if visual_prototypes.size(1) != current_dim:
            raise ValueError(
                'visual prototype dimension for {!r} is {}, expected {}'
                .format(
                    domain_name,
                    visual_prototypes.size(1),
                    current_dim,
                )
            )
        if category_basis.size(0) != current_dim:
            raise ValueError(
                'category basis dimension for {!r} is {}, expected {}'
                .format(
                    domain_name,
                    category_basis.size(0),
                    current_dim,
                )
            )
        if feature_dim is None:
            feature_dim = current_dim
        elif current_dim != feature_dim:
            raise ValueError(
                'routing feature dimensions are inconsistent: {} and {}'
                .format(feature_dim, current_dim)
            )

        for key in ROUTING_ENTRY_KEYS:
            if not torch.isfinite(entry[key]).all():
                raise FloatingPointError(
                    'routing entry {!r} contains non-finite {}'.format(
                        domain_name,
                        key,
                    )
                )

        gram = category_basis.float().t() @ category_basis.float()
        identity = torch.eye(gram.size(0), device=gram.device)
        if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
            raise ValueError(
                'category basis for {!r} is not orthonormal'.format(
                    domain_name
                )
            )
    return feature_dim


def routing_bank_to(routing_bank, device, dtype=torch.float32):
    """Move a validated routing bank without modifying the checkpoint copy."""

    validate_adapter_routing_bank(routing_bank)
    return {
        name: {
            key: value.detach().to(device=device, dtype=dtype)
            for key, value in entry.items()
            if key in ROUTING_ENTRY_KEYS
        }
        for name, entry in routing_bank.items()
    }


def semantic_debiased_residual(residual, category_basis, strength=1.0):
    """Remove a controllable OCIA category-subspace projection."""

    if residual.ndim != 2:
        raise ValueError(
            'adapter residual must have shape [B, D], got {}'.format(
                tuple(residual.shape)
            )
        )
    if category_basis.ndim != 2:
        raise ValueError(
            'category basis must have shape [D, R], got {}'.format(
                tuple(category_basis.shape)
            )
        )
    if residual.size(1) != category_basis.size(0):
        raise ValueError(
            'residual dimension {} does not match category basis {}'
            .format(residual.size(1), category_basis.size(0))
        )
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError('semantic debias strength must be in [0, 1]')

    basis = category_basis.to(
        device=residual.device,
        dtype=residual.dtype,
    )
    projection = (residual @ basis) @ basis.t()
    return residual - strength * projection


def adapter_routing_scores(
    base_projected,
    routing_bank,
    semantic_weight=0.5,
):
    """Combine frozen object-semantic and visual-prototype similarities."""

    feature_dim = validate_adapter_routing_bank(routing_bank)
    if base_projected.ndim != 2 or base_projected.size(1) != feature_dim:
        raise ValueError(
            'base projected features must have shape [B, {}], got {}'
            .format(feature_dim, tuple(base_projected.shape))
        )
    semantic_weight = float(semantic_weight)
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError('adapter semantic weight must be in [0, 1]')

    normalized_base = F.normalize(base_projected.float(), dim=1)
    domain_names = tuple(routing_bank.keys())
    scores = []
    for domain_name in domain_names:
        entry = routing_bank[domain_name]
        semantic_key = F.normalize(
            entry['semantic_key'].float().reshape(1, -1),
            dim=1,
        )
        visual_prototypes = F.normalize(
            entry['visual_prototypes'].float(),
            dim=1,
        )
        semantic_score = normalized_base @ semantic_key.t()
        visual_score = (
            normalized_base @ visual_prototypes.t()
        ).max(dim=1, keepdim=True).values
        scores.append(
            semantic_weight * semantic_score
            + (1.0 - semantic_weight) * visual_score
        )
    return domain_names, torch.cat(scores, dim=1)


def topk_adapter_weights(scores, topk=2, temperature=0.1):
    """Select source adapters and normalize weights within the Top-K set."""

    if scores.ndim != 2 or not scores.size(1):
        raise ValueError(
            'routing scores must have non-empty shape [B, T], got {}'
            .format(tuple(scores.shape))
        )
    topk = int(topk)
    if topk <= 0:
        raise ValueError('adapter top-k must be positive')
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError('adapter routing temperature must be positive')

    selected_count = min(topk, scores.size(1))
    selected_scores, selected_indices = torch.topk(
        scores,
        k=selected_count,
        dim=1,
        largest=True,
        sorted=True,
    )
    weights = F.softmax(selected_scores / temperature, dim=1)
    return selected_indices, weights
