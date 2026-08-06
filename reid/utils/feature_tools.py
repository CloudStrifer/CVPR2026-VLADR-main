from collections import defaultdict

import torch
import torch.nn.functional as F


def _base_model(model):
    wrapped = model.module if hasattr(model, 'module') else model
    return wrapped.base


def initial_classifier(model, data_loader):
    """Initialize current-domain classifier rows with mean global features."""

    was_training = model.training
    model.eval()
    features_by_identity = defaultdict(list)
    main_feature_dim = _base_model(model).in_planes

    with torch.no_grad():
        for images, _, identities, _, _ in data_loader:
            images = images.cuda(non_blocking=True)
            descriptors = model(images)[:, :main_feature_dim]
            for descriptor, identity in zip(descriptors, identities):
                features_by_identity[int(identity)].append(descriptor)

    if was_training:
        model.train()

    if not features_by_identity:
        raise RuntimeError('Cannot initialize classifier from an empty loader')

    identity_order = sorted(features_by_identity)
    expected_order = list(range(len(identity_order)))
    if identity_order != expected_order:
        raise ValueError(
            'Current-domain training identities must be contiguous from 0; '
            'got {}'.format(identity_order[:10])
        )

    centers = [
        torch.stack(features_by_identity[identity]).mean(dim=0)
        for identity in identity_order
    ]
    return F.normalize(torch.stack(centers), dim=1).float()


def visual_identity_statistics(model, data_loader, num_classes):
    """Extract frozen-CLIP visual identity prototypes for one domain.

    Each image projection is L2-normalized before identity-wise averaging.
    The returned prototype and category-center tensors stay on CPU so they can
    be stored together with the Stage 1 prompt checkpoint.
    """

    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError('num_classes must be positive')

    was_training = model.training
    model.eval()
    feature_sums = None
    feature_counts = None
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    with torch.no_grad():
        for images, _, identities, _, _ in data_loader:
            images = images.to(
                device=model_device,
                non_blocking=model_device.type == 'cuda',
            )
            features = model(
                x=images,
                get_image=True,
            )
            features = F.normalize(features.float(), dim=1)
            identities = identities.to(
                device=features.device,
                dtype=torch.long,
                non_blocking=True,
            )

            if identities.numel() == 0:
                continue
            min_identity = int(identities.min().item())
            max_identity = int(identities.max().item())
            if min_identity < 0 or max_identity >= num_classes:
                raise ValueError(
                    'Identity labels must be in [0, {}), got [{}, {}]'
                    .format(
                        num_classes,
                        min_identity,
                        max_identity,
                    )
                )

            if feature_sums is None:
                feature_sums = features.new_zeros(
                    num_classes,
                    features.size(1),
                    dtype=torch.float32,
                )
                feature_counts = features.new_zeros(
                    num_classes,
                    dtype=torch.float32,
                )
            feature_sums.index_add_(0, identities, features)
            feature_counts.index_add_(
                0,
                identities,
                torch.ones_like(identities, dtype=torch.float32),
            )

    if was_training:
        model.train()

    if feature_sums is None:
        raise RuntimeError(
            'Cannot compute visual prototypes from an empty loader'
        )
    missing = torch.nonzero(
        feature_counts == 0,
        as_tuple=False,
    ).flatten()
    if missing.numel():
        raise ValueError(
            'No images were found for identity rows {}'.format(
                missing[:10].tolist()
            )
        )

    prototypes = feature_sums / feature_counts.unsqueeze(1)
    category_center = prototypes.mean(dim=0, keepdim=True)
    if not torch.isfinite(prototypes).all():
        raise FloatingPointError(
            'Non-finite values found in visual identity prototypes'
        )

    return (
        prototypes.detach().cpu(),
        category_center.detach().cpu(),
    )


def category_semantic_subspace(category_text_features, tolerance=None):
    """Build an orthonormal category-common basis from text features.

    ``category_text_features`` stores one CLIP text embedding per row.  The
    returned columns form ``U`` in the OCIA projection ``I - U U^T``.
    """

    if category_text_features.ndim != 2:
        raise ValueError(
            'category_text_features must be a 2-D tensor, got {}'.format(
                tuple(category_text_features.shape)
            )
        )
    if not category_text_features.size(0):
        raise ValueError('at least one category-generic prompt is required')
    if not torch.isfinite(category_text_features).all():
        raise FloatingPointError(
            'Non-finite values found in category text features'
        )

    text_features = F.normalize(
        category_text_features.float(),
        dim=1,
    )
    semantic_matrix = text_features.t().contiguous()
    basis, upper = torch.linalg.qr(semantic_matrix, mode='reduced')
    diagonal = torch.abs(torch.diagonal(upper))
    if diagonal.numel() == 0:
        raise RuntimeError('Cannot construct an empty category subspace')

    if tolerance is None:
        scale = max(float(diagonal.max().item()), 1.0)
        tolerance = (
            max(semantic_matrix.shape)
            * torch.finfo(semantic_matrix.dtype).eps
            * scale
        )
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError('tolerance must be non-negative')

    rank = int(torch.count_nonzero(diagonal > tolerance).item())
    if rank <= 0:
        raise ValueError(
            'Category-generic prompt embeddings have zero numerical rank'
        )
    basis = basis[:, :rank]
    if not torch.isfinite(basis).all():
        raise FloatingPointError(
            'Non-finite values found in category semantic basis'
        )
    return basis


def build_cross_modal_identity_anchors(
    text_features,
    visual_prototypes=None,
    category_center=None,
    category_basis=None,
    mode='text',
    residual_weight=1.0,
):
    """Build fixed identity anchors for Stage 2 global alignment.

    Modes:
      - ``text``: normalized identity-prompt features only;
      - ``prototype``: text features plus visual identity prototypes;
      - ``centered``: text features plus category-centered visual residuals.
      - ``ocia``: text features plus normalized, category-disentangled visual
        residuals obtained with ``I - U U^T``.
    """

    valid_modes = {'text', 'prototype', 'centered', 'ocia'}
    if mode not in valid_modes:
        raise ValueError(
            'Unsupported visual anchor mode {!r}; choose from {}'.format(
                mode,
                sorted(valid_modes),
            )
        )

    output_dtype = text_features.dtype
    text_features = F.normalize(text_features.float(), dim=1)
    if mode == 'text':
        return text_features.to(dtype=output_dtype)

    if visual_prototypes is None:
        raise ValueError(
            'visual_prototypes are required for mode={!r}'.format(mode)
        )
    visual_prototypes = visual_prototypes.to(
        device=text_features.device,
        dtype=text_features.dtype,
    )
    if visual_prototypes.shape != text_features.shape:
        raise ValueError(
            'Visual prototype shape {} does not match text feature shape {}'
            .format(
                tuple(visual_prototypes.shape),
                tuple(text_features.shape),
            )
        )
    if not torch.isfinite(visual_prototypes).all():
        raise FloatingPointError(
            'Non-finite values found in visual identity prototypes'
        )

    if mode == 'prototype':
        visual_component = visual_prototypes
    else:
        if category_center is None:
            category_center = visual_prototypes.mean(
                dim=0,
                keepdim=True,
            )
        category_center = category_center.to(
            device=text_features.device,
            dtype=text_features.dtype,
        )
        expected_center_shape = (1, text_features.size(1))
        if tuple(category_center.shape) != expected_center_shape:
            raise ValueError(
                'Category-center shape {} does not match expected {}'
                .format(
                    tuple(category_center.shape),
                    expected_center_shape,
                )
            )
        visual_component = visual_prototypes - category_center
        if mode == 'ocia':
            if category_basis is None:
                raise ValueError(
                    'category_basis is required for mode={!r}'.format(mode)
                )
            category_basis = category_basis.to(
                device=text_features.device,
                dtype=text_features.dtype,
            )
            if category_basis.ndim != 2:
                raise ValueError(
                    'category_basis must be a 2-D tensor, got {}'.format(
                        tuple(category_basis.shape)
                    )
                )
            if category_basis.size(0) != text_features.size(1):
                raise ValueError(
                    'Category-basis feature dimension {} does not match {}'
                    .format(
                        category_basis.size(0),
                        text_features.size(1),
                    )
                )
            if category_basis.size(1) <= 0:
                raise ValueError('category_basis must have positive rank')
            if not torch.isfinite(category_basis).all():
                raise FloatingPointError(
                    'Non-finite values found in category semantic basis'
                )

            gram = category_basis.t() @ category_basis
            identity = torch.eye(
                category_basis.size(1),
                device=gram.device,
                dtype=gram.dtype,
            )
            if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
                raise ValueError(
                    'category_basis columns must be orthonormal'
                )

            category_projection = (
                visual_component @ category_basis
            ) @ category_basis.t()
            visual_component = F.normalize(
                visual_component - category_projection,
                dim=1,
            )

    residual_weight = float(residual_weight)
    if not torch.isfinite(torch.tensor(residual_weight)):
        raise ValueError('residual_weight must be finite')
    if residual_weight < 0.0:
        raise ValueError('residual_weight must be non-negative')
    visual_weight = residual_weight if mode == 'ocia' else 1.0

    anchors = F.normalize(
        text_features + visual_weight * visual_component,
        dim=1,
    )
    if not torch.isfinite(anchors).all():
        raise FloatingPointError(
            'Non-finite values found in cross-modal identity anchors'
        )
    return anchors.to(dtype=output_dtype)
