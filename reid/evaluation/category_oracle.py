"""Small baseline diagnostic with known categories; not prototype routing."""

import time

import torch
from torch.nn import functional as F

from lreid_dataset.category_stream import EvaluationView, _path_key
from lreid_dataset.category_stream_loaders import build_evaluation_loaders


def oracle_retrieval_metrics(query_features, gallery_features, evaluation, progress=None):
    if not isinstance(evaluation, EvaluationView):
        raise TypeError('expected EvaluationView')
    started = last_progress = time.perf_counter()
    if progress is not None:
        progress(dict(phase='ranking_start', queries=0, total_queries=len(evaluation.query),
                      gallery_images=len(evaluation.gallery), elapsed_seconds=0.))
    if (query_features.ndim != 2 or gallery_features.ndim != 2
            or query_features.shape[1] != gallery_features.shape[1]
            or len(query_features) != len(evaluation.query) or len(gallery_features) != len(evaluation.gallery)):
        raise ValueError('feature shape/order must match the evaluation view')
    for features in (query_features, gallery_features):
        if not torch.isfinite(features).all() or (features.norm(dim=1) <= 1e-12).any():
            raise FloatingPointError('retrieval features must be finite and nonzero')
    query_features = F.normalize(query_features.detach().float().cpu(), dim=1)
    gallery_features = F.normalize(gallery_features.detach().float().cpu(), dim=1)
    aps, rank1 = [], []
    # Resolve each image once, never once per query-gallery pair. Encode the
    # metadata locally so camera/self filtering is vectorized and has no stale
    # filesystem cache across evaluations. Keep scores and stable sorting intact.
    def encoded(values):
        codes = {value: index for index, value in enumerate(dict.fromkeys(values))}
        return codes, torch.tensor([codes[value] for value in values], dtype=torch.long)

    path_codes, gallery_paths = encoded([_path_key(s.path) for s in evaluation.gallery])
    identity_codes, gallery_ids = encoded([s.identity_key for s in evaluation.gallery])
    camera_codes, gallery_cameras = encoded([s.camid for s in evaluation.gallery])
    # Row-wise calculation bounds distance storage to O(N_gallery). Preserve the
    # protocol's self-image and camera rules, including exclude_self datasets.
    for query, features in zip(evaluation.query, query_features):
        scores = gallery_features @ features
        order = torch.argsort(scores, descending=True, stable=True)
        same = gallery_ids.eq(identity_codes.get(query.identity_key, -1))
        valid = gallery_paths.ne(path_codes.get(_path_key(query.path), -1))
        if evaluation.protocol == 'cross_camera':
            valid &= ~(same & gallery_cameras.eq(camera_codes.get(query.camid, -1)))
        matches = same[order][valid[order]].to(dtype=torch.float64)
        if not matches.any():
            raise ValueError('query has no valid gallery positive: {}'.format(query.path))
        precision = matches.cumsum(0) / torch.arange(1, len(matches) + 1, dtype=torch.float64)
        aps.append((precision * matches).sum().item() / matches.sum().item())
        rank1.append(matches[0].item())
        if progress is not None:
            now = time.perf_counter()
            done = len(aps)
            if done == 1 or done == len(evaluation.query) or done % 100 == 0 or now - last_progress >= 10:
                progress(dict(phase='ranking_progress', queries=done, total_queries=len(evaluation.query),
                              gallery_images=len(evaluation.gallery), elapsed_seconds=round(now - started, 2)))
                last_progress = now
    if not aps:
        raise ValueError('empty query set')
    result = {'routing': 'oracle', 'name': evaluation.name, 'category': evaluation.category,
            'split': evaluation.split, 'protocol': evaluation.protocol, 'queries': len(aps),
            'mAP': 100 * sum(aps) / len(aps), 'Rank1': 100 * sum(rank1) / len(rank1)}
    if progress is not None:
        progress(dict(phase='ranking_complete', queries=len(aps), total_queries=len(evaluation.query),
                      elapsed_seconds=round(time.perf_counter() - started, 2),
                      mAP=result['mAP'], Rank1=result['Rank1']))
    return result


def evaluate_category_oracle(model, evaluation, batch_size=128, workers=0):
    model.category_key(evaluation.category)
    _, reference_transform = model.make_transforms()
    loaders = build_evaluation_loaders(evaluation, reference_transform, batch_size, workers)
    previous_mode = model.training
    model.eval()
    output = {}
    try:
        with torch.no_grad():
            for subset, loader in loaders.items():
                output[subset] = torch.cat([
                    model.encode_category(batch['images'].to(model.visual.proj.device), evaluation.category).cpu()
                    for batch in loader
                ])
    finally:
        model.train(previous_mode)
    return oracle_retrieval_metrics(output['query'], output['gallery'], evaluation)
