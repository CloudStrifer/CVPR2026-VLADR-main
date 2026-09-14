"""Lifelong retrieval, routing diagnostics and explicit gallery protocols."""

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import torch

from lreid_dataset.category_stream import _path_key
from lreid_dataset.category_stream_loaders import build_evaluation_loaders
from reid.evaluation.category_oracle import oracle_retrieval_metrics
from reid.evaluation.prototype_router import RoutedEncoder, normalized
from reid.memory.prototype_views import check_summary
from reid.utils.progressive_checkpoint import capture_rng, restore_rng


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = 'test'
    gallery: str = 'per_dataset'
    beta: float = .5
    summary: str = 'ecpm'
    batch_size: int = 128

    def __post_init__(self):
        import math
        if self.split not in ('validation', 'test') or self.gallery not in ('per_dataset', 'mixed', 'both'):
            raise ValueError('invalid evaluation split/gallery scope')
        if type(self.beta) not in (float, int) or not math.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError('beta must be in [0,1]')
        check_summary(self.summary)
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError('evaluation batch_size must be positive')


def check_evaluation_coverage(stream, stage_id, config):
    views = stream.evaluations_at(stage_id, config.split)
    stage = stream.stage(stage_id)
    seen = set(stage.seen_before) | {v.category for v in stage.categories}
    missing = seen - {v.category for v in views}
    if missing:
        raise ValueError('missing {} evaluation for seen categories: {}'.format(config.split, sorted(missing)))
    return views


def routing_diagnostics(samples, predictions, categories):
    if len(samples) != len(predictions):
        raise ValueError('routing prediction count mismatch')
    unique = {}
    for sample, prediction in zip(samples, predictions):
        if sample.category not in categories or prediction not in categories:
            raise ValueError('routing diagnostics include an unseen category')
        key = _path_key(sample.path)
        row = (sample.category, sample.identity_key, prediction)
        if key in unique and unique[key] != row:
            raise ValueError('same evaluation image has inconsistent metadata/prediction')
        unique[key] = row
    confusion = [[0 for _ in categories] for _ in categories]
    indices = {c: i for i, c in enumerate(categories)}
    groups, truth_counts, correct = defaultdict(list), Counter(), Counter()
    for category, identity, prediction in unique.values():
        confusion[indices[category]][indices[prediction]] += 1
        truth_counts[category] += 1
        correct[category] += int(category == prediction)
        groups[identity].append(prediction)
    repeated = [v for v in groups.values() if len(v) >= 2]
    return dict(categories=list(categories), confusion_matrix=confusion, confusion_axes='rows=true, columns=predicted',
        unique_images=len(unique), accuracy=100 * sum(correct.values()) / len(unique),
        per_category_accuracy={c: 100 * correct[c] / truth_counts[c] for c in sorted(truth_counts)},
        multi_image_identities=len(repeated),
        identity_all_same_percent=None if not repeated else 100 * sum(len(set(v)) == 1 for v in repeated) / len(repeated),
        identity_modal_fraction=None if not repeated else sum(max(Counter(v).values()) / len(v) for v in repeated) / len(repeated))


def _merged_gallery(views, features):
    samples, descriptors, seen = [], [], {}
    for view in views:
        for sample, feature in zip(view.gallery, features[view.name]['gallery']):
            key = _path_key(sample.path)
            metadata = (sample.identity_key, sample.camid, sample.category)
            if key in seen:
                if seen[key] != metadata:
                    raise ValueError('mixed gallery has conflicting metadata for the same image')
                continue
            seen[key] = metadata
            samples.append(sample)
            descriptors.append(feature)
    return tuple(samples), torch.stack(descriptors)


def _retrieval_report(views, features, mode, scope, progress=None):
    if progress is not None:
        progress(dict(phase='retrieval_start', routing=mode, gallery_scope=scope))
    gallery, gallery_features = _merged_gallery(views, features) if scope == 'mixed' else (None, None)
    datasets, totals = {}, defaultdict(list)
    for view in views:
        callback = None if progress is None else lambda update: progress(
            dict(update, dataset=view.name, category=view.category, routing=mode, gallery_scope=scope))
        metric = oracle_retrieval_metrics(features[view.name]['query'],
            features[view.name]['gallery'] if scope == 'per_dataset' else gallery_features,
            view if scope == 'per_dataset' else replace(view, gallery=gallery), progress=callback)
        metric.update(routing=mode, gallery_scope=scope,
                      gallery_images=len(view.gallery) if scope == 'per_dataset' else len(gallery))
        datasets[view.name] = metric
        totals[view.category].append(metric)
    by_category = {c: {metric: sum(r[metric] * r['queries'] for r in rows) / sum(r['queries'] for r in rows)
                       for metric in ('mAP', 'Rank1')} for c, rows in totals.items()}
    return dict(datasets=datasets, per_category=by_category,
        category_aggregation='query-weighted within category, equally weighted across categories',
        macro={m: sum(v[m] for v in by_category.values()) / len(by_category) for m in ('mAP', 'Rank1')})


def evaluate_stage(model, memory, stream, stage_id, config=EvaluationConfig()):
    def progress(update):
        # Console diagnostics are deliberately separate from durable result
        # files and checkpoint logs. No RNG use or model/state mutation.
        print(json.dumps(dict(event='evaluation_progress', stage_id=stage_id, **update),
                         ensure_ascii=False, allow_nan=False), flush=True)

    progress(dict(phase='evaluation_start', split=config.split, gallery_scope=config.gallery))
    views = check_evaluation_coverage(stream, stage_id, config)
    index = [s.stage_id for s in stream.stages].index(stage_id)
    if memory.processed_stages != tuple(s.stage_id for s in stream.stages[:index + 1]):
        raise ValueError('evaluation requires exactly this completed stage memory')
    if model.temporary_heads or model.trainable_categories:
        raise ValueError('evaluate only after finish_stage and ECPM commit')
    rng, previous = capture_rng(), model.training
    model.eval()
    device = model.visual.proj.device
    started = time.perf_counter()
    forward_seconds = dict(oracle=0., prototype=0.)
    features = {mode: {} for mode in forward_seconds}
    samples, predictions = [], []
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        encoder = RoutedEncoder(model, memory, config.beta, config.summary)
        _, transform = model.make_transforms()
        with torch.no_grad():
            for view in views:
                progress(dict(phase='dataset_start', dataset=view.name, category=view.category,
                              query_images=len(view.query), gallery_images=len(view.gallery)))
                for mode in features:
                    features[mode][view.name] = {}
                for subset, loader in build_evaluation_loaders(view, transform, config.batch_size, workers=0).items():
                    batches = {mode: [] for mode in features}
                    subset_predictions = []
                    subset_started = last_progress = time.perf_counter()
                    image_count = 0
                    progress(dict(phase='features_start', dataset=view.name, subset=subset,
                                  batches=0, total_batches=len(loader), images=0,
                                  total_images=len(getattr(view, subset))))
                    for batch_index, batch in enumerate(loader, 1):
                        images = batch['images'].to(device)
                        synchronize()
                        before = time.perf_counter()
                        # Only the explicitly named oracle diagnostic sees labels.
                        oracle = normalized(model.encode_category(images, view.category))
                        synchronize()
                        forward_seconds['oracle'] += time.perf_counter() - before
                        before = time.perf_counter()
                        routed, prediction, _ = encoder(images)
                        synchronize()
                        forward_seconds['prototype'] += time.perf_counter() - before
                        batches['oracle'].append(oracle.cpu())
                        batches['prototype'].append(routed.cpu())
                        subset_predictions.extend(prediction)
                        image_count += len(images)
                        now = time.perf_counter()
                        if batch_index == 1 or batch_index == len(loader) or batch_index % 10 == 0 or now - last_progress >= 10:
                            progress(dict(phase='features_progress', dataset=view.name, subset=subset,
                                          batches=batch_index, total_batches=len(loader), images=image_count,
                                          total_images=len(getattr(view, subset)),
                                          elapsed_seconds=round(now - subset_started, 2)))
                            last_progress = now
                    for mode in features:
                        features[mode][view.name][subset] = torch.cat(batches[mode])
                    samples.extend(getattr(view, subset))
                    predictions.extend(subset_predictions)
        scopes = ('per_dataset', 'mixed') if config.gallery == 'both' else (config.gallery,)
        results = {scope: {mode: _retrieval_report(views, features[mode], mode, scope, progress=progress)
                          for mode in features} for scope in scopes}
        for scope in scopes:
            results[scope]['oracle_minus_prototype'] = {
                c: {m: results[scope]['oracle']['per_category'][c][m] - results[scope]['prototype']['per_category'][c][m]
                    for m in ('mAP', 'Rank1')} for c in encoder.router.categories}
        model.assert_reference_unchanged()
        memory_stats = memory.summary()
        progress(dict(phase='routing_diagnostics'))
        result = dict(stage_id=stage_id, stage_index=index, stream_fingerprint=stream.fingerprint,
            split=config.split, routing_summary=config.summary, beta=config.beta,
            seen_categories=list(encoder.router.categories), retrieval=results,
            routing=routing_diagnostics(samples, predictions, encoder.router.categories),
            resources=dict(ecpm=memory_stats,
                adapter_tensor_bytes=sum(p.numel() * p.element_size() for c in model.categories for p in model.adapter_parameters(c)),
                forward_seconds=forward_seconds, evaluation_seconds=time.perf_counter() - started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                peak_memory_scope='evaluation process CUDA allocated tensors; CPU/native FINCH peak not measured',
                stored_descriptor_bytes=sum(t.numel() * t.element_size() for mode in features.values()
                                             for view in mode.values() for t in view.values())))
        progress(dict(phase='evaluation_complete', elapsed_seconds=round(time.perf_counter() - started, 2),
                      prototype_macro={scope: results[scope]['prototype']['macro'] for scope in scopes}))
        return result
    finally:
        model.train(previous)
        restore_rng(rng)


def lifelong_summary(history):
    """R[t,c] and max_so_far(R[:,c])-R[t,c]; unseen entries stay null."""
    if not history:
        return dict(stages=[], categories=[], metrics={})
    if [r['stage_index'] for r in history] != list(range(len(history))):
        raise ValueError('lifelong history must be a contiguous prefix from T1')
    first = history[0]
    for report in history:
        if any(report[k] != first[k] for k in ('stream_fingerprint', 'split', 'routing_summary', 'beta')) or set(report['retrieval']) != set(first['retrieval']):
            raise ValueError('lifelong history mixes protocols or routing settings')
    categories = sorted(set().union(*(set(r['seen_categories']) for r in history)))
    output = dict(stages=[r['stage_id'] for r in history], categories=categories,
                  split=first['split'], routing_summary=first['routing_summary'], beta=first['beta'], metrics={})
    for scope in first['retrieval']:
        for mode in ('oracle', 'prototype'):
            for metric in ('mAP', 'Rank1'):
                best, matrix, forgetting, macro = {}, [], [], []
                for report in history:
                    values = report['retrieval'][scope][mode]['per_category']
                    row, drops = [], []
                    for category in categories:
                        value = values.get(category, {}).get(metric)
                        row.append(value)
                        if value is None:
                            if category in best:
                                raise ValueError('a previously seen category is missing from evaluation')
                            drops.append(None)
                        else:
                            best[category] = max(best.get(category, value), value)
                            drops.append(best[category] - value)
                    matrix.append(row)
                    forgetting.append(drops)
                    macro.append(sum(v for v in row if v is not None) / sum(v is not None for v in row))
                output['metrics']['{}/{}/{}'.format(scope, mode, metric)] = dict(
                    performance=matrix, forgetting=forgetting, macro_per_stage=macro,
                    macro_forgetting=[sum(v for v in row if v is not None) / sum(v is not None for v in row) for row in forgetting])
    return output
