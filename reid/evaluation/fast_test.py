import os.path as osp
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from reid.evaluation import compute_distance_matrix, fast_evaluate_rank
from reid.evaluation.adapter_fusion import (
    adapter_routing_scores,
    routing_bank_to,
    semantic_debiased_residual,
    topk_adapter_weights,
    validate_adapter_routing_bank,
)
from reid.evaluation.distance import euclidean_squared_distance


class CatMeter:
    def __init__(self):
        self.val = None

    def update(self, value):
        if self.val is None:
            self.val = value
        else:
            self.val = torch.cat([self.val, value], dim=0)

    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.detach().cpu().numpy()


def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def _adapter_controller(model):
    wrapped = model.module if hasattr(model, 'module') else model
    required = (
        'domain_adapter_names',
        'get_active_adapter',
        'set_active_adapter',
    )
    if all(hasattr(wrapped, name) for name in required):
        return wrapped
    return None


@contextmanager
def _evaluation_domain_adapter(model, domain_name):
    controller = _adapter_controller(model)
    if controller is None:
        yield None
        return

    names = set(controller.domain_adapter_names())
    if not names:
        yield None
        return

    previous = controller.get_active_adapter()
    selected = domain_name if domain_name in names else None
    controller.set_active_adapter(selected)
    print(
        '[eval] domain adapter for {}: {}'.format(
            domain_name,
            selected if selected is not None else 'base',
        )
    )
    try:
        yield selected
    finally:
        controller.set_active_adapter(previous)


def _available_adapter_routing_bank(model, routing_bank):
    controller = _adapter_controller(model)
    if controller is None:
        raise RuntimeError(
            'OSAF evaluation requires a model with domain adapters'
        )
    available_names = set(controller.domain_adapter_names())
    available_bank = {
        name: entry
        for name, entry in routing_bank.items()
        if name in available_names
    }
    if not available_bank:
        raise RuntimeError(
            'OSAF routing bank has no entry for the learned adapters: {}'
            .format(', '.join(sorted(available_names)))
        )
    validate_adapter_routing_bank(available_bank)
    return available_bank


def _use_osaf_for_domain(
    osaf_bank,
    domain_name,
    learned_domain_names,
    args,
):
    """Return whether this evaluation domain should use OSAF routing."""

    if osaf_bank is None:
        return False
    routing_scope = getattr(args, 'adapter_routing_scope', 'unseen')
    if routing_scope == 'all':
        return True
    if routing_scope == 'unseen':
        return domain_name not in learned_domain_names
    raise ValueError(
        'unsupported adapter routing scope: {}'.format(routing_scope)
    )


def _osaf_descriptor(model, images, routing_bank, args):
    """Build an unknown-domain descriptor from Top-K debiased adapters."""

    controller = _adapter_controller(model)
    if controller is None:
        raise RuntimeError(
            'OSAF evaluation requires a model with domain adapters'
        )
    previous = controller.get_active_adapter()
    try:
        controller.set_active_adapter(None)
        base_descriptor = model(images)
        projected_dim = validate_adapter_routing_bank(routing_bank)
        if base_descriptor.ndim != 2:
            raise ValueError(
                'model descriptor must be 2-D, got {}'.format(
                    tuple(base_descriptor.shape)
                )
            )
        if base_descriptor.size(1) <= projected_dim:
            raise ValueError(
                'raw descriptor dimension {} must be larger than the OSAF '
                'projection dimension {}'.format(
                    base_descriptor.size(1),
                    projected_dim,
                )
            )

        base_main = base_descriptor[:, :-projected_dim]
        base_projected = base_descriptor[:, -projected_dim:]
        domain_names, scores = adapter_routing_scores(
            base_projected,
            routing_bank,
            semantic_weight=args.adapter_semantic_weight,
        )
        selected_indices, selected_weights = topk_adapter_weights(
            scores,
            topk=args.adapter_topk,
            temperature=args.adapter_routing_temperature,
        )

        fused_residual = torch.zeros_like(base_projected)
        unique_indices = torch.unique(selected_indices).tolist()
        for domain_index in unique_indices:
            domain_index = int(domain_index)
            domain_name = domain_names[domain_index]
            controller.set_active_adapter(domain_name)
            adapter_descriptor = model(images)
            adapter_projected = adapter_descriptor[:, -projected_dim:]
            adapter_residual = adapter_projected - base_projected
            debiased_residual = semantic_debiased_residual(
                adapter_residual,
                routing_bank[domain_name]['category_basis'],
                strength=args.adapter_debias_strength,
            )
            sample_weights = (
                selected_weights
                * selected_indices.eq(domain_index).to(
                    dtype=selected_weights.dtype
                )
            ).sum(dim=1, keepdim=True)
            fused_residual = (
                fused_residual
                + sample_weights.to(dtype=debiased_residual.dtype)
                * debiased_residual
            )

        fused_projected = (
            base_projected
            + float(args.adapter_fusion_weight) * fused_residual
        )
        descriptor = torch.cat([base_main, fused_projected], dim=1)
        top1_indices = selected_indices[:, 0].detach().cpu()
        return F.normalize(descriptor, dim=1), domain_names, top1_indices
    finally:
        controller.set_active_adapter(previous)


def _evaluation_options(args):
    use_cython = not getattr(args, 'save_evaluation', False)
    save_dir = None
    if getattr(args, 'save_evaluation', False):
        save_dir = getattr(args, 'logs_dir', getattr(args, 'log_dir', '.'))
    return use_cython, save_dir


def _cmc_at_rank(cmc, rank):
    """Return CMC percentage at a one-based rank, or NaN if unavailable."""
    index = int(rank) - 1
    if index < 0:
        raise ValueError('rank must be a positive integer')
    if len(cmc) <= index:
        return float('nan')
    return float(cmc[index] * 100)


def _rank(
    query_features,
    gallery_features,
    query_pids,
    gallery_pids,
    query_cids,
    gallery_cids,
    args,
):
    pair_count = int(query_features.size(0)) * int(gallery_features.size(0))
    full_matrix_limit = int(
        getattr(args, 'eval_full_matrix_max_elements', 100000000)
    )
    if full_matrix_limit >= 0 and pair_count > full_matrix_limit:
        return _rank_chunked(
            query_features,
            gallery_features,
            query_pids,
            gallery_pids,
            query_cids,
            gallery_cids,
            args,
        )

    distance_matrix = compute_distance_matrix(
        query_features,
        gallery_features,
        'euclidean',
    ).detach().cpu().numpy()
    use_cython, save_dir = _evaluation_options(args)
    cmc, mean_ap = fast_evaluate_rank(
        distance_matrix,
        query_pids.detach().cpu().numpy(),
        gallery_pids.detach().cpu().numpy(),
        query_cids.detach().cpu().numpy(),
        gallery_cids.detach().cpu().numpy(),
        max_rank=50,
        use_metric_cuhk03=False,
        use_cython=use_cython,
        save_dir=save_dir,
    )
    return {
        'mAP': float(mean_ap * 100),
        'Rank1': _cmc_at_rank(cmc, 1),
        'Rank5': _cmc_at_rank(cmc, 5),
        'Rank10': _cmc_at_rank(cmc, 10),
    }


def _valid_market1501_query_indices(
    query_pids,
    gallery_pids,
    query_cids,
    gallery_cids,
):
    gallery_cameras = {}
    for pid, camera in zip(gallery_pids.tolist(), gallery_cids.tolist()):
        gallery_cameras.setdefault(int(pid), set()).add(int(camera))
    return np.asarray(
        [
            index
            for index, (pid, camera) in enumerate(
                zip(query_pids.tolist(), query_cids.tolist())
            )
            if any(
                gallery_camera != int(camera)
                for gallery_camera in gallery_cameras.get(int(pid), ())
            )
        ],
        dtype=np.int64,
    )


def _rank_chunked(
    query_features,
    gallery_features,
    query_pids,
    gallery_pids,
    query_cids,
    gallery_cids,
    args,
):
    """Compute exact Market1501 metrics without a full QxG matrix."""

    chunk_size = int(getattr(args, 'eval_query_chunk_size', 256))
    if chunk_size <= 0:
        raise ValueError('eval query chunk size must be positive')

    query_pids_np = query_pids.detach().cpu().numpy()
    gallery_pids_np = gallery_pids.detach().cpu().numpy()
    query_cids_np = query_cids.detach().cpu().numpy()
    gallery_cids_np = gallery_cids.detach().cpu().numpy()
    valid_indices = _valid_market1501_query_indices(
        query_pids_np,
        gallery_pids_np,
        query_cids_np,
        gallery_cids_np,
    )
    if not len(valid_indices):
        raise AssertionError(
            'Error: all query identities do not appear in gallery'
        )

    pair_count = int(query_features.size(0)) * int(gallery_features.size(0))
    print(
        '[eval] chunked exact ranking: {:,} query-gallery pairs, '
        '{} queries/chunk.'.format(pair_count, chunk_size)
    )
    if getattr(args, 'save_evaluation', False):
        print(
            '[eval] per-match JSON export is disabled for chunked ranking '
            'to keep memory bounded.'
        )

    use_cython, _ = _evaluation_options(args)
    cmc_sum = None
    map_sum = 0.0
    valid_count = 0
    starts = range(0, len(valid_indices), chunk_size)
    with torch.no_grad():
        for start in tqdm(starts, desc='Ranking chunks'):
            chunk_indices_np = valid_indices[start:start + chunk_size]
            chunk_indices = torch.as_tensor(
                chunk_indices_np,
                dtype=torch.long,
                device=query_features.device,
            )
            query_chunk = query_features.index_select(0, chunk_indices)
            distance_matrix = compute_distance_matrix(
                query_chunk,
                gallery_features,
                'euclidean',
            ).detach().cpu().numpy()
            cmc, mean_ap = fast_evaluate_rank(
                distance_matrix,
                query_pids_np[chunk_indices_np],
                gallery_pids_np,
                query_cids_np[chunk_indices_np],
                gallery_cids_np,
                max_rank=50,
                use_metric_cuhk03=False,
                use_cython=use_cython,
                save_dir=None,
                verbose=False,
            )
            current_count = len(chunk_indices_np)
            weighted_cmc = np.asarray(cmc, dtype=np.float64) * current_count
            if cmc_sum is None:
                cmc_sum = weighted_cmc
            else:
                cmc_sum += weighted_cmc
            map_sum += float(mean_ap) * current_count
            valid_count += current_count
            del distance_matrix

    cmc = cmc_sum / valid_count
    mean_ap = map_sum / valid_count
    return {
        'mAP': float(mean_ap * 100),
        'Rank1': _cmc_at_rank(cmc, 1),
        'Rank5': _cmc_at_rank(cmc, 5),
        'Rank10': _cmc_at_rank(cmc, 10),
    }


def fast_eval(features, labels, cameras, args):
    features = torch.stack(features)
    labels = torch.tensor(labels)
    cameras = torch.tensor(cameras)
    distance_matrix = euclidean_squared_distance(
        features,
        features,
    ).detach().cpu().numpy()
    use_cython, save_dir = _evaluation_options(args)
    cmc, mean_ap = fast_evaluate_rank(
        distance_matrix,
        labels.numpy(),
        labels.numpy(),
        cameras.numpy(),
        cameras.numpy(),
        max_rank=50,
        use_metric_cuhk03=False,
        use_cython=use_cython,
        save_dir=save_dir,
    )
    metrics = {
        'mAP': float(mean_ap * 100),
        'Rank1': _cmc_at_rank(cmc, 1),
        'Rank5': _cmc_at_rank(cmc, 5),
        'Rank10': _cmc_at_rank(cmc, 10),
    }
    print(
        'mAP/R1/R5/R10:\t'
        '{mAP:.1f}/{Rank1:.1f}/{Rank5:.1f}/{Rank10:.1f}'.format(
            **metrics
        )
    )
    return mean_ap * 100


def _resolve_feature_index(file_names, dataset, sample_path):
    if sample_path in file_names:
        return file_names[sample_path]
    if dataset.images_dir is not None:
        absolute_path = osp.join(dataset.images_dir, sample_path)
        if absolute_path in file_names:
            return file_names[absolute_path]
    raise KeyError('evaluation image was not loaded: {}'.format(sample_path))


def fast_test_p_s(
    model,
    all_train_sets,
    all_test_only_sets,
    set_index,
    args,
    logger=None,
    writer=None,
    return_avg=False,
    adapter_routing_bank=None,
):
    print('****** start global-feature evaluation ******')
    loaders = all_train_sets[:set_index + 1] + all_test_only_sets
    results = {}
    model.eval()
    learned_domain_names = {
        info[-1] for info in all_train_sets[:set_index + 1]
    }
    osaf_bank = None
    if getattr(args, 'adapter_routing', 'oracle') == 'osaf':
        if adapter_routing_bank is None:
            raise ValueError(
                'OSAF evaluation requires adapter_routing_bank'
            )
        osaf_bank = _available_adapter_routing_bank(
            model,
            adapter_routing_bank,
        )
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device('cuda')
        osaf_bank = routing_bank_to(
            osaf_bank,
            device=model_device,
            dtype=torch.float32,
        )

    for loader_info in loaders:
        (
            dataset,
            _,
            _,
            _,
            test_loader,
            _,
            name,
        ) = loader_info
        features = CatMeter()
        pids = CatMeter()
        camids = CatMeter()
        file_names = {}
        file_count = 0
        osaf_top1_counts = None
        use_osaf = _use_osaf_for_domain(
            osaf_bank,
            name,
            learned_domain_names,
            args,
        )
        if use_osaf:
            osaf_top1_counts = torch.zeros(
                len(osaf_bank),
                dtype=torch.long,
            )
            print(
                '[eval] OSAF {}-domain routing for {}: Top-K={}, '
                'semantic_weight={}, debias_strength={}, fusion_weight={}.'
                .format(
                    (
                        'seen'
                        if name in learned_domain_names
                        else 'unseen'
                    ),
                    name,
                    min(args.adapter_topk, len(osaf_bank)),
                    args.adapter_semantic_weight,
                    args.adapter_debias_strength,
                    args.adapter_fusion_weight,
                )
            )

        torch.cuda.empty_cache()
        print('{} {} feature extraction started'.format(time_now(), name))
        if use_osaf:
            with torch.no_grad():
                for images, paths, targets, cameras, _ in tqdm(test_loader):
                    images = images.cuda(non_blocking=True)
                    descriptor, route_names, top1_indices = (
                        _osaf_descriptor(
                            model,
                            images,
                            osaf_bank,
                            args,
                        )
                    )
                    osaf_top1_counts += torch.bincount(
                        top1_indices,
                        minlength=len(route_names),
                    )
                    features.update(descriptor.detach())
                    pids.update(targets)
                    camids.update(cameras)
                    for path in paths:
                        file_names[path] = file_count
                        file_count += 1
        else:
            with _evaluation_domain_adapter(model, name):
                with torch.no_grad():
                    for images, paths, targets, cameras, _ in tqdm(
                        test_loader
                    ):
                        images = images.cuda(non_blocking=True)
                        descriptor = F.normalize(model(images), dim=1)
                        features.update(descriptor.detach())
                        pids.update(targets)
                        camids.update(cameras)
                        for path in paths:
                            file_names[path] = file_count
                            file_count += 1

        if use_osaf:
            total_routes = max(int(osaf_top1_counts.sum().item()), 1)
            route_summary = ', '.join(
                '{}={:.1f}%'.format(
                    domain_name,
                    100.0 * int(count.item()) / total_routes,
                )
                for domain_name, count in zip(
                    route_names,
                    osaf_top1_counts,
                )
            )
            print('[eval] OSAF Top-1 routes for {}: {}'.format(
                name,
                route_summary,
            ))

        query_indices = [
            _resolve_feature_index(file_names, dataset, path)
            for path, _, _, _ in dataset.query
        ]
        gallery_indices = [
            _resolve_feature_index(file_names, dataset, path)
            for path, _, _, _ in dataset.gallery
        ]
        all_features = features.get_val()
        all_pids = pids.get_val()
        all_camids = camids.get_val()
        metrics = _rank(
            all_features[query_indices],
            all_features[gallery_indices],
            all_pids[query_indices],
            all_pids[gallery_indices],
            all_camids[query_indices],
            all_camids[gallery_indices],
            args,
        )
        print('{} {} feature extraction finished'.format(time_now(), name))
        print(
            'mAP/R1/R5/R10:\t'
            '{mAP:.1f}/{Rank1:.1f}/{Rank5:.1f}/{Rank10:.1f}'.format(
                **metrics
            )
        )
        for metric_name, value in metrics.items():
            results['{}_{}'.format(name, metric_name)] = value

        if writer is not None:
            writer_names = {
                'mAP': 'mAP',
                'Rank1': 'R@1',
                'Rank5': 'R@5',
                'Rank10': 'R@10',
            }
            for metric_name, writer_name in writer_names.items():
                writer.add_scalar(
                    'results/{}_{}'.format(name, writer_name),
                    metrics[metric_name],
                    set_index,
                )

    averages = print_results(
        args,
        results,
        all_train_sets,
        all_test_only_sets,
        set_index,
        logger=logger,
    )
    if writer is not None:
        (
            seen_map,
            seen_r1,
            unseen_map,
            unseen_r1,
            seen_r5,
            seen_r10,
            unseen_r5,
            unseen_r10,
        ) = averages
        writer.add_scalar('results/Seen-Avg_mAP', seen_map, set_index)
        writer.add_scalar('results/Seen-Avg_R@1', seen_r1, set_index)
        writer.add_scalar('results/Seen-Avg_R@5', seen_r5, set_index)
        writer.add_scalar('results/Seen-Avg_R@10', seen_r10, set_index)
        writer.add_scalar('results/UnSeen-Avg_mAP', unseen_map, set_index)
        writer.add_scalar('results/UnSeen-Avg_R@1', unseen_r1, set_index)
        writer.add_scalar('results/UnSeen-Avg_R@5', unseen_r5, set_index)
        writer.add_scalar('results/UnSeen-Avg_R@10', unseen_r10, set_index)
    if return_avg:
        return averages[0], averages[1]
    return metrics['mAP']


def _mean(values):
    return float(np.round(np.mean(values), 1)) if values else float('nan')


def print_results(
    args,
    rank_map_dict,
    all_train_sets,
    all_test_only_sets,
    set_index,
    logger=None,
):
    del args
    seen_names = [info[-1] for info in all_train_sets[:set_index + 1]]
    unseen_names = [info[-1] for info in all_test_only_sets]

    seen_map = [rank_map_dict['{}_mAP'.format(name)] for name in seen_names]
    seen_r1 = [rank_map_dict['{}_Rank1'.format(name)] for name in seen_names]
    seen_r5 = [rank_map_dict['{}_Rank5'.format(name)] for name in seen_names]
    seen_r10 = [rank_map_dict['{}_Rank10'.format(name)] for name in seen_names]
    unseen_map = [
        rank_map_dict['{}_mAP'.format(name)]
        for name in unseen_names
    ]
    unseen_r1 = [
        rank_map_dict['{}_Rank1'.format(name)]
        for name in unseen_names
    ]
    unseen_r5 = [
        rank_map_dict['{}_Rank5'.format(name)]
        for name in unseen_names
    ]
    unseen_r10 = [
        rank_map_dict['{}_Rank10'.format(name)]
        for name in unseen_names
    ]

    average_seen_map = _mean(seen_map)
    average_seen_r1 = _mean(seen_r1)
    average_seen_r5 = _mean(seen_r5)
    average_seen_r10 = _mean(seen_r10)
    average_unseen_map = _mean(unseen_map)
    average_unseen_r1 = _mean(unseen_r1)
    average_unseen_r5 = _mean(unseen_r5)
    average_unseen_r10 = _mean(unseen_r10)

    metric_order = 'Metric order: mAP/R1/R5/R10'
    seen_header = '\t\t'.join(seen_names + ['Average'])
    seen_results = '\t'.join(
        [
            '{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(m, r1, r5, r10)
            for m, r1, r5, r10 in zip(
                seen_map,
                seen_r1,
                seen_r5,
                seen_r10,
            )
        ]
        + [
            '{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(
                average_seen_map,
                average_seen_r1,
                average_seen_r5,
                average_seen_r10,
            )
        ]
    )
    unseen_header = '\t\t'.join(unseen_names + ['Average'])
    unseen_results = '\t'.join(
        [
            '{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(m, r1, r5, r10)
            for m, r1, r5, r10 in zip(
                unseen_map,
                unseen_r1,
                unseen_r5,
                unseen_r10,
            )
        ]
        + [
            '{:.1f}/{:.1f}/{:.1f}/{:.1f}'.format(
                average_unseen_map,
                average_unseen_r1,
                average_unseen_r5,
                average_unseen_r10,
            )
        ]
    )

    print('Average mAP on seen datasets: {}'.format(average_seen_map))
    print('Average Rank-1 on seen datasets: {}'.format(average_seen_r1))
    print('Average Rank-5 on seen datasets: {}'.format(average_seen_r5))
    print('Average Rank-10 on seen datasets: {}'.format(average_seen_r10))
    print(metric_order)
    print(seen_header)
    print(seen_results)
    print('Average mAP on unseen datasets: {}'.format(average_unseen_map))
    print('Average Rank-1 on unseen datasets: {}'.format(average_unseen_r1))
    print('Average Rank-5 on unseen datasets: {}'.format(average_unseen_r5))
    print('Average Rank-10 on unseen datasets: {}'.format(average_unseen_r10))
    print(metric_order)
    print(unseen_header)
    print(unseen_results)

    if logger is not None:
        for line in (
            metric_order,
            seen_header,
            seen_results,
            metric_order,
            unseen_header,
            unseen_results,
        ):
            if hasattr(logger, 'info'):
                logger.info(line)
            else:
                logger.append(line)

    return (
        average_seen_map,
        average_seen_r1,
        average_unseen_map,
        average_unseen_r1,
        average_seen_r5,
        average_seen_r10,
        average_unseen_r5,
        average_unseen_r10,
    )
