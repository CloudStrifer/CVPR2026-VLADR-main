"""Post-training, label-free retrieval on MSMT17 and DogFaceNet.

No optimization, adapter creation, prototype update, or target-domain fitting.
MSMT17 uses the local Market-style test folders. DogFaceNet uses its supplied
test identity list by default, with a documented custom retrieval split.
Use --audit-only to inspect protocols without opening images or loading torch.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lreid_dataset.category_stream import EvaluationView, StreamSample

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _path_key(path):
    # All records are canonical absolute paths from sample()/the stream parser.
    # Avoid repeatedly resolving 94k files against the filesystem during audits.
    return os.path.normcase(str(path))


def sample(path, category, dataset, pid, camera, split):
    return StreamSample(str(Path(path).resolve()), category, dataset, str(pid), str(camera),
                        (category, dataset, str(pid)), 'unseen_evaluation', split)


def msmt17_view(root, allow_subset=False):
    # Reuse the existing metadata-only parser, not the legacy training pipeline.
    from tools.build_msmt17_unseen_manifest import (build_msmt17_unseen_rows,
                                                   validate_standard_counts, MSMT17ManifestError)
    rows, audit = build_msmt17_unseen_rows(root)
    standard = True
    try:
        validate_standard_counts(audit)
    except MSMT17ManifestError:
        standard = False
        if not allow_subset:
            raise
    root = Path(root).resolve()
    if any(not 1 <= int(r['camid']) <= 15 for r in rows):
        raise ValueError('MSMT17 camera IDs must be in 1..15')
    records = [sample(root / r['path'], 'person', 'msmt17', r['pid'], r['camid'], r['split']) for r in rows]
    view = EvaluationView('msmt17_unseen', 'person', 'test', 'cross_camera',
                          tuple(s for s in records if s.split == 'query'),
                          tuple(s for s in records if s.split == 'gallery'))
    audit.update(dataset='MSMT17', protocol='cross_camera', standard_counts_match=standard,
                 split_description='Provided query / bounding_box_test; exclude same-ID same-camera gallery images',
                 ignored_training_directory='bounding_box_train')
    return view, audit


def class_ids(path):
    tokens = Path(path).read_text(encoding='utf-8-sig').split()
    if not tokens or any(not re.fullmatch(r'[0-9]+', token) for token in tokens):
        raise ValueError('Expected a nonempty numeric identity list: ' + str(path))
    # Official files repeat each class once per image. They are not image paths.
    return set(tokens)


def dogfacenet_view(root, class_split='test', protocol='one_query', seed=42):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    folders = {p.name: p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]+', p.name)}
    test_file, train_file = root / 'classes_test.txt', root / 'classes_train.txt'
    if class_split == 'test':
        selected = class_ids(test_file)
        if train_file.exists() and selected & class_ids(train_file):
            raise ValueError('DogFaceNet train/test identity lists overlap')
    else:
        selected = set(folders)
    if not selected or selected - set(folders):
        raise ValueError('DogFaceNet selected identities have missing folders: ' + str(sorted(selected - set(folders))[:10]))
    queries, gallery, excluded = [], [], []
    for pid in sorted(selected):
        images = sorted(p for p in folders[pid].iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < 2:
            excluded.append(dict(identity=pid, images=len(images), reason='fewer_than_two_images'))
            continue
        if protocol == 'one_query':
            digest = hashlib.sha256(('{}\0{}'.format(seed, pid)).encode()).hexdigest()
            q_index = random.Random(int(digest[:16], 16)).randrange(len(images))
            q_paths = [images[q_index]]
            g_paths = images[:q_index] + images[q_index + 1:]
        elif protocol == 'leave_one_out':
            q_paths, g_paths = images, images
        else:
            raise ValueError('Unknown DogFaceNet protocol: ' + protocol)
        # No camera metadata is available; do not manufacture camera IDs.
        queries.extend(sample(p, 'dog', 'dogfacenet', pid, 'unknown', 'query') for p in q_paths)
        gallery.extend(sample(p, 'dog', 'dogfacenet', pid, 'unknown', 'gallery') for p in g_paths)
    view = EvaluationView('dogfacenet_unseen', 'dog', 'test', 'exclude_self', tuple(queries), tuple(gallery))
    audit = dict(dataset='DogFaceNet', root=str(root), class_split=class_split, protocol='exclude_self',
                 retrieval_split=protocol, seed=seed if protocol == 'one_query' else None,
                 supplied_test_list_used=class_split == 'test',
                 selected_identities=len(selected), excluded_identities=excluded, training_images_used=0,
                 split_description=('Custom retrieval: one seeded query per identity, all remaining images in gallery'
                                    if protocol == 'one_query' else 'Custom retrieval: all images query/gallery, exclude self'),
                 camera_metadata_available=False, original_benchmark_protocol=False)
    return view, audit


def audit_view(view, root):
    if not view.query or not view.gallery:
        raise ValueError(view.name + ': query and gallery must both be nonempty')
    root = Path(root).resolve()
    gallery_paths, by_identity = {}, defaultdict(list)
    for subset in ('query', 'gallery'):
        seen = set()
        for s in getattr(view, subset):
            key = _path_key(s.path)
            if key in seen:
                raise ValueError(view.name + ': duplicate path within ' + subset + ': ' + s.path)
            seen.add(key)
            if not Path(s.path).is_relative_to(root):
                raise ValueError('Evaluation image resolves outside its dataset root: ' + s.path)
            if subset == 'gallery':
                gallery_paths[key] = s
                by_identity[s.identity_key].append((key, s.camid))
    for q in view.query:
        if not any(path != _path_key(q.path) and (view.protocol != 'cross_camera' or camera != q.camid)
                   for path, camera in by_identity[q.identity_key]):
            raise ValueError(view.name + ': query has no valid gallery positive: ' + q.path)
        shared = gallery_paths.get(_path_key(q.path))
        if shared and (shared.identity_key != q.identity_key or shared.camid != q.camid):
            raise ValueError('Conflicting query/gallery metadata for ' + q.path)
    return dict(query_images=len(view.query), gallery_images=len(view.gallery),
                query_identities=len({s.identity_key for s in view.query}),
                gallery_identities=len(by_identity), unique_images=len({_path_key(s.path) for s in view.query + view.gallery}),
                query_gallery_path_overlap=len({_path_key(s.path) for s in view.query} & set(gallery_paths)),
                queries_without_valid_positive=0)


def training_overlap_audit(stream, roots, views):
    evaluation_keys = {s.identity_key for v in views for s in v.query + v.gallery}
    training_count = 0
    for stage in stream.stages:
        for category in stage.categories:
            for s in category.samples:
                training_count += 1
                path = Path(s.path)  # Stream records are already resolved.
                source = re.sub(r'[^a-z0-9]', '', s.source_dataset.lower())
                if (any(path.is_relative_to(root) or name in source for name, root in roots.items())
                        or s.identity_key in evaluation_keys):
                    raise ValueError('Dataset is not unseen: training metadata includes ' + s.path)
    return dict(status='passed', training_images_checked=training_count,
                checks=['dataset_root', 'source_dataset_name', 'identity_key'],
                image_content_duplicate_check=False)


def manifest_rows(view, root):
    for subset in ('query', 'gallery'):
        for s in getattr(view, subset):
            yield dict(path=s.path, relative_path=Path(s.path).relative_to(root).as_posix(), split=subset,
                       category=s.category, source_dataset=s.source_dataset, original_pid=s.original_pid,
                       camid=s.camid, identity_key=json.dumps(s.identity_key, ensure_ascii=False))


def manifest_fingerprint(view, root):
    digest = hashlib.sha256()
    for row in manifest_rows(view, root):
        row.pop('path')  # Identical dataset copies have identical protocol fingerprints.
        digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_protocols(output, views, audits):
    output.mkdir(parents=True, exist_ok=True)
    for view in views:
        rows = manifest_rows(view, Path(audits[view.name]['root']))
        with (output / (view.name + '_manifest.csv')).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['path', 'relative_path', 'split', 'category',
                'source_dataset', 'original_pid', 'camid', 'identity_key'])
            writer.writeheader()
            writer.writerows(rows)
    save_json(output / 'protocol_audit.json', audits)


def unique_samples(views):
    records, positions = [], {}
    for view in views:
        for s in view.query + view.gallery:
            key = _path_key(s.path)
            if key in positions:
                old = records[positions[key]]
                if (s.identity_key, s.category, s.camid) != (old.identity_key, old.category, old.camid):
                    raise ValueError('Conflicting metadata for ' + s.path)
            else:
                positions[key] = len(records)
                records.append(s)
    return records, positions


def extract_features(model, memory, views, modes, beta, summary, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader
    from lreid_dataset.category_stream_loaders import CategoryImageDataset, collate_category_samples
    from reid.evaluation.prototype_router import RoutedEncoder, normalized
    records, positions = unique_samples(views)
    labels = {key: i for i, key in enumerate(sorted({s.identity_key for s in records}))}
    _, transform = model.make_transforms()
    loader = DataLoader(CategoryImageDataset(records, labels, transform), batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=workers, collate_fn=collate_category_samples,
        pin_memory=torch.device(device).type == 'cuda')
    encoder = RoutedEncoder(model, memory, beta, summary) if 'prototype' in modes else None
    tables = {mode: torch.empty((len(records), model.feature_dim), dtype=torch.float32) for mode in modes}
    predictions = []
    was_training, offset, started = model.training, 0, time.perf_counter()
    last_progress = started
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch['images'].to(device, non_blocking=True)
                end = offset + len(images)
                # Ground-truth category/identity metadata NEVER enters inference.
                if encoder is not None:
                    descriptors, routes, _ = encoder(images)
                    tables['prototype'][offset:end] = descriptors.cpu()
                    predictions.extend(routes)
                if 'reference' in modes:
                    tables['reference'][offset:end] = normalized(model.encode_reference(images)).cpu()
                offset = end
                now = time.perf_counter()
                if offset == len(images) or offset == len(records) or now - last_progress >= 10:
                    print('Features: {}/{} unique images ({:.1f}s)'.format(offset, len(records), now - started), flush=True)
                    last_progress = now
    finally:
        model.train(was_training)
    return records, positions, tables, predictions


def route_report(view, positions, predictions, categories):
    report = {}
    for subset in ('query', 'gallery'):
        samples = getattr(view, subset)
        routes = [predictions[positions[_path_key(s.path)]] for s in samples]
        report[subset] = dict(counts={c: routes.count(c) for c in categories}, images=len(routes),
            category_accuracy=(100 * sum(r == view.category for r in routes) / len(routes)
                               if view.category in categories else None))
    report['category_accuracy_note'] = ('True category is unseen; routing is selection of an existing expert, '
                                       'not prediction or rejection of a novel class.' if view.category not in categories
                                       else 'Diagnostic only; no category label is supplied to inference.')
    return report


def save_metrics_csv(path, results):
    fields = ['dataset', 'encoder', 'gallery_scope', 'queries', 'gallery_images', 'mAP', 'Rank1']
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, scopes in results.items():
            for scope, datasets in scopes.items():
                for name, metric in datasets.items():
                    writer.writerow(dict(dataset=name, encoder=mode, gallery_scope=scope,
                        **{key: metric[key] for key in ('queries', 'gallery_images', 'mAP', 'Rank1')}))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', help='Completed training directory with latest.pt and reference.pt')
    parser.add_argument('--output-dir', required=True, help='New or empty directory')
    parser.add_argument('--datasets', nargs='+', choices=['msmt17', 'dogfacenet'], default=['msmt17', 'dogfacenet'])
    parser.add_argument('--msmt17-root', default=str(ROOT / 'data/MSMT17'))
    parser.add_argument('--dogfacenet-root', default=str(ROOT / 'data/DogFaceNet'))
    parser.add_argument('--dog-split', choices=['test', 'all'], default='test')
    parser.add_argument('--dog-protocol', choices=['one_query', 'leave_one_out'], default='one_query')
    parser.add_argument('--seed', type=int, default=42, help='DogFaceNet query selection only')
    parser.add_argument('--allow-msmt-subset', action='store_true', help='Explicitly permit nonstandard MSMT17 counts')
    parser.add_argument('--audit-only', action='store_true', help='Metadata only; no images/model/GPU')
    parser.add_argument('--stream-config', help='Audit-only: optionally check absence from this training stream')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--ranking-threads', type=int, default=1, help='CPU threads during memory-bounded ranking')
    parser.add_argument('--encoder', choices=['prototype', 'reference', 'both'], default='prototype')
    parser.add_argument('--gallery', choices=['per_dataset', 'mixed', 'both'], default='per_dataset',
                        help='mixed combines ONLY the selected unseen datasets')
    parser.add_argument('--beta', type=float)
    parser.add_argument('--summary', choices=['ecpm', 'identity_mean'])
    parser.add_argument('--save-features', action='store_true', help='Also save CPU feature tables and row metadata')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.audit_only and not args.run_dir:
        parser.error('--run-dir is required unless --audit-only is used')
    if args.stream_config and not args.audit_only:
        parser.error('--stream-config is only for --audit-only; model evaluation uses its bound training stream')
    if min(args.batch_size, args.ranking_threads) < 1 or args.workers < 0 or args.seed < 0:
        parser.error('batch-size/ranking-threads must be positive; workers/seed must be nonnegative')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('--datasets must not contain duplicates')
    if args.beta is not None and not 0 <= args.beta <= 1:
        parser.error('--beta must be in [0,1]')
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory: ' + str(output))
    views, audits, roots = [], {}, {}
    for name in args.datasets:
        root = Path(getattr(args, name + '_root')).expanduser().resolve()
        print('Auditing ' + name + ' (filenames and labels only)...', flush=True)
        view, audit = (msmt17_view(root, args.allow_msmt_subset) if name == 'msmt17' else
                       dogfacenet_view(root, args.dog_split, args.dog_protocol, args.seed))
        audit.update(audit_view(view, root), manifest_sha256=manifest_fingerprint(view, root))
        views.append(view)
        audits[view.name], roots[name] = audit, root
        print('{}: {} query / {} gallery / {} query identities'.format(
            name, len(view.query), len(view.gallery), audit['query_identities']), flush=True)
    if args.audit_only:
        overlap = dict(status='not_checked', reason='No training stream supplied')
        if args.stream_config:
            from lreid_dataset.category_stream import load_category_stream
            overlap = training_overlap_audit(load_category_stream(args.stream_config), roots, views)
        save_protocols(output, views, audits)
        save_json(output / 'audit_result.json', dict(datasets=audits, training_overlap=overlap))
        print('Audit complete: ' + str(output), flush=True)
        return dict(datasets=audits, training_overlap=overlap)

    import torch
    from tools.evaluate_category_progressive import load_committed_run
    from reid.evaluation.category_oracle import oracle_retrieval_metrics
    from reid.utils.progressive_checkpoint import exact_runtime, file_digest
    directory = Path(args.run_dir).expanduser().resolve()
    digests = {name: file_digest(directory / name) for name in ('latest.pt', 'reference.pt')}
    runtime = exact_runtime(args.device)
    print('Loading the completed model...', flush=True)
    state, reference, stream, model, memory = load_committed_run(directory, args.device)
    if state['phase'] != 'complete' or state['stage_index'] != len(stream.stages):
        raise ValueError('Unseen evaluation requires completion of ALL configured training stages')
    overlap = training_overlap_audit(stream, roots, views)
    for name, digest in digests.items():
        if file_digest(directory / name) != digest:
            raise ValueError('Checkpoint changed during loading: ' + name)
    beta = state['settings'].get('beta', .5) if args.beta is None else args.beta
    summary = args.summary or state['settings'].get('routing_summary', 'ecpm')
    modes = ['prototype', 'reference'] if args.encoder == 'both' else [args.encoder]
    scopes = ['per_dataset', 'mixed'] if args.gallery == 'both' else [args.gallery]
    report = dict(schema_version=1, status='extracting', run_dir=str(directory), checkpoint_sha256=digests,
        stage_id=stream.stages[-1].stage_id, training_stream_fingerprint=stream.fingerprint,
        runtime=runtime, dataset_audits=audits, training_overlap=overlap,
        seen_categories=list(model.categories), beta=beta, routing_summary=summary,
        target_adaptation=False, target_training_images_used=0, results={},
        mixed_gallery_datasets=[v.name for v in views],
        generalization={v.name: ('unseen_dataset_seen_category' if v.category in model.categories
                                else 'unseen_dataset_and_category') for v in views},
        ranking=dict(similarity='cosine', ties='stable_gallery_order', cpu_threads=args.ranking_threads),
        interpretation='Retrieval among enrolled target identities; not open-set unknown rejection. '
                       'DogFaceNet retrieval split is custom, not its original verification benchmark.')
    del reference, state
    save_protocols(output, views, audits)
    save_json(output / 'results.json', report)
    records, positions, features, predictions = extract_features(model, memory, views, modes, beta, summary,
                                                               args.device, args.batch_size, args.workers)
    if predictions:
        report['routing'] = {v.name: route_report(v, positions, predictions, list(model.categories)) for v in views}
    if args.save_features:
        from dataclasses import asdict
        torch.save(dict(features=features, samples=[asdict(s) for s in records], predicted_categories=predictions,
                        checkpoint_sha256=digests, beta=beta, routing_summary=summary,
                        manifest_sha256={name: a['manifest_sha256'] for name, a in audits.items()}), output / 'features.pt')
    report['status'] = 'ranking'
    save_json(output / 'results.json', report)
    merged, seen = [], set()
    for view in views:
        for s in view.gallery:
            key = _path_key(s.path)
            if key not in seen:
                merged.append(s)
                seen.add(key)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.ranking_threads)
    try:
        for mode in modes:
            report['results'][mode] = {}
            table = features[mode]
            for scope in scopes:
                report['results'][mode][scope] = {}
                for view in views:
                    evaluation = replace(view, gallery=tuple(merged)) if scope == 'mixed' else view
                    q = table[[positions[_path_key(s.path)] for s in evaluation.query]]
                    g = table[[positions[_path_key(s.path)] for s in evaluation.gallery]]
                    def progress(update):
                        print(json.dumps(dict(update, dataset=view.name, encoder=mode, gallery_scope=scope)), flush=True)
                    metric = oracle_retrieval_metrics(q, g, evaluation, progress=progress)
                    # The helper's name is historical; only scoring is reused, no oracle encoder is used.
                    metric.update(routing=mode, gallery_scope=scope, gallery_images=len(evaluation.gallery))
                    report['results'][mode][scope][view.name] = metric
                    save_json(output / 'results.json', report)
                    save_metrics_csv(output / 'metrics.csv', report['results'])
                    del q, g
    finally:
        torch.set_num_threads(previous_threads)
    for name, digest in digests.items():
        if file_digest(directory / name) != digest:
            raise ValueError('Checkpoint changed during evaluation: ' + name)
    report.update(status='complete', checkpoint_files_unchanged=True)
    save_json(output / 'results.json', report)
    print('Done: ' + str(output / 'metrics.csv'), flush=True)
    return report


if __name__ == '__main__':
    main()
