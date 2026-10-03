"""Image-only prototype routing followed by mixed-gallery Top-K retrieval.

Ground truth is used only for protocol filtering and result annotation, never
for choosing adapters, scoring, or category-based candidate pruning.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLORS = dict(correct='#209447', same_category_wrong_identity='#ED921D',
              different_category='#D93438', unknown='#8A8A8A')
STATUS_LABELS = dict(correct='Correct identity',
                     same_category_wrong_identity='Same category, wrong ID',
                     different_category='Different category', unknown='Ground truth unavailable')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def path_key(path):
    return os.path.normcase(str(Path(path).expanduser().resolve()))


@dataclass(frozen=True)
class ImageRecord:
    path: str
    category: str = ''
    source_dataset: str = ''
    original_pid: str = ''
    camid: str = ''
    identity_key: tuple = ()
    protocol: str = 'exclude_self'

    @cached_property
    def resolved_key(self):
        return path_key(self.path)


def from_sample(sample, protocol):
    return ImageRecord(sample.path, sample.category, sample.source_dataset,
                       sample.original_pid, sample.camid, sample.identity_key, protocol)


def deduplicate(records):
    """Preserve the same first-occurrence gallery order as the evaluator."""
    found = {}
    for record in records:
        key = record.resolved_key
        previous = found.get(key)
        if previous is not None:
            fields = ('category', 'identity_key', 'camid')
            if any(getattr(previous, f) != getattr(record, f) for f in fields):
                raise ValueError('Conflicting metadata for image: ' + record.path)
        else:
            found[key] = record
    return list(found.values())


def evaluation_records(views):
    queries = [from_sample(s, v.protocol) for v in views for s in v.query]
    gallery = deduplicate([from_sample(s, v.protocol) for v in views for s in v.gallery])
    known = {}
    for record in queries + gallery:
        key = record.resolved_key
        previous = known.get(key)
        if previous is not None and previous != record:
            raise ValueError('Ambiguous evaluation metadata for image: ' + record.path)
        known[key] = record
    return queries, gallery, known


def query_catalog(queries):
    """One-based image indices, in evaluation-view / CSV row order (not PID)."""
    counts = Counter()
    catalog = []
    for index, query in enumerate(queries, 1):
        counts[query.category] += 1
        catalog.append(dict(global_index=index, category_index=counts[query.category],
                            category=query.category, source_dataset=query.source_dataset,
                            original_pid=query.original_pid, camid=query.camid, path=query.path))
    return catalog


def select_catalog_queries(queries, category=None, indices=None, per_category=1):
    catalog = query_catalog(queries)
    pool = [(q, row) for q, row in zip(queries, catalog) if category is None or q.category == category]
    if not pool:
        raise ValueError('No queries for category {!r}; available categories: {}'.format(
            category, ', '.join(sorted({q.category for q in queries})) or '(none)'))
    if indices is not None:
        if any(i < 1 or i > len(pool) for i in indices):
            raise ValueError('Query indices must be within 1..{} ({}); indices are image positions, not PIDs'.format(
                len(pool), 'category ' + category if category else 'global catalog'))
        return [pool[i - 1][0] for i in dict.fromkeys(indices)]
    return [q for q, row in pool if row['category_index'] <= per_category]


def select_query_specs(queries, specs):
    """Select category:index pairs in exactly the requested display order."""
    selected = []
    for spec in specs:
        category, separator, number = spec.rpartition(':')
        if not separator or not category or not number.isdecimal():
            raise ValueError('Invalid query selection {!r}; use category:index, e.g. person:12'.format(spec))
        selected.extend(select_catalog_queries(queries, category, [int(number)]))
    return selected


def list_query_catalog(args, output):
    """Read only checkpoint/stream metadata: no model construction or GPU use."""
    import torch
    from lreid_dataset.category_stream import load_category_stream
    print('Reading checkpoint and query metadata (no GPU or image extraction)...', flush=True)
    checkpoint = Path(args.run_dir).expanduser().resolve() / 'latest.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state.get('kind') != 'ecpm_pgca_continuous' or state.get('schema_version') != 1
            or state.get('phase') == 'training' or state.get('stage_index', 0) < 1):
        raise ValueError('Query listing requires a completed stage checkpoint')
    stream = load_category_stream(state['settings']['stream_config'])
    if stream.fingerprint != state['stream_fingerprint']:
        raise ValueError('Stream protocol changed')
    stage = stream.stages[state['stage_index'] - 1]
    queries = [from_sample(s, v.protocol) for v in stream.evaluations_at(stage.stage_id, args.split) for s in v.query]
    select_catalog_queries(queries, args.query_category)  # Validate requested category.
    catalog = [r for r in query_catalog(queries) if args.query_category is None or r['category'] == args.query_category]
    fields = ['global_index', 'category_index', 'category', 'source_dataset', 'original_pid', 'camid', 'path']
    shown = catalog[:args.list_limit] if args.list_limit else catalog
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(shown)
    print('Shown {}/{} queries; indices start at 1. Use --list-limit 0 to show all.'.format(len(shown), len(catalog)))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'query_catalog.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(catalog)
        print('Full selected catalog: ' + str(output / 'query_catalog.csv'))
    return catalog


def read_manifest(path, known=None, image_root=None):
    """CSV paths are relative to image_root, or to the CSV's own directory."""
    path = Path(path).expanduser().resolve()
    base = Path(image_root).expanduser().resolve() if image_root else path.parent
    known = known or {}
    records = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'path' not in reader.fieldnames:
            raise ValueError('Manifest must contain a path column: ' + str(path))
        for line, row in enumerate(reader, 2):
            if None in row or not (row.get('path') or '').strip():
                raise ValueError('{}:{}: invalid CSV row'.format(path, line))
            values = {k: (row.get(k) or '').strip() for k in
                      ('category', 'source_dataset', 'original_pid', 'camid', 'protocol')}
            image = Path(row['path'].strip()).expanduser()
            image = (image if image.is_absolute() else base / image).resolve()
            existing = known.get(path_key(image))
            if existing is not None:
                for field, value in values.items():
                    if value and value != getattr(existing, field):
                        raise ValueError('{}:{}: {} conflicts with evaluation metadata'.format(path, line, field))
                records.append(existing)
                continue
            labels = [values[k] for k in ('category', 'source_dataset', 'original_pid')]
            if any(labels) and not all(labels):
                raise ValueError('{}:{}: provide category, source_dataset and original_pid together'.format(path, line))
            protocol = values.pop('protocol') or 'exclude_self'
            if protocol not in ('exclude_self', 'cross_camera'):
                raise ValueError('Unsupported query protocol: ' + protocol)
            if protocol == 'cross_camera' and (not all(labels) or not values['camid']):
                raise ValueError('cross_camera requires full identity labels and camid')
            records.append(ImageRecord(str(image), **values, identity_key=tuple(labels) if all(labels) else (),
                                       protocol=protocol))
    if not records:
        raise ValueError('Empty manifest: ' + str(path))
    return deduplicate(records)


def match_status(query, candidate):
    if not query.identity_key or not candidate.identity_key:
        return 'unknown'
    if query.identity_key == candidate.identity_key:
        return 'correct'
    if query.category == candidate.category:
        return 'same_category_wrong_identity'
    return 'different_category'


def valid_gallery_mask(query, gallery):
    """Exactly the evaluator's self/same-identity same-camera exclusions."""
    import torch
    query_path = query.resolved_key
    valid = []
    for candidate in gallery:
        keep = candidate.resolved_key != query_path
        same = bool(query.identity_key) and candidate.identity_key == query.identity_key
        if query.protocol == 'cross_camera' and same:
            if not candidate.camid:
                raise ValueError('cross_camera requires camera metadata for same-identity gallery images')
            keep = keep and candidate.camid != query.camid
        valid.append(keep)
    return torch.tensor(valid, dtype=torch.bool)


def rank_query(query, gallery, query_feature, gallery_features, gallery_routes, top_k):
    import torch
    from reid.evaluation.prototype_router import normalized
    if top_k < 1 or len(gallery_routes) != len(gallery) or len(gallery_features) != len(gallery):
        raise ValueError('Invalid top_k or inconsistent gallery lengths')
    scores = normalized(gallery_features.cpu()) @ normalized(query_feature.cpu().reshape(1, -1))[0]
    order = torch.argsort(scores, descending=True, stable=True)
    valid = valid_gallery_mask(query, gallery)
    order = order[valid[order]][:top_k].tolist()
    if not order:
        raise ValueError('No valid gallery candidates for query: ' + query.path)
    rows = []
    for rank, index in enumerate(order, 1):
        candidate = gallery[index]
        status = match_status(query, candidate)
        rows.append(dict(rank=rank, gallery_index=index, image=asdict(candidate),
                         predicted_category=gallery_routes[index], similarity=float(scores[index]),
                         status=status, border_color=COLORS[status]))
    positives = sum(bool(query.identity_key) and g.identity_key == query.identity_key and bool(v)
                    for g, v in zip(gallery, valid)) if query.identity_key else None
    return dict(results=rows, valid_gallery_count=int(valid.sum()), known_positive_count=positives,
                annotation_complete=bool(query.identity_key) and all(g.identity_key for g in gallery))


def encode_records(encoder, records, transform, device, batch_size, label):
    import torch
    from PIL import Image
    features, routes = [], []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            tensors = []
            for record in records[start:start + batch_size]:
                with Image.open(record.path) as image:
                    tensors.append(transform(image.convert('RGB')))
            # The inference call receives pixels only: no metadata/labels.
            vectors, predictions, _ = encoder(torch.stack(tensors).to(device))
            features.append(vectors.cpu())
            routes.extend(predictions)
            print('{}: {}/{}'.format(label, min(start + batch_size, len(records)), len(records)), flush=True)
    return torch.cat(features), routes


def check_plotting_dependencies():
    """Fail before model/gallery work if the figure backend cannot be imported."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot  # Import backend dependencies now, not after inference.
        from matplotlib.patches import FancyBboxPatch, Patch
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            'Plotting dependencies are missing or broken: {}. In the SAME Python environment run: '
            'python -m pip install matplotlib Pillow. No model loading or feature extraction has started.'
            .format(error)) from error


def render_result(query, query_route, result, output_base, formats=('png', 'pdf'), dpi=300):
    """Single-row compatibility wrapper; no per-image text is drawn."""
    return render_grid([dict(result, query=asdict(query))], output_base, formats, dpi)


def render_grid(reports, output_base, formats=('png', 'pdf'), dpi=300):
    """One query per row, ranked candidates to its right, one bottom legend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
    from PIL import Image

    if not reports or any(not r['results'] for r in reports):
        raise ValueError('Grid requires at least one query and valid candidates for each row')
    count = max(len(r['results']) for r in reports) + 1
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        # Layout distances are in inches. A wider gap separates query from ranks.
        fig_width = max(7.2, count * 1.08 + .25)
        row_height, row_gap, top, bottom = 1.40, .18, .14, .44
        fig_height = top + bottom + len(reports) * row_height + (len(reports) - 1) * row_gap
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
        margin, gap, query_gap = .15, .16, .16
        width = (fig_width - 2 * margin - gap * (count - 1) - query_gap) / count
        paths = []
        try:
            for row_index, report in enumerate(reports):
                y = fig_height - top - row_height - row_index * (row_height + row_gap)
                items = [(report['query']['path'], '#424242')]
                items += [(r['image']['path'], r['border_color']) for r in report['results']]
                for column, (path, color) in enumerate(items):
                    x = margin + column * (width + gap) + (query_gap if column else 0)
                    ax = fig.add_axes([x / fig_width, y / fig_height, width / fig_width, row_height / fig_height])
                    with Image.open(path) as image:
                        ax.imshow(image.convert('RGB'))
                    ax.set_axis_off()
                    ax.add_patch(FancyBboxPatch((-.025, -.025), 1.05, 1.05,
                        boxstyle='round,pad=0.005,rounding_size=0.045', transform=ax.transAxes,
                        facecolor='none', edgecolor=color, linewidth=2.4, clip_on=False))
            has_unknown = any(r['status'] == 'unknown' for report in reports for r in report['results'])
            statuses = list(COLORS) if has_unknown else list(COLORS)[:3]
            fig.legend(handles=[Patch(facecolor='none', edgecolor=COLORS[s], linewidth=2,
                                      label=STATUS_LABELS[s]) for s in statuses],
                       loc='lower center', bbox_to_anchor=(.5, .025 / fig_height), ncol=len(statuses),
                       frameon=False, fontsize=8)
            for extension in formats:
                target = Path(str(output_base) + '.' + extension)
                fig.savefig(target, dpi=dpi, facecolor='white')
                paths.append(str(target.resolve()))
        finally:
            plt.close(fig)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Completed run directory containing latest.pt and reference.pt')
    parser.add_argument('--output-dir', help='Required for retrieval; optional CSV export directory for --list-queries')
    query = parser.add_mutually_exclusive_group()
    query.add_argument('--query', nargs='+', help='One or more query image paths; known evaluation labels are found automatically')
    query.add_argument('--query-manifest', help='CSV with path and optional ground-truth metadata')
    query.add_argument('--query-index', nargs='+', type=int,
                       help='One-based query image indices; scoped to --query-category if provided, otherwise global')
    query.add_argument('--query-select', nargs='+', metavar='CATEGORY:INDEX',
                       help='Mixed-category queries in row order, e.g. person:12 vehicle:1 tiger:2')
    query.add_argument('--list-queries', action='store_true', help='List query indices and labels without running inference')
    parser.add_argument('--query-category', help='Select queries of this category (person, vehicle, panda, tiger, boat); gallery stays mixed')
    parser.add_argument('--list-limit', type=int, default=50, help='Maximum printed catalog rows; 0 shows all; CSV export always includes all')
    parser.add_argument('--queries-per-category', type=int, default=1,
                        help='Without --query/--query-manifest, use first N test queries per seen category')
    gallery = parser.add_mutually_exclusive_group()
    gallery.add_argument('--gallery-manifest', help='Custom gallery CSV; otherwise use all seen evaluation galleries')
    gallery.add_argument('--gallery-dir', help='Recursively retrieve from images in this directory; unknown labels yield gray borders')
    parser.add_argument('--image-root', help='Root for relative image paths in custom CSV files; default is each CSV directory')
    parser.add_argument('--split', choices=('test', 'validation'), default='test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--beta', type=float, help='Default: saved training evaluation beta')
    parser.add_argument('--summary', choices=('ecpm', 'identity_mean'), help='Default: saved routing_summary')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg'), default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_queries and not args.output_dir:
        parser.error('--output-dir is required for retrieval')
    if args.query_category and (args.query or args.query_manifest or args.query_select):
        parser.error('--query-category is for catalog selection; do not combine with --query/--query-manifest/--query-select')
    if args.list_limit < 0:
        parser.error('--list-limit must be nonnegative')
    if min(args.batch_size, args.top_k, args.queries_per_category, args.dpi) < 1:
        raise ValueError('Batch size, top-k, queries-per-category and dpi must be positive')
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output is not None and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory: ' + str(output))
    if args.list_queries:
        return list_query_catalog(args, output)
    check_plotting_dependencies()

    from tools.evaluate_category_progressive import load_committed_run
    from reid.evaluation.prototype_router import RoutedEncoder
    from reid.evaluation.category_progressive import check_evaluation_coverage, EvaluationConfig
    from reid.utils.progressive_checkpoint import atomic_json, exact_runtime, file_digest

    print('Loading committed model and auditing stream metadata...', flush=True)
    exact_runtime(args.device)
    checkpoint = Path(args.run_dir).expanduser().resolve() / 'latest.pt'
    checkpoint_sha = file_digest(checkpoint)
    state, _, stream, model, memory = load_committed_run(args.run_dir, args.device)
    if file_digest(checkpoint) != checkpoint_sha:
        raise ValueError('Checkpoint changed during loading; use a completed, inactive run')
    stage = stream.stages[state['stage_index'] - 1]
    if (not args.query and not args.query_manifest) or (not args.gallery_manifest and not args.gallery_dir):
        views = check_evaluation_coverage(stream, stage.stage_id, EvaluationConfig(split=args.split, gallery='mixed'))
    else:
        views = stream.evaluations_at(stage.stage_id, args.split)
    default_queries, default_gallery, known = evaluation_records(views)
    references = {q.resolved_key: row for q, row in zip(default_queries, query_catalog(default_queries))}
    if args.query_manifest:
        queries = read_manifest(args.query_manifest, known, args.image_root)
    elif args.query:
        queries = deduplicate([known.get(path_key(p), ImageRecord(str(Path(p).expanduser().resolve()))) for p in args.query])
    elif args.query_select:
        queries = select_query_specs(default_queries, args.query_select)
    else:
        queries = select_catalog_queries(default_queries, args.query_category, args.query_index, args.queries_per_category)
    if args.gallery_manifest:
        gallery = read_manifest(args.gallery_manifest, known, args.image_root)
    elif args.gallery_dir:
        directory = Path(args.gallery_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('Gallery directory does not exist: ' + str(directory))
        paths = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        gallery = deduplicate([known.get(path_key(p), ImageRecord(str(p.resolve()))) for p in paths])
    else:
        gallery = default_gallery
    if not queries or not gallery:
        raise ValueError('No query/gallery images; provide custom inputs or configure the selected evaluation split')
    for record in queries + gallery:
        if not Path(record.path).is_file():
            raise FileNotFoundError('Image does not exist: ' + record.path)
    for query in queries:
        if query.protocol == 'cross_camera':
            valid_gallery_mask(query, gallery)  # Fail early on incomplete camera metadata.

    beta = state['settings'].get('beta', .5) if args.beta is None else args.beta
    summary = args.summary or state['settings'].get('routing_summary', 'ecpm')
    encoder = RoutedEncoder(model, memory, beta, summary)
    _, transform = model.make_transforms()
    output.mkdir(parents=True, exist_ok=True)
    print('Retrieving {} queries against {} gallery images; no category pruning.'.format(len(queries), len(gallery)), flush=True)
    gallery_features, gallery_routes = encode_records(encoder, gallery, transform, args.device, args.batch_size, 'Gallery')
    query_features, query_routes = encode_records(encoder, queries, transform, args.device, args.batch_size, 'Query')
    reports = []
    for index, (query, route, feature) in enumerate(zip(queries, query_routes, query_features), 1):
        result = rank_query(query, gallery, feature, gallery_features, gallery_routes, args.top_k)
        name = 'query_{:04d}'.format(index)
        result.update(query=asdict(query), query_reference=references.get(query.resolved_key),
                      predicted_category=route, figure_row=index, figure_files=[])
        reports.append(result)
        if not result['annotation_complete']:
            print('NOTE {}: incomplete ground truth; unknown matches are gray, not inferred from routing.'.format(name), flush=True)
        if result['known_positive_count'] == 0:
            print('NOTE {}: no known valid same-identity gallery image; this is similarity retrieval.'.format(name), flush=True)
    report = dict(schema_version=1, routing='prototype', gallery_scope='mixed',
                  gallery_source='custom' if args.gallery_manifest or args.gallery_dir else args.split + '_evaluation',
                  checkpoint_sha256=checkpoint_sha, run_dir=str(checkpoint.parent), stage_id=stage.stage_id,
                  stage_index=state['stage_index'] - 1, stream_fingerprint=stream.fingerprint,
                  seen_categories=list(model.categories), beta=beta, routing_summary=summary,
                  top_k=args.top_k, gallery_images=len(gallery),
                  category_filter=False, score='cosine_similarity',
                  query_selection=('explicit' if args.query or args.query_manifest else
                                   'category_index_pairs' if args.query_select else
                                   'catalog_indices' if args.query_index else 'first_queries_per_category'),
                  query_category=args.query_category, requested_query_indices=args.query_index,
                  requested_query_selections=args.query_select, figure_layout='query_rows_shared_legend',
                  figure_files=[],
                  color_legend={s: dict(color=c, meaning=STATUS_LABELS[s]) for s, c in COLORS.items()},
                  queries=reports)
    atomic_json(report, output / 'retrieval_results.json')
    print('Rendering {} query rows with one shared legend...'.format(len(reports)), flush=True)
    files = render_grid(reports, output / 'retrieval_grid', args.formats, args.dpi)
    report['figure_files'] = files
    for row in reports:
        row['figure_files'] = files
    atomic_json(report, output / 'retrieval_results.json')
    print('Saved ' + ', '.join(files), flush=True)
    print('Done: ' + str(output / 'retrieval_results.json'), flush=True)
    return report


if __name__ == '__main__':
    main()
