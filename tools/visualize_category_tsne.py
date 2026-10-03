"""Visualize image-only, prototype-routed ReID features from a completed run.

Overview: one joint embedding, colored by category. Identity panels: a separate
embedding per category, colored by identity. Labels never enter the encoder or
t-SNE objective. Cached features allow redraws without loading a model/images.
"""

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CATEGORIES = ['person', 'vehicle', 'panda', 'tiger', 'boat']
NAMES = dict(person='Market1501', vehicle='VeRi', panda='iPanda50', tiger='ATRW', boat='Boat')
CATEGORY_COLORS = dict(person='#4477AA', vehicle='#EE9944', panda='#228833', tiger='#AA3377', boat='#66AACC')


def local_rng(seed, category, purpose):
    digest = hashlib.sha256('{}\0{}\0{}'.format(seed, category, purpose).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def camera_interleaved(records, rng):
    """Randomize within cameras and visit different cameras before repeating."""
    groups = defaultdict(list)
    for record in records:
        groups[record.camid].append(record)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while any(groups.values()):
        for camera in keys:
            if groups[camera]:
                result.append(groups[camera].pop())
    return result


def sample_images(pool, categories, per_category=150, ids_per_category=10, images_per_id=10, seed=42):
    """Metadata-only sampling, independent of model features or success labels."""
    from tools.visualize_category_retrieval import deduplicate
    pool = deduplicate(pool)
    overview, identities, counts = [], [], {}
    for category in categories:
        groups = defaultdict(list)
        for record in pool:
            if record.category == category:
                groups[record.identity_key].append(record)
        if not groups:
            raise ValueError('No evaluation images for requested category: ' + category)
        rng = local_rng(seed, category, 'overview')
        keys = sorted(groups)
        rng.shuffle(keys)
        queues = {key: camera_interleaved(list(groups[key]), rng) for key in keys}
        chosen = []
        # Round-robin identities prevents abundant identities dominating the plot.
        while len(chosen) < per_category and any(queues.values()):
            for key in keys:
                if queues[key] and len(chosen) < per_category:
                    chosen.append(queues[key].pop(0))
        overview.extend(chosen)
        rng = local_rng(seed, category, 'identity_panels')
        eligible = sorted(key for key, records in groups.items() if len(records) >= 2)
        rng.shuffle(eligible)
        selected_ids = eligible[:ids_per_category]
        selected = []
        for key in selected_ids:
            selected.extend(camera_interleaved(list(groups[key]), rng)[:images_per_id])
        identities.extend(selected)
        counts[category] = dict(pool_images=sum(map(len, groups.values())), pool_identities=len(groups),
                                overview_images=len(chosen), identity_panel_images=len(selected),
                                eligible_identities=len(eligible), selected_identities=len(selected_ids))
    union = deduplicate(overview + identities)
    overview_paths = {r.resolved_key for r in overview}
    identity_paths = {r.resolved_key for r in identities}
    samples = [dict(asdict(record), feature_index=i, overview=record.resolved_key in overview_paths,
                    identity_panel=record.resolved_key in identity_paths) for i, record in enumerate(union)]
    return union, samples, counts


def check_dependencies():
    from tools.visualize_category_retrieval import check_plotting_dependencies
    check_plotting_dependencies()
    try:
        import numpy
        import sklearn
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError('Missing t-SNE dependencies. In this Python environment run: '
                           'python -m pip install numpy scikit-learn matplotlib Pillow') from error
    return dict(numpy=numpy.__version__, sklearn=sklearn.__version__)


def embed(features, seed=42, perplexity=30., max_iter=1000):
    import numpy as np
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    features = np.asarray(features, dtype=np.float32)
    n = len(features)
    settings = dict(seed=seed, requested_perplexity=perplexity, metric='cosine',
                    init='pca', learning_rate='auto', max_iter=max_iter, samples=n, cpu_threads=1)
    if n < 3:
        return None, dict(settings, status='skipped_too_few_samples')
    if not np.isfinite(features).all() or features.ndim != 2 or features.shape[1] < 2:
        raise ValueError('t-SNE requires finite [N,D] features with D >= 2')
    if np.max(np.abs(features - features[0])) < 1e-8:
        return None, dict(settings, status='skipped_identical_features')
    effective = min(float(perplexity), max(1., (n - 1) / 3.))
    kwargs = dict(n_components=2, metric='cosine', init='pca', random_state=seed,
                  perplexity=effective, learning_rate='auto')
    # Compatible with sklearn releases before/after the n_iter rename.
    iteration_key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
    kwargs[iteration_key] = max_iter
    estimator = TSNE(**kwargs)
    print('t-SNE: {} images, perplexity={:g}, seed={}'.format(n, effective, seed), flush=True)
    # These small panels run faster without large OpenMP/BLAS thread pools.
    with threadpool_limits(limits=1):
        coordinates = estimator.fit_transform(features)
    if not np.isfinite(coordinates).all():
        raise FloatingPointError('t-SNE produced non-finite coordinates')
    return coordinates, dict(settings, status='complete', effective_perplexity=effective,
                             kl_divergence=float(estimator.kl_divergence_), iterations=int(estimator.n_iter_))


def write_coordinates(path, samples, predictions, indices, coordinates):
    fields = ['feature_index', 'path', 'category', 'source_dataset', 'original_pid', 'camid',
              'identity_key', 'predicted_category', 'x', 'y']
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, index in enumerate(indices):
            sample = samples[index]
            row = {k: sample[k] for k in fields if k in sample}
            row.update(identity_key=json.dumps(sample['identity_key'], ensure_ascii=False),
                       predicted_category=str(predictions[index]),
                       x='' if coordinates is None else float(coordinates[position, 0]),
                       y='' if coordinates is None else float(coordinates[position, 1]))
            writer.writerow(row)


def plot_embeddings(samples, panels, categories, output, formats, dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    def clean(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def save(fig, name):
        paths = []
        try:
            for extension in formats:
                path = output / (name + '.' + extension)
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                paths.append(str(path.resolve()))
        finally:
            plt.close(fig)
        return paths

    files = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        panel = panels['overview']
        if panel['coordinates'] is None:
            ax.text(.5, .5, panel['settings']['status'].replace('_', ' '), ha='center', transform=ax.transAxes)
        else:
            xy = panel['coordinates']
            labels = [samples[i]['category'] for i in panel['indices']]
            fallback = plt.get_cmap('tab10')
            for c_index, category in enumerate(categories):
                mask = np.array([c == category for c in labels])
                ax.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=.8, linewidths=0,
                           color=CATEGORY_COLORS.get(category, fallback(c_index % 10)),
                           label=NAMES.get(category, category))
            fig.legend(loc='lower center', bbox_to_anchor=(.5, .015), ncol=min(3, len(categories)), frameon=False)
        clean(ax)
        fig.subplots_adjust(bottom=.19, top=.97)
        files['overview'] = save(fig, 'tsne_categories')

        for category in categories:
            fig, ax = plt.subplots(figsize=(4.0, 4.2))
            panel = panels[category]
            clean(ax)
            ids = sorted({tuple(samples[i]['identity_key']) for i in panel['indices']})
            ax.set_title(NAMES.get(category, category), fontsize=10)
            if panel['coordinates'] is None:
                ax.text(.5, .5, panel['settings']['status'].replace('_', '\n'), ha='center', va='center', transform=ax.transAxes)
            else:
                xy = panel['coordinates']
                colors = plt.get_cmap('tab10' if len(ids) <= 10 else 'tab20' if len(ids) <= 20 else 'hsv')
                labels = [tuple(samples[i]['identity_key']) for i in panel['indices']]
                for i, identity in enumerate(ids):
                    mask = np.array([key == identity for key in labels])
                    color = colors(i if len(ids) <= 20 else i / len(ids))
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=17, alpha=.85, linewidths=0, color=color,
                               label='ID {}'.format(i + 1))
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.035), ncol=3, fontsize=8,
                          frameon=False, handletextpad=.3, columnspacing=.7)
            fig.subplots_adjust(left=.06, right=.94, top=.91, bottom=.24)
            dataset_name = re.sub(r'[^a-z0-9_-]', '_', NAMES.get(category, category).lower())
            files['identities_' + category] = save(fig, 'tsne_identities_' + dataset_name)
    return files


def load_feature_cache(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as cache:
        features = cache['features'].copy()
        predictions = cache['predicted_categories'].copy()
        manifest = json.loads(str(cache['manifest_json'].item()))
    samples = manifest['samples']
    if (manifest.get('schema_version') != 1 or features.ndim != 2 or features.shape[1] < 2
            or len(features) != len(samples) or predictions.shape != (len(samples),)
            or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1., atol=1e-5)):
        raise ValueError('Invalid feature cache: expected finite normalized features aligned with sample metadata')
    for i, sample in enumerate(samples):
        if sample['feature_index'] != i or sample['category'] not in manifest['categories']:
            raise ValueError('Feature cache metadata does not match feature rows')
    return features, predictions, manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', help='Completed run directory containing latest.pt and reference.pt')
    source.add_argument('--features-file', help='Previously exported features.npz; no model, images, or GPU required')
    parser.add_argument('--output-dir', required=True, help='New or empty output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--categories', nargs='+', default=DEFAULT_CATEGORIES)
    parser.add_argument('--images-per-category', type=int, default=150)
    parser.add_argument('--ids-per-category', type=int, default=10)
    parser.add_argument('--images-per-id', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perplexity', type=float, default=30.)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--summary', choices=['ecpm', 'identity_mean'])
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (min(args.images_per_category, args.ids_per_category, args.batch_size, args.dpi) < 1
            or args.images_per_id < 2 or args.max_iter < 250 or args.seed < 0 or args.seed > 2 ** 32 - 1
            or not math.isfinite(args.perplexity) or args.perplexity <= 0):
        raise ValueError('Invalid sampling/t-SNE settings: positive counts, images-per-id >= 2, max-iter >= 250, uint32 seed required')
    if len(set(args.categories)) != len(args.categories):
        raise ValueError('Categories must be distinct')
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new or empty output directory: ' + str(output))
    versions = check_dependencies()
    import numpy as np

    if args.features_file:
        features, predictions, manifest = load_feature_cache(args.features_file)
        print('Loaded cached features; retaining original sample selection, categories and routing settings.', flush=True)
        source_cache = str(Path(args.features_file).expanduser().resolve())
    else:
        from tools.evaluate_category_progressive import load_committed_run
        from tools.visualize_category_retrieval import evaluation_records, deduplicate, encode_records
        from reid.evaluation.prototype_router import RoutedEncoder
        from reid.utils.progressive_checkpoint import exact_runtime, file_digest
        exact_runtime(args.device)
        checkpoint = Path(args.run_dir).expanduser().resolve() / 'latest.pt'
        checksum = file_digest(checkpoint)
        print('Loading committed model and auditing stream...', flush=True)
        state, reference, stream, model, memory = load_committed_run(checkpoint.parent, args.device)
        if file_digest(checkpoint) != checksum:
            raise ValueError('Checkpoint changed during loading; use an inactive completed run')
        missing = set(args.categories) - set(model.categories)
        if missing:
            raise ValueError('Categories not seen by this checkpoint: {}. Use a final model or set --categories.'.format(sorted(missing)))
        stage = stream.stages[state['stage_index'] - 1]
        queries, gallery, _ = evaluation_records(stream.evaluations_at(stage.stage_id, args.split))
        records, samples, counts = sample_images(deduplicate(queries + gallery), args.categories,
            args.images_per_category, args.ids_per_category, args.images_per_id, args.seed)
        beta = state['settings'].get('beta', .5) if args.beta is None else args.beta
        summary = args.summary or state['settings'].get('routing_summary', 'ecpm')
        encoder = RoutedEncoder(model, memory, beta, summary)
        for record in records:
            if not Path(record.path).is_file():
                raise FileNotFoundError(record.path)
        manifest = dict(schema_version=1, run_dir=str(checkpoint.parent), checkpoint_sha256=checksum,
            stage_id=stage.stage_id, stage_index=state['stage_index'] - 1, stream_fingerprint=stream.fingerprint,
            categories=args.categories, split=args.split, pool='query_gallery_union_deduplicated',
            feature='l2_normalized_prototype_routed_descriptor', routing_summary=summary, beta=beta,
            seen_categories=list(model.categories), sample_seed=args.seed,
            sampling=dict(images_per_category=args.images_per_category, ids_per_category=args.ids_per_category,
                          images_per_id=args.images_per_id, overview='identity_round_robin',
                          identity_panels='random_eligible_identities_camera_interleaved'),
            sample_counts=counts, samples=samples)
        print(json.dumps(counts, ensure_ascii=False), flush=True)
        print('Extracting {} unique sampled images (not the full gallery)...'.format(len(records)), flush=True)
        _, transform = model.make_transforms()
        tensors, predicted = encode_records(encoder, records, transform, args.device, args.batch_size, 'Features')
        features, predictions = tensors.numpy(), np.array(predicted)
        source_cache = str(output / 'features.npz')
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'features.npz', features=features, predicted_categories=predictions,
                            manifest_json=np.array(json.dumps(manifest, ensure_ascii=False)))
        del tensors, encoder, model, memory, state, reference

    output.mkdir(parents=True, exist_ok=True)
    # The cache-only path deliberately avoids all torch/model imports.
    (output / 'sample_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    samples, categories = manifest['samples'], manifest['categories']
    panels = {}
    definitions = [('overview', [i for i, s in enumerate(samples) if s['overview']])]
    definitions += [(c, [i for i, s in enumerate(samples) if s['identity_panel'] and s['category'] == c]) for c in categories]
    settings = {}
    for panel_index, (name, indices) in enumerate(definitions):
        print('Embedding ' + name, flush=True)
        coordinates, info = embed(features[indices], args.seed, args.perplexity, args.max_iter)
        panels[name] = dict(indices=indices, coordinates=coordinates, settings=info)
        filename = 'coordinates_{:02d}_{}.csv'.format(panel_index, re.sub(r'[^A-Za-z0-9_-]', '_', name))
        write_coordinates(output / filename, samples, predictions, indices, coordinates)
        identities = sorted({tuple(samples[i]['identity_key']) for i in indices})
        settings[name] = dict(info, coordinates_file=filename,
                              identity_legend={str(i + 1): list(key) for i, key in enumerate(identities)} if name != 'overview' else {})
    files = plot_embeddings(samples, panels, categories, output, args.formats, args.dpi)
    report = dict(schema_version=1, source_features=source_cache, feature_dim=int(features.shape[1]),
                  extracted_images=len(features), versions=versions, panels=settings, figure_files=files,
                  interpretation='Qualitative visualization only; identity panels have independent coordinate systems.')
    (output / 'tsne_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Done: ' + str(output), flush=True)
    return report


if __name__ == '__main__':
    main()
