"""Standalone redraw of identity t-SNE figures from one or more exported CSVs.

Each CSV must contain one category. Coordinates are used unchanged, and full
JSON identity_key values determine colors and ID numbering, as in the original
plotter. No model, images, feature cache, project modules, or t-SNE fitting are
used. Only matplotlib and numpy are required.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path


NAMES = dict(person='Market1501', vehicle='VeRi', panda='iPanda50', tiger='ATRW', boat='Boat')


def read_coordinates(path):
    categories, identities, coordinates = set(), [], []
    with Path(path).expanduser().open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'category', 'identity_key', 'x', 'y'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError('CSV missing columns: ' + ', '.join(sorted(missing)))
        for row in reader:
            category = (row['category'] or '').strip()
            if not category:
                raise ValueError('CSV line {}: empty category'.format(reader.line_num))
            categories.add(category)
            try:
                xy = (float(row['x']), float(row['y']))
                if not all(math.isfinite(value) for value in xy):
                    raise ValueError('non-finite coordinates')
            except (TypeError, ValueError) as error:
                raise ValueError('CSV line {}: x/y must be finite numbers. Blank coordinates '
                                 'mean no usable embedding was exported.'.format(reader.line_num)) from error
            try:
                identity = json.loads(row['identity_key'])
                if not isinstance(identity, list) or not identity or not all(isinstance(v, str) for v in identity):
                    raise ValueError('expected a nonempty JSON array of strings')
            except (TypeError, ValueError) as error:
                raise ValueError('CSV line {}: invalid identity_key; retain the original exported '
                                 'JSON array, including leading zeros.'.format(reader.line_num)) from error
            identities.append(tuple(identity))
            coordinates.append(xy)
    if not coordinates:
        raise ValueError('CSV contains no coordinate rows')
    if len(categories) != 1:
        raise ValueError('Each identity CSV must contain exactly one category; use the overview script for mixed categories')
    return next(iter(categories)), identities, coordinates


def figure_name(category):
    dataset = re.sub(r'[^a-z0-9_-]', '_', NAMES.get(category, category).lower())
    return 'tsne_identities_' + dataset


def render_identities(category, identities, coordinates, output_dir, formats=('png', 'pdf'), dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    xy = np.asarray(coordinates, dtype=float)
    ids = sorted(set(identities))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(4.0, 4.2))
        try:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(NAMES.get(category, category), fontsize=10)
            colors = plt.get_cmap('tab10' if len(ids) <= 10 else 'tab20' if len(ids) <= 20 else 'hsv')
            for i, identity in enumerate(ids):
                mask = np.array([key == identity for key in identities])
                color = colors(i if len(ids) <= 20 else i / len(ids))
                ax.scatter(xy[mask, 0], xy[mask, 1], s=17, alpha=.85, linewidths=0, color=color,
                           label='ID {}'.format(i + 1))
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.035), ncol=3, fontsize=8,
                      frameon=False, handletextpad=.3, columnspacing=.7)
            fig.subplots_adjust(left=.06, right=.94, top=.91, bottom=.24)
            for extension in formats:
                path = output / (figure_name(category) + '.' + extension)
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                paths.append(str(path.resolve()))
        finally:
            plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', nargs='+', required=True, help='One or more per-category coordinates CSVs')
    parser.add_argument('--output-dir', help='Defaults to csv_plots next to each input CSV')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output figures')
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    formats = list(dict.fromkeys(args.formats))
    jobs, destinations = [], set()
    # Validate the whole batch before writing any figure.
    for filename in args.csv:
        source = Path(filename).expanduser().resolve()
        category, identities, coordinates = read_coordinates(source)
        output = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.parent / 'csv_plots'
        for extension in formats:
            path = output / (figure_name(category) + '.' + extension)
            if path in destinations:
                raise ValueError('Multiple CSVs would write the same figure: ' + str(path))
            destinations.add(path)
            if path.exists() and not args.overwrite:
                raise FileExistsError('Output already exists: {}. Use another directory or --overwrite.'.format(path))
        jobs.append((category, identities, coordinates, output))
    paths = []
    for category, identities, coordinates, output in jobs:
        saved = render_identities(category, identities, coordinates, output, formats, args.dpi)
        paths.extend(saved)
        for path in saved:
            print('Saved: ' + path, flush=True)
    return paths


if __name__ == '__main__':
    main()
