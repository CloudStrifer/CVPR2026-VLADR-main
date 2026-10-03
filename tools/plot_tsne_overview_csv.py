"""Standalone redraw of a category t-SNE overview from exported CSV coordinates.

Only matplotlib and numpy are required. No model, images, feature cache, project
modules, or t-SNE fitting are used. Category order follows first appearance in
the CSV, matching the original export order.
"""

import argparse
import csv
import math
from pathlib import Path


NAMES = dict(person='Market1501', vehicle='VeRi', panda='iPanda50', tiger='ATRW', boat='Boat')
COLORS = dict(person='#4477AA', vehicle='#EE9944', panda='#228833', tiger='#AA3377', boat='#66AACC')


def read_coordinates(path):
    categories, labels, coordinates = [], [], []
    with Path(path).expanduser().open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'category', 'x', 'y'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError('CSV missing columns: ' + ', '.join(sorted(missing)))
        for row in reader:
            category = (row['category'] or '').strip()
            if not category:
                raise ValueError('CSV line {}: empty category'.format(reader.line_num))
            try:
                xy = (float(row['x']), float(row['y']))
                if not all(math.isfinite(value) for value in xy):
                    raise ValueError('non-finite coordinates')
            except (TypeError, ValueError) as error:
                raise ValueError('CSV line {}: x/y must be finite numbers. Blank coordinates '
                                 'mean no usable embedding was exported.'.format(reader.line_num)) from error
            if category not in categories:
                categories.append(category)
            labels.append(category)
            coordinates.append(xy)
    if not coordinates:
        raise ValueError('CSV contains no coordinate rows')
    return categories, labels, coordinates


def render_overview(categories, labels, coordinates, output_dir, formats=('png', 'pdf'), dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    xy = np.asarray(coordinates, dtype=float)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        try:
            fallback = plt.get_cmap('tab10')
            for c_index, category in enumerate(categories):
                mask = np.array([c == category for c in labels])
                ax.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=.8, linewidths=0,
                           color=COLORS.get(category, fallback(c_index % 10)),
                           label=NAMES.get(category, category))
            fig.legend(loc='lower center', bbox_to_anchor=(.5, .015), ncol=min(3, len(categories)), frameon=False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.subplots_adjust(bottom=.19, top=.97)
            for extension in formats:
                path = output / ('tsne_categories.' + extension)
                fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
                paths.append(str(path.resolve()))
        finally:
            plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='coordinates_00_overview.csv')
    parser.add_argument('--output-dir', help='Defaults to csv_plots next to the CSV')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output figures')
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    source = Path(args.csv).expanduser().resolve()
    categories, labels, coordinates = read_coordinates(source)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.parent / 'csv_plots'
    formats = list(dict.fromkeys(args.formats))
    for extension in formats:
        path = output / ('tsne_categories.' + extension)
        if path.exists() and not args.overwrite:
            raise FileExistsError('Output already exists: {}. Use another directory or --overwrite.'.format(path))
    paths = render_overview(categories, labels, coordinates, output, formats, args.dpi)
    for path in paths:
        print('Saved: ' + path, flush=True)
    return paths


if __name__ == '__main__':
    main()
