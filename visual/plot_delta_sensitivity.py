"""Plot delta sensitivity: seen-category average mAP and R1 (percent).

Standalone: python plot_delta_sensitivity.py
Dependency: matplotlib. Data transcribed from the supplied workbook,
Sheet1!C18:E22 (mixed/prototype, seen-category averages).
Only measured points are joined; no smoothing or invented error bars.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Edit these rows to plot updated experiments: (delta, mAP, R1).
DATA = [
    (0.30, 54.16, 73.49),
    (0.40, 54.48, 74.29),
    (0.50, 54.88, 75.22),
    (0.60, 54.54, 74.84),
    (0.70, 54.17, 73.90),
]
# Match the supplied stage-wise performance figure.
MAP_COLOR = '#0072B2'
R1_COLOR = '#D55E00'


def make_figure():
    x, map_values, r1_values = zip(*DATA)
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'mathtext.fontset': 'dejavusans',
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    }):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='#B0B0B0', linestyle='--', linewidth=.5, alpha=.3)
        ax.plot(x, map_values, color=MAP_COLOR, marker='o', linestyle='-', label='mAP',
                linewidth=2.2, markersize=6, markerfacecolor=MAP_COLOR, zorder=3)
        ax.plot(x, r1_values, color=R1_COLOR, marker='s', linestyle='-', label='Rank-1',
                linewidth=2.2, markersize=6, markerfacecolor=R1_COLOR, zorder=3)
        ax.set_xlabel(r'$\delta$', labelpad=5)
        ax.set_ylabel('Performance(%)', labelpad=6)
        ax.set_xticks(x, [f'{value:g}' for value in x])
        ax.set_xlim(.274, .726)
        # Same limits as the alpha figure; no separate rescaling of mAP and R1.
        ax.set_ylim(45, 80)
        ax.set_yticks(range(50, 81, 10))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='in', length=3.5, width=1.0, pad=3)
        ax.legend(loc='lower right', frameon=False, fontsize=10)
        fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'parameter_sensitivity')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf', 'svg'])
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--no-values', action='store_true', help='Compatibility option: point labels are always hidden')
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    try:
        with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
            for extension in dict.fromkeys(args.formats):
                path = output / ('delta_sensitivity.' + extension)
                fig.savefig(path, dpi=args.dpi, facecolor='white')
                print('Saved: ' + str(path))
    finally:
        plt.close(fig)
    with (output / 'delta_source_data.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['delta', 'mAP_percent', 'R1_percent'])
        writer.writerows(DATA)


if __name__ == '__main__':
    main()
