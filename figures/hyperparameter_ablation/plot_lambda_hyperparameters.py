from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3


EXPERIMENTS = [
    {
        "parameter": [0, 0.25, 0.5, 1, 2],
        "mAP": [55.14, 56.95, 57.24, 58.88, 56.12],
        "R1": [79.79, 80.52, 81.37, 83.34, 80.64],
        "symbol": r"$\lambda_a$",
        "panel": "a",
    },
    {
        "parameter": [0, 0.25, 0.5, 1, 2],
        "mAP": [56.21, 56.98, 57.43, 58.88, 57.10],
        "R1": [80.22, 80.53, 80.97, 83.34, 82.73],
        "symbol": r"$\lambda_r$",
        "panel": "b",
    },
]


COLORS = {
    "mAP": "#B64342",
    "R1": "#0F4D92",
    "optimum": "#2E9E44",
    "grid": "#D5D5D5",
    "axis": "#4D4D4D",
}


def style_broken_axes(ax_top, ax_bottom) -> None:
    for ax in (ax_top, ax_bottom):
        ax.grid(
            True,
            which="major",
            axis="both",
            color=COLORS["grid"],
            linestyle=(0, (2, 2)),
            linewidth=0.55,
            alpha=0.8,
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.spines["left"].set_color(COLORS["axis"])
        ax.spines["bottom"].set_color(COLORS["axis"])
        ax.tick_params(labelsize=7.5, colors="#272727")

    ax_top.set_ylim(79.3, 83.8)
    ax_top.set_yticks([80, 81, 82, 83])
    ax_bottom.set_ylim(54.5, 59.4)
    ax_bottom.set_yticks([55, 56, 57, 58, 59])

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    break_size = 0.013
    break_style = dict(color="#272727", clip_on=False, linewidth=0.8)
    ax_top.plot(
        (-break_size, +break_size),
        (-break_size, +break_size),
        transform=ax_top.transAxes,
        **break_style,
    )
    ax_top.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, +break_size),
        transform=ax_top.transAxes,
        **break_style,
    )
    ax_bottom.plot(
        (-break_size, +break_size),
        (1 - break_size, 1 + break_size),
        transform=ax_bottom.transAxes,
        **break_style,
    )
    ax_bottom.plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        transform=ax_bottom.transAxes,
        **break_style,
    )


def draw_panel(ax_top, ax_bottom, experiment: dict) -> None:
    parameters = np.asarray(experiment["parameter"], dtype=float)
    map_values = np.asarray(experiment["mAP"], dtype=float)
    r1_values = np.asarray(experiment["R1"], dtype=float)
    x = np.arange(parameters.size)
    best_index = int(np.argmax((map_values + r1_values) / 2.0))

    ax_top.plot(
        x,
        r1_values,
        color=COLORS["R1"],
        marker="s",
        markersize=4.8,
        markeredgewidth=0.8,
        markeredgecolor="white",
        linewidth=1.6,
        zorder=3,
    )
    ax_bottom.plot(
        x,
        map_values,
        color=COLORS["mAP"],
        marker="o",
        markersize=5.0,
        markeredgewidth=0.8,
        markeredgecolor="white",
        linewidth=1.6,
        zorder=3,
    )

    for ax in (ax_top, ax_bottom):
        ax.axvline(
            best_index,
            color=COLORS["optimum"],
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            zorder=1,
        )

    ax_top.scatter(
        best_index,
        r1_values[best_index],
        s=55,
        facecolors="none",
        edgecolors=COLORS["optimum"],
        linewidths=1.1,
        zorder=4,
    )
    ax_bottom.scatter(
        best_index,
        map_values[best_index],
        s=55,
        facecolors="none",
        edgecolors=COLORS["optimum"],
        linewidths=1.1,
        zorder=4,
    )

    style_broken_axes(ax_top, ax_bottom)
    ax_bottom.set_xlim(-0.22, len(x) - 0.78)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([f"{value:g}" for value in parameters])
    ax_bottom.set_xlabel(experiment["symbol"], fontsize=9, labelpad=6)

    ax_top.text(
        -0.11,
        1.02,
        experiment["panel"],
        transform=ax_top.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "lambda_ablation_combined.jpg"

    fig = plt.figure(figsize=(7.2, 3.45))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 1],
        hspace=0.06,
        wspace=0.18,
    )

    ax_a_top = fig.add_subplot(grid[0, 0])
    ax_a_bottom = fig.add_subplot(grid[1, 0], sharex=ax_a_top)
    ax_b_top = fig.add_subplot(grid[0, 1], sharey=ax_a_top)
    ax_b_bottom = fig.add_subplot(grid[1, 1], sharex=ax_b_top, sharey=ax_a_bottom)

    draw_panel(ax_a_top, ax_a_bottom, EXPERIMENTS[0])
    draw_panel(ax_b_top, ax_b_bottom, EXPERIMENTS[1])

    ax_b_top.tick_params(axis="y", labelleft=False)
    ax_b_bottom.tick_params(axis="y", labelleft=False)

    fig.text(
        0.025,
        0.59,
        "Average performance (%)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.5,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color=COLORS["mAP"],
            marker="o",
            markersize=5,
            linewidth=1.6,
            markeredgecolor="white",
            label="Average mAP",
        ),
        Line2D(
            [],
            [],
            color=COLORS["R1"],
            marker="s",
            markersize=4.8,
            linewidth=1.6,
            markeredgecolor="white",
            label="Average R1",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        columnspacing=1.8,
        handlelength=2.2,
        handletextpad=0.5,
        fontsize=7.8,
    )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.25)
    fig.savefig(
        output_path,
        format="jpg",
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
