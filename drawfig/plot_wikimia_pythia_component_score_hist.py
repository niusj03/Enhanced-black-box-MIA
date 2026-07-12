#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'simmia')}"
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


PLOTS = [
    {"method": "ll"},
    {"method": "seq_nm_ratio"},
    {"method": "seq_contrast"},
]

MEMBER_COLOR = "#4169E1"
NONMEMBER_COLOR = "#FF9900"


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing score CSV: {path}")

    labels = []
    scores = []
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))

    if not scores:
        raise ValueError(f"No scores found in {path}")
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if math.isclose(min_score, max_score):
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros_like(grid)

    std = float(np.std(values, ddof=1))
    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    if not math.isfinite(bandwidth) or bandwidth <= 1e-8:
        bandwidth = 0.05

    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).mean(axis=1)
    return density / (bandwidth * math.sqrt(2 * math.pi))


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_panel(
    ax,
    labels: np.ndarray,
    scores: np.ndarray,
    config: dict,
) -> None:
    normalized = minmax_normalize(scores)
    member = normalized[labels == 1]
    nonmember = normalized[labels == 0]
    if len(member) == 0 or len(nonmember) == 0:
        raise ValueError(f"Need both member and non-member scores for {config['method']}")

    bins = np.linspace(0.0, 1.0, 22)
    grid = np.linspace(-0.08, 1.08, 400)

    ax.hist(
        member,
        bins=bins,
        density=True,
        color=MEMBER_COLOR,
        alpha=0.45,
        label="Member",
        edgecolor="white",
        linewidth=0.45,
    )
    ax.hist(
        nonmember,
        bins=bins,
        density=True,
        color=NONMEMBER_COLOR,
        alpha=0.45,
        label="Non-member",
        edgecolor="white",
        linewidth=0.45,
    )

    ax.plot(grid, gaussian_kde(member, grid), color=MEMBER_COLOR, linewidth=2.4)
    ax.plot(grid, gaussian_kde(nonmember, grid), color=NONMEMBER_COLOR, linewidth=2.4)

    member_mean = float(np.mean(member))
    nonmember_mean = float(np.mean(nonmember))
    ax.axvline(member_mean, color=MEMBER_COLOR, linestyle="--", linewidth=2.05)
    ax.axvline(nonmember_mean, color=NONMEMBER_COLOR, linestyle="--", linewidth=2.05)

    ax.set_xlim(-0.08, 1.08)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", width=1.4, length=6, direction="out")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.35)
    ax.spines[side].set_color("#333333")

    y_top = ax.get_ylim()[1]
    y_diff = y_top * 0.36
    left = min(member_mean, nonmember_mean)
    right = max(member_mean, nonmember_mean)
    diff = abs(member_mean - nonmember_mean)

    ax.annotate(
        "",
        xy=(left, y_diff),
        xytext=(right, y_diff),
        arrowprops={
            "arrowstyle": "<->",
            "color": "black",
            "linewidth": 1.55,
            "mutation_scale": 8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(0.77, 0.455),
        xycoords="axes fraction",
        xytext=(right + 0.012, y_diff),
        textcoords="data",
        arrowprops={
            "arrowstyle": "simple,head_length=0.85,head_width=0.85,tail_width=0.16",
            "connectionstyle": "arc3,rad=0.0",
            "facecolor": "#D62728",
            "edgecolor": "#D62728",
            "linewidth": 0.0,
            "mutation_scale": 20,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    ax.text(
        0.77,
        0.50,
        f"Difference: {diff:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="black",
    )


def plot_length(
    score_root: Path,
    output_dir: Path,
    length: str,
    formats: list[str],
    output_suffix: str,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2), sharey=True)

    for ax, config in zip(axes, PLOTS):
        score_path = score_root / f"wikimia_len{length}_{config['method']}.csv"
        labels, scores = load_scores(score_path)
        plot_panel(ax, labels, scores, config)

    axes[0].set_ylabel("Density", fontweight="bold", fontsize=27)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            edgecolor="#CCCCCC",
            prop={"size": 18, "weight": "bold"},
        )
        legend.get_frame().set_linewidth(1.1)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.965, wspace=0.08)

    saved_paths = []
    for fmt in formats:
        out_path = output_dir / (
            f"wikimia_pythia_component_scores_len{length}{output_suffix}.{fmt}"
        )
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot min-max normalized component-wise WPMIA score distributions."
    )
    parser.add_argument(
        "--score_root",
        default="logs/ablation/wikimia_pythia69b_component_scores",
    )
    parser.add_argument("--output_dir", default="drawfig")
    parser.add_argument("--lengths", nargs="+", default=["32", "64", "128"])
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Optional suffix inserted before the image extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output image formats, e.g. png pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_root = Path(args.score_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    saved = []
    for length in args.lengths:
        saved.extend(
            plot_length(
                score_root,
                output_dir,
                length,
                args.formats,
                args.output_suffix,
            )
        )

    for path in saved:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
