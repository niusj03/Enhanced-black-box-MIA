#!/usr/bin/env python3
"""Plot the 7-shot WPMIA prefix-mechanism diagnostic."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'simmia')}"
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MEMBER_COLOR = "#4C72DF"
NONMEMBER_COLOR = "#F5A33B"


def load_rows(path: Path, shot: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as fin:
        rows = [row for row in csv.DictReader(fin) if int(row["shot"]) == shot]
    if not rows:
        raise ValueError(f"No rows for shot={shot}: {path}")
    return rows


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 17,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7, zorder=0)
    ax.tick_params(axis="both", width=1.2, length=5, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")


def add_paired_difference(
    ax,
    x: float,
    y_first: float,
    y_second: float,
    value: float,
) -> None:
    y_low, y_high = sorted((y_first, y_second))
    ax.text(
        x - 0.035,
        (y_low + y_high) / 2,
        f"{value:.2f}",
        ha="right",
        va="center",
        rotation=90,
        fontsize=16,
        fontweight="bold",
    )
    ax.plot(
        [x, x],
        [y_low, y_high],
        color="#333333",
        linewidth=1.6,
        solid_capstyle="round",
        zorder=6,
    )
    ax.scatter([x], [y_high], marker="^", s=16, color="#333333", zorder=7)
    ax.scatter([x], [y_low], marker="v", s=16, color="#333333", zorder=7)


def plot_mechanism(ax, rows: list[dict[str, str]]) -> None:
    specs = [
        (1, "delta_m", 0.82, MEMBER_COLOR),
        (1, "delta_nm", 1.18, NONMEMBER_COLOR),
        (0, "delta_m", 1.82, MEMBER_COLOR),
        (0, "delta_nm", 2.18, NONMEMBER_COLOR),
    ]
    data = []
    positions = []
    colors = []
    for label, field, position, color in specs:
        values = np.asarray(
            [float(row[field]) for row in rows if int(row["target_label"]) == label],
            dtype=float,
        )
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid violin data for label={label}, field={field}")
        data.append(values)
        positions.append(position)
        colors.append(color)

    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.30,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
        body.set_linewidth(1.0)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.085,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
    )
    for values, position, color in zip(data, positions, colors):
        ax.scatter(
            [position],
            [float(np.mean(values))],
            marker="D",
            s=34,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    ax.axhline(0.0, color="black", linewidth=1.2, zorder=3)
    ax.set_xticks([1.0, 2.0], ["Member target", "Non-member target"])
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(18)
        tick_label.set_fontweight("bold")
    ax.set_ylabel(
        r"$\Delta \widehat{\mathrm{LL}}^{q}="
        r"\widehat{\mathrm{LL}}^{q}-\widehat{\mathrm{LL}}^{0}$",
        fontweight="bold",
        fontsize=21,
    )
    ax.legend(
        handles=[
            Patch(facecolor=MEMBER_COLOR, alpha=0.55, label="Member prefix"),
            Patch(
                facecolor=NONMEMBER_COLOR,
                alpha=0.55,
                label="Non-member prefix",
            ),
            Line2D(
                [],
                [],
                color="#333333",
                marker=r"$\updownarrow$",
                markersize=17,
                linestyle="None",
                label="Paired mean difference",
            ),
        ],
        loc="upper right",
        ncol=1,
        frameon=True,
        prop={"size": 13, "weight": "bold"},
    )
    style_axes(ax)

    y_min = min(float(np.min(values)) for values in data)
    y_max = max(float(np.max(values)) for values in data)
    value_span = max(y_max - y_min, 1e-8)
    ax.set_ylim(y_min - 0.08 * value_span, y_max + 0.12 * value_span)

    member_rows = [row for row in rows if int(row["target_label"]) == 1]
    nonmember_rows = [row for row in rows if int(row["target_label"]) == 0]
    member_difference = float(np.mean(
        [float(row["delta_m"]) - float(row["delta_nm"]) for row in member_rows]
    ))
    nonmember_difference = float(np.mean(
        [float(row["delta_nm"]) - float(row["delta_m"]) for row in nonmember_rows]
    ))
    means = [float(np.mean(values)) for values in data]
    add_paired_difference(ax, 0.50, means[0], means[1], member_difference)
    add_paired_difference(ax, 1.50, means[2], means[3], nonmember_difference)


def write_table(rows: list[dict[str, str]], output_path: Path) -> None:
    member_rows = [row for row in rows if int(row["target_label"]) == 1]
    nonmember_rows = [row for row in rows if int(row["target_label"]) == 0]

    def mean(selected: list[dict[str, str]], field: str) -> float:
        return float(np.mean([float(row[field]) for row in selected]))

    member_prefix_member = mean(member_rows, "delta_m")
    member_prefix_nonmember = mean(nonmember_rows, "delta_m")
    nonmember_prefix_member = mean(member_rows, "delta_nm")
    nonmember_prefix_nonmember = mean(nonmember_rows, "delta_nm")
    table_rows = [
        (
            "Member prefix",
            member_prefix_member,
            member_prefix_nonmember,
        ),
        (
            "Non-member prefix",
            nonmember_prefix_member,
            nonmember_prefix_nonmember,
        ),
        (
            "Contrast prefixes",
            member_prefix_member - nonmember_prefix_member,
            member_prefix_nonmember - nonmember_prefix_nonmember,
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["Prefix condition", "Member target", "Non-member target", "Mean difference"]
        )
        for condition, member_value, nonmember_value in table_rows:
            writer.writerow(
                [
                    condition,
                    f"{member_value:.2f}",
                    f"{nonmember_value:.2f}",
                    f"{member_value - nonmember_value:.2f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the WPMIA target-level prefix mechanism."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(
            "scripts_output_rebuttal/mechanism/"
            "wikimia_pythia69b_prefix_mechanism_per_target.csv"
        ),
    )
    parser.add_argument("--shot", type=int, default=7)
    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("drawfig/wikimia_pythia69b_prefix_mechanism.png"),
    )
    parser.add_argument(
        "--output_table_csv",
        type=Path,
        default=Path(
            "scripts_output_rebuttal/mechanism/"
            "wikimia_pythia69b_prefix_mechanism_table.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, args.shot)
    apply_style()

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    plot_mechanism(ax, rows)
    fig.tight_layout()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output_png}")
    plt.close(fig)
    write_table(rows, args.output_table_csv)
    print(f"Saved {args.output_table_csv}")


if __name__ == "__main__":
    main()
