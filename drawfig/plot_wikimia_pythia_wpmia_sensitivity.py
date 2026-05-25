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


LENGTHS = [
    {
        "value": "32",
        "label": "32",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "value": "64",
        "label": "64",
        "color": "#E86E32",
        "marker": "s",
        "linestyle": ":",
    },
    {
        "value": "128",
        "label": "128",
        "color": "#45A032",
        "marker": "D",
        "linestyle": "--",
    },
]


PLOTS = {
    "tau": {
        "vary": "tau",
        "fixed": "gamma",
        "fixed_value": 1.0,
        "output": "wikimia_pythia_wpmia_tau_sensitivity_auc.png",
        "xlabel": r"temperature parameter ($\tau$)",
        "xticks": [0.03, 0.05, 0.1, 0.2, 0.5],
    },
    "gamma": {
        "vary": "gamma",
        "fixed": "tau",
        "fixed_value": 0.2,
        "output": "wikimia_pythia_wpmia_gamma_sensitivity_auc.png",
        "xlabel": r"contrast strength parameter ($\gamma$)",
        "xticks": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
}


def float_eq(lhs: float, rhs: float) -> bool:
    return abs(lhs - rhs) < 1e-9


def load_points(
    csv_path: Path,
    *,
    length: str,
    vary: str,
    fixed: str,
    fixed_value: float,
) -> list[tuple[float, float]]:
    points: dict[float, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok" or row.get("length") != length:
                continue
            if not float_eq(float(row[fixed]), fixed_value):
                continue
            points[float(row[vary])] = float(row["auc_pct"])

    if not points:
        raise ValueError(
            f"No rows found for length={length}, {fixed}={fixed_value} in {csv_path}"
        )
    return sorted(points.items())


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 20,
            "axes.labelsize": 30,
            "axes.labelweight": "bold",
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def padded_ylim(values: list[float]) -> tuple[float, float]:
    lower = math.floor((min(values) - 3.0) / 5.0) * 5.0
    upper = math.ceil((max(values) + 3.0) / 5.0) * 5.0
    return max(0.0, lower), upper


def format_value_label(value: float) -> str:
    return f"{value:g}"


def plot_sensitivity(
    csv_path: Path,
    output_dir: Path,
    plot_name: str,
    formats: list[str],
) -> list[Path]:
    config = PLOTS[plot_name]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    x_values = config["xticks"]
    x_positions = list(range(len(x_values)))

    all_auc: list[float] = []
    for item in LENGTHS:
        points = load_points(
            csv_path,
            length=item["value"],
            vary=config["vary"],
            fixed=config["fixed"],
            fixed_value=config["fixed_value"],
        )
        auc_by_x = {x: auc for x, auc in points}
        aucs = [auc_by_x[x] for x in x_values]
        all_auc.extend(aucs)

        ax.plot(
            x_positions,
            aucs,
            label=item["label"],
            color=item["color"],
            marker=item["marker"],
            linestyle=item["linestyle"],
            linewidth=4.2,
            markersize=8.8,
            markeredgewidth=1.6,
            markeredgecolor=item["color"],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_value_label(x) for x in x_values])
    ax.set_xlabel(config["xlabel"], fontweight="bold", labelpad=10)
    ax.set_ylabel("AUC", fontweight="bold", labelpad=4)
    ax.set_ylim(*padded_ylim(all_auc))

    ax.grid(True, color="#D9D9D9", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.tick_params(axis="both", which="major", width=1.8, length=7, direction="out")
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.4)
        ax.spines[side].set_color("#555555")

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.035, 0.985),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#555555",
        handlelength=1.45,
        handletextpad=0.18,
        labelspacing=0.34,
        prop={"size": 23, "weight": "bold"},
    )
    legend.get_frame().set_linewidth(1.2)
    for text in legend.get_texts():
        text.set_color("black")

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.96)

    saved_paths: list[Path] = []
    stem = Path(config["output"]).with_suffix("")
    for fmt in formats:
        out_path = output_dir / f"{stem.name}.{fmt}"
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WPMIA tau/gamma sensitivity on WikiMIA Pythia-6.9B."
    )
    parser.add_argument(
        "--csv_path",
        default="logs/wpmia/wikimia_pythia_wpmia_sweep.csv",
        help="CSV produced by scripts/run_wpmia_wikimia_cache_sweep.sh.",
    )
    parser.add_argument("--output_dir", default="drawfig")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output image formats, e.g. png pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    apply_plot_style()
    saved: list[Path] = []
    for plot_name in PLOTS:
        saved.extend(plot_sensitivity(csv_path, output_dir, plot_name, args.formats))

    for path in saved:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
