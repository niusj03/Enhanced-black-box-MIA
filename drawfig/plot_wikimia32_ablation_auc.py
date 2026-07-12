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
from matplotlib.ticker import FuncFormatter


METHODS = [
    {
        "key": "wpmia",
        "label": "WPMIA",
        "color": "#009E73",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "key": "simmia_star",
        "label": "SimMIA*",
        "color": "#D62728",
        "marker": "s",
        "linestyle": ":",
    },
    {
        "key": "simmia",
        "label": "SimMIA",
        "color": "#0072B2",
        "marker": "D",
        "linestyle": "--",
    },
]


ABLATIONS = {
    "prefix_ratio": {
        "output": "wikimia32_pythia69b_prefix_ratio_auc.png",
        "xticks": [0.1, 0.3, 0.5, 0.7, 0.9],
        "xformatter": lambda x, _pos: f"{x:.1f}",
        "legend_loc": "lower right",
        "use_continuation_ratio": True,
    },
    "num_samples": {
        "output": "wikimia32_pythia69b_num_samples_auc.png",
        "xticks": [0, 20, 40, 60, 80, 100],
        "xformatter": lambda x, _pos: f"{int(x)}",
        "xlim": (0, 105),
        "legend_loc": "lower right",
    },
    "num_shots": {
        "output": "wikimia32_pythia69b_num_shots_auc.png",
        "xticks": list(range(1, 11)),
        "xformatter": lambda x, _pos: f"{int(x)}",
        "legend_loc": "lower right",
    },
}


def load_points(csv_path: Path) -> list[tuple[float, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    points: dict[float, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            x_value = row.get("ablation_value")
            auc_value = row.get("auc_pct")
            if not x_value or not auc_value:
                continue
            points[float(x_value)] = float(auc_value)

    if not points:
        raise ValueError(f"No ok AUC rows found in {csv_path}")
    return sorted(points.items())


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 20,
            "axes.labelsize": 42,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def padded_ylim(values: list[float]) -> tuple[float, float]:
    lower = math.floor((min(values) - 4.0) / 5.0) * 5.0
    upper = math.ceil((max(values) + 4.0) / 5.0) * 5.0
    return max(0.0, lower), upper


def plot_ablation(
    csv_root: Path,
    output_dir: Path,
    prefix: str,
    ablation: str,
    formats: list[str],
) -> list[Path]:
    config = ABLATIONS[ablation]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    all_auc: list[float] = []
    for method in METHODS:
        csv_path = csv_root / f"{prefix}_{ablation}_{method['key']}.csv"
        points = load_points(csv_path)
        if config.get("use_continuation_ratio"):
            points = sorted((round(1.0 - x, 10), auc) for x, auc in points)
        xs = [x for x, _auc in points]
        aucs = [auc for _x, auc in points]
        all_auc.extend(aucs)

        ax.plot(
            xs,
            aucs,
            label=method["label"],
            color=method["color"],
            marker=method["marker"],
            linestyle=method["linestyle"],
            linewidth=3.2,
            markersize=10,
            markeredgewidth=1.8,
            markeredgecolor=method["color"],
        )

    ax.set_xticks(config["xticks"])
    ax.xaxis.set_major_formatter(FuncFormatter(config["xformatter"]))
    if "xlim" in config:
        ax.set_xlim(*config["xlim"])
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
        loc=config["legend_loc"],
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#555555",
        columnspacing=1.1,
        borderpad=0.32,
        handlelength=1.9,
        labelspacing=0.45,
        prop={"size": 20, "weight": "bold"},
    )
    legend.get_frame().set_linewidth(1.2)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.96)

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
        description="Plot WikiMIA length-32 Pythia-6.9B ablation AUC curves."
    )
    parser.add_argument("--csv_root", default="logs/ablation")
    parser.add_argument("--output_dir", default="drawfig")
    parser.add_argument("--prefix", default="wikimia32_pythia69b")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output image formats, e.g. png pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_root = Path(args.csv_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    saved: list[Path] = []
    for ablation in ABLATIONS:
        saved.extend(
            plot_ablation(
                csv_root=csv_root,
                output_dir=output_dir,
                prefix=args.prefix,
                ablation=ablation,
                formats=args.formats,
            )
        )

    for path in saved:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
