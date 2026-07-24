#!/usr/bin/env python3
r"""
Plot RSP/RSE boxplots from NanoJ-SQUIRREL batch summary Excel.

Default input:
  <repo>/visualization/Experimental_data_eval/Micro_Mito/
  resized results/squirrel_batch_results/squirrel_batch_summary.xlsx

The plot groups boxes by resolution and colors them by alpha, where alpha is
the Poisson noise value in the `poisson_noise` column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DESM_ROOT = SCRIPT_DIR.parent
DEFAULT_EXCEL = Path(
    DESM_ROOT
    / "visualization"
    / "Experimental_data_eval"
    / "Micro_Mito"
    / "resized results"
    / "squirrel_batch_results"
    / "squirrel_batch_summary.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SQUIRREL RSP/RSE boxplots.")
    parser.add_argument(
        "--excel",
        default=str(DEFAULT_EXCEL),
        help="Path to squirrel_batch_summary.xlsx.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG path. Default: <excel_dir>/squirrel_rsp_rse_boxplot.png.",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Optional PDF output path.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        choices=[1, 2],
        default=None,
        help="Optionally plot only one channel. Default: use both channels.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )
    return parser.parse_args()


def alpha_label(value: float) -> str:
    return r"$\alpha$ = " + f"{float(value):.3f}"


def numeric_sorted(values) -> list:
    return sorted(values, key=lambda x: float(x))


def load_data(excel_path: Path, channel: int | None) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    required = {"resolution", "poisson_noise", "RSP", "RSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {excel_path}: {sorted(missing)}")

    df = df.copy()
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df["poisson_noise"] = pd.to_numeric(df["poisson_noise"], errors="coerce")
    df["RSP"] = pd.to_numeric(df["RSP"], errors="coerce")
    df["RSE"] = pd.to_numeric(df["RSE"], errors="coerce")
    df = df.dropna(subset=["resolution", "poisson_noise", "RSP", "RSE"])

    if channel is not None:
        if "channel" not in df.columns:
            raise ValueError("--channel was given, but the Excel file has no channel column.")
        df = df[df["channel"] == channel]

    if df.empty:
        raise ValueError("No rows left to plot after filtering.")
    return df


def draw_metric_boxplot(
    ax,
    df: pd.DataFrame,
    metric: str,
    resolutions: list[float],
    alphas: list[float],
    colors: list[str],
) -> None:
    n_alpha = len(alphas)
    box_width = 0.16 if n_alpha <= 5 else 0.12
    inner_step = box_width * 1.42
    group_gap = max(1.15, n_alpha * inner_step + 0.55)
    centers = np.arange(len(resolutions), dtype=float) * group_gap + 1.0
    offsets = (np.arange(n_alpha) - (n_alpha - 1) / 2.0) * inner_step

    all_values = []
    all_positions = []
    box_colors = []
    for center, resolution in zip(centers, resolutions):
        for offset, alpha, color in zip(offsets, alphas, colors):
            values = df[
                (df["resolution"] == resolution)
                & (np.isclose(df["poisson_noise"], alpha))
            ][metric].dropna().to_numpy()
            all_values.append(values)
            all_positions.append(center + offset)
            box_colors.append(color)

    bp = ax.boxplot(
        all_values,
        positions=all_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={
            "marker": "s",
            "markerfacecolor": "white",
            "markeredgecolor": "#777777",
            "markersize": 2.2,
            "markeredgewidth": 0.7,
        },
        medianprops={"linewidth": 1.2, "color": "#555555"},
    )

    for box, color in zip(bp["boxes"], box_colors):
        box.set(facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.2)
    for whisker, color in zip(bp["whiskers"], np.repeat(box_colors, 2)):
        whisker.set(color=color, linewidth=1.1)
    for cap, color in zip(bp["caps"], np.repeat(box_colors, 2)):
        cap.set(color=color, linewidth=1.1)

    for boundary in (centers[:-1] + centers[1:]) / 2.0:
        ax.axvline(boundary, color="black", linewidth=1.4)

    ax.set_title(metric, fontsize=24, pad=8)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{int(r)} nm" for r in resolutions], fontsize=20)
    ax.tick_params(axis="y", labelsize=16, width=1.1, length=4)
    ax.tick_params(axis="x", width=0, length=0, pad=8)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#222222")


def make_plot(
    df: pd.DataFrame,
    output_path: Path,
    pdf_path: Path | None,
    title: str | None,
    dpi: int,
    rsp_ylim: tuple[float, float] | None = None,
    rse_ylim: tuple[float, float] | None = None,
) -> None:
    resolutions = numeric_sorted(df["resolution"].dropna().unique())
    alphas = numeric_sorted(df["poisson_noise"].dropna().unique())
    colors = ["#666666", "#ff3b3b", "#0068ff", "#2cb66f", "#b36bff", "#ff9d00", "#00a5a5"]
    colors = colors[: len(alphas)]

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.linewidth": 1.1,
        "mathtext.default": "regular",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=False)
    draw_metric_boxplot(axes[0], df, "RSP", resolutions, alphas, colors)
    draw_metric_boxplot(axes[1], df, "RSE", resolutions, alphas, colors)
    if rsp_ylim is not None:
        axes[0].set_ylim(*rsp_ylim)
    if rse_ylim is not None:
        axes[1].set_ylim(*rse_ylim)

    handles = [
        mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.22, label=alpha_label(alpha))
        for alpha, color in zip(alphas, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        frameon=False,
        fontsize=22,
        handlelength=1.35,
        columnspacing=1.8,
    )

    if title:
        fig.suptitle(title, y=1.10, fontsize=24)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.78, wspace=0.17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main(rsp_ylim = (0.75, 0.96), rse_ylim = (0, 30)) -> int:
    args = parse_args()
    

    excel_path = Path(args.excel).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else excel_path.with_name("squirrel_rsp_rse_boxplot.png")
    )
    pdf_path = Path(args.pdf).expanduser().resolve() if args.pdf else None

    df = load_data(excel_path, args.channel)
    make_plot(df, output_path, pdf_path, args.title, args.dpi, rsp_ylim, rse_ylim)
    print(f"Saved plot: {output_path}")
    if pdf_path is not None:
        print(f"Saved PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    if __name__ == "__main__":
        import sys

        sys.argv = [
            "plot_squirrel_summary.py",

            "--excel",
            str(DEFAULT_EXCEL),

            "--output",
            str(DEFAULT_EXCEL.with_name("squirrel_rsp_rse_boxplot.png")),

            # 鍙€夛細鍚屾椂淇濆瓨 PDF
            # "--pdf",
            # str(DEFAULT_EXCEL.with_name("squirrel_rsp_rse_boxplot.pdf")),

            # 鍙€夛細鍙敾 channel 1 鎴?channel 2锛涗笉鍐欏氨鏄袱涓?channel 涓€璧风粺璁?            # "--channel",
            # "1",

            "--dpi",
            "300",

            # 鍙€夋爣棰?            # "--title",
            # "Micro Mito",
        ]

        raise SystemExit(main(
            rsp_ylim = (0.7, 0.96), rse_ylim = (0, 35)
        ))
