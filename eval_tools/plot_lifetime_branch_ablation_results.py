import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_TOOLS_DIR = SCRIPT_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--metric", default="lifetime_branch", choices=["lifetime_branch", "whole_gate"])
    parser.add_argument("--min-overlap-pixels", type=int, default=50)
    parser.add_argument("--save-prefix", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--keep-duplicate-samples",
        action="store_true",
        help="Keep duplicated sample rows. By default, duplicated samples are reduced to the last row.",
    )
    return parser.parse_args()


def find_latest_result_dir(metric):
    csv_name = get_metric_config(metric)["csv_name"]
    candidates = []
    for name in os.listdir(EVAL_TOOLS_DIR):
        path = os.path.join(EVAL_TOOLS_DIR, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, csv_name)):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No result directory with {csv_name} was found in {EVAL_TOOLS_DIR}.")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def get_metric_config(metric):
    if metric == "lifetime_branch":
        return {
            "csv_name": "flim_lifetime_branch_ablation.csv",
            "overlap_col": "delta_when_lifetime_branch_neutralized_in_overlap",
            "non_overlap_col": "delta_when_lifetime_branch_neutralized_in_non_overlap",
            "ratio_col": "lifetime_branch_delta_overlap_to_non_overlap",
            "figure_stem": "lifetime_branch_overlap_analysis",
            "ylabel": "Output change after\nlifetime-branch ablation",
        }
    return {
        "csv_name": "flim_gate_ablation.csv",
        "overlap_col": "delta_when_FLIM_gates_neutralized_in_overlap",
        "non_overlap_col": "delta_when_FLIM_gates_neutralized_in_non_overlap",
        "ratio_col": "gate_ablation_delta_overlap_to_non_overlap",
        "figure_stem": "whole_gate_overlap_analysis",
        "ylabel": "Output change after\nFLIM gate neutralization",
    }


def make_stats(filtered, cfg):
    overlap = filtered[cfg["overlap_col"]]
    non_overlap = filtered[cfg["non_overlap_col"]]
    ratio = filtered[cfg["ratio_col"]]
    diff = overlap - non_overlap

    if ratio.median() < 1:
        alternative = "less"
        direction_text = "overlap < non-overlap"
    else:
        alternative = "greater"
        direction_text = "overlap > non-overlap"

    if wilcoxon is not None and len(filtered) > 0:
        p_value = wilcoxon(overlap, non_overlap, alternative=alternative, zero_method="wilcox").pvalue
        p_two_sided = wilcoxon(overlap, non_overlap, alternative="two-sided", zero_method="wilcox").pvalue
    else:
        p_value = np.nan
        p_two_sided = np.nan

    cohen_dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    stats = {
        "n": len(filtered),
        "overlap_delta_mean": overlap.mean(),
        "overlap_delta_median": overlap.median(),
        "overlap_delta_q1": overlap.quantile(0.25),
        "overlap_delta_q3": overlap.quantile(0.75),
        "non_overlap_delta_mean": non_overlap.mean(),
        "non_overlap_delta_median": non_overlap.median(),
        "non_overlap_delta_q1": non_overlap.quantile(0.25),
        "non_overlap_delta_q3": non_overlap.quantile(0.75),
        "ratio_mean": ratio.mean(),
        "ratio_median": ratio.median(),
        "ratio_q1": ratio.quantile(0.25),
        "ratio_q3": ratio.quantile(0.75),
        "fraction_ratio_gt_1": (ratio > 1).mean(),
        "fraction_ratio_lt_1": (ratio < 1).mean(),
        "wilcoxon_alternative": alternative,
        "wilcoxon_direction": direction_text,
        "wilcoxon_p_value": p_value,
        "wilcoxon_two_sided_p_value": p_two_sided,
        "paired_cohen_dz": cohen_dz,
    }
    return stats


def save_figure(fig, save_prefix, panel_name, show=False):
    png_path = f"{save_prefix}_{panel_name}.png"
    pdf_path = f"{save_prefix}_{panel_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return png_path, pdf_path


def plot_overlap_analysis(filtered, cfg, stats, save_prefix, show=False):
    overlap = filtered[cfg["overlap_col"]]
    non_overlap = filtered[cfg["non_overlap_col"]]
    ratio = filtered[cfg["ratio_col"]]

    plt.rcParams.update({
        "font.size": 16,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    rng = np.random.default_rng(321)
    output_paths = []

    fig0, ax0 = plt.subplots(figsize=(4, 2.85), constrained_layout=True)
    for _, row in filtered.iterrows():
        x0 = rng.normal(0, 0.02)
        x1 = 1 + rng.normal(0, 0.02)
        ax0.plot([x0, x1], [row[cfg["non_overlap_col"]], row[cfg["overlap_col"]]],
                 color="0.78", linewidth=0.65, zorder=1)

    ax0.scatter(np.zeros(len(filtered)) + rng.normal(0, 0.032, len(filtered)),
                non_overlap, s=12, color="#4c78a8", alpha=0.9, zorder=2)
    ax0.scatter(np.ones(len(filtered)) + rng.normal(0, 0.032, len(filtered)),
                overlap, s=12, color="#e45756", alpha=0.9, zorder=2)

    for x, values in [(0, non_overlap), (1, overlap)]:
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        ax0.errorbar([x], [median], yerr=[[median - q1], [q3 - median]],
                     fmt="s", markersize=4.8, color="black", ecolor="black",
                     elinewidth=1.2, capsize=3, zorder=5)

    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Non-overlap", "Overlap"], rotation=18)
    ax0.set_ylabel(cfg["ylabel"])
    ax0.spines[["top", "right"]].set_visible(False)
    output_paths.append(save_figure(fig0, save_prefix, "paired_response", show=show))

    fig1, ax1 = plt.subplots(figsize=(4, 2.85), constrained_layout=True)
    parts = ax1.violinplot([ratio], positions=[0], widths=0.68,
                           showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#72b7b2")
        body.set_edgecolor("black")
        body.set_alpha(0.75)

    ax1.scatter(np.zeros(len(filtered)) + rng.normal(0, 0.043, len(filtered)),
                ratio, s=12, color="black", alpha=0.55)
    median = ratio.median()
    q1 = ratio.quantile(0.25)
    q3 = ratio.quantile(0.75)
    ax1.errorbar([0], [median], yerr=[[median - q1], [q3 - median]],
                 fmt="s", color="black", ecolor="black", markersize=4.8, capsize=3)
    ax1.axhline(1, color="0.45", linestyle="--", linewidth=0.9)
    ax1.set_xlim(-0.55, 0.55)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Overlap /\nnon-overlap"])
    ax1.set_ylabel("Ratio")
    ax1.spines[["top", "right"]].set_visible(False)
    output_paths.append(save_figure(fig1, save_prefix, "ratio", show=show))

    fig2, ax2 = plt.subplots(figsize=(4, 2.85), constrained_layout=True)
    ax2.scatter(non_overlap, overlap, s=18, color="#f58518", alpha=0.78,
                edgecolor="black", linewidth=0.25)
    lim_high = max(non_overlap.max(), overlap.max()) * 1.08
    ax2.plot([0, lim_high], [0, lim_high], color="0.45", linestyle="--", linewidth=0.9)
    ax2.set_xlim(0, lim_high)
    ax2.set_ylim(0, lim_high)
    ax2.set_xlabel("Non-overlap change")
    ax2.set_ylabel("Overlap change")
    ax2.spines[["top", "right"]].set_visible(False)
    output_paths.append(save_figure(fig2, save_prefix, "scatter", show=show))

    return output_paths


def main():
    args = parse_args()
    result_dir = args.result_dir or find_latest_result_dir(args.metric)
    cfg = get_metric_config(args.metric)
    csv_path = os.path.join(result_dir, cfg["csv_name"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if not args.keep_duplicate_samples and "sample" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["sample"], keep="last")
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"Removed duplicated sample rows: {n_removed} (kept last row per sample)")
    filtered = df[df["overlap_pixels"] >= args.min_overlap_pixels].copy()
    if filtered.empty:
        raise ValueError(f"No samples remain after overlap_pixels >= {args.min_overlap_pixels}.")

    save_prefix = args.save_prefix
    if save_prefix is None:
        save_prefix = os.path.join(
            result_dir,
            f"{cfg['figure_stem']}_overlap_ge{args.min_overlap_pixels}",
        )

    filtered_csv = f"{save_prefix}_filtered_samples.csv"
    stats_csv = f"{save_prefix}_stats.csv"
    filtered.to_csv(filtered_csv, index=False)

    stats = make_stats(filtered, cfg)
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)
    figure_paths = plot_overlap_analysis(filtered, cfg, stats, save_prefix, show=args.show)

    print(f"Result directory: {result_dir}")
    print(f"Input CSV: {csv_path}")
    print(f"Filtered samples: {len(filtered)}")
    print(f"Median ratio: {stats['ratio_median']:.4f}")
    print(f"Mean ratio: {stats['ratio_mean']:.4f}")
    print(f"Wilcoxon: {stats['wilcoxon_direction']}, P = {stats['wilcoxon_p_value']:.4e}")
    for png_path, pdf_path in figure_paths:
        print(f"Wrote: {png_path}")
        print(f"Wrote: {pdf_path}")
    print(f"Wrote: {stats_csv}")
    print(f"Wrote: {filtered_csv}")


if __name__ == "__main__":
    RUN_FROM_IDE = True

    if RUN_FROM_IDE:
        import sys
        sys.argv = [
            __file__,
            "--result-dir",
            os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_synthetic_Mito_Lyso_ide_run"),
            "--metric", "lifetime_branch",
            "--min-overlap-pixels", "50",
        ]
        main()
        
        sys.argv = [
            __file__,
            "--result-dir",
            os.path.join(SCRIPT_DIR, "DESM_IL_lifetime_branch_synthetic_Micro_Mito_Lyso_ide_run"),
            "--metric", "lifetime_branch",
            "--min-overlap-pixels", "50",
        ]
        main()
