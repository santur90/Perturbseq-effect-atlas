from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():

    comparison = pd.read_csv(
        TABLE_DIR / "corum_benchmark_comparison.csv"
    )

    pairwise = pd.read_csv(
        TABLE_DIR / "corum_pairwise_metrics.csv"
    ).iloc[0]

    prediction = pd.read_csv(
        TABLE_DIR / "corum_complex_prediction_summary.csv"
    ).iloc[0]

    diagnostics = pd.read_csv(
        TABLE_DIR / "in_module_vs_scattered.csv"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    # ==================================================
    # A. Same-complex neighbor recovery
    # ==================================================

    ax = axes[0, 0]

    ax.plot(
        comparison["k"],
        comparison["hit_rate_at_k"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Perturb-seq phenotype",
    )

    ax.errorbar(
        comparison["k"],
        comparison["random_hit_mean"],
        yerr=comparison["random_hit_sd"],
        marker="o",
        linestyle="--",
        linewidth=1.5,
        capsize=3,
        label="Permuted labels",
    )

    ax.set_xlabel(
        "Number of nearest neighbors (k)"
    )

    ax.set_ylabel(
        "Same-complex hit rate"
    )

    ax.set_title(
        "A  Functional neighbor recovery",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylim(
        0,
        0.8,
    )

    ax.legend(
        frameon=False,
    )

    # ==================================================
    # B. Neighbor precision
    # ==================================================

    ax = axes[0, 1]

    ax.plot(
        comparison["k"],
        comparison["mean_precision_at_k"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Perturb-seq phenotype",
    )

    ax.errorbar(
        comparison["k"],
        comparison["random_precision_mean"],
        yerr=comparison["random_precision_sd"],
        marker="o",
        linestyle="--",
        linewidth=1.5,
        capsize=3,
        label="Permuted labels",
    )

    ax.set_xlabel(
        "Number of nearest neighbors (k)"
    )

    ax.set_ylabel(
        "Mean same-complex precision"
    )

    ax.set_title(
        "B  Neighbor precision",
        loc="left",
        fontweight="bold",
    )

    ax.set_ylim(
        0,
        0.48,
    )

    ax.legend(
        frameon=False,
    )

    # ==================================================
    # C. Held-out complex prediction
    # ==================================================

    ax = axes[1, 0]

    metrics = [
        prediction["top1_accuracy"],
        prediction["top5_accuracy"],
        prediction["top10_accuracy"],
        prediction["mrr"],
    ]

    labels = [
        "Top-1",
        "Top-5",
        "Top-10",
        "MRR",
    ]

    bars = ax.bar(
        labels,
        metrics,
        color=[
            "#4C78A8",
            "#59A14F",
            "#F28E2B",
            "#E15759",
        ],
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        0.78,
    )

    ax.set_title(
        "C  Held-out complex prioritization",
        loc="left",
        fontweight="bold",
    )

    for bar, value in zip(
        bars,
        metrics,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )

    # ==================================================
    # D. Pairwise benchmark
    # ==================================================

    ax = axes[1, 1]

    prevalence = pairwise[
        "positive_prevalence"
    ]

    values = [
        prevalence,
        pairwise[
            "average_precision"
        ],
        pairwise[
            "auroc"
        ],
    ]

    labels = [
        "Random AP\n(prevalence)",
        "Perturb-seq\nAP",
        "Perturb-seq\nAUROC",
    ]

    bars = ax.bar(
        labels,
        values,
        color=[
            "#BAB0AC",
            "#4C78A8",
            "#59A14F",
        ],
    )

    ax.set_ylim(
        0,
        0.9,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "D  Pairwise same-complex prediction",
        loc="left",
        fontweight="bold",
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )

    ax.text(
        0.50,
        0.62,
        (
            f"AP enrichment\n"
            f"{pairwise['ap_over_prevalence']:.1f}×"
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    # ==================================================
    # Final layout
    # ==================================================

    fig.suptitle(
        "Perturb-seq phenotypes recover curated protein-complex relationships",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    fig.text(
        0.5,
        -0.015,
        (
            "Responsive K562 perturbations evaluated against curated CORUM "
            "protein-complex annotations; random baseline generated by gene-label permutation."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout()

    output = (
        FIG_DIR
        / "07_portfolio_summary.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", output)

    print("\nKey metrics")
    print("-" * 50)

    print(
        "Top-1 same-complex hit:",
        f"{comparison.loc[comparison['k'] == 1, 'hit_rate_at_k'].iloc[0]:.3f}",
    )

    print(
        "Top-20 same-complex hit:",
        f"{comparison.loc[comparison['k'] == 20, 'hit_rate_at_k'].iloc[0]:.3f}",
    )

    print(
        "Held-out Top-1:",
        f"{prediction['top1_accuracy']:.3f}",
    )

    print(
        "Held-out Top-10:",
        f"{prediction['top10_accuracy']:.3f}",
    )

    print(
        "Pairwise AP:",
        f"{pairwise['average_precision']:.3f}",
    )

    print(
        "Pairwise AUROC:",
        f"{pairwise['auroc']:.3f}",
    )

    print(
        "AP / prevalence:",
        f"{pairwise['ap_over_prevalence']:.1f}x",
    )


if __name__ == "__main__":
    main()
