from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import median_abs_deviation, spearmanr

INPUT = Path("results/tables/perturbation_summary.csv")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fit_control_background(control):
    """
    Fit expected log(effect_l2) as a smooth function of log(cell count)
    using non-targeting controls.
    """

    x = np.log10(control["num_cells_filtered"].to_numpy())
    y = np.log10(control["effect_l2"].to_numpy())

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Aggregate controls into cell-count bins to stabilize the fit.
    bins = pd.qcut(
        x,
        q=min(25, len(np.unique(x))),
        duplicates="drop",
    )

    binned = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "bin": bins,
        }
    ).groupby(
        "bin",
        observed=True,
    ).agg(
        x=("x", "median"),
        y=("y", "median"),
        n=("y", "size"),
    )

    binned = binned.sort_values("x")

    spline = UnivariateSpline(
        binned["x"],
        binned["y"],
        w=np.sqrt(binned["n"]),
        k=3,
        s=len(binned) * 0.002,
    )

    return spline, binned


def main():
    df = pd.read_csv(INPUT, index_col=0)

    required = [
        "num_cells_filtered",
        "effect_l2",
        "is_control",
        "qc_pass",
    ]

    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    valid = (
        df["num_cells_filtered"].notna()
        & df["effect_l2"].notna()
        & (df["num_cells_filtered"] > 0)
        & (df["effect_l2"] > 0)
    )

    controls = df[
        valid
        & df["is_control"].astype(bool)
    ].copy()

    print("Valid non-targeting controls:", len(controls))

    if len(controls) < 100:
        raise ValueError(
            "Too few non-targeting controls for background fitting."
        )

    spline, binned = fit_control_background(controls)

    df["log10_cells"] = np.nan
    df["log10_effect_l2"] = np.nan
    df["expected_log10_effect"] = np.nan
    df["effect_residual"] = np.nan

    df.loc[valid, "log10_cells"] = np.log10(
        df.loc[valid, "num_cells_filtered"]
    )

    df.loc[valid, "log10_effect_l2"] = np.log10(
        df.loc[valid, "effect_l2"]
    )

    df.loc[valid, "expected_log10_effect"] = spline(
        df.loc[valid, "log10_cells"]
    )

    df.loc[valid, "effect_residual"] = (
        df.loc[valid, "log10_effect_l2"]
        - df.loc[valid, "expected_log10_effect"]
    )

    control_residual = df.loc[
        valid & df["is_control"].astype(bool),
        "effect_residual",
    ]

    center = control_residual.median()
    mad = median_abs_deviation(
        control_residual,
        scale="normal",
        nan_policy="omit",
    )

    if not np.isfinite(mad) or mad <= 0:
        raise ValueError("Invalid control residual MAD.")

    df["adjusted_effect_z"] = (
        df["effect_residual"] - center
    ) / mad

    # Descriptive empirical tail probability against controls.
    # This is not a replicate-level inferential p-value.
    sorted_control = np.sort(control_residual.to_numpy())

    residual_values = df["effect_residual"].to_numpy()
    empirical_tail = np.full(len(df), np.nan)

    finite_residual = np.isfinite(residual_values)

    ranks = np.searchsorted(
        sorted_control,
        residual_values[finite_residual],
        side="left",
    )

    empirical_tail[finite_residual] = (
        len(sorted_control) - ranks + 1
    ) / (
        len(sorted_control) + 1
    )

    df["control_empirical_tail"] = empirical_tail

    df["strong_adjusted_effect"] = (
        df["qc_pass"].astype(bool)
        & ~df["is_control"].astype(bool)
        & (df["adjusted_effect_z"] >= 3)
    )

    before = df[
        valid & ~df["is_control"].astype(bool)
    ]

    rho_before, _ = spearmanr(
        before["num_cells_filtered"],
        before["effect_l2"],
    )

    after = before.dropna(
        subset=["adjusted_effect_z"]
    )

    rho_after, _ = spearmanr(
        after["num_cells_filtered"],
        after["adjusted_effect_z"],
    )

    print(f"Before adjustment: rho = {rho_before:.3f}")
    print(f"After adjustment:  rho = {rho_after:.3f}")

    print(
        "Targeting perturbations with adjusted_effect_z >= 3:",
        int(df["strong_adjusted_effect"].sum()),
    )

    print("\nTop adjusted effects:")
    display_columns = [
        "target_gene",
        "num_cells_filtered",
        "knockdown_efficiency",
        "effect_l2",
        "adjusted_effect_z",
        "control_empirical_tail",
        "energy_test_p_value",
        "mean_leverage_score",
    ]

    top = (
        df[
            df["qc_pass"].astype(bool)
            & ~df["is_control"].astype(bool)
        ]
        .sort_values(
            "adjusted_effect_z",
            ascending=False,
        )
        [display_columns]
        .head(25)
    )

    print(top.round(4).to_string())

    df.to_csv(
        TABLE_DIR / "perturbation_summary_adjusted.csv"
    )

    top.to_csv(
        TABLE_DIR / "top_adjusted_effects.csv"
    )

    binned.to_csv(
        TABLE_DIR / "control_background_bins.csv",
        index=False,
    )

    # -------------------------------------------------
    # Diagnostic figure
    # -------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
    )

    target = df[
        valid
        & ~df["is_control"].astype(bool)
    ]

    control = df[
        valid
        & df["is_control"].astype(bool)
    ]

    axes[0].scatter(
        target["num_cells_filtered"],
        target["effect_l2"],
        s=5,
        alpha=0.2,
        color="grey",
        label="targeting",
    )

    axes[0].scatter(
        control["num_cells_filtered"],
        control["effect_l2"],
        s=10,
        alpha=0.65,
        color="crimson",
        label="non-targeting",
    )

    grid_cells = np.logspace(
        np.log10(
            df.loc[valid, "num_cells_filtered"].min()
        ),
        np.log10(
            df.loc[valid, "num_cells_filtered"].max()
        ),
        300,
    )

    expected_effect = 10 ** spline(
        np.log10(grid_cells)
    )

    axes[0].plot(
        grid_cells,
        expected_effect,
        color="black",
        linewidth=2,
        label="control background",
    )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Filtered cells")
    axes[0].set_ylabel("Effect magnitude")
    axes[0].set_title("Cell-count-dependent background")
    axes[0].legend(fontsize=8)

    axes[1].scatter(
        target["num_cells_filtered"],
        target["adjusted_effect_z"],
        s=5,
        alpha=0.25,
        color="grey",
    )

    axes[1].scatter(
        control["num_cells_filtered"],
        control["adjusted_effect_z"],
        s=10,
        alpha=0.6,
        color="crimson",
    )

    axes[1].axhline(
        3,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Filtered cells")
    axes[1].set_ylabel("Adjusted effect z-score")
    axes[1].set_title("Effect after cell-count adjustment")

    plot_df = df[
        df["qc_pass"].astype(bool)
        & ~df["is_control"].astype(bool)
    ].dropna(
        subset=[
            "adjusted_effect_z",
            "knockdown_efficiency",
        ]
    )

    color = -np.log10(
        plot_df["energy_test_p_value"]
        .clip(lower=1e-4)
    )

    scatter = axes[2].scatter(
        plot_df["knockdown_efficiency"].clip(-1, 1),
        plot_df["adjusted_effect_z"],
        c=color,
        cmap="viridis",
        s=7,
        alpha=0.45,
    )

    axes[2].axhline(
        3,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    axes[2].set_xlabel(
        "Target knockdown efficiency"
    )

    axes[2].set_ylabel(
        "Adjusted effect z-score"
    )

    axes[2].set_title(
        "Adjusted effect and knockdown"
    )

    cbar = fig.colorbar(
        scatter,
        ax=axes[2],
    )

    cbar.set_label(
        "-log10(energy-test p-value)"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "03_adjusted_effect_score.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSaved:")
    print(
        TABLE_DIR
        / "perturbation_summary_adjusted.csv"
    )
    print(
        TABLE_DIR
        / "top_adjusted_effects.csv"
    )
    print(
        FIG_DIR
        / "03_adjusted_effect_score.png"
    )


if __name__ == "__main__":
    main()
