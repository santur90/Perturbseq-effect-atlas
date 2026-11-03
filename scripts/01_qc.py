from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


RAW_FILE = Path("data/raw/K562_gwps_raw_bulk_01.h5ad")
NORM_FILE = Path("data/raw/K562_gwps_normalized_bulk_01.h5ad")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def parse_perturbation_ids(obs_names):
    rows = []

    for name in obs_names:
        parts = name.split("_")

        if len(parts) >= 4:
            screen_id = parts[0]
            target = parts[1]
            guide = parts[2]
            ensembl_id = parts[-1]
        else:
            screen_id = None
            target = None
            guide = None
            ensembl_id = None

        rows.append(
            {
                "perturbation_id": name,
                "screen_id": screen_id,
                "target_gene": target,
                "guide": guide,
                "target_ensembl": ensembl_id,
            }
        )

    return pd.DataFrame(rows).set_index("perturbation_id")


def main():
    print("Loading metadata...")

    raw = sc.read_h5ad(RAW_FILE, backed="r")
    norm = sc.read_h5ad(NORM_FILE, backed="r")

    print("\nDataset")
    print(raw)

    # --------------------------------------------------
    # Perturbation metadata
    # --------------------------------------------------

    meta = raw.obs.copy()

    parsed = parse_perturbation_ids(raw.obs_names)
    meta = meta.join(parsed)

    meta.to_csv(TABLE_DIR / "perturbation_metadata.csv")

    print("\nNumber of perturbations:", len(meta))
    print("Unique target genes:", meta["target_gene"].nunique())

    print("\nCell-count summary:")
    print(meta["num_cells_filtered"].describe())

    print("\nPerturbation efficiency fields:")
    print(
        meta[
            [
                "control_expr",
                "fold_expr",
                "pct_expr",
                "mean_leverage_score",
                "energy_test_p_value",
            ]
        ].describe()
    )

    # --------------------------------------------------
    # Check pct_expr relation
    # --------------------------------------------------

    relation_error = np.nanmax(
        np.abs(
            meta["pct_expr"]
            - (meta["fold_expr"] - 1)
        )
    )

    print(
        "\nMax |pct_expr - (fold_expr - 1)|:",
        relation_error,
    )

    # --------------------------------------------------
    # Scan normalized matrix for non-finite values
    # --------------------------------------------------

    print("\nScanning normalized matrix for NaN/Inf...")

    n_inf = 0
    n_nan = 0
    genes_with_inf = set()
    perturbations_with_inf = set()

    chunk_size = 500

    for start in range(0, norm.n_obs, chunk_size):
        end = min(start + chunk_size, norm.n_obs)

        X = np.asarray(norm.X[start:end, :])

        inf_mask = np.isinf(X)
        nan_mask = np.isnan(X)

        n_inf += inf_mask.sum()
        n_nan += nan_mask.sum()

        if inf_mask.any():
            row_idx, col_idx = np.where(inf_mask)

            for i in np.unique(row_idx):
                perturbations_with_inf.add(
                    norm.obs_names[start + i]
                )

            for j in np.unique(col_idx):
                genes_with_inf.add(
                    norm.var_names[j]
                )

    total_values = norm.n_obs * norm.n_vars

    print("\nNormalized matrix diagnostics")
    print("--------------------------------")
    print("Total values:", total_values)
    print("Inf values:", n_inf)
    print("NaN values:", n_nan)
    print(
        "Inf fraction:",
        n_inf / total_values,
    )
    print(
        "Perturbations containing Inf:",
        len(perturbations_with_inf),
    )
    print(
        "Genes containing Inf:",
        len(genes_with_inf),
    )

    # Save affected genes

    inf_gene_table = pd.DataFrame(
        {
            "ensembl_id": sorted(genes_with_inf)
        }
    )

    if len(inf_gene_table) > 0:
        inf_gene_table["gene_name"] = (
            inf_gene_table["ensembl_id"]
            .map(norm.var["gene_name"])
        )

    inf_gene_table.to_csv(
        TABLE_DIR / "genes_with_inf.csv",
        index=False,
    )

    # --------------------------------------------------
    # QC figures
    # --------------------------------------------------

    sns.set_context("talk")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    sns.histplot(
        meta["num_cells_filtered"].dropna(),
        bins=60,
        ax=axes[0, 0],
    )

    axes[0, 0].set_title(
        "Cells per perturbation"
    )

    axes[0, 0].set_xlabel(
        "Filtered cells"
    )

    sns.histplot(
        meta["pct_expr"].dropna(),
        bins=60,
        ax=axes[0, 1],
    )

    axes[0, 1].set_title(
        "Target expression change"
    )

    axes[0, 1].set_xlabel(
        "pct_expr"
    )

    sns.histplot(
        meta["mean_leverage_score"].dropna(),
        bins=60,
        ax=axes[1, 0],
    )

    axes[1, 0].set_title(
        "Perturbation leverage score"
    )

    p = (
        meta["energy_test_p_value"]
        .clip(lower=1e-300)
    )

    axes[1, 1].scatter(
        meta["mean_leverage_score"],
        -np.log10(p),
        s=8,
        alpha=0.35,
    )

    axes[1, 1].set_xlabel(
        "Mean leverage score"
    )

    axes[1, 1].set_ylabel(
        "-log10(energy-test p-value)"
    )

    axes[1, 1].set_title(
        "Perturbation effect evidence"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "01_perturbation_qc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSaved:")
    print(
        TABLE_DIR / "perturbation_metadata.csv"
    )
    print(
        TABLE_DIR / "genes_with_inf.csv"
    )
    print(
        FIG_DIR / "01_perturbation_qc.png"
    )


if __name__ == "__main__":
    main()
