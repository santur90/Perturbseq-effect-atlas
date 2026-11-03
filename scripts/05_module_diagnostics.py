import argparse
import re
from pathlib import Path

import anndata as ad
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors


DATA = Path("data/processed/effect_matrix_clean.h5ad")
SUMMARY = Path("results/tables/perturbation_summary_adjusted.csv")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


FAMILIES = {
    "Ribosome": r"^RP[LS]\d",
    "Mito ribosome": r"^MRP[LS]\d",
    "Proteasome": r"^PSM[ABCD]\d",
    "Complex I": r"^NDUF",
    "RNA Pol II": r"^POLR2",
    "Exosome": r"^EXOSC\d",
}


def family_of(gene):
    """Assign simple prefix-based functional family."""

    if not isinstance(gene, str):
        return "Other"

    for family, pattern in FAMILIES.items():
        if re.match(pattern, gene):
            return family

    return "Other"


def knn(X, k):
    """Cosine k-nearest-neighbor search."""

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        n_jobs=-1,
    )

    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    # Remove self-neighbor.
    similarities = 1.0 - distances[:, 1:]
    indices = indices[:, 1:]

    return similarities, indices


def build_graph(similarities, indices):
    """Build undirected weighted kNN graph."""

    edges = {}

    for i in range(indices.shape[0]):
        for similarity, j in zip(
            similarities[i],
            indices[i],
        ):
            j = int(j)

            if i == j:
                continue

            edge = (
                min(i, j),
                max(i, j),
            )

            edges[edge] = max(
                edges.get(edge, -1),
                float(similarity),
            )

    graph = ig.Graph(
        n=indices.shape[0],
        edges=list(edges.keys()),
        directed=False,
    )

    graph.es["weight"] = [
        max(weight, 0.0)
        for weight in edges.values()
    ]

    return graph


def best_f1(families, clusters):
    """
    For each known family, find the Leiden cluster
    recovering that family with the best F1 score.
    """

    rows = []

    for family in FAMILIES:
        member = families == family
        n_family = int(member.sum())

        if n_family < 5:
            continue

        best = {
            "f1": 0.0,
            "cluster": None,
            "precision": 0.0,
            "recall": 0.0,
        }

        candidate_clusters = np.unique(
            clusters[member]
        )

        for cluster_id in candidate_clusters:
            in_cluster = clusters == cluster_id

            tp = int(
                (member & in_cluster).sum()
            )

            precision = (
                tp / in_cluster.sum()
                if in_cluster.sum() > 0
                else 0.0
            )

            recall = tp / n_family

            if precision + recall > 0:
                f1 = (
                    2
                    * precision
                    * recall
                    / (precision + recall)
                )
            else:
                f1 = 0.0

            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "cluster": int(cluster_id),
                    "precision": precision,
                    "recall": recall,
                }

        rows.append(
            {
                "family": family,
                "n": n_family,
                "best_f1": best["f1"],
                "cluster": best["cluster"],
                "precision": best["precision"],
                "recall": best["recall"],
            }
        )

    return pd.DataFrame(rows)


def main(k, min_z, max_p):
    print("Loading data...")

    adata = ad.read_h5ad(DATA)

    summary = pd.read_csv(
        SUMMARY,
        index_col=0,
    )

    summary = summary.loc[
        adata.obs_names
    ]

    # ==================================================
    # Base QC set
    # ==================================================

    base = (
        summary["qc_pass"].astype(bool)
        & ~summary["is_control"].astype(bool)
    )

    X_all = adata.obsm["X_pca"]

    # ==================================================
    # Part 1
    # Known-family members:
    # recovered vs scattered
    # ==================================================

    names = adata.obs_names[base]

    X = X_all[base.values]

    meta = summary.loc[
        names
    ].copy()

    meta["family"] = (
        meta["target_gene"]
        .map(family_of)
    )

    genes = (
        meta["target_gene"]
        .fillna("")
        .values
    )

    families = meta["family"].values

    # Use additional neighbors because same-target
    # promoter perturbations will later be excluded.
    _, indices = knn(
        X,
        k + 20,
    )

    same_family_fraction = np.full(
        len(meta),
        np.nan,
    )

    for i in np.where(
        families != "Other"
    )[0]:

        # Exclude another perturbation of the same gene.
        neighbors = [
            j
            for j in indices[i]
            if genes[j] != genes[i]
        ][:k]

        if len(neighbors) == 0:
            continue

        same_family_fraction[i] = np.mean(
            families[neighbors]
            == families[i]
        )

    meta[
        "frac_neighbors_same_family"
    ] = same_family_fraction

    family_members = meta[
        meta["family"] != "Other"
    ].copy()

    # Descriptive threshold:
    # >=30% of top-k neighbors belong to same family.
    family_members["in_module"] = (
        family_members[
            "frac_neighbors_same_family"
        ]
        >= 0.30
    )

    print(
        f"Family members: {len(family_members)}; "
        f"in-module (>=30% same-family neighbours): "
        f"{int(family_members['in_module'].sum())}"
    )

    # ==================================================
    # Compare recovered vs scattered family members
    # ==================================================

    diagnostic_rows = []

    diagnostic_columns = [
        "adjusted_effect_z",
        "knockdown_efficiency",
        "num_cells_filtered",
    ]

    for column in diagnostic_columns:

        recovered = family_members.loc[
            family_members["in_module"],
            column,
        ].dropna()

        scattered = family_members.loc[
            ~family_members["in_module"],
            column,
        ].dropna()

        if (
            len(recovered) > 0
            and len(scattered) > 0
        ):
            test = mannwhitneyu(
                recovered,
                scattered,
                alternative="two-sided",
            )

            p_value = test.pvalue
        else:
            p_value = np.nan

        diagnostic_rows.append(
            {
                "metric": column,
                "in_module_median":
                    recovered.median(),
                "scattered_median":
                    scattered.median(),
                "mannwhitney_p":
                    p_value,
                "n_in_module":
                    len(recovered),
                "n_scattered":
                    len(scattered),
            }
        )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    print(
        "\nIn-module vs scattered members:"
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    family_members.to_csv(
        TABLE_DIR
        / "family_member_module_status.csv"
    )

    diagnostics.to_csv(
        TABLE_DIR
        / "in_module_vs_scattered.csv",
        index=False,
    )

    # ==================================================
    # Diagnostic figure
    # ==================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
    )

    plot_specs = [
        (
            axes[0],
            "adjusted_effect_z",
            "Adjusted effect score",
        ),
        (
            axes[1],
            "knockdown_efficiency",
            "Knockdown efficiency",
        ),
    ]

    for axis, column, ylabel in plot_specs:

        recovered = family_members.loc[
            family_members["in_module"],
            column,
        ].dropna()

        scattered = family_members.loc[
            ~family_members["in_module"],
            column,
        ].dropna()

        # Only clip knockdown efficiency for
        # visualization.
        #
        # Do NOT clip adjusted_effect_z.
        if column == "knockdown_efficiency":
            recovered_plot = recovered.clip(
                -1,
                1,
            )

            scattered_plot = scattered.clip(
                -1,
                1,
            )

        else:
            recovered_plot = recovered
            scattered_plot = scattered

        data = [
            recovered_plot,
            scattered_plot,
        ]

        axis.boxplot(
            data,
            tick_labels=[
                (
                    "in module\n"
                    f"(n={len(recovered_plot)})"
                ),
                (
                    "scattered\n"
                    f"(n={len(scattered_plot)})"
                ),
            ],
            showfliers=False,
        )

        # Add raw points with deterministic jitter.
        rng = np.random.default_rng(42)

        for x_position, values in enumerate(
            data,
            start=1,
        ):
            jitter = rng.normal(
                x_position,
                0.04,
                size=len(values),
            )

            axis.scatter(
                jitter,
                values,
                s=10,
                alpha=0.35,
                color="steelblue",
            )

        axis.set_ylabel(ylabel)

    axes[0].set_title(
        "Transcriptional effect strength"
    )

    axes[1].set_title(
        "Target knockdown"
    )

    fig.suptitle(
        "Known-complex members: "
        "recovered vs. scattered",
        fontsize=14,
    )

    plt.tight_layout()

    figure_path = (
        FIG_DIR
        / "05_in_module_vs_scattered.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================
    # Part 2
    # Responsive-only network
    # ==================================================

    responsive = (
        base
        & (
            summary["adjusted_effect_z"]
            >= min_z
        )
        & (
            summary["energy_test_p_value"]
            <= max_p
        )
    )

    n_responsive = int(
        responsive.sum()
    )

    n_base = int(
        base.sum()
    )

    print(
        "\nResponsive perturbations "
        f"(z>={min_z}, "
        f"energy p<={max_p}): "
        f"{n_responsive}/{n_base}"
    )

    if n_responsive < 100:
        raise ValueError(
            "Too few responsive perturbations. "
            "Consider lowering --min-z or "
            "relaxing --max-p."
        )

    X_responsive = X_all[
        responsive.values
    ]

    responsive_meta = summary.loc[
        responsive
    ].copy()

    responsive_families = (
        responsive_meta[
            "target_gene"
        ]
        .map(family_of)
        .values
    )

    similarities, responsive_indices = knn(
        X_responsive,
        k,
    )

    graph = build_graph(
        similarities,
        responsive_indices,
    )

    print(
        "Responsive graph nodes:",
        graph.vcount(),
    )

    print(
        "Responsive graph edges:",
        graph.ecount(),
    )

    # ==================================================
    # Leiden resolution sweep
    # ==================================================

    resolutions = [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ]

    sweep_rows = []
    memberships = {}

    for resolution in resolutions:

        partition = (
            leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=42,
            )
        )

        membership = np.array(
            partition.membership
        )

        memberships[resolution] = membership

        family_scores = best_f1(
            responsive_families,
            membership,
        )

        family_scores[
            "resolution"
        ] = resolution

        family_scores.to_csv(
            TABLE_DIR
            / (
                "family_f1_res"
                f"{resolution}.csv"
            ),
            index=False,
        )

        cluster_sizes = np.bincount(
            membership
        )

        mean_family_f1 = (
            family_scores["best_f1"].mean()
            if len(family_scores) > 0
            else np.nan
        )

        sweep_rows.append(
            {
                "resolution":
                    resolution,
                "n_clusters":
                    len(
                        np.unique(
                            membership
                        )
                    ),
                "median_cluster_size":
                    int(
                        np.median(
                            cluster_sizes
                        )
                    ),
                "mean_family_best_f1":
                    mean_family_f1,
            }
        )

    sweep = pd.DataFrame(
        sweep_rows
    )

    print(
        "\nResolution sweep "
        "(responsive graph):"
    )

    print(
        sweep.round(3)
        .to_string(
            index=False
        )
    )

    sweep.to_csv(
        TABLE_DIR
        / "leiden_resolution_sweep.csv",
        index=False,
    )

    # ==================================================
    # Pick best resolution
    # ==================================================

    valid_sweep = sweep.dropna(
        subset=[
            "mean_family_best_f1"
        ]
    )

    if len(valid_sweep) == 0:
        raise ValueError(
            "No valid family F1 scores "
            "were produced."
        )

    best_resolution = valid_sweep.loc[
        valid_sweep[
            "mean_family_best_f1"
        ].idxmax(),
        "resolution",
    ]

    best_membership = memberships[
        best_resolution
    ]

    best_scores = best_f1(
        responsive_families,
        best_membership,
    )

    print(
        "\nBest resolution by "
        "mean family F1:",
        best_resolution,
    )

    print(
        "\nFamily recovery at "
        "best resolution:"
    )

    print(
        best_scores
        .round(3)
        .to_string(
            index=False
        )
    )

    best_scores.to_csv(
        TABLE_DIR
        / "family_recovery_best_resolution.csv",
        index=False,
    )

    # ==================================================
    # Save responsive network nodes
    # ==================================================

    output_nodes = (
        responsive_meta[
            [
                "target_gene",
                "adjusted_effect_z",
                "knockdown_efficiency",
                "num_cells_filtered",
                "energy_test_p_value",
                "mean_leverage_score",
            ]
        ]
        .copy()
    )

    output_nodes[
        "family"
    ] = responsive_families

    output_nodes[
        "cluster"
    ] = best_membership

    output_nodes.to_csv(
        TABLE_DIR
        / "responsive_network_nodes.csv"
    )

    # ==================================================
    # Save responsive network edges
    # ==================================================

    responsive_names = (
        responsive_meta.index.to_numpy()
    )

    responsive_genes = (
        responsive_meta[
            "target_gene"
        ]
        .fillna("")
        .to_numpy()
    )

    edge_rows = []

    seen = set()

    for i in range(
        responsive_indices.shape[0]
    ):
        for sim, j in zip(
            similarities[i],
            responsive_indices[i],
        ):
            j = int(j)

            edge = (
                min(i, j),
                max(i, j),
            )

            if edge in seen:
                continue

            seen.add(edge)

            edge_rows.append(
                {
                    "source":
                        responsive_names[
                            edge[0]
                        ],
                    "target":
                        responsive_names[
                            edge[1]
                        ],
                    "source_gene":
                        responsive_genes[
                            edge[0]
                        ],
                    "target_gene":
                        responsive_genes[
                            edge[1]
                        ],
                    "cosine_similarity":
                        float(sim),
                }
            )

    pd.DataFrame(
        edge_rows
    ).to_csv(
        TABLE_DIR
        / "responsive_network_edges.csv",
        index=False,
    )

    print("\nSaved:")

    outputs = [
        TABLE_DIR
        / "family_member_module_status.csv",

        TABLE_DIR
        / "in_module_vs_scattered.csv",

        TABLE_DIR
        / "leiden_resolution_sweep.csv",

        TABLE_DIR
        / "family_recovery_best_resolution.csv",

        TABLE_DIR
        / "responsive_network_nodes.csv",

        TABLE_DIR
        / "responsive_network_edges.csv",

        FIG_DIR
        / "05_in_module_vs_scattered.png",
    ]

    for output in outputs:
        print(" ", output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help=(
            "Number of cosine nearest "
            "neighbors."
        ),
    )

    parser.add_argument(
        "--min-z",
        type=float,
        default=5.0,
        help=(
            "Minimum cell-count-adjusted "
            "effect score for responsive "
            "network."
        ),
    )

    parser.add_argument(
        "--max-p",
        type=float,
        default=0.001,
        help=(
            "Maximum energy-test p-value "
            "for responsive network."
        ),
    )

    args = parser.parse_args()

    main(
        args.k,
        args.min_z,
        args.max_p,
    )
