from pathlib import Path
import re

import anndata as ad
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

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


def assign_family(gene):
    if not isinstance(gene, str):
        return "Other"

    for family, pattern in FAMILIES.items():
        if re.match(pattern, gene):
            return family

    return "Other"


def main():
    print("Loading clean matrix...")

    adata = ad.read_h5ad(DATA)
    summary = pd.read_csv(
        SUMMARY,
        index_col=0,
    )

    summary = summary.loc[adata.obs_names]

    # --------------------------------------------------
    # Select perturbations used for network analysis
    # --------------------------------------------------

    mask = (
        summary["qc_pass"].astype(bool)
        & ~summary["is_control"].astype(bool)
    )

    # Do NOT filter by adjusted_effect_z yet.
    # Weak perturbations remain useful for evaluating
    # whether transcriptomic similarity is meaningful.

    names = adata.obs_names[mask]
    genes = summary.loc[
        names,
        "target_gene"
    ].fillna("unknown")

    print("Perturbations in network:", len(names))
    print("Unique target genes:", genes.nunique())

    # --------------------------------------------------
    # PCA representation generated previously
    # --------------------------------------------------

    X = adata.obsm["X_pca"][mask]

    finite = np.isfinite(X).all(axis=1)

    X = X[finite]
    names = names[finite]
    genes = genes.loc[names]

    print(
        "Finite PCA embeddings:",
        len(names),
    )

    # --------------------------------------------------
    # kNN graph using cosine distance
    # --------------------------------------------------

    k = 15

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        n_jobs=-1,
    )

    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    edges = {}

    for i in range(len(names)):
        for d, j in zip(
            distances[i, 1:],
            indices[i, 1:],
        ):
            if i == j:
                continue

            edge = tuple(
                sorted((i, int(j)))
            )

            similarity = 1.0 - float(d)

            if edge not in edges:
                edges[edge] = similarity
            else:
                edges[edge] = max(
                    edges[edge],
                    similarity,
                )

    edge_list = list(edges.keys())
    weights = [
        edges[e]
        for e in edge_list
    ]

    graph = ig.Graph(
        n=len(names),
        edges=edge_list,
        directed=False,
    )

    graph.vs["name"] = names.tolist()
    graph.vs["target_gene"] = genes.tolist()
    graph.es["weight"] = weights

    print("Graph nodes:", graph.vcount())
    print("Graph edges:", graph.ecount())

    # --------------------------------------------------
    # Leiden clustering
    # --------------------------------------------------

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=1.0,
        seed=42,
    )

    cluster = np.array(
        partition.membership
    )

    print(
        "Leiden clusters:",
        len(np.unique(cluster)),
    )

    cluster_sizes = (
        pd.Series(cluster)
        .value_counts()
        .sort_index()
    )

    print("\nCluster sizes:")
    print(cluster_sizes.to_string())

    # --------------------------------------------------
    # Node metadata
    # --------------------------------------------------

    nodes = summary.loc[names].copy()

    nodes["cluster"] = cluster

    nodes["family"] = [
        assign_family(g)
        for g in nodes["target_gene"]
    ]

    nodes.to_csv(
        TABLE_DIR
        / "network_nodes.csv"
    )

    # --------------------------------------------------
    # Save edges
    # --------------------------------------------------

    edge_rows = []

    for (i, j), weight in zip(
        edge_list,
        weights,
    ):
        edge_rows.append(
            {
                "source": names[i],
                "target": names[j],
                "source_gene": genes.iloc[i],
                "target_gene": genes.iloc[j],
                "cosine_similarity": weight,
            }
        )

    edge_df = pd.DataFrame(edge_rows)

    edge_df.to_csv(
        TABLE_DIR
        / "network_edges.csv",
        index=False,
    )

    # --------------------------------------------------
    # Family enrichment by cluster
    # --------------------------------------------------

    enrichment_rows = []

    for family in FAMILIES:
        global_fraction = (
            nodes["family"] == family
        ).mean()

        if global_fraction == 0:
            continue

        for cluster_id, group in nodes.groupby(
            "cluster"
        ):
            n_family = (
                group["family"] == family
            ).sum()

            if n_family == 0:
                continue

            fraction = n_family / len(group)

            enrichment_rows.append(
                {
                    "family": family,
                    "cluster": cluster_id,
                    "cluster_size": len(group),
                    "n_family": n_family,
                    "fraction": fraction,
                    "global_fraction": global_fraction,
                    "fold_enrichment":
                        fraction / global_fraction,
                }
            )

    enrichment = pd.DataFrame(
        enrichment_rows
    )

    enrichment = enrichment.sort_values(
        [
            "fold_enrichment",
            "n_family",
        ],
        ascending=False,
    )

    enrichment.to_csv(
        TABLE_DIR
        / "cluster_family_enrichment.csv",
        index=False,
    )

    print("\nTop family enrichments:")
    print(
        enrichment.head(25)
        .round(2)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 2D visualization
    #
    # Use t-SNE only for visualization.
    # Clustering was performed on PCA kNN graph.
    # --------------------------------------------------

    print("\nCreating 2D visualization...")

    tsne = TSNE(
        n_components=2,
        perplexity=50,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    xy = tsne.fit_transform(X)

    nodes["TSNE1"] = xy[:, 0]
    nodes["TSNE2"] = xy[:, 1]

    nodes.to_csv(
        TABLE_DIR
        / "network_nodes.csv"
    )

    # --------------------------------------------------
    # Figure 1: clusters
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
    )

    axes[0].scatter(
        xy[:, 0],
        xy[:, 1],
        c=cluster,
        cmap="tab20",
        s=5,
        alpha=0.65,
    )

    axes[0].set_title(
        "Perturbation effect landscape"
    )

    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # --------------------------------------------------
    # Figure 2: known functional families
    # --------------------------------------------------

    axes[1].scatter(
        xy[:, 0],
        xy[:, 1],
        color="lightgrey",
        s=4,
        alpha=0.25,
    )

    colors = {
        "Ribosome": "#1f77b4",
        "Mito ribosome": "#ff7f0e",
        "Proteasome": "#2ca02c",
        "Complex I": "#d62728",
        "RNA Pol II": "#9467bd",
        "Exosome": "#8c564b",
    }

    for family, color in colors.items():
        m = nodes["family"] == family

        axes[1].scatter(
            xy[m, 0],
            xy[m, 1],
            s=14,
            alpha=0.9,
            label=family,
            color=color,
        )

    axes[1].set_title(
        "Recovery of known functional modules"
    )

    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    axes[1].legend(
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "04_perturbation_modules.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSaved:")
    print(
        TABLE_DIR
        / "network_nodes.csv"
    )
    print(
        TABLE_DIR
        / "network_edges.csv"
    )
    print(
        TABLE_DIR
        / "cluster_family_enrichment.csv"
    )
    print(
        FIG_DIR
        / "04_perturbation_modules.png"
    )


if __name__ == "__main__":
    main()
