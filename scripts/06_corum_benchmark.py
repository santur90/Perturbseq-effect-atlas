import argparse
from collections import defaultdict
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


DATA = Path("data/processed/effect_matrix_clean.h5ad")
SUMMARY = Path("results/tables/perturbation_summary_adjusted.csv")
CORUM = Path("data/reference/corum_human_core.csv")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility
# ============================================================

def safe_median(values):
    """
    Median that returns NaN when every value is missing.

    Avoids RuntimeWarning: Mean of empty slice.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


# ============================================================
# CORUM parsing
# ============================================================

def parse_corum(
    corum_file,
    available_genes,
    min_complex_size=3,
    max_complex_size=50,
    min_screen_members=3,
):
    corum = pd.read_csv(corum_file)

    required = [
        "ComplexID",
        "ComplexName",
        "Organism",
        "subunits.Gene.name.",
    ]

    missing = [
        col
        for col in required
        if col not in corum.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CORUM columns: {missing}"
        )

    corum = corum[
        corum["Organism"].eq("Human")
    ].copy()

    available_genes = {
        str(gene).upper()
        for gene in available_genes
    }

    records = []

    for _, row in corum.iterrows():

        members = {
            gene.strip().upper()
            for gene in str(
                row["subunits.Gene.name."]
            ).split(";")
            if gene.strip()
            and gene.strip().lower() != "nan"
        }

        if not (
            min_complex_size
            <= len(members)
            <= max_complex_size
        ):
            continue

        represented = (
            members
            & available_genes
        )

        if (
            len(represented)
            < min_screen_members
        ):
            continue

        records.append(
            {
                "complex_id":
                    str(row["ComplexID"]),

                "complex_name":
                    row["ComplexName"],

                "all_members":
                    members,

                "screen_members":
                    represented,

                "n_all_members":
                    len(members),

                "n_screen_members":
                    len(represented),
            }
        )

    return records


# ============================================================
# Gene-level perturbation embeddings
# ============================================================

def build_gene_embeddings(
    adata,
    summary,
    min_z,
    max_p,
):
    """
    Collapse multiple promoter perturbations for the
    same target gene into a median PCA embedding.

    Requirements:
      - QC-pass
      - targeting perturbation
      - target gene available
      - adjusted_effect_z >= min_z
      - energy_test_p_value <= max_p
      - finite PCA coordinates
    """

    summary = summary.loc[
        adata.obs_names
    ].copy()

    X_pca = np.asarray(
        adata.obsm["X_pca"]
    )

    mask = (
        summary["qc_pass"].astype(bool)
        & ~summary["is_control"].astype(bool)
        & summary["target_gene"].notna()
        & (
            summary["adjusted_effect_z"]
            >= min_z
        )
        & (
            summary["energy_test_p_value"]
            <= max_p
        )
    )

    positions = np.flatnonzero(
        mask.to_numpy()
    )

    selected = summary.iloc[
        positions
    ].copy()

    selected[
        "matrix_position"
    ] = positions

    finite_embedding = np.array(
        [
            np.isfinite(
                X_pca[position]
            ).all()
            for position in positions
        ],
        dtype=bool,
    )

    n_removed = int(
        (~finite_embedding).sum()
    )

    if n_removed > 0:
        print(
            "Removing responsive perturbations "
            "with non-finite PCA embeddings:",
            n_removed,
        )

    selected = selected.iloc[
        np.flatnonzero(
            finite_embedding
        )
    ].copy()

    gene_rows = []
    gene_embeddings = []

    for gene, group in selected.groupby(
        "target_gene",
        sort=True,
    ):

        gene_positions = group[
            "matrix_position"
        ].to_numpy(
            dtype=int
        )

        gene_matrix = X_pca[
            gene_positions
        ]

        if not np.isfinite(
            gene_matrix
        ).all():
            continue

        # All values are finite here, so regular median
        # is appropriate and should not produce warnings.
        embedding = np.median(
            gene_matrix,
            axis=0,
        )

        if not np.isfinite(
            embedding
        ).all():
            continue

        gene_rows.append(
            {
                "target_gene":
                    str(gene).upper(),

                "n_perturbations":
                    len(group),

                "median_adjusted_effect_z":
                    safe_median(
                        group[
                            "adjusted_effect_z"
                        ]
                    ),

                "median_knockdown_efficiency":
                    safe_median(
                        group[
                            "knockdown_efficiency"
                        ]
                    ),

                "median_num_cells":
                    safe_median(
                        group[
                            "num_cells_filtered"
                        ]
                    ),

                "min_energy_test_p_value":
                    pd.to_numeric(
                        group[
                            "energy_test_p_value"
                        ],
                        errors="coerce",
                    ).min(),
            }
        )

        gene_embeddings.append(
            embedding
        )

    if not gene_embeddings:
        raise ValueError(
            "No finite gene embeddings generated."
        )

    gene_metadata = (
        pd.DataFrame(
            gene_rows
        )
        .set_index(
            "target_gene"
        )
    )

    gene_embeddings = np.vstack(
        gene_embeddings
    ).astype(
        np.float32
    )

    print(
        "Finite gene embeddings:",
        len(gene_metadata),
    )

    return (
        gene_metadata,
        gene_embeddings,
        selected,
    )


# ============================================================
# CORUM annotation maps
# ============================================================

def build_annotations(
    genes,
    complexes,
):
    gene_set = set(
        genes
    )

    gene_to_complexes = defaultdict(
        set
    )

    complex_to_genes = {}
    complex_names = {}

    for record in complexes:

        complex_id = record[
            "complex_id"
        ]

        members = (
            record["screen_members"]
            & gene_set
        )

        if len(members) < 3:
            continue

        complex_to_genes[
            complex_id
        ] = members

        complex_names[
            complex_id
        ] = record[
            "complex_name"
        ]

        for gene in members:
            gene_to_complexes[
                gene
            ].add(
                complex_id
            )

    return (
        gene_to_complexes,
        complex_to_genes,
        complex_names,
    )


# ============================================================
# Cosine nearest neighbors
# ============================================================

def find_neighbors(
    embeddings,
    k_max,
):
    n_neighbors = min(
        k_max + 1,
        len(embeddings),
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        n_jobs=-1,
    )

    model.fit(
        embeddings
    )

    distances, indices = (
        model.kneighbors(
            embeddings
        )
    )

    # First neighbor is self.
    similarities = (
        1.0
        - distances[:, 1:]
    )

    indices = (
        indices[:, 1:]
    )

    return (
        similarities,
        indices,
    )


# ============================================================
# Neighbor recovery
# ============================================================

def evaluate_neighbor_recovery(
    genes,
    similarities,
    indices,
    gene_to_complexes,
    k_values,
):
    summary_rows = []
    query_rows = []

    max_k = max(
        k_values
    )

    for i, query_gene in enumerate(
        genes
    ):

        truth = (
            gene_to_complexes.get(
                query_gene,
                set(),
            )
        )

        if not truth:
            continue

        neighbor_indices = indices[
            i,
            :max_k,
        ]

        neighbor_genes = genes[
            neighbor_indices
        ]

        labels = np.array(
            [
                int(
                    bool(
                        truth
                        & gene_to_complexes.get(
                            neighbor_gene,
                            set(),
                        )
                    )
                )
                for neighbor_gene
                in neighbor_genes
            ],
            dtype=int,
        )

        positive_ranks = np.where(
            labels == 1
        )[0]

        first_positive_rank = (
            int(
                positive_ranks[0]
                + 1
            )
            if len(
                positive_ranks
            ) > 0
            else np.nan
        )

        row = {
            "query_gene":
                query_gene,

            "n_true_complexes":
                len(
                    truth
                ),

            "first_positive_rank":
                first_positive_rank,

            "reciprocal_rank":
                (
                    1.0
                    / first_positive_rank
                    if np.isfinite(
                        first_positive_rank
                    )
                    else 0.0
                ),
        }

        for k in k_values:

            top_labels = labels[
                :k
            ]

            row[
                f"hit_at_{k}"
            ] = int(
                top_labels.sum()
                > 0
            )

            row[
                f"precision_at_{k}"
            ] = float(
                top_labels.mean()
            )

        query_rows.append(
            row
        )

    query_table = pd.DataFrame(
        query_rows
    )

    if len(
        query_table
    ) == 0:
        raise ValueError(
            "No CORUM-annotated queries available."
        )

    for k in k_values:

        summary_rows.append(
            {
                "method":
                    "Perturb-seq cosine",

                "k":
                    k,

                "n_queries":
                    len(
                        query_table
                    ),

                "hit_rate_at_k":
                    query_table[
                        f"hit_at_{k}"
                    ].mean(),

                "mean_precision_at_k":
                    query_table[
                        f"precision_at_{k}"
                    ].mean(),

                "mrr":
                    query_table[
                        "reciprocal_rank"
                    ].mean(),
            }
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        query_table,
    )


# ============================================================
# Held-out complex-label prediction
# ============================================================

def evaluate_complex_prediction(
    genes,
    similarities,
    indices,
    gene_to_complexes,
    k_neighbors=20,
):
    """
    Neighbor identities come only from Perturb-seq
    phenotype geometry.

    Query CORUM annotation is used only to score the
    neighbor-voting result.
    """

    rows = []

    for i, query_gene in enumerate(
        genes
    ):

        truth = (
            gene_to_complexes.get(
                query_gene,
                set(),
            )
        )

        if not truth:
            continue

        votes = defaultdict(
            float
        )

        for (
            similarity,
            neighbor_index,
        ) in zip(
            similarities[
                i,
                :k_neighbors,
            ],
            indices[
                i,
                :k_neighbors,
            ],
        ):

            neighbor_gene = genes[
                neighbor_index
            ]

            neighbor_complexes = (
                gene_to_complexes.get(
                    neighbor_gene,
                    set(),
                )
            )

            if not neighbor_complexes:
                continue

            # Similarity-weighted neighbor vote.
            # Split the vote if the neighbor has
            # overlapping CORUM annotations.
            weight = max(
                float(
                    similarity
                ),
                0.0,
            ) / len(
                neighbor_complexes
            )

            for complex_id in (
                neighbor_complexes
            ):
                votes[
                    complex_id
                ] += weight

        ranked = sorted(
            votes.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        true_ranks = [
            rank
            for rank, (
                complex_id,
                _
            ) in enumerate(
                ranked,
                start=1,
            )
            if complex_id in truth
        ]

        best_rank = (
            min(
                true_ranks
            )
            if true_ranks
            else np.nan
        )

        rows.append(
            {
                "query_gene":
                    query_gene,

                "n_true_complexes":
                    len(
                        truth
                    ),

                "n_predicted_complexes":
                    len(
                        ranked
                    ),

                "best_true_complex_rank":
                    best_rank,

                "top1_correct":
                    int(
                        np.isfinite(
                            best_rank
                        )
                        and best_rank <= 1
                    ),

                "top5_correct":
                    int(
                        np.isfinite(
                            best_rank
                        )
                        and best_rank <= 5
                    ),

                "top10_correct":
                    int(
                        np.isfinite(
                            best_rank
                        )
                        and best_rank <= 10
                    ),

                "reciprocal_rank":
                    (
                        1.0
                        / best_rank
                        if np.isfinite(
                            best_rank
                        )
                        else 0.0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Pairwise benchmark
# ============================================================

def evaluate_pairwise(
    genes,
    embeddings,
    gene_to_complexes,
    random_state=42,
    max_negative_pairs=500_000,
):
    """
    Positive pair:
        both genes share >=1 retained CORUM complex.

    Negative pair:
        both genes have retained CORUM annotation,
        but they share no retained complex.

    All positives are retained.
    Negatives are reservoir-sampled when needed.
    """

    rng = np.random.default_rng(
        random_state
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    normalized = (
        embeddings
        / np.clip(
            norms,
            1e-12,
            None,
        )
    )

    positive_scores = []
    negative_scores = []

    negative_seen = 0

    n_genes = len(
        genes
    )

    for i in range(
        n_genes - 1
    ):

        query_complexes = (
            gene_to_complexes.get(
                genes[i],
                set(),
            )
        )

        if not query_complexes:
            continue

        similarities = (
            normalized[
                i + 1:
            ]
            @ normalized[i]
        )

        for other_index, score in enumerate(
            similarities,
            start=i + 1,
        ):

            other_complexes = (
                gene_to_complexes.get(
                    genes[
                        other_index
                    ],
                    set(),
                )
            )

            if not other_complexes:
                continue

            score = float(
                score
            )

            if (
                query_complexes
                & other_complexes
            ):

                positive_scores.append(
                    score
                )

            else:

                negative_seen += 1

                if (
                    len(
                        negative_scores
                    )
                    < max_negative_pairs
                ):
                    negative_scores.append(
                        score
                    )

                else:

                    replacement = int(
                        rng.integers(
                            0,
                            negative_seen,
                        )
                    )

                    if (
                        replacement
                        < max_negative_pairs
                    ):
                        negative_scores[
                            replacement
                        ] = score

    labels = np.concatenate(
        [
            np.ones(
                len(
                    positive_scores
                ),
                dtype=int,
            ),

            np.zeros(
                len(
                    negative_scores
                ),
                dtype=int,
            ),
        ]
    )

    scores = np.concatenate(
        [
            np.asarray(
                positive_scores,
                dtype=float,
            ),

            np.asarray(
                negative_scores,
                dtype=float,
            ),
        ]
    )

    if len(
        np.unique(
            labels
        )
    ) < 2:
        raise ValueError(
            "Pairwise benchmark needs both "
            "positive and negative pairs."
        )

    prevalence = float(
        labels.mean()
    )

    ap = float(
        average_precision_score(
            labels,
            scores,
        )
    )

    auroc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    return {
        "n_positive_pairs":
            int(
                len(
                    positive_scores
                )
            ),

        "n_negative_pairs":
            int(
                len(
                    negative_scores
                )
            ),

        "positive_prevalence":
            prevalence,

        "average_precision":
            ap,

        "ap_over_prevalence":
            (
                ap / prevalence
                if prevalence > 0
                else np.nan
            ),

        "auroc":
            auroc,
    }


# ============================================================
# Random-label baseline
# ============================================================

def random_baseline(
    embeddings,
    genes,
    gene_to_complexes,
    k_values,
    n_permutations=100,
    random_state=42,
):
    """
    Keep Perturb-seq geometry fixed and shuffle gene
    identities across the geometry.

    This preserves annotation frequency while breaking
    correspondence between phenotype and gene identity.
    """

    rng = np.random.default_rng(
        random_state
    )

    max_k = max(
        k_values
    )

    similarities, indices = (
        find_neighbors(
            embeddings,
            k_max=max_k,
        )
    )

    rows = []

    for permutation in range(
        n_permutations
    ):

        shuffled_genes = (
            rng.permutation(
                genes
            )
        )

        metrics, _ = (
            evaluate_neighbor_recovery(
                shuffled_genes,
                similarities,
                indices,
                gene_to_complexes,
                k_values,
            )
        )

        metrics[
            "method"
        ] = (
            "Permuted gene labels"
        )

        metrics[
            "permutation"
        ] = permutation

        rows.append(
            metrics
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main(
    min_z,
    max_p,
    min_complex_size,
    max_complex_size,
    min_screen_members,
    n_permutations,
):

    print(
        "Loading Perturb-seq data..."
    )

    adata = ad.read_h5ad(
        DATA
    )

    summary = pd.read_csv(
        SUMMARY,
        index_col=0,
    )

    (
        gene_metadata,
        embeddings,
        perturbation_metadata,
    ) = build_gene_embeddings(
        adata,
        summary,
        min_z=min_z,
        max_p=max_p,
    )

    genes = (
        gene_metadata.index
        .to_numpy()
    )

    print(
        "Responsive perturbations:",
        len(
            perturbation_metadata
        ),
    )

    print(
        "Responsive target genes:",
        len(
            genes
        ),
    )

    # --------------------------------------------------------
    # CORUM
    # --------------------------------------------------------

    print(
        "\nLoading CORUM..."
    )

    complexes = parse_corum(
        CORUM,
        available_genes=genes,
        min_complex_size=(
            min_complex_size
        ),
        max_complex_size=(
            max_complex_size
        ),
        min_screen_members=(
            min_screen_members
        ),
    )

    (
        gene_to_complexes,
        complex_to_genes,
        complex_names,
    ) = build_annotations(
        genes,
        complexes,
    )

    annotated_genes = [
        gene
        for gene in genes
        if gene in gene_to_complexes
    ]

    print(
        "Retained CORUM complexes:",
        len(
            complex_to_genes
        ),
    )

    print(
        "CORUM-annotated responsive genes:",
        len(
            annotated_genes
        ),
    )

    # --------------------------------------------------------
    # Neighbors
    # --------------------------------------------------------

    k_values = [
        1,
        5,
        10,
        20,
    ]

    max_k = max(
        k_values
    )

    similarities, indices = (
        find_neighbors(
            embeddings,
            k_max=max_k,
        )
    )

    # --------------------------------------------------------
    # Neighbor recovery
    # --------------------------------------------------------

    print(
        "\nEvaluating nearest-neighbor "
        "complex recovery..."
    )

    (
        neighbor_metrics,
        query_metrics,
    ) = evaluate_neighbor_recovery(
        genes,
        similarities,
        indices,
        gene_to_complexes,
        k_values,
    )

    print(
        neighbor_metrics
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Held-out complex prediction
    # --------------------------------------------------------

    print(
        "\nEvaluating held-out complex "
        "label prediction..."
    )

    complex_prediction = (
        evaluate_complex_prediction(
            genes,
            similarities,
            indices,
            gene_to_complexes,
            k_neighbors=20,
        )
    )

    complex_summary = {
        "n_queries":
            len(
                complex_prediction
            ),

        "top1_accuracy":
            complex_prediction[
                "top1_correct"
            ].mean(),

        "top5_accuracy":
            complex_prediction[
                "top5_correct"
            ].mean(),

        "top10_accuracy":
            complex_prediction[
                "top10_correct"
            ].mean(),

        "mrr":
            complex_prediction[
                "reciprocal_rank"
            ].mean(),
    }

    print(
        pd.Series(
            complex_summary
        )
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Pairwise benchmark
    # --------------------------------------------------------

    print(
        "\nEvaluating pairwise "
        "same-complex prediction..."
    )

    pairwise = evaluate_pairwise(
        genes,
        embeddings,
        gene_to_complexes,
    )

    print(
        pd.Series(
            pairwise
        )
        .round(5)
        .to_string()
    )

    # --------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------

    print(
        f"\nRunning random-label baseline "
        f"({n_permutations} permutations)..."
    )

    random_metrics = random_baseline(
        embeddings,
        genes,
        gene_to_complexes,
        k_values,
        n_permutations=(
            n_permutations
        ),
    )

    random_summary = (
        random_metrics
        .groupby("k")
        .agg(
            random_hit_mean=(
                "hit_rate_at_k",
                "mean",
            ),

            random_hit_sd=(
                "hit_rate_at_k",
                "std",
            ),

            random_precision_mean=(
                "mean_precision_at_k",
                "mean",
            ),

            random_precision_sd=(
                "mean_precision_at_k",
                "std",
            ),

            random_mrr_mean=(
                "mrr",
                "mean",
            ),

            random_mrr_sd=(
                "mrr",
                "std",
            ),
        )
        .reset_index()
    )

    comparison = (
        neighbor_metrics
        .merge(
            random_summary,
            on="k",
            how="left",
        )
    )

    print(
        "\nPerturb-seq versus "
        "random-label baseline:"
    )

    print(
        comparison
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save metadata
    # ========================================================

    gene_metadata[
        "n_corum_complexes"
    ] = [
        len(
            gene_to_complexes.get(
                gene,
                set(),
            )
        )
        for gene in genes
    ]

    gene_metadata.to_csv(
        TABLE_DIR
        / "corum_gene_metadata.csv"
    )

    complex_rows = []

    for (
        complex_id,
        members,
    ) in complex_to_genes.items():

        complex_rows.append(
            {
                "complex_id":
                    complex_id,

                "complex_name":
                    complex_names[
                        complex_id
                    ],

                "n_screen_members":
                    len(
                        members
                    ),

                "screen_members":
                    ";".join(
                        sorted(
                            members
                        )
                    ),
            }
        )

    complex_table = (
        pd.DataFrame(
            complex_rows
        )
        .sort_values(
            "n_screen_members",
            ascending=False,
        )
    )

    complex_table.to_csv(
        TABLE_DIR
        / "corum_complexes_retained.csv",
        index=False,
    )

    # ========================================================
    # Save benchmark results
    # ========================================================

    neighbor_metrics.to_csv(
        TABLE_DIR
        / "corum_neighbor_metrics.csv",
        index=False,
    )

    query_metrics.to_csv(
        TABLE_DIR
        / "corum_query_metrics.csv",
        index=False,
    )

    complex_prediction.to_csv(
        TABLE_DIR
        / "corum_complex_prediction.csv",
        index=False,
    )

    pd.DataFrame(
        [complex_summary]
    ).to_csv(
        TABLE_DIR
        / "corum_complex_prediction_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [pairwise]
    ).to_csv(
        TABLE_DIR
        / "corum_pairwise_metrics.csv",
        index=False,
    )

    random_metrics.to_csv(
        TABLE_DIR
        / "corum_random_baseline.csv",
        index=False,
    )

    comparison.to_csv(
        TABLE_DIR
        / "corum_benchmark_comparison.csv",
        index=False,
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    # Hit rate@k
    axes[0].plot(
        comparison["k"],
        comparison[
            "hit_rate_at_k"
        ],
        marker="o",
        linewidth=2,
        label="Perturb-seq",
    )

    axes[0].errorbar(
        comparison["k"],
        comparison[
            "random_hit_mean"
        ],
        yerr=comparison[
            "random_hit_sd"
        ],
        marker="o",
        linestyle="--",
        capsize=3,
        label="Permuted labels",
    )

    axes[0].set_xlabel(
        "k neighbors"
    )

    axes[0].set_ylabel(
        "Hit rate@k"
    )

    axes[0].set_title(
        "Same-complex neighbor recovery"
    )

    axes[0].legend(
        frameon=False
    )

    # Precision@k
    axes[1].plot(
        comparison["k"],
        comparison[
            "mean_precision_at_k"
        ],
        marker="o",
        linewidth=2,
        label="Perturb-seq",
    )

    axes[1].errorbar(
        comparison["k"],
        comparison[
            "random_precision_mean"
        ],
        yerr=comparison[
            "random_precision_sd"
        ],
        marker="o",
        linestyle="--",
        capsize=3,
        label="Permuted labels",
    )

    axes[1].set_xlabel(
        "k neighbors"
    )

    axes[1].set_ylabel(
        "Mean precision@k"
    )

    axes[1].set_title(
        "CORUM neighbor precision"
    )

    axes[1].legend(
        frameon=False
    )

    # Held-out complex prediction
    bars = [
        complex_summary[
            "top1_accuracy"
        ],
        complex_summary[
            "top5_accuracy"
        ],
        complex_summary[
            "top10_accuracy"
        ],
        complex_summary[
            "mrr"
        ],
    ]

    labels = [
        "Top-1",
        "Top-5",
        "Top-10",
        "MRR",
    ]

    axes[2].bar(
        labels,
        bars,
        color=[
            "#4C78A8",
            "#59A14F",
            "#F28E2B",
            "#E15759",
        ],
    )

    axes[2].set_ylim(
        0,
        min(
            1.0,
            max(
                bars
            )
            * 1.2
            + 0.02,
        ),
    )

    axes[2].set_ylabel(
        "Score"
    )

    axes[2].set_title(
        "Held-out complex prediction"
    )

    for i, value in enumerate(
        bars
    ):
        axes[2].text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    figure_path = (
        FIG_DIR
        / "06_corum_benchmark.png"
    )

    plt.savefig(
        figure_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Final console summary
    # ========================================================

    print(
        "\nSaved outputs:"
    )

    outputs = [
        TABLE_DIR
        / "corum_gene_metadata.csv",

        TABLE_DIR
        / "corum_complexes_retained.csv",

        TABLE_DIR
        / "corum_neighbor_metrics.csv",

        TABLE_DIR
        / "corum_query_metrics.csv",

        TABLE_DIR
        / "corum_complex_prediction.csv",

        TABLE_DIR
        / "corum_complex_prediction_summary.csv",

        TABLE_DIR
        / "corum_pairwise_metrics.csv",

        TABLE_DIR
        / "corum_random_baseline.csv",

        TABLE_DIR
        / "corum_benchmark_comparison.csv",

        FIG_DIR
        / "06_corum_benchmark.png",
    ]

    for output in outputs:
        print(
            " ",
            output,
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-z",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-p",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--min-complex-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-complex-size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--min-screen-members",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    main(
        min_z=args.min_z,
        max_p=args.max_p,
        min_complex_size=(
            args.min_complex_size
        ),
        max_complex_size=(
            args.max_complex_size
        ),
        min_screen_members=(
            args.min_screen_members
        ),
        n_permutations=(
            args.n_permutations
        ),
    )
