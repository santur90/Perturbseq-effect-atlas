import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

NORM_FILE = Path("data/raw/K562_gwps_normalized_bulk_01.h5ad")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")
PROC_DIR = Path("data/processed")

for d in (FIG_DIR, TABLE_DIR, PROC_DIR):
    d.mkdir(parents=True, exist_ok=True)

ID_PATTERN = re.compile(r"^(\d+)_(.+)_(P1P2|P1|P2)_(ENSG\d+)$")

FAMILIES = {
    "cytosolic_ribosome (RPL/RPS)": r"^RP[LS]\d",
    "mito_ribosome (MRPL/MRPS)": r"^MRP[LS]\d",
    "proteasome (PSMA/B/C/D)": r"^PSM[ABCD]\d",
    "complex_I (NDUF)": r"^NDUF",
    "RNA_Pol_II (POLR2)": r"^POLR2",
    "SF3B/SF3A splicing": r"^SF3[AB]\d",
    "eIF translation": r"^EIF\d",
    "exosome (EXOSC)": r"^EXOSC\d",
}


def parse_ids(names):
    rows = []
    for n in names:
        m = ID_PATTERN.match(n)
        is_ctrl = "non-targeting" in n.lower()
        if m:
            rows.append(
                dict(
                    perturbation_id=n,
                    screen_index=m.group(1),
                    target_gene=m.group(2),
                    promoter=m.group(3),
                    target_ensembl=m.group(4),
                    is_control=is_ctrl,
                )
            )
        else:
            rows.append(
                dict(
                    perturbation_id=n,
                    screen_index=None,
                    target_gene="non-targeting" if is_ctrl else None,
                    promoter=None,
                    target_ensembl=None,
                    is_control=is_ctrl,
                )
            )
    return pd.DataFrame(rows).set_index("perturbation_id")


def family_sanity_check(embedding, obs, k=10):
    """Do nearest neighbours of family members belong to the same family?
    Prefix-based sanity check only; replace with CORUM later."""
    genes = obs["target_gene"].fillna("").values
    nn = NearestNeighbors(n_neighbors=k + 30, metric="cosine")
    nn.fit(embedding)
    _, idx = nn.kneighbors(embedding)

    rows = []
    for fam, pattern in FAMILIES.items():
        regex = re.compile(pattern)
        is_member = np.array([bool(regex.match(g)) for g in genes])
        n_members = int(is_member.sum())
        if n_members < 5:
            continue

        base_rate = n_members / len(genes)
        fractions = []
        for i in np.where(is_member)[0]:
            # drop self and other rows of the same target gene (P1/P2 rows)
            neigh = [j for j in idx[i] if genes[j] != genes[i]][:k]
            fractions.append(is_member[neigh].mean())

        mean_frac = float(np.mean(fractions))
        rows.append(
            dict(
                family=fam,
                n_perturbations=n_members,
                mean_fraction_neighbors_in_family=mean_frac,
                random_expectation=base_rate,
                fold_enrichment=mean_frac / base_rate,
            )
        )
    return pd.DataFrame(rows)


def main(min_cells, n_pcs, k):
    print("Loading normalized matrix...")
    adata = sc.read_h5ad(NORM_FILE)
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)
    print("Matrix:", X.shape)

    # ---- 1. remove genes with any non-finite value ----------------------
    finite_gene = np.isfinite(X).all(axis=0)
    dropped = adata.var.loc[~finite_gene, ["gene_name"]].copy()
    dropped["n_nonfinite_perturbations"] = (~np.isfinite(X[:, ~finite_gene])).sum(axis=0)
    dropped.to_csv(TABLE_DIR / "dropped_genes_nonfinite.csv")
    print(f"Dropped {int((~finite_gene).sum())} genes with non-finite values")

    Xc = X[:, finite_gene]
    var_clean = adata.var.loc[finite_gene].copy()

    # ---- 2. perturbation metadata ---------------------------------------
    obs = adata.obs.join(parse_ids(adata.obs_names))
    obs["knockdown_assessable"] = obs["fold_expr"].notna()
    obs["knockdown_efficiency"] = 1.0 - obs["fold_expr"]  # = -pct_expr
    obs["qc_pass"] = obs["num_cells_filtered"].fillna(0) >= min_cells

    n_ctrl = int(obs["is_control"].sum())
    print(f"\nNon-targeting control rows detected by name: {n_ctrl}")
    if n_ctrl:
        print(obs.index[obs["is_control"]][:5].tolist())
    print("core_control value counts:")
    print(obs["core_control"].value_counts(dropna=False))
    print(
        "crosstab is_control x core_control:\n",
        pd.crosstab(obs["is_control"], obs["core_control"]),
    )

    # ---- 3. effect magnitude + empirical control background -------------
    obs["effect_l2"] = np.linalg.norm(Xc, axis=1)
    obs["effect_rms"] = obs["effect_l2"] / np.sqrt(Xc.shape[1])

    if n_ctrl >= 20:
        ctrl_vals = obs.loc[obs["is_control"], "effect_l2"]
        q95 = ctrl_vals.quantile(0.95)
        obs["above_control_q95"] = obs["effect_l2"] > q95
        print(f"\nControl effect_l2: median={ctrl_vals.median():.2f}, q95={q95:.2f}")
        print(
            "Control rows mean |z|: "
            f"{np.abs(Xc[obs['is_control'].values]).mean():.4f}"
        )
        n_above = int(obs.loc[~obs["is_control"] & obs["qc_pass"], "above_control_q95"].sum())
        n_tot = int((~obs["is_control"] & obs["qc_pass"]).sum())
        print(f"Targeting perturbations (qc_pass) above control q95: {n_above}/{n_tot}")
    else:
        obs["above_control_q95"] = np.nan
        print("\nToo few control rows detected; skipping control background.")

    tgt = obs[~obs["is_control"]].dropna(subset=["num_cells_filtered"])
    rho_cells, _ = spearmanr(tgt["num_cells_filtered"], tgt["effect_l2"])
    kd = tgt.dropna(subset=["knockdown_efficiency"])
    rho_kd, _ = spearmanr(kd["knockdown_efficiency"], kd["effect_l2"])
    print(f"\nSpearman(effect_l2, num_cells_filtered) = {rho_cells:.3f}")
    print(f"Spearman(effect_l2, knockdown_efficiency) = {rho_kd:.3f}")
    print(f"qc_pass (cells >= {min_cells}): {int(obs['qc_pass'].sum())}/{len(obs)}")

    # ---- 4. PCA on qc_pass perturbations --------------------------------
    keep = obs["qc_pass"].values
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0)
    emb = pca.fit_transform(Xc[keep])
    evr = pca.explained_variance_ratio_
    print(f"\nPCA: first 10 PCs explain {evr[:10].sum():.1%}; all {n_pcs}: {evr.sum():.1%}")

    emb_full = np.full((Xc.shape[0], n_pcs), np.nan, dtype=np.float32)
    emb_full[keep] = emb

    # ---- 5. neighbour sanity check --------------------------------------
    obs_keep = obs.loc[keep]
    sanity = family_sanity_check(emb, obs_keep, k=k)
    sanity.to_csv(TABLE_DIR / "neighbor_family_sanity_check.csv", index=False)
    print(f"\nNeighbour sanity check (top-{k} cosine neighbours in PCA space):")
    print(sanity.round(3).to_string(index=False))

    # ---- save -----------------------------------------------------------
    summary_cols = [
        "target_gene", "promoter", "target_ensembl", "is_control",
        "num_cells_filtered", "knockdown_assessable", "knockdown_efficiency",
        "effect_l2", "effect_rms", "above_control_q95", "qc_pass",
        "energy_test_p_value", "mean_leverage_score",
    ]
    obs[summary_cols].to_csv(TABLE_DIR / "perturbation_summary.csv")

    obs_out = obs.copy()
    for c in obs_out.columns:
        if obs_out[c].dtype == object:
            obs_out[c] = obs_out[c].fillna("").astype(str)
    clean = ad.AnnData(X=Xc, obs=obs_out, var=var_clean)
    clean.obsm["X_pca"] = emb_full
    clean.write_h5ad(PROC_DIR / "effect_matrix_clean.h5ad")

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    t = obs[~obs["is_control"]]
    c = obs[obs["is_control"]]
    ax[0].scatter(t["num_cells_filtered"], t["effect_l2"], s=5, alpha=0.25, c="grey", label="targeting")
    if len(c):
        ax[0].scatter(c["num_cells_filtered"], c["effect_l2"], s=10, alpha=0.7, c="crimson", label="non-targeting")
    ax[0].axvline(min_cells, ls="--", c="k", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Filtered cells")
    ax[0].set_ylabel("Effect magnitude (L2 norm)")
    ax[0].set_title("Effect size vs. cell number")
    ax[0].legend()

    ax[1].scatter(
        t["knockdown_efficiency"].clip(-1, 1), t["effect_l2"],
        s=5, alpha=0.25, c="grey",
    )
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Target knockdown (1 - fold_expr; clipped to [-1, 1])")
    ax[1].set_ylabel("Effect magnitude (L2 norm)")
    ax[1].set_title("Effect size vs. knockdown")

    ax[2].plot(np.arange(1, n_pcs + 1), np.cumsum(evr), marker="o", ms=3)
    ax[2].set_xlabel("Principal component")
    ax[2].set_ylabel("Cumulative explained variance")
    ax[2].set_title("PCA of perturbation effects")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_effect_magnitude.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    for p in [
        PROC_DIR / "effect_matrix_clean.h5ad",
        TABLE_DIR / "perturbation_summary.csv",
        TABLE_DIR / "dropped_genes_nonfinite.csv",
        TABLE_DIR / "neighbor_family_sanity_check.csv",
        FIG_DIR / "02_effect_magnitude.png",
    ]:
        print(" ", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    main(args.min_cells, args.n_pcs, args.k)
