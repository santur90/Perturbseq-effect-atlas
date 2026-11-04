import argparse
from collections import defaultdict
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold

DATA = Path("data/processed/effect_matrix_clean.h5ad")
SUMMARY = Path("results/tables/perturbation_summary_adjusted.csv")
CORUM = Path("data/reference/corum_human_core.csv")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 100


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def build_gene_effects(adata, summary, min_z, max_p):
    """One median effect vector per target gene (responsive, QC-pass)."""
    summary = summary.loc[adata.obs_names]
    mask = (
        summary["qc_pass"].astype(bool)
        & ~summary["is_control"].astype(bool)
        & summary["target_gene"].notna()
        & (summary["adjusted_effect_z"] >= min_z)
        & (summary["energy_test_p_value"] <= max_p)
    )
    positions = np.flatnonzero(mask.to_numpy())
    sel = summary.iloc[positions].copy()
    sel["pos"] = positions
    sel["target_gene"] = sel["target_gene"].astype(str).str.upper()

    X = adata.X
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)

    genes, vectors, zs = [], [], []
    for gene, grp in sel.groupby("target_gene", sort=True):
        rows = X[grp["pos"].to_numpy()]
        if not np.isfinite(rows).all():
            continue
        genes.append(gene)
        vectors.append(np.median(rows, axis=0))
        zs.append(float(np.nanmedian(grp["adjusted_effect_z"])))
    return np.array(genes), np.vstack(vectors).astype(np.float32), np.array(zs)


def build_complex_matrix(genes, min_size=3, max_size=50, min_screen=3):
    """Binary gene x complex matrix, same filters as the CORUM benchmark."""
    corum = pd.read_csv(CORUM)
    corum = corum[corum["Organism"].eq("Human")]
    gene_set = set(genes)
    complexes = {}
    for _, row in corum.iterrows():
        members = {
            g.strip().upper()
            for g in str(row["subunits.Gene.name."]).split(";")
            if g.strip() and g.strip().lower() != "nan"
        }
        if not (min_size <= len(members) <= max_size):
            continue
        present = members & gene_set
        if len(present) >= min_screen:
            complexes[str(row["ComplexID"])] = present

    ids = sorted(complexes)
    index = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(genes), len(ids)), dtype=np.float32)
    for j, cid in enumerate(ids):
        for g in complexes[cid]:
            M[index[g], j] = 1.0
    return M, ids


# ------------------------------------------------------------------
# Models: each returns predictions for the test genes
# ------------------------------------------------------------------
def predict_mean(Mtr, Ytr, Mte, **_):
    return np.tile(Ytr.mean(axis=0), (Mte.shape[0], 1)), None


def predict_neighbor(Mtr, Ytr, Mte, **_):
    """Jaccard-weighted mean over training genes sharing a complex."""
    inter = Mte @ Mtr.T
    union = Mte.sum(1, keepdims=True) + Mtr.sum(1)[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    wsum = jac.sum(1, keepdims=True)
    mean = Ytr.mean(axis=0, keepdims=True)
    pred = np.where(wsum > 0, (jac @ Ytr) / np.clip(wsum, 1e-12, None), mean)
    return pred.astype(np.float32), (wsum.ravel() > 0)


def predict_ridge(Mtr, Ytr, Mte, **_):
    model = RidgeCV(alphas=np.logspace(-2, 3, 12))
    model.fit(Mtr, Ytr)
    return model.predict(Mte).astype(np.float32), (Mte.sum(1) > 0)


def predict_ridge_shuffled(Mtr, Ytr, Mte, rng=None, **_):
    shuffled = Mtr[rng.permutation(Mtr.shape[0])]
    model = RidgeCV(alphas=np.logspace(-2, 3, 12))
    model.fit(shuffled, Ytr)
    return model.predict(Mte).astype(np.float32), None


METHODS = {
    "Train mean": predict_mean,
    "Complex-neighbor mean": predict_neighbor,
    "Ridge (CORUM features)": predict_ridge,
    "Ridge (shuffled features)": predict_ridge_shuffled,
}


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def rowwise_corr(A, B):
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def topk_overlap(pred, obs, k=TOP_K):
    out = np.empty(pred.shape[0])
    for i in range(pred.shape[0]):
        p = set(np.argpartition(-np.abs(pred[i]), k - 1)[:k])
        o = set(np.argpartition(-np.abs(obs[i]), k - 1)[:k])
        out[i] = len(p & o) / k
    return out


# ------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------
def split_gene_holdout(n, n_splits, seed, **_):
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, te


def split_complex_disjoint(n, n_splits, M, **_):
    """Hold out groups of genes; drop from training every gene that shares
    any complex with any test gene."""
    sizes = M.sum(0)
    groups = []
    for i in range(n):
        cols = np.flatnonzero(M[i])
        groups.append(int(cols[np.argmin(sizes[cols])]))  # smallest complex
    groups = np.array(groups)
    for _, te in GroupKFold(n_splits).split(np.arange(n), groups=groups):
        test_cols = np.flatnonzero(M[te].sum(0) > 0)
        blocked = M[:, test_cols].sum(1) > 0
        tr = np.flatnonzero(~blocked)
        yield tr, te


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def run_split(name, splitter, genes, Y, Z, M, n_splits, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for fold, (tr, te) in enumerate(
        splitter(len(genes), n_splits=n_splits, seed=seed, M=M)
    ):
        print(f"  [{name}] fold {fold}: train={len(tr)} test={len(te)}")
        Mtr, Ytr, Mte, Yte = M[tr], Y[tr], M[te], Y[te]
        train_mean = Ytr.mean(axis=0, keepdims=True)

        for method, fn in METHODS.items():
            pred, covered = fn(Mtr, Ytr, Mte, rng=rng)
            pear = rowwise_corr(pred, Yte)
            delta = rowwise_corr(pred - train_mean, Yte - train_mean)
            if method == "Train mean":
                delta = np.full(len(te), np.nan)
            top = topk_overlap(pred, Yte)
            for k, gi in enumerate(te):
                rows.append(
                    dict(
                        split=name, fold=fold, method=method, gene=genes[gi],
                        effect_z=Z[gi], pearson=pear[k], delta_pearson=delta[k],
                        top100_overlap=top[k],
                        has_feature_support=(
                            np.nan if covered is None else bool(covered[k])
                        ),
                    )
                )
    return pd.DataFrame(rows)


def paired_vs_mean(df, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = []
    for split, g in df.groupby("split"):
        base = g[g["method"] == "Train mean"].set_index("gene")
        for method in g["method"].unique():
            if method == "Train mean":
                continue
            cur = g[g["method"] == method].set_index("gene")
            for metric in ["pearson", "top100_overlap"]:
                d = (cur[metric] - base[metric]).dropna().to_numpy()
                if len(d) == 0:
                    continue
                boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
                out.append(
                    dict(
                        split=split, method=method, metric=metric,
                        n_genes=len(d), mean_diff_vs_train_mean=d.mean(),
                        ci_low=np.percentile(boots, 2.5),
                        ci_high=np.percentile(boots, 97.5),
                        fraction_improved=(d > 0).mean(),
                    )
                )
    return pd.DataFrame(out)


def make_figure(df, path):
    splits = list(df["split"].unique())
    fig, axes = plt.subplots(2, len(splits), figsize=(6.5 * len(splits), 9), squeeze=False)
    for c, split in enumerate(splits):
        g = df[df["split"] == split]
        for r, (metric, title) in enumerate(
            [("pearson", "Pearson (full effect vector)"),
             ("delta_pearson", "Pearson after removing training mean")]
        ):
            ax = axes[r, c]
            methods = [m for m in METHODS if not (metric == "delta_pearson" and m == "Train mean")]
            data = [g.loc[g["method"] == m, metric].dropna() for m in methods]
            ax.boxplot(data, tick_labels=[f"{m}\n(n={len(d)})" for m, d in zip(methods, data)],
                       showfliers=False)
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_title(f"{split}: {title}", fontsize=11)
            ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main(min_z, max_p, n_splits, seed, n_boot):
    print("Loading data...")
    adata = ad.read_h5ad(DATA)
    summary = pd.read_csv(SUMMARY, index_col=0)
    genes_all, Y_all, Z_all = build_gene_effects(adata, summary, min_z, max_p)
    print("Responsive genes:", len(genes_all), "| effect dim:", Y_all.shape[1])

    M_all, complex_ids = build_complex_matrix(genes_all)
    keep = M_all.sum(1) > 0
    genes, Y, Z, M = genes_all[keep], Y_all[keep], Z_all[keep], M_all[keep]
    print(f"CORUM complexes: {len(complex_ids)} | annotated genes modeled: {len(genes)}")

    frames = []
    for name, splitter in [
        ("A: gene holdout", split_gene_holdout),
        ("B: complex-disjoint holdout", split_complex_disjoint),
    ]:
        print(f"\nSplit {name}")
        frames.append(run_split(name, splitter, genes, Y, Z, M, n_splits, seed))

    df = pd.concat(frames, ignore_index=True)
    df["has_feature_support"] = df["has_feature_support"].astype(float)
    df.to_csv(TABLE_DIR / "heldout_prediction_per_gene.csv", index=False)

    summ = (
        df.groupby(["split", "method"])
        .agg(n_genes=("gene", "nunique"),
             pearson_mean=("pearson", "mean"),
             pearson_median=("pearson", "median"),
             delta_pearson_mean=("delta_pearson", "mean"),
             top100_overlap_mean=("top100_overlap", "mean"))
        .reset_index()
    )
    summ.to_csv(TABLE_DIR / "heldout_prediction_summary.csv", index=False)
    print("\nSummary:")
    print(summ.round(4).to_string(index=False))

    paired = paired_vs_mean(df, n_boot, seed)
    paired.to_csv(TABLE_DIR / "heldout_prediction_vs_mean.csv", index=False)
    print("\nPaired difference vs. train-mean baseline (95% bootstrap CI over genes):")
    print(paired.round(4).to_string(index=False))

    cov = (
        df[df["method"] == "Complex-neighbor mean"]
        .groupby("split")["has_feature_support"].mean()
    )
    print("\nFraction of test genes with a training neighbor (complex-neighbor model):")
    print(cov.round(3).to_string())

    r = df[(df["method"] == "Ridge (CORUM features)") & (df["split"].str.startswith("A"))].copy()
    r["effect_tertile"] = pd.qcut(r["effect_z"], 3, labels=["low", "mid", "high"])
    print("\nSplit A ridge by transcriptional-effect strength:")
    print(r.groupby("effect_tertile", observed=True)[["pearson", "delta_pearson"]]
          .mean().round(4).to_string())

    make_figure(df, FIG_DIR / "08_heldout_prediction.png")
    print("\nSaved tables to results/tables and results/figures/08_heldout_prediction.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-z", type=float, default=5.0)
    p.add_argument("--max-p", type=float, default=0.001)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    main(a.min_z, a.max_p, a.n_splits, a.seed, a.n_boot)
