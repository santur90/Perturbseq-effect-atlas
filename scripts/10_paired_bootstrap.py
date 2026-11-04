import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DATA = Path("data/processed/effect_matrix_clean.h5ad")
SUMMARY = Path("results/tables/perturbation_summary_adjusted.csv")
TABLE_DIR = Path("results/tables")

QUERY_METRICS = ["top1_hit", "top20_hit", "heldout_top1",
                 "heldout_top10", "heldout_mrr"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_table(bench, genes, emb, gene_to_cx):
    """Per-gene outcomes, one row per CORUM-annotated query gene."""
    sims, idx = bench.find_neighbors(emb, k_max=20)
    _, q = bench.evaluate_neighbor_recovery(
        genes, sims, idx, gene_to_cx, [1, 5, 10, 20])
    pred = bench.evaluate_complex_prediction(genes, sims, idx, gene_to_cx, 20)
    q = q.set_index("query_gene")
    pred = pred.set_index("query_gene")
    if not q.index.equals(pred.index):
        raise ValueError("Query gene order differs between evaluations.")
    return pd.DataFrame({
        "top1_hit": q["hit_at_1"].astype(float),
        "top20_hit": q["hit_at_20"].astype(float),
        "heldout_top1": pred["top1_correct"].astype(float),
        "heldout_top10": pred["top10_correct"].astype(float),
        "heldout_mrr": pred["reciprocal_rank"].astype(float),
    })


def build_pair_labels(annotated_genes, gene_to_cx):
    """Same positive definition as script 06: share >=1 retained complex."""
    cx_ids = sorted({c for g in annotated_genes for c in gene_to_cx[g]})
    col = {c: i for i, c in enumerate(cx_ids)}
    M = np.zeros((len(annotated_genes), len(cx_ids)), dtype=np.float32)
    for i, g in enumerate(annotated_genes):
        for c in gene_to_cx[g]:
            M[i, col[c]] = 1.0
    shared = (M @ M.T) > 0
    i0, i1 = np.triu_indices(len(annotated_genes), k=1)
    return i0, i1, shared[i0, i1].astype(int)


def pair_scores(emb, i0, i1):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    e = emb / np.clip(norm, 1e-12, None)
    return (e @ e.T)[i0, i1]


def paired_bootstrap(query_ref, query_alts, score_ref, score_alts,
                     labels, i0, i1, n_boot, seed):
    """Resample genes with replacement, identical draws for every method.

    Query metrics: weighted mean of per-gene differences.
    Pairwise AP: pairs weighted by w_i * w_j (i<j), so a gene drawn twice
    contributes its pairs twice. Self-pairs of duplicated genes are ignored.
    The alternative is the mean over the supplied runs (seeds).
    """
    n = len(query_ref)
    rng = np.random.default_rng(seed)

    ref = query_ref[QUERY_METRICS].to_numpy()
    alt = np.mean([q[QUERY_METRICS].to_numpy() for q in query_alts], axis=0)
    diff = alt - ref

    point = {m: diff[:, k].mean() for k, m in enumerate(QUERY_METRICS)}
    ap_ref = average_precision_score(labels, score_ref)
    ap_alt = np.mean([average_precision_score(labels, s) for s in score_alts])
    point["pairwise_ap"] = ap_alt - ap_ref

    boots = {m: np.empty(n_boot) for m in point}
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        for k, m in enumerate(QUERY_METRICS):
            boots[m][b] = (w @ diff[:, k]) / n
        pw = w[i0] * w[i1]
        keep = pw > 0
        y, sw = labels[keep], pw[keep]
        a_ref = average_precision_score(y, score_ref[keep], sample_weight=sw)
        a_alt = np.mean([
            average_precision_score(y, s[keep], sample_weight=sw)
            for s in score_alts])
        boots["pairwise_ap"][b] = a_alt - a_ref

    rows = []
    for m in point:
        rows.append(dict(
            metric=m,
            mean_diff=point[m],
            ci_low=np.percentile(boots[m], 2.5),
            ci_high=np.percentile(boots[m], 97.5),
            frac_bootstraps_positive=(boots[m] > 0).mean(),
        ))
    return pd.DataFrame(rows)


def main(min_z, max_p, latent, seeds, n_boot, epochs, lr, weight_decay,
         noise_sd, patience):
    import anndata as ad
    import torch

    bench = load_module("bench06", "scripts/06_corum_benchmark.py")
    ae = load_module("ae09", "scripts/09_autoencoder_benchmark.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adata = ad.read_h5ad(DATA)
    summary = pd.read_csv(SUMMARY, index_col=0).loc[adata.obs_names]
    X = np.asarray(adata.X, dtype=np.float32)
    pca_original = np.asarray(adata.obsm["X_pca"]).copy()

    eligible = np.flatnonzero(
        summary["qc_pass"].astype(bool).to_numpy()
        & ~summary["is_control"].astype(bool).to_numpy())

    def gene_level(embedding_full):
        adata.obsm["X_pca"] = embedding_full.astype(np.float32)
        meta, emb, _ = bench.build_gene_embeddings(adata, summary, min_z, max_p)
        return meta.index.to_numpy(), emb

    genes, emb_pca = gene_level(pca_original)
    complexes = bench.parse_corum(bench.CORUM, genes)
    gene_to_cx, _, _ = bench.build_annotations(genes, complexes)
    annotated = np.array([i for i, g in enumerate(genes) if g in gene_to_cx])
    annotated_genes = genes[annotated]
    i0, i1, labels = build_pair_labels(annotated_genes, gene_to_cx)
    print(f"Annotated genes: {len(annotated)} | pairs: {len(labels)} | "
          f"positives: {int(labels.sum())}")

    q_ref = query_table(bench, genes, emb_pca, gene_to_cx)
    s_ref = pair_scores(emb_pca[annotated], i0, i1)

    q_alts, s_alts, per_seed = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(eligible)
        n_val = max(200, int(0.1 * len(perm)))
        model, val_loss, n_ep = ae.train_autoencoder(
            X, perm[n_val:], perm[:n_val], latent, seed, epochs, lr,
            weight_decay, noise_sd, patience, device)
        z = np.full((X.shape[0], latent), np.nan, dtype=np.float32)
        z[eligible] = ae.encode(model, X[eligible], device)
        g, emb_ae = gene_level(z)
        if not np.array_equal(g, genes):
            raise ValueError("Gene sets differ between PCA and autoencoder.")
        q = query_table(bench, genes, emb_ae, gene_to_cx)
        s = pair_scores(emb_ae[annotated], i0, i1)
        q_alts.append(q)
        s_alts.append(s)
        per_seed.append(dict(
            seed=seed, val_mse=val_loss, epochs=n_ep,
            top1_diff=q["top1_hit"].mean() - q_ref["top1_hit"].mean(),
            ap_diff=(average_precision_score(labels, s)
                     - average_precision_score(labels, s_ref))))
        print(f"seed {seed}: top1 diff {per_seed[-1]['top1_diff']:+.4f}, "
              f"AP diff {per_seed[-1]['ap_diff']:+.4f}")

    pd.DataFrame(per_seed).to_csv(
        TABLE_DIR / "autoencoder_minus_pca_per_seed.csv", index=False)

    result = paired_bootstrap(q_ref, q_alts, s_ref, s_alts, labels, i0, i1,
                              n_boot, seed=42)
    result.to_csv(TABLE_DIR / "autoencoder_minus_pca_bootstrap.csv", index=False)
    print(f"\nAutoencoder (mean of {seeds} seeds) minus PCA, "
          f"95% gene-level bootstrap CI ({n_boot} draws):")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-z", type=float, default=5.0)
    p.add_argument("--max-p", type=float, default=0.001)
    p.add_argument("--latent", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--noise-sd", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    a = p.parse_args()
    main(a.min_z, a.max_p, a.latent, a.seeds, a.n_boot, a.epochs, a.lr,
         a.weight_decay, a.noise_sd, a.patience)
