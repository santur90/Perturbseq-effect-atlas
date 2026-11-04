import argparse
import importlib.util
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

DATA = Path("data/processed/effect_matrix_clean.h5ad")
SUMMARY = Path("results/tables/perturbation_summary_adjusted.csv")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmark_module():
    """Reuse the exact CORUM benchmark functions from script 06."""
    spec = importlib.util.spec_from_file_location(
        "bench06", "scripts/06_corum_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoEncoder(nn.Module):
    def __init__(self, n_in, latent=50, hidden=512, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(X, train_idx, val_idx, latent, seed, epochs, lr,
                      weight_decay, noise_sd, patience, device):
    """Unsupervised training on expression vectors only. No CORUM labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AutoEncoder(X.shape[1], latent=latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X[train_idx]).to(device)
    Xva = torch.from_numpy(X[val_idx]).to(device)

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for start in range(0, len(perm), 256):
            batch = Xtr[perm[start:start + 256]]
            noisy = batch + noise_sd * torch.randn_like(batch)
            recon, _ = model(noisy)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva)[0], Xva).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, epoch + 1


def encode(model, X, device):
    with torch.no_grad():
        return model.encoder(torch.from_numpy(X).to(device)).cpu().numpy()


def evaluate_embedding(bench, adata, summary, embedding_full, label, min_z, max_p):
    """Run the same gene-level CORUM evaluation used in script 06."""
    adata.obsm["X_pca"] = embedding_full.astype(np.float32)
    gene_meta, emb, _ = bench.build_gene_embeddings(adata, summary, min_z, max_p)
    genes = gene_meta.index.to_numpy()

    complexes = bench.parse_corum(bench.CORUM, genes)
    gene_to_cx, _, _ = bench.build_annotations(genes, complexes)

    k_values = [1, 5, 10, 20]
    sims, idx = bench.find_neighbors(emb, k_max=max(k_values))
    nb, _ = bench.evaluate_neighbor_recovery(genes, sims, idx, gene_to_cx, k_values)
    pred = bench.evaluate_complex_prediction(genes, sims, idx, gene_to_cx, 20)
    pair = bench.evaluate_pairwise(genes, emb, gene_to_cx)

    hit = nb.set_index("k")["hit_rate_at_k"]
    return genes, dict(
        representation=label,
        n_genes=len(genes),
        top1_hit=hit[1], top5_hit=hit[5], top20_hit=hit[20],
        heldout_top1=pred["top1_correct"].mean(),
        heldout_top10=pred["top10_correct"].mean(),
        heldout_mrr=pred["reciprocal_rank"].mean(),
        pairwise_ap=pair["average_precision"],
        pairwise_auroc=pair["auroc"],
    )


def main(min_z, max_p, latent, seeds, epochs, lr, weight_decay, noise_sd,
         patience, skip_raw):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    bench = load_benchmark_module()
    adata = ad.read_h5ad(DATA)
    summary = pd.read_csv(SUMMARY, index_col=0).loc[adata.obs_names]

    X = np.asarray(adata.X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Effect matrix contains non-finite values.")
    pca_original = np.asarray(adata.obsm["X_pca"]).copy()

    # Unsupervised training set: QC-pass targeting perturbations.
    eligible = np.flatnonzero(
        summary["qc_pass"].astype(bool).to_numpy()
        & ~summary["is_control"].astype(bool).to_numpy()
    )
    print("Perturbations used to train the autoencoder:", len(eligible))

    rows, reference_genes = [], None

    genes, row = evaluate_embedding(
        bench, adata, summary, pca_original,
        f"PCA-{pca_original.shape[1]} (script 02)", min_z, max_p)
    reference_genes = genes
    row.update(seed=-1)
    rows.append(row)
    print("PCA done")

    if not skip_raw:
        raw = np.full_like(X, np.nan)
        raw[eligible] = X[eligible]
        genes, row = evaluate_embedding(
            bench, adata, summary, raw, "Raw effect vector (cosine)", min_z, max_p)
        assert np.array_equal(genes, reference_genes), "gene sets differ (raw)"
        row.update(seed=-1)
        rows.append(row)
        print("Raw done")

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(eligible)
        n_val = max(200, int(0.1 * len(perm)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        model, val_loss, n_epochs = train_autoencoder(
            X, train_idx, val_idx, latent, seed, epochs, lr,
            weight_decay, noise_sd, patience, device)

        z = np.full((X.shape[0], latent), np.nan, dtype=np.float32)
        z[eligible] = encode(model, X[eligible], device)

        genes, row = evaluate_embedding(
            bench, adata, summary, z, f"Autoencoder-{latent}", min_z, max_p)
        assert np.array_equal(genes, reference_genes), "gene sets differ (AE)"
        row.update(seed=seed, val_mse=val_loss, epochs_run=n_epochs)
        rows.append(row)
        print(f"AE seed {seed}: val MSE={val_loss:.4f}, epochs={n_epochs}, "
              f"top1={row['top1_hit']:.3f}, AP={row['pairwise_ap']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "autoencoder_vs_pca_per_run.csv", index=False)

    metrics = ["top1_hit", "top5_hit", "top20_hit", "heldout_top1",
               "heldout_top10", "heldout_mrr", "pairwise_ap", "pairwise_auroc"]
    summ = results.groupby("representation")[metrics].agg(["mean", "std"])
    summ.to_csv(TABLE_DIR / "autoencoder_vs_pca_summary.csv")
    print("\nSummary (mean, std across seeds; PCA and raw are single runs):")
    print(summ.round(4).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(
        axes, ["top1_hit", "heldout_top10", "pairwise_ap"],
        ["Top-1 same-complex hit", "Held-out Top-10", "Pairwise AP"],
    ):
        for i, rep in enumerate(results["representation"].unique()):
            vals = results.loc[results["representation"] == rep, metric]
            ax.scatter(np.full(len(vals), i), vals, alpha=0.7)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="k")
        ax.set_xticks(range(results["representation"].nunique()))
        ax.set_xticklabels(results["representation"].unique(),
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_autoencoder_vs_pca.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("\nSaved tables and results/figures/09_autoencoder_vs_pca.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-z", type=float, default=5.0)
    p.add_argument("--max-p", type=float, default=0.001)
    p.add_argument("--latent", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--noise-sd", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--skip-raw", action="store_true")
    a = p.parse_args()
    main(a.min_z, a.max_p, a.latent, a.seeds, a.epochs, a.lr,
         a.weight_decay, a.noise_sd, a.patience, a.skip_raw)
