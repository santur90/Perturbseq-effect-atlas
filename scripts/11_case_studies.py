"""Case studies: do genes reported in Replogle et al. 2022 sit near the expected
functional group in this project's perturbation-effect embedding?

Pre-specified pairings (written down before looking at any result):
  CCDC86, ZNF236, SPATA5L1 -> ribosome biogenesis
  C7orf26                  -> transcription (Integrator)
  TMEM242                  -> mitochondrial respiration

This is a consistency check on the SAME dataset the paper analysed, using
reference sets defined here by gene-name prefix and CORUM complex-name keywords.
It is not an independent replication and it does not establish gene function.
"""
import argparse
import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

# gene label -> (names to look for, expected reference sets)
CASES = {
    "CCDC86": (["CCDC86"], ["ribosome_biogenesis"]),
    "ZNF236": (["ZNF236"], ["ribosome_biogenesis"]),
    "SPATA5L1": (["SPATA5L1"], ["ribosome_biogenesis"]),
    "C7orf26": (["C7ORF26", "INTS15"], ["integrator"]),
    "TMEM242": (["TMEM242"], ["respiratory_chain"]),
}

# Reference sets from gene-name prefixes (heuristic; members must be in the embedding).
PREFIX_SETS = {
    "integrator": r"^INTS\d+",
    "respiratory_chain": r"^(NDUF|UQCR|COX\d|SDH[ABCD]$|ATP5)",
    "cytosolic_ribosome": r"^RP[LS]\d",
    "proteasome": r"^PSM[ABCD]\d",  # negative control for every case gene
}

# Reference sets from CORUM complex names (the matched names are printed and saved).
CORUM_KEYWORD_SETS = {
    "ribosome_biogenesis": r"processome|pre-?ribosom|ribosome biogenesis|pre-?[46]0S",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRUCTURAL_RIBOSOMAL = re.compile(r"^(RP[LS]\d|MRP[LS]\d)")


def drop_structural_ribosomal(members):
    """Remove cytosolic and mitochondrial ribosomal-protein genes (name-prefix heuristic).

    GO 'ribosome biogenesis' includes some structural ribosomal proteins. Those genes
    share a very strong common response with every other ribosome-related knockdown, so
    the primary biogenesis test excludes them; the unfiltered set is kept as a sensitivity.
    """
    return {g for g in members if not STRUCTURAL_RIBOSOMAL.match(g)}


def count_primary_tests(matrix):
    """Number of pre-specified tests that could actually be run (set large enough)."""
    return int((matrix["expected"].astype(bool) & matrix["p_perm"].notna()).sum())


def load_gmt(path):
    sets = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            sets[parts[0]] = {g.strip().upper() for g in parts[2:] if g.strip()}
    return sets


def build_reference_sets(genes, complexes, extra_sets=None):
    """Return ({set name: member genes present in `genes`}, matched CORUM names)."""
    present = set(genes)
    sets, matched = {}, []
    for name, pattern in PREFIX_SETS.items():
        regex = re.compile(pattern)
        sets[name] = {g for g in present if regex.match(g)}
    for name, pattern in CORUM_KEYWORD_SETS.items():
        regex = re.compile(pattern, re.IGNORECASE)
        members = set()
        for rec in complexes:
            if regex.search(str(rec["complex_name"])):
                members |= set(rec["screen_members"]) & present
                matched.append(dict(
                    reference_set=name, complex_id=rec["complex_id"],
                    complex_name=rec["complex_name"],
                    n_screen_members=len(set(rec["screen_members"]) & present)))
        sets[name] = members
    for name, members in (extra_sets or {}).items():
        sets[name] = set(members) & present
    return sets, pd.DataFrame(
        matched, columns=["reference_set", "complex_id", "complex_name", "n_screen_members"])


def cosine_matrix(emb):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.clip(norm, 1e-12, None)
    return unit @ unit.T


def make_effect_bins(effect, n_bins=10):
    """Quantile bin id (0..n_bins-1) for every gene, from its gene-level effect strength."""
    effect = np.asarray(effect, dtype=float)
    edges = np.unique(np.quantile(effect, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, effect, side="right")


def matched_null_draws(candidates, bins, member_bins, n_null, rng):
    """Random gene sets with the same effect-strength composition as the reference set.

    For every reference-set member in bin b, one candidate from bin b is drawn
    (without replacement within a draw). Returns an (n_null, k) array of gene indices.
    """
    cand_bins = bins[candidates]
    columns = []
    for b, count in zip(*np.unique(member_bins, return_counts=True)):
        pool = candidates[cand_bins == b]
        take = min(int(count), len(pool))
        if take == 0:
            continue
        order = rng.random((n_null, len(pool))).argsort(axis=1)[:, :take]
        columns.append(pool[order])
    return np.concatenate(columns, axis=1) if columns else np.empty((n_null, 0), dtype=int)


def set_test(sim_row, self_idx, member_idx, n_null, rng, top_n, bins=None):
    """Is the case gene more similar to this reference set than to random gene sets?

    The case gene is removed from both the reference set and the candidate pool.
    Null: random gene sets of the same size drawn from all other genes, so the
    comparison already accounts for the case gene's overall level of similarity.

    If `bins` (effect-strength bin per gene) is given, a second, matched null is also
    computed in which every random set has the same effect-strength composition as the
    reference set. This removes the advantage a strongly responding reference set has
    over randomly drawn, mostly weakly responding genes.
    """
    candidates = np.delete(np.arange(len(sim_row)), self_idx)
    members = np.array([m for m in member_idx if m != self_idx], dtype=int)
    k = len(members)
    result = dict(n_set_members=k, mean_sim_obs=np.nan, null_mean=np.nan, null_sd=np.nan,
                  z=np.nan, p_perm=np.nan, top_overlap=np.nan, expected_overlap=np.nan,
                  p_hyper=np.nan, matched_null_mean=np.nan, z_matched=np.nan,
                  p_matched=np.nan, expected_overlap_matched=np.nan,
                  p_overlap_matched=np.nan, n_matched_draw_size=np.nan)
    if k < 1:
        return result
    cand_sims = sim_row[candidates]
    obs = float(sim_row[members].mean())
    null = np.empty(n_null)
    for b in range(n_null):
        null[b] = cand_sims[rng.choice(len(candidates), size=k, replace=False)].mean()
    order = candidates[np.argsort(-cand_sims, kind="stable")][:top_n]
    overlap = int(np.isin(order, members).sum())
    result.update(
        mean_sim_obs=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
        z=(obs - null.mean()) / null.std() if null.std() > 0 else np.nan,
        p_perm=(1 + int((null >= obs).sum())) / (n_null + 1),
        top_overlap=overlap, expected_overlap=top_n * k / len(candidates),
        p_hyper=float(hypergeom.sf(overlap - 1, len(candidates), k, top_n)))

    if bins is not None:
        draws = matched_null_draws(candidates, bins, bins[members], n_null, rng)
        if draws.shape[1] == k:                      # every member found a same-bin pool
            m_null = sim_row[draws].mean(axis=1)
            in_top = np.zeros(len(sim_row), dtype=bool)
            in_top[order] = True
            m_overlap = in_top[draws].sum(axis=1)
            result.update(
                matched_null_mean=float(m_null.mean()),
                z_matched=(obs - m_null.mean()) / m_null.std() if m_null.std() > 0 else np.nan,
                p_matched=(1 + int((m_null >= obs).sum())) / (n_null + 1),
                expected_overlap_matched=float(m_overlap.mean()),
                p_overlap_matched=(1 + int((m_overlap >= overlap).sum())) / (n_null + 1),
                n_matched_draw_size=draws.shape[1])
    return result


def run_case_studies(genes, emb, status, ref_sets, gene_to_cx, cases=None,
                     n_null=10000, top_n=20, show_neighbors=15, min_set_size=5, seed=0,
                     effect=None, n_bins=10):
    """Returns (case status table, gene x reference-set test matrix, neighbor table)."""
    cases = cases or CASES
    genes = np.asarray(genes)
    index = {g: i for i, g in enumerate(genes)}
    sim = cosine_matrix(emb)
    rng = np.random.default_rng(seed)
    bins = make_effect_bins(effect, n_bins) if effect is not None else None
    set_idx = {name: np.array(sorted(index[g] for g in members if g in index), dtype=int)
               for name, members in ref_sets.items()}

    status_rows, matrix_rows, neighbor_rows = [], [], []
    for label, (aliases, expected_sets) in cases.items():
        hit = next((a for a in aliases if a in index), None)
        row = status.get(label, {}).copy() if isinstance(status, dict) else {}
        row.update(case_gene=label, name_found_in_embedding=hit,
                   in_embedding=hit is not None,
                   n_corum_complexes=len(gene_to_cx.get(hit, set())) if hit else np.nan)
        status_rows.append(row)
        if hit is None:
            continue
        i = index[hit]
        for name, members in set_idx.items():
            n_members = int(np.sum(members != i))
            if n_members < min_set_size:
                res = set_test(sim[i], i, [], n_null, rng, top_n, bins)
                res["n_set_members"] = n_members
            else:
                res = set_test(sim[i], i, members, n_null, rng, top_n, bins)
            matrix_rows.append(dict(case_gene=label, reference_set=name,
                                    expected=name in expected_sets, **res))
        order = [j for j in np.argsort(-sim[i], kind="stable") if j != i][:show_neighbors]
        for rank, j in enumerate(order, start=1):
            neighbor_rows.append(dict(
                case_gene=label, rank=rank, neighbor=genes[j], cosine=float(sim[i, j]),
                in_sets=";".join(n for n, m in set_idx.items() if j in set(m.tolist())),
                n_corum_complexes=len(gene_to_cx.get(genes[j], set()))))
    return pd.DataFrame(status_rows), pd.DataFrame(matrix_rows), pd.DataFrame(neighbor_rows)


SET_LABELS = {
    "ribosome_biogenesis_gmt_noRP": "GO ribosome biogenesis\n(no ribosomal proteins)",
    "integrator": "Integrator\n(INTS genes)",
    "respiratory_chain": "Respiratory chain\n(NDUF/UQCR/COX/SDH/ATP5)",
    "cytosolic_ribosome": "Cytosolic ribosomal\nproteins",
    "ribosome_biogenesis_gmt": "GO ribosome biogenesis\n(full set)",
    "proteasome": "Proteasome\n(negative control)",
}
SET_ORDER = list(SET_LABELS)          # pre-specified sets first, then the others
CASE_ORDER = list(CASES)


def overlap_log2_enrichment(observed, expected, pseudo=0.5):
    """log2((observed + pseudo) / (expected + pseudo)); the pseudocount keeps 0 finite."""
    return np.log2((np.asarray(observed, dtype=float) + pseudo)
                   / (np.asarray(expected, dtype=float) + pseudo))


def heatmap_tables(matrix, pseudo=0.5):
    """Pivoted tables used by the figure. Cells with no matched-null result stay NaN.

    The figure is driven by nearest-neighbor overlap against the effect-strength-matched
    expectation, not by mean cosine similarity, because mean similarity was not specific
    (strongly responding gene sets are similar to each other for non-specific reasons).
    """
    m = matrix.copy()
    m = m[m["reference_set"].isin(SET_ORDER) & m["p_overlap_matched"].notna()]
    m["log2_enrichment"] = overlap_log2_enrichment(
        m["top_overlap"], m["expected_overlap_matched"], pseudo)

    def pivot(col):
        t = m.pivot(index="case_gene", columns="reference_set", values=col)
        rows = [c for c in CASE_ORDER if c in t.index] + [c for c in t.index if c not in CASE_ORDER]
        cols = [c for c in SET_ORDER if c in t.columns]
        return t.loc[rows, cols]

    return {k: pivot(k) for k in ["log2_enrichment", "top_overlap", "expected_overlap_matched",
                                  "p_overlap_matched", "expected"]}


def make_heatmap(matrix, path, alpha):
    """Heatmap of top-20 neighbor overlap versus the effect-strength-matched expectation.

    Only enrichment is coloured. Observed counts below expectation are shown as white:
    with expectations of 0.2 to 4 neighbors, a count of 0 is not evidence of depletion.
    """
    t = heatmap_tables(matrix)
    enr, obs, exp_ov = t["log2_enrichment"], t["top_overlap"], t["expected_overlap_matched"]
    pv, boxed = t["p_overlap_matched"], t["expected"]
    shown = enr.clip(lower=0.0)
    vmax = max(float(np.nanmax(shown.to_numpy(dtype=float))) if shown.size else 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(1.75 * enr.shape[1] + 2.5, 0.85 * enr.shape[0] + 3.2))
    im = ax.imshow(shown.to_numpy(dtype=float), cmap="Reds", vmin=0, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.grid(False)
    for r in range(enr.shape[0]):
        for c in range(enr.shape[1]):
            if np.isnan(enr.iat[r, c]):
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=8, color="grey")
                continue
            star = " *" if pv.iat[r, c] < alpha else ""
            dark = shown.iat[r, c] > 0.55 * vmax
            ax.text(c, r, f"{int(obs.iat[r, c])} / {exp_ov.iat[r, c]:.1f}{star}",
                    ha="center", va="center", fontsize=10,
                    color="white" if dark else "black", fontweight="bold" if star else "normal")
            if bool(boxed.iat[r, c]):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=2.8))
    ax.set_xticks(range(enr.shape[1]))
    ax.set_xticklabels([SET_LABELS.get(c, c) for c in enr.columns], fontsize=8.5)
    ax.set_yticks(range(enr.shape[0]))
    ax.set_yticklabels(enr.index, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("log2 enrichment of top-20 neighbors\n(observed vs. matched expectation;"
                   " values below 0 shown as 0)")
    fig.suptitle("Nearest neighbors of five case genes in the perturbation-effect embedding",
                 fontsize=11, y=0.985)
    fig.text(
        0.5, 0.005,
        "Cell text: neighbors in set among the top 20 / expected under a null matched on effect "
        "strength.  Boxed = pre-specified pairing.  * = matched permutation P for overlap < "
        f"{alpha:.3g}\n(descriptive for non-boxed cells).  Colour = log2((observed + 0.5) / "
        "(expected + 0.5)).  P-values are bounded below by the number of permutations.",
        ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close()


def main(min_z, max_p, n_null, top_n, seed, gmt, gmt_term):
    import anndata as ad

    bench = load_module("bench06", "scripts/06_corum_benchmark.py")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(bench.DATA)
    summary = pd.read_csv(bench.SUMMARY, index_col=0).loc[adata.obs_names]
    meta, emb, _ = bench.build_gene_embeddings(adata, summary, min_z, max_p)
    genes = meta.index.to_numpy()

    complexes = bench.parse_corum(bench.CORUM, genes)
    gene_to_cx, _, _ = bench.build_annotations(genes, complexes)

    extra = {}
    cases = {k: (a, list(e)) for k, (a, e) in CASES.items()}
    if gmt and gmt_term:
        gmt_sets = load_gmt(gmt)
        if gmt_term not in gmt_sets:
            raise ValueError(f"'{gmt_term}' not found in {gmt}")
        full = gmt_sets[gmt_term]
        extra["ribosome_biogenesis_gmt"] = full                      # sensitivity only
        extra["ribosome_biogenesis_gmt_noRP"] = drop_structural_ribosomal(full)  # primary
        for k in ("CCDC86", "ZNF236", "SPATA5L1"):
            cases[k] = (cases[k][0], ["ribosome_biogenesis_gmt_noRP"])

    ref_sets, matched = build_reference_sets(genes, complexes, extra)
    print("\nReference set sizes (members present in the embedding):")
    for name, members in ref_sets.items():
        print(f"  {name}: {len(members)}")
    print("\nCORUM complexes matched by name for ribosome_biogenesis:")
    print(matched.to_string(index=False) if len(matched) else "  none")

    # Status of each case gene in the full screen, whether or not it is "responsive".
    summary = summary.assign(_gene=summary["target_gene"].astype(str).str.upper())
    status = {}
    for label, (aliases, _) in cases.items():
        rows = summary[summary["_gene"].isin(aliases)]
        status[label] = dict(
            n_rows_in_screen=len(rows),
            num_cells_filtered=rows["num_cells_filtered"].median() if len(rows) else np.nan,
            knockdown_efficiency=rows["knockdown_efficiency"].median() if len(rows) else np.nan,
            adjusted_effect_z=rows["adjusted_effect_z"].max() if len(rows) else np.nan,
            min_energy_p=rows["energy_test_p_value"].min() if len(rows) else np.nan)

    status_df, matrix, neighbors = run_case_studies(
        genes, emb, status, ref_sets, gene_to_cx, cases=cases,
        n_null=n_null, top_n=top_n, seed=seed,
        effect=meta["median_adjusted_effect_z"].to_numpy())
    print("\nCase gene status:")
    print(status_df.round(3).to_string(index=False))
    if matrix.empty:
        print("\nNone of the case genes are in the responsive embedding; no tests were run.")
        status_df.to_csv(TABLE_DIR / "case_study_status.csv", index=False)
        return

    n_primary = count_primary_tests(matrix)
    alpha = 0.05 / max(n_primary, 1)
    matrix["bonferroni_alpha"] = alpha
    matrix.to_csv(TABLE_DIR / "case_study_matrix.csv", index=False)
    neighbors.to_csv(TABLE_DIR / "case_study_neighbors.csv", index=False)
    status_df.to_csv(TABLE_DIR / "case_study_status.csv", index=False)
    matched.to_csv(TABLE_DIR / "case_study_corum_matches.csv", index=False)

    cols = ["case_gene", "reference_set", "n_set_members", "mean_sim_obs", "null_mean",
            "z", "p_perm", "top_overlap", "expected_overlap", "p_hyper"]
    print(f"\nPRIMARY (pre-specified) tests, Bonferroni alpha = {alpha:.4f} "
          f"for {n_primary} tests:")
    print(matrix[matrix["expected"]][cols].round(4).to_string(index=False))
    untested = matrix[matrix["expected"] & matrix["p_perm"].isna()]
    if len(untested):
        print(f"\n{len(untested)} pre-specified test(s) could not be run (reference set too small).")
    print("\nSPECIFICITY (all other gene x set pairs, descriptive only):")
    print(matrix[~matrix["expected"]][cols].round(4).to_string(index=False))
    mcols = ["case_gene", "reference_set", "n_set_members", "mean_sim_obs", "matched_null_mean",
             "z_matched", "p_matched", "top_overlap", "expected_overlap_matched",
             "p_overlap_matched"]
    print("\nEFFECT-STRENGTH-MATCHED null (added after seeing the unmatched results; "
          "sensitivity analysis). Random sets have the reference set's effect-strength mix:")
    shown = matrix.dropna(subset=["p_matched"])
    print(shown[shown["expected"]][mcols].round(4).to_string(index=False))
    print("\nMatched null, all other gene x set pairs:")
    print(shown[~shown["expected"]][mcols].round(4).to_string(index=False))
    print("\nTop neighbors:")
    for label, grp in neighbors.groupby("case_gene", sort=False):
        print(f"  {label}: " + ", ".join(
            f"{r.neighbor}{'[' + r.in_sets + ']' if r.in_sets else ''}"
            for r in grp.itertuples()))

    make_heatmap(matrix, FIG_DIR / "11_case_studies.png", alpha)
    print("\nSaved tables to results/tables/case_study_*.csv and "
          "results/figures/11_case_studies.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-z", type=float, default=5.0)
    p.add_argument("--max-p", type=float, default=0.001)
    p.add_argument("--n-null", type=int, default=10000)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gmt", default=None, help="optional GMT file with a ribosome-biogenesis set")
    p.add_argument("--gmt-term", default=None, help="set name inside the GMT file")
    a = p.parse_args()
    main(a.min_z, a.max_p, a.n_null, a.top_n, a.seed, a.gmt, a.gmt_term)
