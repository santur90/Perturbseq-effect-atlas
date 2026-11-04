import json
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from conftest import SCRIPTS


def clustered_embedding(n_clusters=4, per_cluster=6, dim=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 3
    labels = np.repeat(np.arange(n_clusters), per_cluster)
    emb = centers[labels] + rng.normal(scale=noise, size=(len(labels), dim))
    genes = np.array([f"G{i}" for i in range(len(labels))])
    gene_to_cx = {g: {str(labels[i])} for i, g in enumerate(genes)}
    return genes, emb.astype(np.float32), gene_to_cx


def write_corum(path, rows):
    df = pd.DataFrame(rows, columns=["ComplexID", "ComplexName", "Organism",
                                     "subunits.Gene.name."])
    df.to_csv(path, index=False)


def test_safe_median_returns_nan_without_warning(bench):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(bench.safe_median(pd.Series([np.nan, np.nan])))
        assert bench.safe_median(pd.Series([1.0, 3.0, np.nan])) == 2.0


def test_parse_corum_applies_all_filters(bench, tmp_path):
    big = ";".join(f"BIG{i}" for i in range(60))
    path = tmp_path / "corum.csv"
    write_corum(path, [
        (1, "kept", "Human", "aaa;BBB;ccc;ZZZ"),
        (2, "too small", "Human", "AAA;BBB"),
        (3, "too few in screen", "Human", "AAA;QQQ;RRR;SSS"),
        (4, "wrong organism", "Mouse", "AAA;BBB;CCC"),
        (5, "too large", "Human", big + ";AAA;BBB;CCC"),
    ])
    records = bench.parse_corum(path, available_genes=["AAA", "BBB", "CCC"])
    assert [r["complex_id"] for r in records] == ["1"]
    assert records[0]["screen_members"] == {"AAA", "BBB", "CCC"}
    assert records[0]["n_all_members"] == 4


def test_parse_corum_rejects_missing_columns(bench, tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(path, index=False)
    try:
        bench.parse_corum(path, available_genes=["A"])
    except ValueError as err:
        assert "Missing CORUM columns" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_build_annotations_is_consistent_in_both_directions(bench):
    records = [
        dict(complex_id="1", complex_name="c1", screen_members={"A", "B", "C"}),
        dict(complex_id="2", complex_name="c2", screen_members={"A", "B"}),
    ]
    g2c, c2g, names = bench.build_annotations(["A", "B", "C", "D"], records)
    assert set(c2g) == {"1"}  # complex 2 has <3 members and is dropped
    assert g2c["A"] == {"1"} and "D" not in g2c
    assert names["1"] == "c1"


def test_find_neighbors_excludes_self(bench):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 8))
    sims, idx = bench.find_neighbors(emb, k_max=5)
    assert idx.shape == (30, 5)
    assert all(i not in idx[i] for i in range(30))
    assert (np.diff(sims, axis=1) <= 1e-9).all()  # most similar first


def test_neighbor_recovery_perfect_on_clean_clusters(bench):
    genes, emb, g2c = clustered_embedding()
    sims, idx = bench.find_neighbors(emb, k_max=5)
    summary, query = bench.evaluate_neighbor_recovery(
        genes, sims, idx, g2c, [1, 5])
    assert summary.loc[summary["k"] == 1, "hit_rate_at_k"].iloc[0] == 1.0
    assert summary.loc[summary["k"] == 5, "mean_precision_at_k"].iloc[0] == 1.0
    assert len(query) == len(genes)


def test_neighbor_recovery_collapses_when_labels_are_shuffled(bench):
    genes, emb, g2c = clustered_embedding(n_clusters=6, per_cluster=8)
    sims, idx = bench.find_neighbors(emb, k_max=5)
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(30):
        shuffled = rng.permutation(genes)
        s, _ = bench.evaluate_neighbor_recovery(shuffled, sims, idx, g2c, [1])
        hits.append(s["hit_rate_at_k"].iloc[0])
    assert np.mean(hits) < 0.4  # chance is about 1/6 here


def test_complex_prediction_correct_on_clean_clusters(bench):
    genes, emb, g2c = clustered_embedding()
    sims, idx = bench.find_neighbors(emb, k_max=10)
    pred = bench.evaluate_complex_prediction(genes, sims, idx, g2c, 10)
    assert pred["top1_correct"].mean() == 1.0
    assert (pred["reciprocal_rank"] == 1.0).all()


def test_complex_prediction_breaks_ties_by_complex_id(bench):
    genes = np.array(["Q", "N"])
    sims = np.array([[0.9], [0.9]])
    idx = np.array([[1], [0]])
    # Neighbour N belongs to two complexes -> both get identical votes.
    low = bench.evaluate_complex_prediction(
        genes, sims, idx, {"Q": {"1"}, "N": {"2", "1"}}, 1)
    high = bench.evaluate_complex_prediction(
        genes, sims, idx, {"Q": {"2"}, "N": {"2", "1"}}, 1)
    assert low.loc[low["query_gene"] == "Q", "best_true_complex_rank"].iloc[0] == 1
    assert high.loc[high["query_gene"] == "Q", "best_true_complex_rank"].iloc[0] == 2


def test_complex_prediction_does_not_depend_on_hash_seed():
    """Regression test for tie-breaking that used to follow set order."""
    code = (
        "import importlib.util, json, numpy as np\n"
        f"spec = importlib.util.spec_from_file_location('b', r'{SCRIPTS}/06_corum_benchmark.py')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "genes = np.array(['Q', 'N'])\n"
        "sims = np.array([[0.9], [0.9]]); idx = np.array([[1], [0]])\n"
        "ids = {str(i) for i in range(40)}\n"
        "g2c = {'Q': {'17'}, 'N': ids}\n"
        "p = m.evaluate_complex_prediction(genes, sims, idx, g2c, 1)\n"
        "print(json.dumps(p['best_true_complex_rank'].tolist()))\n"
    )
    outputs = set()
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed}, check=True)
        outputs.add(json.loads(result.stdout.strip().splitlines()[-1])[0])
    # All 40 complexes tie, so rank follows sorted (string) complex id.
    expected = sorted(str(i) for i in range(40)).index('17') + 1
    assert outputs == {expected}


def test_pairwise_ap_far_above_prevalence_for_structured_embedding(bench):
    genes, emb, g2c = clustered_embedding(n_clusters=5, per_cluster=10)
    result = bench.evaluate_pairwise(genes, emb, g2c)
    assert result["average_precision"] > 0.95
    assert result["auroc"] > 0.99
    assert result["ap_over_prevalence"] > 3


def test_pairwise_ap_near_prevalence_for_random_embedding(bench):
    genes, _, g2c = clustered_embedding(n_clusters=5, per_cluster=10)
    rng = np.random.default_rng(3)
    result = bench.evaluate_pairwise(genes, rng.normal(size=(50, 16)), g2c)
    assert result["average_precision"] < 3 * result["positive_prevalence"]
    assert 0.3 < result["auroc"] < 0.7


def make_fake_adata(pca):
    names = [f"p{i}" for i in range(len(pca))]
    return SimpleNamespace(obs_names=pd.Index(names), obsm={"X_pca": pca})


def test_build_gene_embeddings_collapses_promoters_and_filters(bench):
    pca = np.array([
        [1.0, 0.0],      # p0 GENEA P1
        [3.0, 2.0],      # p1 GENEA P2  -> median with p0
        [5.0, 5.0],      # p2 GENEB
        [np.nan, 1.0],   # p3 GENEC non-finite embedding -> dropped
        [9.0, 9.0],      # p4 control -> dropped
        [7.0, 7.0],      # p5 GENED low effect -> dropped
        [8.0, 8.0],      # p6 GENEE fails qc -> dropped
    ])
    summary = pd.DataFrame({
        "target_gene": ["GENEA", "GENEA", "GENEB", "GENEC", "non-targeting",
                        "GENED", "GENEE"],
        "qc_pass": [True, True, True, True, True, True, False],
        "is_control": [False, False, False, False, True, False, False],
        "adjusted_effect_z": [8, 9, 20, 10, 50, 1, 30],
        "energy_test_p_value": [1e-4] * 7,
        "knockdown_efficiency": [0.8, 0.6, np.nan, 0.5, 0.9, 0.5, 0.5],
        "num_cells_filtered": [100, 200, 150, 120, 300, 90, 80],
    }, index=[f"p{i}" for i in range(7)])
    meta, emb, _ = bench.build_gene_embeddings(
        make_fake_adata(pca), summary, min_z=5.0, max_p=0.001)
    assert list(meta.index) == ["GENEA", "GENEB"]
    np.testing.assert_allclose(emb[0], [2.0, 1.0])
    assert meta.loc["GENEA", "n_perturbations"] == 2
    assert np.isnan(meta.loc["GENEB", "median_knockdown_efficiency"])
    assert np.isfinite(emb).all()
