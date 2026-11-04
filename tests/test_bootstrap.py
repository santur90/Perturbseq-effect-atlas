import numpy as np
from sklearn.metrics import average_precision_score


def toy_problem(seed=0, n=150, dim=20, n_cx=12):
    rng = np.random.default_rng(seed)
    member = rng.integers(0, n_cx, n)
    genes = np.array([f"G{i}" for i in range(n)])
    gene_to_cx = {g: {str(member[i])} for i, g in enumerate(genes)}
    centers = rng.normal(size=(n_cx, dim))
    good = centers[member] + rng.normal(scale=0.8, size=(n, dim))
    bad = centers[member] + rng.normal(scale=2.5, size=(n, dim))
    return genes, member, gene_to_cx, good, bad


def test_pair_labels_match_direct_definition(boot):
    genes, member, g2c, _, _ = toy_problem()
    i0, i1, labels = boot.build_pair_labels(genes, g2c)
    direct = np.array([member[a] == member[b] for a, b in zip(i0, i1)]).astype(int)
    assert len(labels) == len(genes) * (len(genes) - 1) // 2
    assert (labels == direct).all()


def test_pair_labels_handle_multi_complex_genes(boot):
    genes = np.array(["A", "B", "C"])
    g2c = {"A": {"1", "2"}, "B": {"2"}, "C": {"3"}}
    i0, i1, labels = boot.build_pair_labels(genes, g2c)
    pairs = {(genes[a], genes[b]): int(y) for a, b, y in zip(i0, i1, labels)}
    assert pairs == {("A", "B"): 1, ("A", "C"): 0, ("B", "C"): 0}


def test_unit_weights_reproduce_plain_average_precision(boot):
    genes, _, g2c, good, _ = toy_problem()
    i0, i1, labels = boot.build_pair_labels(genes, g2c)
    scores = boot.pair_scores(good, i0, i1)
    weighted = average_precision_score(labels, scores, sample_weight=np.ones(len(labels)))
    assert abs(average_precision_score(labels, scores) - weighted) < 1e-12


def test_bootstrap_detects_a_real_improvement(boot, bench):
    genes, _, g2c, good, bad = toy_problem()
    i0, i1, labels = boot.build_pair_labels(genes, g2c)
    q_bad = boot.query_table(bench, genes, bad, g2c)
    q_good = boot.query_table(bench, genes, good, g2c)
    res = boot.paired_bootstrap(
        q_bad, [q_good], boot.pair_scores(bad, i0, i1),
        [boot.pair_scores(good, i0, i1)], labels, i0, i1, n_boot=60, seed=1)
    ap = res.set_index("metric").loc["pairwise_ap"]
    assert ap["ci_low"] > 0 and ap["mean_diff"] > 0.2
    assert (res["mean_diff"] > 0).all()


def test_bootstrap_of_identical_methods_is_exactly_zero(boot, bench):
    genes, _, g2c, good, _ = toy_problem()
    i0, i1, labels = boot.build_pair_labels(genes, g2c)
    q = boot.query_table(bench, genes, good, g2c)
    s = boot.pair_scores(good, i0, i1)
    res = boot.paired_bootstrap(q, [q], s, [s], labels, i0, i1, n_boot=30, seed=1)
    np.testing.assert_allclose(res[["mean_diff", "ci_low", "ci_high"]].to_numpy(), 0.0)


def test_bootstrap_is_reproducible_for_a_fixed_seed(boot, bench):
    genes, _, g2c, good, bad = toy_problem()
    i0, i1, labels = boot.build_pair_labels(genes, g2c)
    args = (boot.query_table(bench, genes, bad, g2c),
            [boot.query_table(bench, genes, good, g2c)],
            boot.pair_scores(bad, i0, i1), [boot.pair_scores(good, i0, i1)],
            labels, i0, i1)
    a = boot.paired_bootstrap(*args, n_boot=25, seed=7)
    b = boot.paired_bootstrap(*args, n_boot=25, seed=7)
    assert a.equals(b)
