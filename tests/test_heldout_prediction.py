import numpy as np
import pandas as pd


def toy_matrix(n_genes=60, n_cx=12, seed=0):
    """Binary gene x complex matrix; every gene is in 1-2 complexes."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n_genes, n_cx), dtype=np.float32)
    for i in range(n_genes):
        M[i, rng.integers(0, n_cx)] = 1.0
        if rng.random() < 0.3:
            M[i, rng.integers(0, n_cx)] = 1.0
    return M


def test_rowwise_corr_matches_numpy(heldout):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 40)), rng.normal(size=(5, 40))
    expected = [np.corrcoef(a[i], b[i])[0, 1] for i in range(5)]
    np.testing.assert_allclose(heldout.rowwise_corr(a, b), expected, atol=1e-10)


def test_rowwise_corr_is_nan_for_constant_rows(heldout):
    a = np.ones((1, 10))
    b = np.arange(10, dtype=float)[None, :]
    assert np.isnan(heldout.rowwise_corr(a, b)).all()


def test_topk_overlap_bounds(heldout):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 500))
    assert (heldout.topk_overlap(x, x, k=20) == 1.0).all()
    a = np.zeros((1, 100)); a[0, :10] = 5
    b = np.zeros((1, 100)); b[0, 90:] = 5
    assert heldout.topk_overlap(a, b, k=10)[0] == 0.0


def test_neighbor_model_falls_back_to_train_mean_without_shared_complex(heldout):
    Mtr = np.array([[1, 0], [1, 0]], dtype=np.float32)
    Ytr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    Mte = np.array([[0, 1]], dtype=np.float32)
    pred, covered = heldout.predict_neighbor(Mtr, Ytr, Mte)
    np.testing.assert_allclose(pred[0], Ytr.mean(axis=0))
    assert not covered[0]


def test_neighbor_model_uses_jaccard_weights(heldout):
    Mtr = np.array([[1, 0], [1, 1]], dtype=np.float32)  # Jaccard with test: 1.0, 0.5
    Ytr = np.array([[2.0, 0.0], [8.0, 0.0]], dtype=np.float32)
    Mte = np.array([[1, 0]], dtype=np.float32)
    pred, covered = heldout.predict_neighbor(Mtr, Ytr, Mte)
    np.testing.assert_allclose(pred[0], [(1.0 * 2 + 0.5 * 8) / 1.5, 0.0], rtol=1e-6)
    assert covered[0]


def test_gene_holdout_split_partitions_all_genes(heldout):
    tests = []
    for tr, te in heldout.split_gene_holdout(50, n_splits=5, seed=0):
        assert not set(tr) & set(te)
        tests.extend(te)
    assert sorted(tests) == list(range(50))


def test_complex_disjoint_split_has_no_shared_complex_between_train_and_test(heldout):
    M = toy_matrix()
    seen = []
    for tr, te in heldout.split_complex_disjoint(len(M), n_splits=3, M=M):
        assert not set(tr) & set(te)
        test_complexes = M[te].sum(0) > 0
        assert M[tr][:, test_complexes].sum() == 0  # no leakage via any complex
        seen.extend(te)
    assert sorted(seen) == list(range(len(M)))


def test_ridge_in_complex_disjoint_fold_cannot_discriminate_test_genes(heldout):
    """Documents the Split B finding: unseen complexes give one constant vector."""
    rng = np.random.default_rng(1)
    M = toy_matrix()
    Y = rng.normal(size=(len(M), 30)).astype(np.float32)
    for tr, te in heldout.split_complex_disjoint(len(M), n_splits=3, M=M):
        pred, _ = heldout.predict_ridge(M[tr], Y[tr], M[te])
        assert pred.std(axis=0).max() < 1e-4


def test_ridge_beats_train_mean_when_complex_signal_exists(heldout):
    rng = np.random.default_rng(2)
    M = toy_matrix(n_genes=200, n_cx=10)
    loadings = rng.normal(size=(10, 40))
    Y = (M @ loadings + rng.normal(scale=0.3, size=(200, 40))).astype(np.float32)
    gains = []
    for tr, te in heldout.split_gene_holdout(200, n_splits=4, seed=0):
        ridge, _ = heldout.predict_ridge(M[tr], Y[tr], M[te])
        base, _ = heldout.predict_mean(M[tr], Y[tr], M[te])
        gains.append(np.nanmean(heldout.rowwise_corr(ridge, Y[te]))
                     - np.nanmean(heldout.rowwise_corr(base, Y[te])))
    assert min(gains) > 0.2


def test_build_complex_matrix_reads_corum_table(heldout, tmp_path, monkeypatch):
    path = tmp_path / "corum.csv"
    pd.DataFrame({
        "ComplexID": [10, 20, 30],
        "ComplexName": ["a", "b", "c"],
        "Organism": ["Human", "Human", "Mouse"],
        "subunits.Gene.name.": ["G1;G2;G3", "G2;G3;G4;G9", "G1;G2;G3"],
    }).to_csv(path, index=False)
    monkeypatch.setattr(heldout, "CORUM", path)
    M, ids = heldout.build_complex_matrix(np.array(["G1", "G2", "G3", "G4"]))
    assert ids == ["10", "20"]  # mouse complex excluded
    assert M.shape == (4, 2)
    np.testing.assert_array_equal(
        M, [[1, 0], [1, 1], [1, 1], [0, 1]])
