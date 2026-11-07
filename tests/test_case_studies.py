import numpy as np
import pandas as pd


def synthetic_world(seed=0, dim=24, n_background=250):
    """Background genes plus five planted groups; some case genes sit near a group."""
    rng = np.random.default_rng(seed)
    # Orthogonal centers: unrelated groups have no built-in similarity, so any
    # significant off-target result would indicate a real bug in the test.
    basis, _ = np.linalg.qr(rng.normal(size=(dim, 5)))
    names = ["integrator", "respiratory", "proteasome", "ribosome", "riboBG"]
    centers = {k: basis[:, i] * 8 for i, k in enumerate(names)}
    genes, rows = [], []

    def add(names, center, noise=0.5):
        for n in names:
            genes.append(n)
            rows.append(centers[center] + rng.normal(scale=noise, size=dim))

    add([f"INTS{i}" for i in range(1, 13)], "integrator")
    add([f"NDUFA{i}" for i in range(1, 16)], "respiratory")
    add([f"PSMA{i}" for i in range(1, 8)] + [f"PSMB{i}" for i in range(1, 8)], "proteasome")
    add([f"RPL{i}" for i in range(1, 15)], "ribosome")
    add([f"RBG{i}" for i in range(1, 13)], "riboBG")
    add(["CCDC86", "SPATA5L1"], "riboBG", noise=0.6)      # near ribosome biogenesis
    add(["C7ORF26"], "integrator", noise=0.6)              # near Integrator
    add(["TMEM242"], "respiratory", noise=0.6)             # near respiratory chain
    for i in range(n_background):
        genes.append(f"BG{i}")
        rows.append(rng.normal(size=dim) * 3)
    genes.append("ZNF236")                                  # unrelated gene
    rows.append(rng.normal(size=dim) * 3)
    complexes = [dict(complex_id="1", complex_name="SSU processome",
                      screen_members={f"RBG{i}" for i in range(1, 13)}),
                 dict(complex_id="2", complex_name="Unrelated complex",
                      screen_members={"BG1", "BG2", "BG3"})]
    return np.array(genes), np.array(rows, dtype=np.float32), complexes


def test_build_reference_sets_uses_prefixes_and_keywords(cases_mod):
    genes, _, complexes = synthetic_world()
    sets, matched = cases_mod.build_reference_sets(genes, complexes)
    assert sets["integrator"] == {f"INTS{i}" for i in range(1, 13)}
    assert sets["ribosome_biogenesis"] == {f"RBG{i}" for i in range(1, 13)}
    assert list(matched["complex_name"]) == ["SSU processome"]
    assert len(sets["proteasome"]) == 14
    # prefix sets only contain genes that are actually in the embedding
    assert sets["respiratory_chain"] <= set(genes)


def test_keyword_match_is_case_insensitive_and_ignores_genes_not_in_embedding(cases_mod):
    complexes = [dict(complex_id="9", complex_name="Pre-60S ribosomal particle",
                      screen_members={"A", "B", "NOTPRESENT"})]
    sets, matched = cases_mod.build_reference_sets(np.array(["A", "B", "C"]), complexes)
    assert sets["ribosome_biogenesis"] == {"A", "B"}
    assert len(matched) == 1


def test_set_test_detects_a_nearby_set(cases_mod):
    genes, emb, complexes = synthetic_world()
    sets, _ = cases_mod.build_reference_sets(genes, complexes)
    sim = cases_mod.cosine_matrix(emb)
    idx = {g: i for i, g in enumerate(genes)}
    members = [idx[g] for g in sets["integrator"]]
    res = cases_mod.set_test(sim[idx["C7ORF26"]], idx["C7ORF26"], members,
                             500, np.random.default_rng(0), 20)
    assert res["p_perm"] < 0.01 and res["z"] > 5
    assert res["top_overlap"] >= 10 and res["p_hyper"] < 1e-6


def test_set_test_excludes_the_case_gene_from_its_own_reference_set(cases_mod):
    genes, emb, _ = synthetic_world()
    sim = cases_mod.cosine_matrix(emb)
    idx = {g: i for i, g in enumerate(genes)}
    members = [idx[f"INTS{i}"] for i in range(1, 13)]
    self_idx = members[0]
    res = cases_mod.set_test(sim[self_idx], self_idx, members, 100,
                             np.random.default_rng(0), 20)
    assert res["n_set_members"] == 11


def test_p_values_are_not_inflated_under_the_null(cases_mod):
    rng = np.random.default_rng(5)
    emb = rng.normal(size=(300, 20)).astype(np.float32)
    sim = cases_mod.cosine_matrix(emb)
    members = rng.choice(np.arange(100, 300), size=12, replace=False)
    ps = [cases_mod.set_test(sim[i], i, members, 300, np.random.default_rng(i), 20)["p_perm"]
          for i in range(60)]
    assert 0.0 <= np.mean(np.array(ps) < 0.05) <= 0.15   # nominal rate is 0.05


def test_end_to_end_primary_hits_and_specificity(cases_mod):
    genes, emb, complexes = synthetic_world()
    sets, _ = cases_mod.build_reference_sets(genes, complexes)
    status, matrix, neighbors = cases_mod.run_case_studies(
        genes, emb, {}, sets, {}, n_null=600, seed=1)
    m = matrix.set_index(["case_gene", "reference_set"])

    for gene, ref in [("CCDC86", "ribosome_biogenesis"), ("SPATA5L1", "ribosome_biogenesis"),
                      ("C7orf26", "integrator"), ("TMEM242", "respiratory_chain")]:
        assert m.loc[(gene, ref), "p_perm"] < 0.01, (gene, ref)
        assert bool(m.loc[(gene, ref), "expected"])

    # the unrelated gene is not specifically close to any set
    assert (m.loc["ZNF236", "p_perm"] > 0.01).all()
    # negative control set: no case gene is close to the proteasome
    assert (m.xs("proteasome", level="reference_set")["p_perm"] > 0.01).all()
    # specificity: a ribosome-biogenesis gene is not close to Integrator
    assert m.loc[("CCDC86", "integrator"), "p_perm"] > 0.01


def test_alias_lookup_and_missing_gene_handling(cases_mod):
    genes, emb, complexes = synthetic_world()
    keep = [i for i, g in enumerate(genes) if g != "TMEM242"]
    genes = np.array([("INTS15" if genes[i] == "C7ORF26" else genes[i]) for i in keep])
    emb = emb[keep]
    sets, _ = cases_mod.build_reference_sets(genes, complexes)
    status, matrix, _ = cases_mod.run_case_studies(genes, emb, {}, sets, {}, n_null=50)
    s = status.set_index("case_gene")
    assert s.loc["C7orf26", "name_found_in_embedding"] == "INTS15"
    assert not s.loc["TMEM242", "in_embedding"]
    assert "TMEM242" not in set(matrix["case_gene"])
    # INTS15 matches the Integrator prefix but must not count itself as a member
    row = matrix[(matrix["case_gene"] == "C7orf26") & (matrix["reference_set"] == "integrator")].iloc[0]
    assert row["n_set_members"] == 12


def test_small_reference_sets_are_not_tested(cases_mod):
    genes, emb, _ = synthetic_world()
    sets = {"tiny": {"INTS1", "INTS2", "INTS3"}, "integrator": {f"INTS{i}" for i in range(1, 13)}}
    _, matrix, _ = cases_mod.run_case_studies(
        genes, emb, {}, sets, {}, cases={"C7orf26": (["C7ORF26"], ["integrator"])},
        n_null=50, min_set_size=5)
    tiny = matrix[matrix["reference_set"] == "tiny"].iloc[0]
    assert np.isnan(tiny["p_perm"]) and tiny["n_set_members"] == 3


def test_gmt_loader_and_heatmap(cases_mod, tmp_path):
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("SET_A\tna\tg1\tg2\tG3\nSET_B\tna\tx\n")
    sets = cases_mod.load_gmt(gmt)
    assert sets["SET_A"] == {"G1", "G2", "G3"}

    genes, emb, complexes = synthetic_world()
    ref, _ = cases_mod.build_reference_sets(genes, complexes)
    effect = np.random.default_rng(0).uniform(5, 40, len(genes))
    _, matrix, _ = cases_mod.run_case_studies(genes, emb, {}, ref, {}, n_null=50, effect=effect)
    out = tmp_path / "fig.png"
    cases_mod.make_heatmap(matrix, out, 0.01)
    assert out.stat().st_size > 1000


def test_drop_structural_ribosomal(cases_mod):
    members = {"RPL3", "RPS19", "MRPL12", "MRPS9", "NOP56", "UTP11", "RPLP0", "WDR3"}
    kept = cases_mod.drop_structural_ribosomal(members)
    assert kept == {"NOP56", "UTP11", "RPLP0", "WDR3"}  # RPLP0 is not matched by the prefix rule


def test_count_primary_tests_ignores_rows_that_could_not_run(cases_mod):
    matrix = pd.DataFrame({
        "expected": [True, True, True, False],
        "p_perm": [0.01, np.nan, 0.5, 0.2]})
    assert cases_mod.count_primary_tests(matrix) == 2


def test_filtered_biogenesis_set_is_the_primary_test_and_full_set_is_sensitivity(cases_mod):
    genes, emb, complexes = synthetic_world()
    full = {f"RBG{i}" for i in range(1, 13)} | {"RPL1", "RPL2", "RPL3", "RPL4", "RPL5"}
    extra = {"gmt_full": full, "gmt_noRP": cases_mod.drop_structural_ribosomal(full)}
    sets, _ = cases_mod.build_reference_sets(genes, complexes, extra)
    cases = {"CCDC86": (["CCDC86"], ["gmt_noRP"])}
    _, matrix, _ = cases_mod.run_case_studies(genes, emb, {}, sets, {}, cases=cases, n_null=300)
    m = matrix.set_index("reference_set")
    assert bool(m.loc["gmt_noRP", "expected"]) and not bool(m.loc["gmt_full", "expected"])
    assert m.loc["gmt_noRP", "n_set_members"] == 12
    assert m.loc["gmt_full", "n_set_members"] == 17
    assert cases_mod.count_primary_tests(matrix) == 1


# ---------------------------------------------------------------------------
# Effect-strength-matched null
# ---------------------------------------------------------------------------
def confounded_world(seed=0, dim=30):
    """Strongly responding genes share one common direction; weak genes are mostly noise.

    The reference set R is made only of strongly responding genes.
      * X_strong : strongly responding, but unrelated to R beyond the shared direction
      * X_member : strongly responding AND shares R's specific signature
      * X_weak   : weakly responding, unrelated
    Most random genes are weak, so an unmatched null makes X_strong look 'close to R'.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=dim); common /= np.linalg.norm(common)
    sig_r = rng.normal(size=dim); sig_r -= sig_r @ common * common; sig_r /= np.linalg.norm(sig_r)
    genes, rows, effect = [], [], []

    def add(name, strength, signature=None, sig_weight=0.0):
        vec = strength * common + rng.normal(scale=1.0, size=dim)
        if signature is not None:
            vec = vec + sig_weight * signature
        genes.append(name); rows.append(vec); effect.append(strength)

    for i in range(20):
        add(f"R{i}", rng.uniform(25, 35), sig_r, 8.0)           # reference set, strong
    for i in range(60):
        add(f"STRONG{i}", rng.uniform(25, 35))                  # strong, unrelated
    for i in range(240):
        add(f"WEAK{i}", rng.uniform(0.5, 4))                    # weak, unrelated
    add("X_strong", 30.0)
    add("X_member", 30.0, sig_r, 8.0)
    add("X_weak", 2.0)
    return np.array(genes), np.array(rows, dtype=np.float32), np.array(effect)


def run_confounded(cases_mod, n_null=800, seed=1):
    genes, emb, effect = confounded_world()
    ref = {"R": {g for g in genes if g.startswith("R") and g[1:].isdigit()}}
    cases = {"X_strong": (["X_STRONG"], []), "X_member": (["X_MEMBER"], ["R"]),
             "X_weak": (["X_WEAK"], [])}
    genes = np.array([g.upper() for g in genes])
    ref = {"R": {g.upper() for g in ref["R"]}}
    _, matrix, _ = cases_mod.run_case_studies(
        genes, emb, {}, ref, {}, cases=cases, n_null=n_null, seed=seed, effect=effect,
        min_set_size=5)
    return matrix.set_index("case_gene")


def test_unmatched_null_is_fooled_by_shared_response_but_matched_null_is_not(cases_mod):
    m = run_confounded(cases_mod)
    row = m.loc["X_strong"]
    assert row["p_perm"] < 0.01          # unmatched: looks specifically close to R
    assert row["p_matched"] > 0.05       # matched: no specific relation
    assert row["z_matched"] < 2


def test_matched_null_still_detects_a_genuine_relationship(cases_mod):
    m = run_confounded(cases_mod)
    row = m.loc["X_member"]
    assert row["p_matched"] < 0.01 and row["z_matched"] > 3
    assert row["p_overlap_matched"] < 0.05
    assert row["top_overlap"] > row["expected_overlap_matched"]


def test_matched_expected_overlap_exceeds_unmatched_for_strong_sets(cases_mod):
    m = run_confounded(cases_mod)
    row = m.loc["X_strong"]
    assert row["expected_overlap_matched"] > row["expected_overlap"]


def test_matched_draws_reproduce_the_effect_strength_composition(cases_mod):
    rng = np.random.default_rng(0)
    effect = np.concatenate([rng.uniform(0, 1, 300), rng.uniform(9, 10, 100)])
    bins = cases_mod.make_effect_bins(effect, 4)
    candidates = np.arange(1, 400)
    members = np.array([310, 320, 330, 340, 5, 6])              # 4 strong, 2 weak
    draws = cases_mod.matched_null_draws(candidates, bins, bins[members], 200, rng)
    assert draws.shape == (200, 6)
    assert ((effect[draws] > 5).sum(axis=1) == 4).all()          # always 4 strong + 2 weak
    assert all(len(set(row)) == 6 for row in draws)              # no duplicates within a draw
    assert 0 not in draws                                         # case gene (index 0) excluded


def test_matched_fields_are_nan_without_effect_vector(cases_mod):
    genes, emb, _ = synthetic_world()
    sets = {"integrator": {f"INTS{i}" for i in range(1, 13)}}
    _, matrix, _ = cases_mod.run_case_studies(
        genes, emb, {}, sets, {}, cases={"C7orf26": (["C7ORF26"], ["integrator"])}, n_null=50)
    assert matrix["z_matched"].isna().all() and matrix["p_perm"].notna().all()


def test_set_test_never_offers_the_case_gene_to_the_matched_null(cases_mod, monkeypatch):
    seen = {}
    original = cases_mod.matched_null_draws

    def spy(candidates, bins, member_bins, n_null, rng):
        seen["candidates"] = np.array(candidates)
        return original(candidates, bins, member_bins, n_null, rng)

    monkeypatch.setattr(cases_mod, "matched_null_draws", spy)
    genes, emb, _ = synthetic_world()
    sim = cases_mod.cosine_matrix(emb)
    bins = cases_mod.make_effect_bins(np.random.default_rng(0).uniform(5, 40, len(genes)), 5)
    idx = {g: i for i, g in enumerate(genes)}
    members = [idx[f"INTS{i}"] for i in range(1, 13)]
    case = idx["C7ORF26"]
    cases_mod.set_test(sim[case], case, members, 50, np.random.default_rng(0), 20, bins)
    assert case not in seen["candidates"]
    assert len(seen["candidates"]) == len(genes) - 1


# ---------------------------------------------------------------------------
# Heatmap data and layout
# ---------------------------------------------------------------------------
def tiny_matrix():
    rows = [
        ("CCDC86", "ribosome_biogenesis_gmt_noRP", 13, 2.7, 0.0001, True),
        ("CCDC86", "cytosolic_ribosome", 0, 1.4, 1.0, False),
        ("C7orf26", "integrator", 3, 0.2, 0.0006, True),
        ("C7orf26", "ribosome_biogenesis_gmt_noRP", 0, 1.9, 1.0, False),
        ("C7orf26", "not_in_order_list", 1, 1.0, 0.5, False),
    ]
    return pd.DataFrame(rows, columns=["case_gene", "reference_set", "top_overlap",
                                       "expected_overlap_matched", "p_overlap_matched", "expected"])


def test_overlap_log2_enrichment_values(cases_mod):
    np.testing.assert_allclose(cases_mod.overlap_log2_enrichment([3, 0], [0.5, 0.5]), [np.log2(3.5), -1.0])
    assert np.isfinite(cases_mod.overlap_log2_enrichment(0, 2.0))     # zero observed stays finite


def test_heatmap_tables_order_filter_and_values(cases_mod):
    t = cases_mod.heatmap_tables(tiny_matrix())
    enr = t["log2_enrichment"]
    assert list(enr.index) == ["CCDC86", "C7orf26"]                  # follows the case order
    assert list(enr.columns) == ["ribosome_biogenesis_gmt_noRP", "integrator", "cytosolic_ribosome"]
    assert "not_in_order_list" not in enr.columns
    assert enr.loc["CCDC86", "ribosome_biogenesis_gmt_noRP"] > 2
    assert np.isnan(enr.loc["CCDC86", "integrator"])                  # untested cell stays NaN
    assert t["top_overlap"].loc["C7orf26", "integrator"] == 3


def test_heatmap_skips_cells_without_matched_results(cases_mod):
    m = tiny_matrix()
    m.loc[0, "p_overlap_matched"] = np.nan
    t = cases_mod.heatmap_tables(m)
    assert np.isnan(t["log2_enrichment"].loc["CCDC86", "ribosome_biogenesis_gmt_noRP"])


def test_heatmap_file_is_written_for_small_input(cases_mod, tmp_path):
    out = tmp_path / "h.png"
    cases_mod.make_heatmap(tiny_matrix(), out, 0.01)
    assert out.stat().st_size > 1000
