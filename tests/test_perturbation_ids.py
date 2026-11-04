def test_parse_ids_standard_targeting_perturbation(effect):
    df = effect.parse_ids(["10000_ZBTB46_P1P2_ENSG00000130584"])
    row = df.iloc[0]
    assert row["target_gene"] == "ZBTB46"
    assert row["promoter"] == "P1P2"
    assert row["target_ensembl"] == "ENSG00000130584"
    assert not row["is_control"]


def test_parse_ids_non_targeting_control(effect):
    row = effect.parse_ids(
        ["10747_non-targeting_non-targeting_non-targeting"]).iloc[0]
    assert row["is_control"]
    assert row["target_gene"] == "non-targeting"


def test_parse_ids_keeps_hyphens_in_gene_names(effect):
    row = effect.parse_ids(["123_HLA-A_P1_ENSG00000206503"]).iloc[0]
    assert row["target_gene"] == "HLA-A"


def test_parse_ids_leaves_unparseable_ids_without_a_target(effect):
    """Known limitation: IDs without an Ensembl suffix get no target gene."""
    row = effect.parse_ids(["7155_RBM14-RBM4_P1_nan"]).iloc[0]
    assert row["target_gene"] is None
    assert not row["is_control"]


def test_parse_ids_preserves_order_and_index(effect):
    names = ["1_A_P1_ENSG00000000001", "2_B_P2_ENSG00000000002"]
    df = effect.parse_ids(names)
    assert list(df.index) == names
