import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def make_controls(n=500, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.round(10 ** rng.uniform(1.3, 3.0, n)).astype(int)
    effect = 60 * cells ** -0.5 * np.exp(rng.normal(scale=0.08, size=n))
    return pd.DataFrame({"num_cells_filtered": cells, "effect_l2": effect})


def test_control_background_removes_cell_count_dependence(adjust):
    controls = make_controls()
    spline, binned = adjust.fit_control_background(controls)
    log_cells = np.log10(controls["num_cells_filtered"].to_numpy())
    residual = np.log10(controls["effect_l2"].to_numpy()) - spline(log_cells)

    raw_rho = spearmanr(controls["num_cells_filtered"], controls["effect_l2"])[0]
    adj_rho = spearmanr(controls["num_cells_filtered"], residual)[0]
    assert raw_rho < -0.9          # strong confounding before adjustment
    assert abs(adj_rho) < 0.1      # essentially gone afterwards
    assert abs(np.median(residual)) < 0.02


def test_control_background_tracks_the_true_curve(adjust):
    controls = make_controls()
    spline, _ = adjust.fit_control_background(controls)
    for cells in (30, 100, 300, 1000):
        expected = np.log10(60 * cells ** -0.5)
        assert abs(spline(np.log10(cells)) - expected) < 0.05


def test_background_bins_are_sorted_and_cover_the_range(adjust):
    controls = make_controls()
    _, binned = adjust.fit_control_background(controls)
    assert binned["x"].is_monotonic_increasing
    assert binned["n"].sum() == len(controls)
