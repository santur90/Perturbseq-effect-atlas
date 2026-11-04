import numpy as np
import pytest

torch = pytest.importorskip("torch")


def low_rank_data(n=300, dim=40, rank=4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, rank)) @ rng.normal(size=(rank, dim))).astype(np.float32)


def fit(autoenc, X, seed, epochs=60):
    idx = np.arange(len(X))
    return autoenc.train_autoencoder(
        X, idx[40:], idx[:40], latent=4, seed=seed, epochs=epochs, lr=1e-3,
        weight_decay=1e-4, noise_sd=0.0, patience=100, device="cpu")


def test_encoder_output_shape(autoenc):
    X = low_rank_data()
    model, _, _ = fit(autoenc, X, seed=0, epochs=3)
    assert autoenc.encode(model, X, "cpu").shape == (len(X), 4)


def test_training_beats_predicting_zero(autoenc):
    X = low_rank_data()
    _, val_mse, _ = fit(autoenc, X, seed=0)
    baseline = float(np.mean(X[:40] ** 2))
    assert val_mse < baseline


def test_same_seed_gives_same_embedding(autoenc):
    X = low_rank_data()
    a = autoenc.encode(fit(autoenc, X, seed=3, epochs=5)[0], X, "cpu")
    b = autoenc.encode(fit(autoenc, X, seed=3, epochs=5)[0], X, "cpu")
    np.testing.assert_allclose(a, b, atol=1e-5)
