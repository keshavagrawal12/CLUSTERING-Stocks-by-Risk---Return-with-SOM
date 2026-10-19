import numpy as np
import pytest

from app import config
from app.models.clustering import cluster_neurons
from app.models.som import SelfOrganizingMap, exp_decay


def test_bmu_hand_made_example():
    """2x2 map in 2-D with known weights: the closest neuron must win."""
    som = SelfOrganizingMap(rows=2, cols=2, dim=2)
    som.weights = np.array([[[0, 0], [0, 10]], [[10, 0], [10, 10]]], dtype=float)
    assert som.bmu(np.array([9.0, 1.0])) == (1, 0)
    assert som.bmu(np.array([1.0, 8.0])) == (0, 1)
    assert som.bmu(np.array([0.2, 0.1])) == (0, 0)


def test_neighbourhood_is_gaussian_on_grid():
    som = SelfOrganizingMap(rows=3, cols=3, dim=2)
    h = som.neighbourhood((1, 1), sigma=1.0)
    assert h[1, 1] == pytest.approx(1.0)
    assert h[0, 1] == pytest.approx(np.exp(-1 / 2))       # grid distance 1
    assert h[0, 0] == pytest.approx(np.exp(-2 / 2))       # grid distance sqrt(2)


def test_kohonen_update_rule_per_neuron():
    """Vectorised update must equal w + eta*h*(x - w) neuron by neuron."""
    som = SelfOrganizingMap(rows=3, cols=3, dim=2, seed=0)
    som.weights = np.random.default_rng(0).normal(size=(3, 3, 2))
    before = som.weights.copy()
    x, eta, sigma = np.array([1.0, -1.0]), 0.3, 1.2
    h = som.update(x, (0, 2), eta, sigma)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(som.weights[i, j], before[i, j] + eta * h[i, j] * (x - before[i, j]))


def test_eta_and_sigma_decay_as_specified():
    som = SelfOrganizingMap(sigma0=2.0, eta0=0.5, ordering_iters=100, convergence_iters=400)
    phase0, eta0, sig0 = som.schedule(0)
    assert phase0 == "ordering" and eta0 == pytest.approx(0.5) and sig0 == pytest.approx(2.0)
    # End of ordering phase reaches the convergence-phase start values.
    _, eta_e, sig_e = som.schedule(99)
    assert eta_e == pytest.approx(config.SOM_CONV_ETA0, rel=0.05)
    assert sig_e == pytest.approx(config.SOM_CONV_SIGMA0, rel=0.05)
    phase, eta_c, sig_c = som.schedule(100)
    assert phase == "convergence" and eta_c == pytest.approx(config.SOM_CONV_ETA0)
    # Monotone exponential decay: ratio between equally spaced steps is constant.
    vals = [som.schedule(t)[2] for t in (0, 20, 40, 60)]
    ratios = np.array(vals[1:]) / np.array(vals[:-1])
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)
    assert all(np.diff([som.schedule(t)[1] for t in range(0, 500, 25)]) <= 0)
    _, eta_end, sig_end = som.schedule(499)
    assert eta_end == pytest.approx(config.SOM_ETA_MIN, rel=0.05)
    assert sig_end == pytest.approx(config.SOM_SIGMA_MIN, rel=0.05)


def test_exp_decay_formula():
    assert exp_decay(1.0, 0, 10, 0.1) == pytest.approx(1.0)
    assert exp_decay(1.0, 10, 10, 0.1) == pytest.approx(0.1)
    assert exp_decay(1.0, 5, 10, 0.1) == pytest.approx(np.sqrt(0.1))


def test_qe_decreases_on_clustered_data(blobs):
    X, _ = blobs
    som = SelfOrganizingMap(rows=6, cols=6, init="random", ordering_iters=500, convergence_iters=1500, seed=3)
    hist = som.fit(X)
    qes = [c.qe for c in hist.checkpoints]
    assert qes[-1] < 0.5 * qes[0]
    assert qes[-1] <= min(qes[:5])


def test_topographic_error_low_on_well_ordered_map(blobs):
    X, _ = blobs
    som = SelfOrganizingMap(rows=6, cols=6, init="pca", seed=0)
    som.fit(X, store_weights=False)
    assert som.topographic_error(X) < 0.1


def test_topographic_error_detects_scrambled_map(blobs):
    X, _ = blobs
    som = SelfOrganizingMap(rows=6, cols=6, init="pca", seed=0)
    som.fit(X, store_weights=False)
    good = som.topographic_error(X)
    flat = som.weights.reshape(-1, 7)
    som.weights = flat[np.random.default_rng(0).permutation(len(flat))].reshape(6, 6, 7)
    assert som.topographic_error(X) > good


def test_reproducible_with_seed(blobs):
    X, _ = blobs
    a = SelfOrganizingMap(init="random", seed=7, ordering_iters=200, convergence_iters=300)
    b = SelfOrganizingMap(init="random", seed=7, ordering_iters=200, convergence_iters=300)
    a.fit(X, store_weights=False)
    b.fit(X, store_weights=False)
    np.testing.assert_array_equal(a.weights, b.weights)
    c = SelfOrganizingMap(init="random", seed=8, ordering_iters=200, convergence_iters=300)
    c.fit(X, store_weights=False)
    assert not np.allclose(a.weights, c.weights)


def test_qe_in_same_range_as_minisom(blobs):
    """Cross-check against MiniSom (tests only) with comparable settings."""
    from minisom import MiniSom

    X, _ = blobs
    ours = SelfOrganizingMap(rows=7, cols=7, init="pca", seed=1)
    ours.fit(X, store_weights=False)
    ms = MiniSom(7, 7, 7, sigma=2.0, learning_rate=0.5, random_seed=1)
    ms.pca_weights_init(X)
    ms.train_random(X, 10_000)
    q_ours, q_ms = ours.quantization_error(X), ms.quantization_error(X)
    assert 0.5 * q_ms < q_ours < 1.5 * q_ms, (q_ours, q_ms)


def test_outputs_shapes(blobs):
    X, _ = blobs
    som = SelfOrganizingMap(rows=5, cols=4, ordering_iters=100, convergence_iters=100)
    hist = som.fit(X)
    assert som.u_matrix().shape == (5, 4)
    assert som.component_planes().shape == (7, 5, 4)
    assert som.hit_map(X).sum() == len(X)
    assert len(hist.checkpoints) >= 2 and hist.checkpoints[0].weights.shape == (5, 4, 7)


def test_neuron_clustering_recovers_blobs(blobs):
    X, y = blobs
    som = SelfOrganizingMap(rows=6, cols=6, seed=0)
    som.fit(X, store_weights=False)
    res = cluster_neurons(som.weights, som.bmus(X), k_max=6)
    assert res.k == 3
    assert [p["k"] for p in res.silhouette] == list(range(3, 7))
    # Each blob maps to exactly one cluster.
    for lbl in "ABC":
        assert len(set(res.stock_labels[y == lbl])) == 1
