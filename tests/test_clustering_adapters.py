"""External estimator boundary checks and small real clustering regressions."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

from toponymy.clustering import (
    EVoCClusterer,
    KMeansClusterer,
    PLSCANClusterer,
    ToponymyClusterer,
    validate_cluster_tree,
)


@pytest.mark.parametrize("factory", [PLSCANClusterer, EVoCClusterer, KMeansClusterer])
@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 0)),
        np.zeros((1, 2, 3)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1 + 0j]]),
        np.array([[True]]),
        np.array([["1"]]),
        np.array([[0]], dtype=object),
        sparse.eye(3),
    ],
)
def test_invalid_vectors_fail_before_external_calls(factory, vectors):
    with pytest.raises(ValueError):
        factory().fit(vectors)


@pytest.mark.parametrize("factory", [PLSCANClusterer, EVoCClusterer, KMeansClusterer])
def test_empty_fit_and_estimator_clone(factory):
    estimator = clone(factory())
    assert estimator.fit(np.empty((0, 2))) is estimator
    assert estimator.cluster_layers_ == []
    assert estimator.cluster_tree_ == {}
    assert list(estimator) == []


@pytest.mark.parametrize("factory", [PLSCANClusterer, EVoCClusterer])
def test_undersized_density_data_is_noise_without_optional_import(factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "fast_hdbscan", None)
    monkeypatch.setitem(sys.modules, "evoc", None)
    estimator = factory().fit(np.ones((1, 2)))
    assert estimator.cluster_tree_ == {}
    np.testing.assert_array_equal(estimator.cluster_layers_[0].labels, [-1])


@pytest.mark.parametrize("factory", [PLSCANClusterer, EVoCClusterer])
@pytest.mark.parametrize(
    "options",
    [
        {"min_samples": 0},
        {"base_min_cluster_size": 1},
        {"max_layers": 0},
        {"min_samples": True},
        {"max_layers": 1.5},
    ],
)
def test_invalid_density_parameters_are_not_masked_by_empty_data(factory, options):
    with pytest.raises(ValueError):
        factory(**options).fit(np.empty((0, 2)))


@pytest.mark.parametrize(
    "options",
    [
        {"min_clusters": 0},
        {"base_n_clusters": 0},
        {"min_clusters": 5, "base_n_clusters": 4},
    ],
)
def test_invalid_kmeans_parameters(options):
    with pytest.raises(ValueError):
        KMeansClusterer(**options).fit(np.ones((3, 2)))


def test_plscan_forwards_options_and_detaches_returned_labels(monkeypatch):
    calls = []
    emitted_labels = [np.array([81, 81, -1, 4, 4, 4])]

    class ExternalPLSCAN:
        def __init__(self, **options):
            calls.append(options)

        def fit(self, vectors):
            calls.append(vectors)
            self.cluster_layers_ = emitted_labels
            return self

    monkeypatch.setitem(
        sys.modules, "fast_hdbscan", SimpleNamespace(PLSCAN=ExternalPLSCAN)
    )
    vectors = np.arange(24.0).reshape(6, 4)[:, ::2]
    estimator = PLSCANClusterer(reproducible=True, metric_kwds={"p": 2})
    assert estimator.fit(vectors, verbose=True) is estimator
    assert calls[0]["reproducible"] is True
    assert calls[0]["metric_kwds"] == {"p": 2}
    assert calls[0]["verbose"] is True
    assert calls[1] is vectors
    emitted_labels[0][:] = -1
    assert [cluster.label for cluster in estimator.cluster_layers_[0]] == [4, 81]
    np.testing.assert_array_equal(
        estimator.cluster_layers_[0].labels, [81, 81, -1, 4, 4, 4]
    )


def test_evoc_uses_actual_constructor_options_and_shared_tree_rule(monkeypatch):
    calls = []

    class ExternalEVoC:
        def __init__(self, **options):
            assert "verbose" not in options
            calls.append(options)

        def fit(self, vectors):
            self.cluster_layers_ = [
                np.array([4, 4, 19, 19, -1, -1]),
                np.array([2, 3, 2, 2, -1, -1]),
            ]
            self.cluster_tree_ = {(1, 2): [(0, 4), (0, 19)]}
            return self

    monkeypatch.setitem(sys.modules, "evoc", SimpleNamespace(EVoC=ExternalEVoC))
    estimator = EVoCClusterer(random_state=31, approx_n_clusters=2).fit(np.ones((6, 2)))
    assert calls[0]["random_state"] == 31
    assert calls[0]["approx_n_clusters"] == 2
    assert (0, 4) in estimator.cluster_tree_[(2, 0)]
    assert estimator.cluster_tree_[(1, 2)] == [(0, 19)]
    validate_cluster_tree(estimator.cluster_tree_, estimator.cluster_layers_)


def test_missing_evoc_reports_optional_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "evoc", None)
    with pytest.raises(ImportError, match=r"toponymy\[evoc\]"):
        EVoCClusterer().fit(np.ones((6, 2)))


def test_evoc_does_not_disguise_broken_dependency_import(monkeypatch):
    original_import = builtins.__import__

    def import_with_missing_dependency(name, *args, **kwargs):
        if name == "evoc":
            raise ModuleNotFoundError("EVoC dependency is missing", name="numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="dependency is missing"):
        EVoCClusterer().fit(np.ones((6, 2)))


@pytest.mark.parametrize(
    "factory,module,attribute",
    [
        (PLSCANClusterer, "fast_hdbscan", "PLSCAN"),
        (EVoCClusterer, "evoc", "EVoC"),
    ],
)
def test_external_programming_errors_propagate(factory, module, attribute, monkeypatch):
    class BrokenEstimator:
        def __init__(self, **options):
            pass

        def fit(self, vectors):
            raise RuntimeError("external implementation failed")

    monkeypatch.setitem(
        sys.modules, module, SimpleNamespace(**{attribute: BrokenEstimator})
    )
    with pytest.raises(RuntimeError, match="external implementation"):
        factory().fit(np.ones((6, 2)))


@pytest.mark.parametrize(
    "graph",
    [
        np.zeros((2, 2)),
        sparse.csr_matrix(np.ones((2, 3))),
        sparse.csr_matrix([[-1.0]]),
        sparse.csr_matrix([[np.inf]]),
    ],
)
def test_invalid_precomputed_graph(graph):
    with pytest.raises(ValueError):
        PLSCANClusterer(metric="precomputed").fit(graph)


def test_precomputed_sparse_graph_reaches_external_estimator(monkeypatch):
    captured = []

    class ExternalPLSCAN:
        def __init__(self, **options):
            assert options["metric"] == "precomputed"

        def fit(self, graph):
            captured.append(graph)
            self.cluster_layers_ = [np.zeros(graph.shape[0], dtype=int)]

    monkeypatch.setitem(
        sys.modules, "fast_hdbscan", SimpleNamespace(PLSCAN=ExternalPLSCAN)
    )
    graph = sparse.dok_matrix((6, 6), dtype=float)
    graph[0, 1] = 2.0
    estimator = PLSCANClusterer(metric="precomputed").fit(graph)
    assert sparse.isspmatrix_csr(captured[0])
    assert captured[0][0, 1] == 2.0
    assert estimator.cluster_tree_ == {(1, 0): [(0, 0)]}
    empty = PLSCANClusterer(metric="precomputed").fit(sparse.csr_matrix((0, 0)))
    assert empty.cluster_layers_ == []


def test_toponymy_clusterer_migration_is_explicit():
    with pytest.warns(FutureWarning, match="now uses PLSCAN"):
        estimator = ToponymyClusterer().fit(np.ones((1, 2)))
    assert isinstance(estimator, PLSCANClusterer)


def test_real_kmeans_small_data_and_repeated_fit():
    vectors = np.array([[0.0, 0.1], [0.1, 0.0], [8.0, 8.1], [8.1, 8.0]])
    estimator = KMeansClusterer(min_clusters=1, base_n_clusters=16, random_state=7)
    estimator.fit(vectors)
    assert [len(layer) for layer in estimator] == [4, 1]
    validate_cluster_tree(estimator.cluster_tree_, estimator.cluster_layers_)
    estimator.fit(np.array([[1.0, 2.0]]))
    assert [len(layer) for layer in estimator] == [1]
    np.testing.assert_array_equal(estimator.cluster_layers_[0].labels, [0])


@pytest.mark.real_clustering
def test_real_plscan_small_blobs():
    random = np.random.default_rng(1729)
    vectors = np.concatenate(
        [
            random.normal(-4, 0.1, (48, 2)),
            random.normal(4, 0.1, (48, 2)),
        ]
    )
    estimator = PLSCANClusterer(reproducible=True).fit(vectors)
    assert estimator.plscan_ is not None
    assert estimator.cluster_layers_
    assert all(len(layer.labels) == 96 for layer in estimator)
    assert any(len(layer) >= 2 for layer in estimator)
    validate_cluster_tree(estimator.cluster_tree_, estimator.cluster_layers_)


@pytest.mark.real_clustering
def test_real_evoc_small_blobs():
    random = np.random.default_rng(1729)
    vectors = np.concatenate(
        [
            random.normal(-4, 0.1, (48, 8)),
            random.normal(4, 0.1, (48, 8)),
        ]
    ).astype(np.float32)
    estimator = EVoCClusterer(
        random_state=31, n_neighbors=8, n_epochs=10, approx_n_clusters=2
    ).fit(vectors)
    assert estimator.evoc_ is not None
    assert len(estimator.cluster_layers_) == 1
    assert len(estimator.cluster_layers_[0].labels) == 96
    assert len(estimator.cluster_layers_[0]) >= 1
    validate_cluster_tree(estimator.cluster_tree_, estimator.cluster_layers_)
