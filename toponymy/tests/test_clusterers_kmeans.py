"""Tests for KMeansClusterer.

Most tests use a stand-in for scikit-learn's ``KMeans`` that records how it is
called. The others run the real estimator on small or well-separated data.
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_mutual_info_score

from toponymy import clusterers
from toponymy.clusterers import KMeansClusterer, validate_cluster_tree


@pytest.fixture
def kmeans_double(monkeypatch):
    """Install a stand-in for ``KMeans`` and return its class.

    A fit labels observation ``i`` with ``i % n_clusters``, unless ``labels``
    maps the cluster count to other labels. The class lists the instances
    created; set ``fail_at`` to the index of a fit to make that fit raise.
    """

    class KMeansDouble:
        instances = []
        labels = {}
        fail_at = None

        def __init__(self, **options):
            self.options = options
            type(self).instances.append(self)

        def fit_predict(self, vectors):
            self.vectors = vectors
            if self.instances.index(self) == self.fail_at:
                raise RuntimeError("KMeans failed")
            n_clusters = self.options["n_clusters"]
            return self.labels.get(n_clusters, np.arange(len(vectors)) % n_clusters)

    monkeypatch.setattr(clusterers, "KMeans", KMeansDouble)
    return KMeansDouble


@pytest.mark.parametrize(
    "n_observations,options,n_clusters",
    [
        (1000, {"base_n_clusters": 64, "min_clusters": 4}, [64, 16, 4]),
        (1000, {}, [1000, 250, 62, 15]),
        (20, {"base_n_clusters": 64, "min_clusters": 4}, [20, 5]),
        (3, {"base_n_clusters": 16, "min_clusters": 6}, [3]),
        (1, {"base_n_clusters": 16, "min_clusters": 1}, [1]),
        (100, {"base_n_clusters": 4, "min_clusters": 1}, [4, 1]),
        (100, {"base_n_clusters": 5, "min_clusters": 5}, [5]),
    ],
    ids=[
        "64-16-4",
        "defaults-capped",
        "capped",
        "fewer-than-min-clusters",
        "single",
        "down-to-one",
        "one-layer",
    ],
)
def test_each_layer_asks_for_a_quarter_of_the_clusters_below_it(
    kmeans_double, n_observations, options, n_clusters
):
    vectors = np.ones((n_observations, 2))
    clusterer = KMeansClusterer(**options, random_state=7, verbose=True)
    assert clusterer.fit(vectors) is clusterer

    assert [len(layer) for layer in clusterer] == n_clusters
    assert [kmeans.options for kmeans in kmeans_double.instances] == [
        {"n_clusters": n, "random_state": 7, "verbose": True} for n in n_clusters
    ]
    assert all(kmeans.vectors is vectors for kmeans in kmeans_double.instances)


def test_no_observations_give_no_layers(kmeans_double):
    clusterer = KMeansClusterer().fit(np.ones((0, 2)))
    assert clusterer.cluster_layers_ == []
    assert clusterer.cluster_tree_ == {}
    assert not kmeans_double.instances


def test_crossing_resolutions_are_joined_by_containment(kmeans_double):
    kmeans_double.labels = {
        8: np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]),
        2: np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    }
    clusterer = KMeansClusterer(min_clusters=2, base_n_clusters=8)
    clusterer.fit(np.ones((12, 2)))

    # Cluster 0 of the finest layer straddles both coarse clusters, so it
    # attaches to the root rather than to the first cluster it overlaps.
    assert clusterer.cluster_tree_ == {
        (1, 0): [(0, 1), (0, 2)],
        (1, 1): [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
        (2, 0): [(0, 0), (1, 0), (1, 1)],
    }
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)


def test_refitting_replaces_every_layer(kmeans_double):
    clusterer = KMeansClusterer(min_clusters=4, base_n_clusters=64)
    assert len(clusterer.fit(np.ones((1000, 2))).cluster_layers_) == 3
    assert [len(layer) for layer in clusterer.fit(np.ones((2, 2)))] == [2]
    assert clusterer.fit(np.ones((0, 2))).cluster_layers_ == []
    assert clusterer.cluster_tree_ == {}


@pytest.mark.parametrize(
    "options",
    [
        {"min_clusters": 0},
        {"base_n_clusters": 0},
        {"min_clusters": -1},
        {"min_clusters": True},
        {"base_n_clusters": 64.0},
        {"base_n_clusters": None},
        {"min_clusters": 5, "base_n_clusters": 4},
    ],
)
def test_invalid_parameters_are_rejected_even_without_observations(
    kmeans_double, options
):
    for n_observations in (0, 10):
        with pytest.raises(ValueError):
            KMeansClusterer(**options).fit(np.ones((n_observations, 2)))
    assert not kmeans_double.instances


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 0)),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[1 + 0j]]),
        np.array([[True]]),
        np.array([[0]], dtype=object),
        sparse.eye(3),
    ],
)
def test_invalid_vectors_are_rejected_before_kmeans_runs(kmeans_double, vectors):
    with pytest.raises(ValueError):
        KMeansClusterer().fit(vectors)
    assert not kmeans_double.instances


def test_failed_or_rejected_fits_keep_the_previous_fit(kmeans_double):
    clusterer = KMeansClusterer(min_clusters=4, base_n_clusters=64)
    clusterer.fit(np.ones((100, 2)))
    before = clusterer.cluster_layers_, clusterer.cluster_tree_

    # KMeans fails on the second resolution, after the first one succeeded.
    kmeans_double.fail_at = len(kmeans_double.instances) + 1
    with pytest.raises(RuntimeError, match="KMeans failed"):
        clusterer.fit(np.ones((100, 2)))
    with pytest.raises(ValueError):
        clusterer.fit(np.full((100, 2), np.nan))
    with pytest.raises(ValueError):
        clusterer.set_params(min_clusters=0).fit(np.ones((100, 2)))

    after = clusterer.cluster_layers_, clusterer.cluster_tree_
    assert all(old is new for old, new in zip(before, after))


def test_parameters_cloning_and_fit_predict(kmeans_double):
    options = {
        "min_clusters": 2,
        "base_n_clusters": np.int64(8),
        "random_state": np.random.RandomState(0),
        "verbose": True,
    }
    clusterer = KMeansClusterer(**options)
    assert clusterer.get_params() == options
    with pytest.raises(NotFittedError):
        iter(clusterer)

    layers, tree = clusterer.fit_predict(np.ones((10, 2)))
    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_
    cloned = clone(clusterer)
    assert not hasattr(cloned, "cluster_layers_")
    assert cloned.get_params().keys() == options.keys()


def _blobs():
    # The data of the legacy test_kmeans_clusterer_class.
    return make_blobs(
        n_samples=1000,
        n_features=2,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=0,
    )


def test_blobs_are_clustered_at_64_16_and_4_clusters():
    vectors, ground_truth = _blobs()
    before = vectors.copy()
    clusterer = KMeansClusterer(min_clusters=4, base_n_clusters=64, random_state=42)
    clusterer.fit(vectors)

    assert [len(layer) for layer in clusterer] == [64, 16, 4]
    for index, layer in enumerate(clusterer):
        assert np.all(layer.labels >= 0)
        # The legacy thresholds. Five blobs split into 64, 16 or 4 clusters
        # cannot match exactly; every seed tried scored about 0.49, 0.64, 0.82.
        score = adjusted_mutual_info_score(layer.labels, ground_truth)
        assert score >= 0.25 * (index + 1)
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)
    np.testing.assert_array_equal(vectors, before)


def test_a_random_state_makes_fits_repeatable():
    vectors, _ = _blobs()
    first = KMeansClusterer(min_clusters=4, base_n_clusters=64, random_state=3)
    second = clone(first)
    for layer, other in zip(first.fit(vectors), second.fit(vectors), strict=True):
        np.testing.assert_array_equal(layer.labels, other.labels)


def test_small_inputs_and_an_invalid_random_state():
    vectors = np.array([[0.0, 0.1], [0.1, 0.0], [8.0, 8.1], [8.1, 8.0]])
    clusterer = KMeansClusterer(min_clusters=1, base_n_clusters=16, random_state=7)
    assert [len(layer) for layer in clusterer.fit(vectors)] == [4, 1]
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)

    clusterer.fit(np.array([[1.0, 2.0]]))
    assert [layer.labels.tolist() for layer in clusterer] == [[0]]

    # scikit-learn's own errors reach the caller, and the previous fit stays.
    before = clusterer.cluster_layers_
    with pytest.raises(ValueError, match="random_state"):
        clusterer.set_params(random_state="seed").fit(vectors)
    assert clusterer.cluster_layers_ is before
