"""Tests for ToponymyClusterer, PLSCANClusterer under the legacy name.

PLSCANClusterer itself is tested in test_clusterers_plscan.py. These tests
check that ToponymyClusterer takes the same parameters, fits in the same way,
and leaves the legacy class alone.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs

import toponymy
from toponymy import clustering
from toponymy.clusterers import (
    Clusterer,
    PLSCANClusterer,
    ToponymyClusterer,
    validate_cluster_tree,
)


def _blobs():
    # Three well-separated groups of three blobs each.
    centers = [
        (x + dx, y + dy)
        for x, y in ((0, 0), (30, 0), (15, 26))
        for dx, dy in ((0, 0), (5, 0), (2.5, 4))
    ]
    vectors, _ = make_blobs(
        n_samples=600, centers=centers, cluster_std=0.8, random_state=0
    )
    return vectors


def test_it_is_plscan_clusterer_with_the_same_parameters():
    assert issubclass(ToponymyClusterer, PLSCANClusterer)
    assert ToponymyClusterer.__init__ is PLSCANClusterer.__init__
    assert ToponymyClusterer().get_params() == PLSCANClusterer().get_params()
    # The options of the legacy algorithm are not accepted.
    with pytest.raises(TypeError):
        ToponymyClusterer(next_cluster_size_quantile=0.85)


def test_cloning_keeps_the_class_and_the_parameters():
    clusterer = ToponymyClusterer(min_clusters=2, reproducible=True)
    cloned = clone(clusterer).set_params(max_layers=4)

    assert type(cloned) is ToponymyClusterer
    assert repr(cloned).startswith("ToponymyClusterer(")
    assert cloned.get_params() == {**clusterer.get_params(), "max_layers": 4}


def test_fit_and_fit_predict_follow_the_clusterer_contract_without_warnings():
    clusterer = ToponymyClusterer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert clusterer.fit(np.ones((1, 2))) is clusterer
        layers, tree = clusterer.fit_predict(np.ones((1, 2)))

    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_


@pytest.mark.parametrize("method", ["fit", "fit_predict"])
def test_it_fits_like_plscan_clusterer_with_the_same_options(method):
    vectors = _blobs()
    options = {"min_clusters": 4, "reproducible": True}
    expected = PLSCANClusterer(**options).fit(vectors)
    clusterer = ToponymyClusterer(**options)
    getattr(clusterer, method)(vectors)

    # min_clusters drops some of PLSCAN's layers, so the metadata is filtered.
    n_native = len(clusterer.plscan_.cluster_layers_)
    assert 0 < len(clusterer.cluster_layers_) < n_native
    assert clusterer.plscan_.get_params() == expected.plscan_.get_params()
    for layer, other in zip(clusterer, expected, strict=True):
        np.testing.assert_array_equal(layer.labels, other.labels)
    assert clusterer.cluster_tree_ == expected.cluster_tree_
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)
    for strengths, other in zip(
        clusterer.cluster_probabilities_,
        expected.cluster_probabilities_,
        strict=True,
    ):
        np.testing.assert_array_equal(strengths, other)
    assert clusterer.cluster_persistence_scores_ == expected.cluster_persistence_scores_
    np.testing.assert_array_equal(
        clusterer.plscan_min_cluster_sizes_, expected.plscan_min_cluster_sizes_
    )


def test_the_legacy_toponymy_clusterer_is_unchanged():
    legacy = clustering.ToponymyClusterer
    assert toponymy.ToponymyClusterer is legacy
    assert legacy is not ToponymyClusterer
    assert not issubclass(legacy, Clusterer)
