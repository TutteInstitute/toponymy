"""Retain synthetic clustering quality expectations across the API migration.

These cases come from the stable and refactor clustering suites. They measure
agreement with known blob assignments, not the semantic quality of topic names.
Each fit uses exactly 1,000 observations; the largest input is 1,000 by 128.
Run this marked suite serially with numerical thread counts fixed to one. EVoC
uses the adapter's default isolated process, which avoids the old Numba cache
conflict without xfail or replacement clustering kernels.
"""

from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score

from toponymy.clustering import EVoCClusterer, KMeansClusterer, PLSCANClusterer

pytestmark = pytest.mark.real_clustering


def _blobs(*, centers, n_features=2, cluster_std=0.05):
    return make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=centers,
        center_box=(0.0, 1.0),
        cluster_std=cluster_std,
        random_state=0,
    )


def _assert_assigned_ami(labels, ground_truth, minimum):
    """Use the original non-noise metric and reject vacuous assignments."""
    assert labels.shape == ground_truth.shape
    assigned = labels >= 0
    assert assigned.any(), "The density clusterer assigned no observations"
    assert np.unique(labels[assigned]).size >= 2
    assert np.unique(ground_truth[assigned]).size >= 2
    score = adjusted_mutual_info_score(labels[assigned], ground_truth[assigned])
    assert score >= minimum, (
        f"Assigned-point AMI {score:.6f} is below {minimum}; "
        f"assigned {assigned.sum()}/{labels.size} observations"
    )


def _require_evoc():
    # Absence is optional; an installed but broken dependency must fail the fit.
    if find_spec("evoc") is None:
        pytest.skip("EVoC quality checks require the optional toponymy[evoc] extra")


def test_kmeans_layer_quality_matches_original_thresholds():
    """The refactor seed makes the stable 64/16/4 resolution check repeatable."""
    vectors, ground_truth = _blobs(centers=5)
    clusterer = KMeansClusterer(
        min_clusters=4, base_n_clusters=64, random_state=42
    ).fit(vectors)

    assert [len(layer) for layer in clusterer.cluster_layers_] == [64, 16, 4]
    for index, layer in enumerate(clusterer.cluster_layers_):
        # KMeans includes every observation; unlike density clustering, this
        # original expectation must never discard observations as noise.
        assert np.all(layer.labels >= 0)
        score = adjusted_mutual_info_score(layer.labels, ground_truth)
        minimum = 0.25 * (index + 1)
        assert score >= minimum, f"Layer {index} AMI {score:.6f} is below {minimum}"


@pytest.mark.parametrize(
    "centers, options",
    [
        pytest.param(
            5,
            {"min_clusters": 4, "min_samples": 5, "base_min_cluster_size": 10},
            id="stable-explicit-resolution",
        ),
        pytest.param(10, {}, id="refactor-default-resolution"),
    ],
)
def test_plscan_final_layer_quality_matches_original_threshold(centers, options):
    """Preserve both original datasets and their final-layer AMI >= 0.9.

    The stable case explicitly keeps its minimum of four clusters. The refactor
    case exercises current defaults, including min_clusters=1: filtering to four
    here would change which final resolution its original expectation measured.
    """
    vectors, ground_truth = _blobs(centers=centers)
    clusterer = PLSCANClusterer(**options).fit(vectors)

    assert len(clusterer.cluster_layers_) > 1
    _assert_assigned_ami(clusterer.cluster_layers_[-1].labels, ground_truth, 0.9)


def test_evoc_stable_data_quality_with_equivalent_external_options():
    """Translate stable min_clusters to the external option it actually used.

    The stable wrapper passed min_clusters=4 as approx_n_clusters=4, and stored
    next_cluster_size_quantile without using it. Its global NumPy seed of zero
    becomes the explicit random_state=0. Other effective options and the dataset
    are unchanged. The original 5..7 unique-label count includes the noise label.
    """
    _require_evoc()
    vectors, ground_truth = _blobs(centers=5, n_features=128, cluster_std=0.05)
    clusterer = EVoCClusterer(
        approx_n_clusters=4,
        base_min_cluster_size=10,
        min_samples=5,
        random_state=0,
    ).fit(vectors)

    assert clusterer.cluster_layers_
    labels = clusterer.cluster_layers_[-1].labels
    assert 5 <= np.unique(labels).size <= 7
    _assert_assigned_ami(labels, ground_truth, 0.75)


def test_evoc_refactor_default_hierarchy_quality_matches_original_threshold():
    """Preserve native granularity and the refactor's ground-truth AMI threshold."""
    _require_evoc()
    vectors, ground_truth = _blobs(centers=5, n_features=128, cluster_std=0.001)
    clusterer = EVoCClusterer(random_state=42).fit(vectors)

    assert clusterer.cluster_layers_
    final_layer = clusterer.cluster_layers_[-1]
    # EVoC selects granularity: its maintained version can split one true blob.
    # The adapter must retain those clusters rather than truncate native output.
    assert len(final_layer) >= 5
    np.testing.assert_array_equal(
        final_layer.labels, clusterer.evoc_.cluster_layers_[-1]
    )
    assert (
        len(final_layer) == np.unique(final_layer.labels[final_layer.labels >= 0]).size
    )
    _assert_assigned_ami(final_layer.labels, ground_truth, 0.75)
