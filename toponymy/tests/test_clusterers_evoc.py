"""EVoCClusterer against the real EVoC estimator.

These tests run EVoC's compiled kernels in the same process as fast_hdbscan's.
That works from evoc 0.3.2: earlier versions define numba types with the same
names as fast_hdbscan (evoc PR 46), and whichever library runs second fails with
a numba ``TypeError``. Only the tests that run EVoC are skipped for such a
version, or when ``evoc`` is not installed. The stand-in tests of
``test_clusterers.py`` always run.

EVoC's result can differ with the seed, the number of numba threads and the
platform. Quality is therefore asserted only on well-separated blobs, where its
answer was stable across seeds and thread counts; elsewhere only the structure
of the result is checked.
"""

import copy
import inspect
from importlib.metadata import PackageNotFoundError, version
import pickle
import re

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score

from toponymy.clusterers import EVoCClusterer, validate_cluster_tree

try:
    EVOC_VERSION = version("evoc")
except PackageNotFoundError:
    EVOC_VERSION = None

if EVOC_VERSION is None:
    CANNOT_RUN_EVOC = "evoc is not installed"
elif tuple(map(int, re.findall(r"\d+", EVOC_VERSION)[:3])) < (0, 3, 2):
    CANNOT_RUN_EVOC = (
        f"evoc {EVOC_VERSION} shares numba type names with fast_hdbscan; "
        "evoc 0.3.2 fixed this"
    )
else:
    CANNOT_RUN_EVOC = None

requires_evoc = pytest.mark.skipif(EVOC_VERSION is None, reason="evoc is not installed")
runs_alongside_fast_hdbscan = pytest.mark.skipif(
    CANNOT_RUN_EVOC is not None, reason=CANNOT_RUN_EVOC or ""
)

TWO_CLUSTERS = {"approx_n_clusters": 2, "random_state": 31}
TWO_BLOBS_TRUTH = np.repeat([0, 1], 100)


def _two_blobs():
    rng = np.random.default_rng(1729)
    vectors = np.concatenate(
        [rng.normal(-4, 0.1, (100, 16)), rng.normal(4, 0.1, (100, 16))]
    )
    return vectors.astype(np.float32)


def _read_only(vectors):
    vectors = vectors.copy()
    vectors.flags.writeable = False
    return vectors


def _blobs(*, n_features, cluster_std):
    return make_blobs(
        n_samples=1000,
        n_features=n_features,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=cluster_std,
        random_state=0,
    )


def _assert_assigned_ami(labels, ground_truth, minimum):
    """Score only the assigned observations, and reject a vacuous assignment."""
    assert labels.shape == ground_truth.shape
    assigned = labels >= 0
    assert np.unique(labels[assigned]).size >= 2
    assert np.unique(ground_truth[assigned]).size >= 2
    score = adjusted_mutual_info_score(labels[assigned], ground_truth[assigned])
    assert score >= minimum, (
        f"Assigned-point AMI {score:.6f} is below {minimum}; "
        f"assigned {assigned.sum()}/{labels.size} observations"
    )


@requires_evoc
def test_parameters_defaults_and_order_are_those_of_evoc():
    from evoc import EVoC

    def defaults(estimator):
        parameters = inspect.signature(estimator.__init__).parameters
        return [(name, parameter.default) for name, parameter in parameters.items()]

    assert defaults(EVoCClusterer) == defaults(EVoC)


@runs_alongside_fast_hdbscan
def test_two_blobs_are_recovered_and_the_fitted_evoc_is_exposed():
    clusterer = EVoCClusterer(**TWO_CLUSTERS).fit(_two_blobs())

    # Every parameter reached EVoC.
    assert clusterer.evoc_.get_params() == clusterer.get_params()
    (layer,) = clusterer.cluster_layers_
    np.testing.assert_array_equal(layer.labels, clusterer.evoc_.cluster_layers_[0])
    _assert_assigned_ami(layer.labels, TWO_BLOBS_TRUTH, 0.9)
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)


@runs_alongside_fast_hdbscan
@pytest.mark.parametrize(
    "prepare",
    [
        lambda vectors: vectors,
        lambda vectors: vectors.astype(np.float64),
        np.asfortranarray,
        lambda vectors: np.repeat(vectors, 2, axis=1)[:, ::2],
        _read_only,
        # Unit-length float32 input reaches EVoC's kernels without being copied.
        lambda vectors: _read_only(
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        ),
        lambda vectors: (vectors * 10).astype(np.int8),
        lambda vectors: ((vectors + 8) * 10).astype(np.uint8),
        lambda vectors: vectors.tolist(),
    ],
    ids=[
        "float32",
        "float64",
        "fortran-ordered",
        "strided-view",
        "read-only",
        "read-only-unit-length",
        "int8",
        "uint8",
        "nested-lists",
    ],
)
def test_vectors_are_accepted_as_given_and_never_modified(prepare):
    vectors = prepare(_two_blobs())
    before = np.array(vectors, copy=True)
    clusterer = EVoCClusterer(**TWO_CLUSTERS).fit(vectors)

    _assert_assigned_ami(clusterer.cluster_layers_[-1].labels, TWO_BLOBS_TRUTH, 0.9)
    np.testing.assert_array_equal(vectors, before)


@runs_alongside_fast_hdbscan
@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_fitted_clusterer_survives_copying_and_pickling(duplicate):
    clusterer = EVoCClusterer(**TWO_CLUSTERS).fit(_two_blobs())
    duplicated = duplicate(clusterer)

    for layer, original in zip(duplicated.cluster_layers_, clusterer.cluster_layers_):
        np.testing.assert_array_equal(layer.labels, original.labels)
        assert not layer.labels.flags.writeable
    assert duplicated.cluster_tree_ == clusterer.cluster_tree_
    np.testing.assert_array_equal(duplicated.evoc_.labels_, clusterer.evoc_.labels_)


@runs_alongside_fast_hdbscan
def test_blobs_are_clustered_with_the_options_the_legacy_wrapper_used():
    """The data of the legacy ``test_evoc_clusterer_class``.

    The legacy wrapper passed ``min_clusters=4`` to EVoC as
    ``approx_n_clusters=4``, and seeded NumPy with 0 rather than passing a seed.
    EVoC treats the count as a target: for these five blobs it returns four or
    five clusters, depending on the seed.
    """
    vectors, ground_truth = _blobs(n_features=128, cluster_std=0.05)
    clusterer = EVoCClusterer(
        approx_n_clusters=4, base_min_cluster_size=10, min_samples=5, random_state=0
    ).fit(vectors)

    final = clusterer.cluster_layers_[-1]
    assert 4 <= len(final) <= 7
    _assert_assigned_ami(final.labels, ground_truth, 0.75)


@runs_alongside_fast_hdbscan
def test_default_hierarchy_keeps_the_layers_and_ids_evoc_chose():
    # Tightly packed blobs: EVoC assigns few points, and its layers vary with the
    # seed and the thread count. Only the structure is asserted: layers and IDs
    # are kept exactly as EVoC returned them, and not cut down to the true count.
    vectors, _ = _blobs(n_features=128, cluster_std=0.001)
    clusterer = EVoCClusterer(random_state=42).fit(vectors)

    native_layers = clusterer.evoc_.cluster_layers_
    assert len(clusterer.cluster_layers_) == len(native_layers) >= 1
    for layer, native_labels in zip(clusterer.cluster_layers_, native_layers):
        np.testing.assert_array_equal(layer.labels, native_labels)
    assert any(len(layer) for layer in clusterer)
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)


def _fit_evoc(vectors):
    assert EVoCClusterer(**TWO_CLUSTERS).fit(vectors).cluster_layers_


def _fit_fast_hdbscan(vectors):
    from fast_hdbscan import PLSCAN

    assert PLSCAN().fit(vectors).cluster_layers_


@runs_alongside_fast_hdbscan
@pytest.mark.parametrize(
    "first,second",
    [(_fit_evoc, _fit_fast_hdbscan), (_fit_fast_hdbscan, _fit_evoc)],
    ids=["evoc-first", "fast_hdbscan-first"],
)
def test_evoc_and_fast_hdbscan_run_in_one_process_in_either_order(first, second):
    """Fit one library, then the other, then the first again, in this process.

    Other tests share the process, so the library used first is not controlled;
    the three fits cover both transitions between the libraries. With evoc
    before 0.3.2 the library that runs second raises a numba ``TypeError``.
    """
    vectors = _two_blobs()
    first(vectors)
    second(vectors)
    first(vectors)
