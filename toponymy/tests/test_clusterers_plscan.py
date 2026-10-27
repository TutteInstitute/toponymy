"""Tests for PLSCANClusterer.

Most tests use a stand-in for ``fast_hdbscan.PLSCAN`` that records how it is
called, and returns layers whose metadata differs from layer to layer. The
tests at the end run the real estimator on well-separated blobs, where its
result was the same for every numba thread count tried.
"""

import inspect

import fast_hdbscan
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph

from toponymy import clusterers
from toponymy.clusterers import PLSCANClusterer, validate_cluster_tree

# The stand-in's layers, with 3, 1 and 2 clusters, and the metadata of each.
# Like PLSCAN, it scores its base layer 0.0 and the others with float32.
NATIVE_LAYERS = [
    np.array([4, 4, 19, 19, 81, -1]),
    np.array([7, 7, 7, 7, -1, -1]),
    np.array([0, 0, 1, 1, 1, -1]),
]
NATIVE_STRENGTHS = [np.full(6, 0.25), np.full(6, 0.5), np.full(6, 0.75)]
NATIVE_SCORES = [0.0, np.float32(1.5), np.float32(2.5)]


@pytest.fixture
def plscan_double(monkeypatch):
    """Install a stand-in for ``fast_hdbscan.PLSCAN`` and return its class.

    The class lists the instances created. Set ``layers``, ``strengths`` or
    ``scores`` on it to change what a fit returns, ``omit`` to leave fitted
    attributes out, or ``error`` to make a fit raise.
    """

    class PLSCANDouble:
        instances = []
        layers = NATIVE_LAYERS
        strengths = NATIVE_STRENGTHS
        scores = NATIVE_SCORES
        omit = ()
        error = None

        def __init__(self, **options):
            self.options = options
            type(self).instances.append(self)

        def fit(self, vectors):
            self.fitted_on = vectors
            if self.error is not None:
                raise self.error
            self.cluster_layers_ = [layer.copy() for layer in self.layers]
            self.membership_strength_layers_ = [s.copy() for s in self.strengths]
            self.layer_persistence_scores_ = list(self.scores)
            self.min_cluster_sizes_ = np.array([5.0, 8.0, 13.0])
            # PLSCAN's own tree, which PLSCANClusterer must not use.
            self.cluster_tree_ = {(1, 7): [(0, 4), (0, 19), (0, 81)]}
            for name in self.omit:
                delattr(self, name)
            return self

    monkeypatch.setattr(clusterers, "PLSCAN", PLSCANDouble)
    return PLSCANDouble


def _vectors(n_observations=6):
    return np.ones((n_observations, 2))


def _fitted(plscan_double):
    # PLSCAN is created for every input, to check its parameters, but fitted
    # only on enough observations.
    return [
        plscan for plscan in plscan_double.instances if hasattr(plscan, "fitted_on")
    ]


def _fitted_state(clusterer):
    return (
        clusterer.cluster_layers_,
        clusterer.cluster_tree_,
        clusterer.cluster_probabilities_,
        clusterer.cluster_persistence_scores_,
        clusterer.plscan_min_cluster_sizes_,
        clusterer.plscan_,
    )


def test_every_option_but_min_clusters_reaches_plscan(plscan_double):
    vectors = _vectors()
    cannot_link = sparse.csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(6, 6))
    clusterer = PLSCANClusterer(
        min_clusters=2,
        max_layers=4,
        algorithm="kruskal",
        cannot_link=cannot_link,
        metric_kwds={},
        verbose=True,
    )
    assert clusterer.fit(vectors) is clusterer

    (plscan,) = plscan_double.instances
    expected = clusterer.get_params()
    del expected["min_clusters"]
    assert plscan.options == expected
    assert plscan.fitted_on is vectors
    assert clusterer.plscan_ is plscan


def test_layers_keep_plscan_ids_and_the_tree_uses_containment(plscan_double):
    clusterer = PLSCANClusterer().fit(_vectors())

    assert [[cluster.label for cluster in layer] for layer in clusterer] == [
        [4, 19, 81],
        [7],
        [0, 1],
    ]
    # Cluster 81 is noise in layer 1, so it attaches to layer 2, although
    # PLSCAN's own tree puts it under cluster 7.
    assert clusterer.cluster_tree_ == {
        (1, 7): [(0, 4), (0, 19)],
        (2, 1): [(0, 81)],
        (3, 0): [(1, 7), (2, 0), (2, 1)],
    }
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)


@pytest.mark.parametrize("min_clusters", [1, 2, 3])
@pytest.mark.parametrize(
    "order", [(0, 1, 2), (1, 0, 2), (1, 2, 0)], ids=["0-1-2", "1-0-2", "1-2-0"]
)
def test_each_kept_layer_has_the_metadata_of_the_same_plscan_layer(
    plscan_double, order, min_clusters
):
    plscan_double.layers = [NATIVE_LAYERS[i] for i in order]
    plscan_double.strengths = [NATIVE_STRENGTHS[i] for i in order]
    plscan_double.scores = [NATIVE_SCORES[i] for i in order]
    clusterer = PLSCANClusterer(min_clusters=min_clusters).fit(_vectors())

    n_clusters = [(3, 1, 2)[i] for i in order]
    assert [len(layer) for layer in clusterer] == [
        n for n in n_clusters if n >= min_clusters
    ]
    metadata = {
        tuple(labels): (strengths.tolist(), score)
        for labels, strengths, score in zip(
            NATIVE_LAYERS, NATIVE_STRENGTHS, NATIVE_SCORES
        )
    }
    for layer, probabilities, score in zip(
        clusterer,
        clusterer.cluster_probabilities_,
        clusterer.cluster_persistence_scores_,
        strict=True,
    ):
        assert (probabilities.tolist(), score) == metadata[tuple(layer.labels)]


@pytest.mark.parametrize(
    "native",
    [{}, {"layers": [], "strengths": [], "scores": []}],
    ids=["every-layer-dropped", "no-layers"],
)
def test_without_a_kept_layer_the_result_is_noise_without_metadata(
    plscan_double, native
):
    for name, value in native.items():
        setattr(plscan_double, name, value)
    clusterer = PLSCANClusterer(min_clusters=4).fit(_vectors())

    assert [layer.labels.tolist() for layer in clusterer] == [[-1] * 6]
    assert clusterer.cluster_tree_ == {}
    # The layer of noise is not one of PLSCAN's, so it has no PLSCAN metadata.
    assert clusterer.cluster_probabilities_ == []
    assert clusterer.cluster_persistence_scores_ == []
    (plscan,) = plscan_double.instances
    assert clusterer.plscan_ is plscan
    np.testing.assert_array_equal(
        clusterer.plscan_min_cluster_sizes_, plscan.min_cluster_sizes_
    )


def test_fitted_metadata_does_not_share_memory_with_plscan(plscan_double):
    clusterer = PLSCANClusterer().fit(_vectors())
    plscan = clusterer.plscan_
    for array in [
        *plscan.cluster_layers_,
        *plscan.membership_strength_layers_,
        plscan.min_cluster_sizes_,
    ]:
        array[...] = -1

    for layer, labels in zip(clusterer, NATIVE_LAYERS, strict=True):
        np.testing.assert_array_equal(layer.labels, labels)
    for probabilities, strengths in zip(
        clusterer.cluster_probabilities_, NATIVE_STRENGTHS, strict=True
    ):
        np.testing.assert_array_equal(probabilities, strengths)
    np.testing.assert_array_equal(clusterer.plscan_min_cluster_sizes_, [5, 8, 13])
    assert all(type(score) is float for score in clusterer.cluster_persistence_scores_)


@pytest.mark.parametrize(
    "options,runs_plscan",
    [
        ({}, True),
        ({"min_samples": 6}, True),
        ({"min_samples": 7}, False),
        ({"base_min_cluster_size": 6}, True),
        ({"base_min_cluster_size": 7}, False),
    ],
)
def test_plscan_runs_only_on_enough_observations(plscan_double, options, runs_plscan):
    clusterer = PLSCANClusterer(**options).fit(_vectors())
    assert bool(_fitted(plscan_double)) is runs_plscan
    assert (clusterer.plscan_ is not None) is runs_plscan
    if not runs_plscan:
        assert [layer.labels.tolist() for layer in clusterer] == [[-1] * 6]
        assert _fitted_state(clusterer)[1:] == ({}, [], [], None, None)


@pytest.mark.parametrize(
    "vectors,metric",
    [
        (_vectors(0), "euclidean"),
        (_vectors(1), "euclidean"),
        (sparse.csr_matrix((0, 0)), "precomputed"),
        (sparse.csr_matrix((1, 1)), "precomputed"),
    ],
    ids=["empty", "single", "empty-graph", "single-graph"],
)
def test_empty_and_single_inputs_do_not_run_plscan(plscan_double, vectors, metric):
    clusterer = clone(PLSCANClusterer(metric=metric))
    assert clusterer.fit(vectors) is clusterer

    expected = [[-1]] if vectors.shape[0] else []
    assert [layer.labels.tolist() for layer in clusterer] == expected
    assert _fitted_state(clusterer)[1:] == ({}, [], [], None, None)
    assert not _fitted(plscan_double)


def test_refitting_replaces_the_layers_and_all_their_metadata(plscan_double):
    clusterer = PLSCANClusterer().fit(_vectors())
    assert len(clusterer.cluster_probabilities_) == 3

    clusterer.fit(_vectors(0))
    assert _fitted_state(clusterer) == ([], {}, [], [], None, None)

    clusterer.set_params(min_clusters=2).fit(_vectors())
    assert [len(layer) for layer in clusterer] == [3, 2]
    assert clusterer.cluster_persistence_scores_ == [0.0, 2.5]
    assert clusterer.plscan_ is _fitted(plscan_double)[-1]

    clusterer.set_params(min_clusters=4).fit(_vectors())
    assert [len(layer) for layer in clusterer] == [0]
    assert clusterer.cluster_probabilities_ == []
    assert clusterer.cluster_persistence_scores_ == []
    assert clusterer.plscan_ is _fitted(plscan_double)[-1]
    assert len(_fitted(plscan_double)) == 3


@pytest.mark.parametrize(
    "attribute,value,error",
    [
        ("error", RuntimeError("PLSCAN failed"), RuntimeError),
        ("layers", [layer + 0.5 for layer in NATIVE_LAYERS], ValueError),
        ("strengths", NATIVE_STRENGTHS[:2], ValueError),
        ("omit", ["membership_strength_layers_"], AttributeError),
        ("omit", ["min_cluster_sizes_"], AttributeError),
    ],
    ids=[
        "plscan-fails",
        "non-integer-labels",
        "fewer-strengths-than-layers",
        "no-strengths",
        "no-min-cluster-sizes",
    ],
)
def test_failed_fit_propagates_and_keeps_the_previous_fit(
    plscan_double, attribute, value, error
):
    clusterer = PLSCANClusterer().fit(_vectors())
    before = _fitted_state(clusterer)

    setattr(plscan_double, attribute, value)
    with pytest.raises(error) as caught:
        clusterer.fit(_vectors())

    if attribute == "error":
        assert caught.value is value
    assert all(old is new for old, new in zip(before, _fitted_state(clusterer)))


def test_rejected_input_and_parameters_keep_the_previous_fit(plscan_double):
    clusterer = PLSCANClusterer().fit(_vectors())
    before = _fitted_state(clusterer)

    with pytest.raises(ValueError):
        clusterer.fit(np.full((6, 2), np.nan))
    with pytest.raises(ValueError, match="sparse distance graph"):
        clusterer.set_params(metric="precomputed").fit(_vectors())
    with pytest.raises(ValueError, match="max_layers"):
        clusterer.set_params(metric="euclidean", max_layers=0).fit(_vectors())

    assert all(old is new for old, new in zip(before, _fitted_state(clusterer)))
    assert len(plscan_double.instances) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"min_clusters": 0},
        {"min_clusters": True},
        {"min_samples": 0},
        {"min_samples": 2.0},
        {"base_min_cluster_size": 1},
        {"max_layers": 0},
        {"max_layers": None},
        {"max_layers": 1.5},
    ],
)
def test_invalid_parameters_are_not_masked_by_empty_data(plscan_double, options):
    with pytest.raises(ValueError):
        PLSCANClusterer(**options).fit(_vectors(0))
    assert not plscan_double.instances


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 0)),
        np.zeros((1, 2, 3)),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[1 + 0j]]),
        np.array([[True]]),
        np.array([["1"]]),
        sparse.eye(6),
    ],
)
def test_invalid_vectors_are_rejected_before_plscan_runs(plscan_double, vectors):
    with pytest.raises(ValueError):
        PLSCANClusterer().fit(vectors)
    assert not plscan_double.instances


@pytest.mark.parametrize(
    "graph",
    [
        np.zeros((6, 6)),
        sparse.csr_matrix(np.ones((6, 5))),
        sparse.csr_matrix([[0.0, -1.0], [1.0, 0.0]]),
        sparse.csr_matrix([[0.0, np.nan], [1.0, 0.0]]),
        sparse.csr_matrix([[0.0, np.inf], [1.0, 0.0]]),
        sparse.csr_matrix([[0, 1j], [1, 0]]),
        sparse.csr_matrix(np.eye(6, dtype=bool)),
    ],
    ids=["dense", "not-square", "negative", "nan", "infinite", "complex", "boolean"],
)
def test_invalid_distance_graphs_are_rejected_before_plscan_runs(plscan_double, graph):
    with pytest.raises(ValueError):
        PLSCANClusterer(metric="precomputed").fit(graph)
    assert not plscan_double.instances


# Distances along a chain, with an explicitly stored zero distance.
GRAPH = sparse.csr_matrix(
    ([0.0, 1.0, 2.0, 2.0, 1.0, 0.5], ([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 5])),
    shape=(6, 6),
)


@pytest.mark.parametrize(
    "convert",
    [
        sparse.csr_matrix,
        sparse.csr_array,
        sparse.coo_matrix,
        sparse.csc_array,
        sparse.lil_matrix,
        sparse.dok_matrix,
    ],
    ids=["csr", "csr-array", "coo", "csc-array", "lil", "dok"],
)
def test_distance_graphs_reach_plscan_as_given(plscan_double, convert):
    graph = convert(GRAPH)
    before = graph.copy()
    PLSCANClusterer(metric="precomputed").fit(graph)

    assert plscan_double.instances[0].fitted_on is graph
    # Comparing values cannot see a stored zero that was dropped; nnz can.
    assert graph.nnz == before.nnz
    assert (graph != before).nnz == 0


def test_distance_graphs_are_never_made_dense(plscan_double):
    # As a dense array, this graph would need 8 TB.
    graph = sparse.csr_matrix(([1.0, 2.0], ([0, 1], [1, 2])), shape=(10**6, 10**6))
    PLSCANClusterer(metric="precomputed").fit(graph)
    assert plscan_double.instances[0].fitted_on is graph


@pytest.mark.parametrize(
    "vectors,copied",
    [
        (np.arange(12.0).reshape(6, 2), False),
        (np.arange(12, dtype=np.float32).reshape(6, 2), False),
        (np.asfortranarray(np.arange(12.0).reshape(6, 2)), True),
        (np.arange(24.0).reshape(6, 4)[:, ::2], True),
    ],
    ids=["float64", "float32", "fortran-ordered", "strided"],
)
def test_vectors_reach_plscan_c_contiguous(plscan_double, vectors, copied):
    PLSCANClusterer().fit(vectors)

    received = plscan_double.instances[0].fitted_on
    assert received.flags.c_contiguous
    np.testing.assert_array_equal(received, vectors)
    assert (received is not vectors) is copied


def test_parameters_cloning_and_fit_predict(plscan_double):
    options = {
        "min_clusters": 2,
        "min_samples": np.int64(3),
        "max_layers": 4,
        "base_min_cluster_size": 6,
        "base_n_clusters": 30,
        "layer_similarity_threshold": 0.4,
        "reproducible": True,
        "metric": "manhattan",
        "algorithm": "kruskal",
        "knn_k": 12,
        "cannot_link": None,
        "validate_cannot_link": False,
        "metric_kwds": {"p": 3},
        "verbose": True,
    }
    assert options.keys() == PLSCANClusterer().get_params().keys()
    clusterer = PLSCANClusterer(**options)
    assert all(getattr(clusterer, name) is value for name, value in options.items())
    with pytest.raises(NotFittedError):
        iter(clusterer)

    layers, tree = clusterer.fit_predict(_vectors())
    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_
    cloned = clone(clusterer)
    assert cloned.get_params() == options
    assert not hasattr(cloned, "cluster_layers_")


def test_parameters_and_defaults_are_those_of_plscan_plus_min_clusters():
    def defaults(estimator):
        parameters = inspect.signature(estimator.__init__).parameters
        return {name: parameter.default for name, parameter in parameters.items()}

    expected = {"self": inspect.Parameter.empty, "min_clusters": 1}
    expected.update(defaults(fast_hdbscan.PLSCAN))
    assert defaults(PLSCANClusterer) == expected


@pytest.mark.parametrize(
    "options",
    [
        {"algorithm": "prim"},
        {"base_n_clusters": 0},
        {"cannot_link": sparse.eye(2, format="csr")},
    ],
    ids=["algorithm", "base-n-clusters", "cannot-link-without-kruskal"],
)
def test_plscan_checks_its_own_parameters_even_on_tiny_inputs(options):
    with pytest.raises(ValueError):
        PLSCANClusterer(**options).fit(_vectors(1))


def _five_blobs():
    # The data of the legacy test_plscan_clusterer_fit_returns_self.
    return make_blobs(
        n_samples=1000,
        n_features=2,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=0,
    )


def _two_blobs():
    rng = np.random.default_rng(1729)
    vectors = np.concatenate(
        [rng.normal(-4, 0.1, (100, 16)), rng.normal(4, 0.1, (100, 16))]
    )
    return vectors.astype(np.float32), np.repeat([0, 1], 100)


def _assigned_ami(labels, ground_truth):
    assigned = labels >= 0
    assert np.unique(labels[assigned]).size >= 2
    return adjusted_mutual_info_score(labels[assigned], ground_truth[assigned])


def test_blobs_are_clustered_with_the_options_the_legacy_test_used():
    vectors, ground_truth = _five_blobs()
    before = vectors.copy()
    clusterer = PLSCANClusterer(min_clusters=4, min_samples=5, base_min_cluster_size=10)
    clusterer.fit(vectors)

    native = clusterer.plscan_
    kept = [
        i
        for i, labels in enumerate(native.cluster_layers_)
        if np.unique(labels[labels >= 0]).size >= 4
    ]
    assert len(clusterer.cluster_layers_) == len(kept) > 1
    for layer, probabilities, score, i in zip(
        clusterer,
        clusterer.cluster_probabilities_,
        clusterer.cluster_persistence_scores_,
        kept,
        strict=True,
    ):
        np.testing.assert_array_equal(layer.labels, native.cluster_layers_[i])
        np.testing.assert_array_equal(
            probabilities, native.membership_strength_layers_[i]
        )
        assert score == native.layer_persistence_scores_[i]
    np.testing.assert_array_equal(
        clusterer.plscan_min_cluster_sizes_, native.min_cluster_sizes_
    )
    # The legacy threshold; the final layer scored 0.929 with 1 to 16 threads.
    assert _assigned_ami(clusterer.cluster_layers_[-1].labels, ground_truth) >= 0.9
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)
    np.testing.assert_array_equal(vectors, before)


def test_two_blobs_are_recovered_with_the_default_options():
    vectors, ground_truth = _two_blobs()
    clusterer = PLSCANClusterer().fit(vectors)

    assert _assigned_ami(clusterer.cluster_layers_[-1].labels, ground_truth) >= 0.9
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)


def test_two_blobs_are_recovered_from_a_sparse_distance_graph():
    vectors, ground_truth = _two_blobs()
    graph = kneighbors_graph(vectors, 15, mode="distance")
    graph = graph.maximum(graph.T).tocsr()
    before = graph.copy()
    clusterer = PLSCANClusterer(metric="precomputed").fit(graph)

    assert _assigned_ami(clusterer.cluster_layers_[-1].labels, ground_truth) >= 0.9
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)
    assert (graph != before).nnz == 0


def test_fortran_ordered_vectors_are_clustered_like_c_ordered_ones():
    # fast_hdbscan 0.3.2 raises a numba TypeError for Fortran-ordered arrays.
    vectors, _ = _two_blobs()
    expected = PLSCANClusterer(reproducible=True).fit(vectors)
    clusterer = PLSCANClusterer(reproducible=True).fit(np.asfortranarray(vectors))
    for layer, other in zip(clusterer, expected, strict=True):
        np.testing.assert_array_equal(layer.labels, other.labels)


def test_dropping_every_layer_leaves_noise_and_the_fitted_plscan():
    vectors, _ = _two_blobs()
    clusterer = PLSCANClusterer(min_clusters=3).fit(vectors)

    assert [len(layer) for layer in clusterer] == [0]
    assert clusterer.cluster_probabilities_ == []
    assert clusterer.cluster_persistence_scores_ == []
    native_layers = clusterer.plscan_.cluster_layers_
    assert native_layers
    assert all(np.unique(labels[labels >= 0]).size < 3 for labels in native_layers)
