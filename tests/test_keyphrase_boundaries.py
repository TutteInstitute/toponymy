import numpy as np
import pytest
from scipy import sparse

from toponymy.keyphrases import (
    KeyphraseBuilder,
    central_keyphrases,
    subset_matrix_and_class_labels,
)
from toponymy.clustering import PrecomputedClusterer
from toponymy.feature_extraction import TextKeyphraseExtractor
from toponymy.exemplar_texts import diverse_exemplars, submodular_selection_exemplars


class Embedder:
    def __init__(self):
        self.calls = []
        self.response = None

    def encode(self, names, **options):
        self.calls.append(tuple(names))
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is not None:
            return self.response
        return np.ones((len(names), 2))


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("offline"),
        np.ones((1, 2)),
        np.full((3, 2), np.inf),
        np.ones((3, 0)),
    ],
)
def test_failed_builder_refit_exposes_no_old_or_partial_fitted_state(failure):
    embedder = Embedder()
    builder = KeyphraseBuilder(
        ngram_range=(1, 1), min_occurrences=1, n_jobs=1, embedder=embedder
    ).fit(["apple banana"])
    assert builder.keyphrase_vectors_.shape == (2, 2)
    embedder.response = failure
    with pytest.raises((ValueError, RuntimeError)):
        builder.fit(["carrot dates elderberry"])
    assert not any(
        hasattr(builder, name)
        for name in (
            "object_x_keyphrase_matrix_",
            "keyphrase_list_",
            "keyphrase_vectors_",
        )
    )


def test_failed_converter_clears_previous_builder_state():
    builder = KeyphraseBuilder(ngram_range=(1, 1), min_occurrences=1, n_jobs=1).fit(
        ["apple"]
    )
    builder.object_to_text = lambda obj: 42
    with pytest.raises(TypeError, match="returning strings"):
        builder.fit([object()])
    assert not hasattr(builder, "keyphrase_list_")


def test_empty_builder_is_explicit_and_does_no_encoding():
    embedder = Embedder()
    builder = KeyphraseBuilder(embedder=embedder)
    counts, names, vectors = builder.fit_transform([])
    assert sparse.isspmatrix_csr(counts) and counts.shape == (0, 0)
    assert names == [] and vectors is None and embedder.calls == []


def test_central_keyphrases_use_the_same_center_for_ranking_and_diversification():
    angles = np.deg2rad([0.0, 1.0, 30.0])
    positive = np.column_stack((np.cos(angles), np.sin(angles)))
    vectors = np.vstack((positive, -positive))
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    counts = sparse.diags([10, 1, 1, 10, 1, 1], format="csr")
    labels = np.array([0, 0, 0, 1, 1, 1])
    for offset in (np.zeros(2), np.array([100.0, 80.0])):
        actual = central_keyphrases(
            labels,
            counts,
            names,
            vectors + offset,
            None,
            n_keyphrases=2,
            diversify_alpha=1.0,
        )
        assert actual == [["bravo", "charlie"], ["echo", "foxtrot"]]


def test_missing_used_keyphrases_encode_once_before_global_center():
    embedder = Embedder()
    vectors = np.zeros((3, 2))
    counts = sparse.csr_matrix([[1, 1, 0], [0, 1, 0]])
    central_keyphrases(
        np.array([0, 1]),
        counts,
        ["apple", "banana", "unused"],
        vectors,
        embedder,
        n_keyphrases=2,
    )
    assert embedder.calls == [("apple", "banana")]
    np.testing.assert_array_equal(vectors, [[1, 1], [1, 1], [0, 0]])


def test_on_demand_embedding_cannot_silently_underflow_the_vector_table():
    embedder = Embedder()
    embedder.response = np.array([[1e-300, 0.0]])
    vectors = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype"):
        central_keyphrases(
            np.array([0]),
            sparse.csr_matrix([[1]]),
            ["tiny"],
            vectors,
            embedder,
            n_keyphrases=1,
        )
    np.testing.assert_array_equal(vectors, np.zeros((1, 2)))


def test_finite_large_counts_keep_their_weighted_direction():
    counts = sparse.csr_matrix([[1.0, 1e308], [1.0, 1e308]])
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with np.errstate(over="raise", invalid="raise"):
        selected = central_keyphrases(
            np.array([0, 0]), counts, ["weak", "strong"], vectors, None, n_keyphrases=1
        )
    assert selected == [["strong"]]
    np.testing.assert_array_equal(counts.toarray(), [[1.0, 1e308], [1.0, 1e308]])


@pytest.mark.parametrize(
    "response",
    [
        np.ones((1, 2)),
        np.ones((2, 3)),
        [[1.0, 0.0], [np.nan, 0.0]],
        [[1.0, 0.0], [1e300, 0.0]],
    ],
)
def test_malformed_embedding_batch_leaves_the_entire_table_unchanged(response):
    embedder = Embedder()
    embedder.response = response
    vectors = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        central_keyphrases(
            np.array([0, 0]),
            sparse.eye(2, format="csr"),
            ["apple", "banana"],
            vectors,
            embedder,
        )
    np.testing.assert_array_equal(vectors, np.zeros((2, 2)))


@pytest.mark.parametrize("counts", [[[-1.0]], [[np.nan]], [[np.inf]]])
def test_invalid_counts_are_rejected_before_embedding(counts):
    embedder = Embedder()
    with pytest.raises(ValueError, match="counts"):
        central_keyphrases(
            np.array([0]),
            sparse.csr_matrix(counts),
            ["apple"],
            np.zeros((1, 2)),
            embedder,
        )
    assert embedder.calls == []


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "lil", "dok"])
def test_count_subset_preserves_sparse_formats_and_input(format):
    source = sparse.csr_matrix([[2, 0, 0], [0, 3, 0], [0, 0, 7]]).asformat(format)
    counts, labels, columns = subset_matrix_and_class_labels(
        np.array([7, 9, -1]), source
    )
    np.testing.assert_array_equal(counts.toarray(), [[2, 0], [0, 3]])
    np.testing.assert_array_equal(labels, [7, 9])
    np.testing.assert_array_equal(columns, [0, 1])
    counts.data[:] = 0
    np.testing.assert_array_equal(source.toarray(), [[2, 0, 0], [0, 3, 0], [0, 0, 7]])
    assert source.format == format


def test_empty_untyped_labels_have_no_invalid_values():
    counts, labels, columns = subset_matrix_and_class_labels(
        np.array([]), sparse.csr_matrix((0, 2))
    )
    assert counts.shape == (0, 0) and len(labels) == len(columns) == 0


def test_complete_readonly_vectors_and_genuine_zero_response_are_supported():
    vectors = np.array([[1.0, 0.0], [0.0, 0.0]])
    vectors.setflags(write=False)
    selected = central_keyphrases(
        np.array([0, 0]),
        sparse.eye(2, format="csr"),
        ["apple", "banana"],
        vectors,
        None,
    )
    assert set(selected[0]) == {"apple", "banana"}
    embedder = Embedder()
    embedder.response = np.zeros((1, 2))
    selected = central_keyphrases(
        np.array([0]), sparse.csr_matrix([[1]]), ["apple"], np.zeros((1, 2)), embedder
    )
    assert selected == [["apple"]] and embedder.calls == [("apple",)]


@pytest.mark.parametrize("weights", [[np.nan, 1], [np.inf, 1], [-1, 2], [0, 0]])
def test_weighted_mean_rejects_invalid_weights(weights):
    from toponymy.utility_functions import _mean_vector

    with pytest.raises(ValueError, match="weights"):
        _mean_vector(np.eye(2), weights=weights)


@pytest.mark.parametrize(
    "method",
    [
        "central",
        "information_weighted",
        "bm25",
        "facility_location",
        "saturated_coverage",
    ],
)
def test_supplied_complete_zero_vectors_are_not_reembedded_per_layer(method):
    embedder = Embedder()
    clusterer = PrecomputedClusterer([[7, 70, 700], [900, 900, 900]]).fit(np.eye(3))
    original = np.zeros((3, 2))
    TextKeyphraseExtractor(method, n_keyphrases=1).fit(
        ["one", "two", "three"],
        clusterer,
        embedder=embedder,
        object_x_keyphrase_matrix=sparse.eye(3, format="csr"),
        keyphrase_list=["apple", "banana", "carrot"],
        keyphrase_vectors=original,
    )
    assert embedder.calls == []
    np.testing.assert_array_equal(original, np.zeros((3, 2)))


def test_central_exemplar_keeps_tiny_direction_after_offset_cancels():
    vectors = np.array(
        [[1e300, -1e-300], [1e300, 2e-300], [1e300, 3e-300], [1e300, -4e-300]]
    )
    names, indices = diverse_exemplars(
        np.array([0, 0, 0, 1]), list("abcd"), vectors, n_exemplars=1
    )
    assert indices[0] == [1] and names[0] == ["b"]


@pytest.mark.parametrize("method", ["facility_location", "saturated_coverage"])
def test_submodular_exemplar_centering_keeps_mixed_magnitude_directions(method):
    y = np.array([-1.0, 2.0, 3.0, -4.0])
    ordinary = np.column_stack((np.ones(4), y))
    extreme = np.column_stack((np.full(4, 1e300), y * 1e-300))
    arguments = (np.array([0, 0, 0, 1]), list("abcd"))
    options = dict(n_exemplars=1, submodular_function=method, random_state=42)
    expected = submodular_selection_exemplars(*arguments, ordinary, **options)
    actual = submodular_selection_exemplars(*arguments, extreme, **options)
    assert actual == expected
