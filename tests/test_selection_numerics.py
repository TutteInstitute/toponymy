"""Analytic selection oracles at finite floating-point and token boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from toponymy import Toponymy
from toponymy.feature_extraction import TextExemplarExtractor
from toponymy.keyphrases import create_tokenizers_ngrammer
from toponymy.utility_functions import (
    _mean_vector,
    centroids_from_labels,
    diversify_max_alpha,
)


@pytest.mark.parametrize("encoded_object", [False, True])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("amber", ["amber"]),
        (
            "amber birch cedar",
            [
                "amber",
                "birch",
                "cedar",
                "amber birch",
                "birch cedar",
                "amber birch cedar",
            ],
        ),
    ],
)
def test_tokenizer_includes_terminal_windows(encoded_object, text, expected):
    class Tokenizer:
        def encode(self, value):
            tokens = value.split()
            return SimpleNamespace(ids=tokens) if encoded_object else tokens

        def decode(self, tokens):
            return " ".join(tokens)

    assert create_tokenizers_ngrammer(Tokenizer(), (1, 3))(text) == expected


@pytest.mark.parametrize(
    "dtype,scale",
    [
        (np.float64, 1.0),
        (np.float64, 1e-300),
        (np.float64, 1e300),
        (np.float32, 1e-30),
        (np.float32, 1e30),
    ],
)
def test_central_exemplars_select_analytic_directions(dtype, scale):
    # Global mean is zero; only members 2 and 5 point along their centroids.
    geometry = np.array([[3, 0], [0, 3], [2, 2], [-3, 0], [0, -3], [-2, -2]])
    storage = np.zeros((6, 4), dtype=dtype)
    storage[:, ::2] = geometry * scale
    vectors = storage[:, ::2]
    before = vectors.copy()
    vectors.flags.writeable = False
    objects = ["east", "north", "northeast", "west", "south", "southwest"]
    layers = [SimpleNamespace(labels=np.array([7, 7, 7, 90, 90, 90]))]
    extractor = TextExemplarExtractor(n_exemplars=1, diversify_alpha=0)
    assert extractor.fit_predict(objects, layers, embedding_vectors=vectors) == [
        [["northeast"], ["southwest"]]
    ]
    assert extractor.indices_ == [[[2], [5]]]
    np.testing.assert_array_equal(vectors, before)
    assert not vectors.flags.writeable


@pytest.mark.parametrize("scale", [1.0, 1e-300, 1e300])
@pytest.mark.parametrize("strided", [False, True])
def test_name_similarity_preserves_proportional_pairs(scale, strided):
    vectors = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, -1.0], [-1.0, -1.0]]) * scale
    if strided:
        backing = np.zeros((4, 4))
        backing[:, ::2] = vectors
        vectors = backing[:, ::2]
    vectors.flags.writeable = False
    before = vectors.copy()
    calls = []

    def encode(names):
        calls.append(list(names))
        return vectors

    model = Toponymy(object(), SimpleNamespace(encode=encode))
    model.request_counts_ = {"name_embeddings": 0}
    topics = [
        SimpleNamespace(name=name) for name in ["first", "second", "third", "fourth"]
    ]
    assert model._similar_groups(topics) == [[0, 1]]
    assert calls == [["first", "second", "third", "fourth"]]
    np.testing.assert_array_equal(vectors, before)


@pytest.mark.parametrize(
    "minimum,maximum", [(1e17, np.nextafter(1e17, np.inf)), (1e308, 1.6e308)]
)
@pytest.mark.parametrize("requested", [1, 2])
def test_alpha_search_stops_when_interval_cannot_shrink(minimum, maximum, requested):
    assert diversify_max_alpha(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        requested,
        min_alpha=minimum,
        max_alpha=maximum,
    ) == [0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_centroids_preserve_finite_extremes_and_empty_label_gaps(dtype):
    largest = np.finfo(dtype).max
    vectors = np.array([[largest, largest], [largest, -largest], [0, 0]], dtype=dtype)
    vectors.flags.writeable = False
    result = centroids_from_labels(np.array([2, 2, -1]), vectors)
    np.testing.assert_array_equal(result, [[0, 0], [0, 0], [largest, 0]])


def test_centroid_coordinates_keep_independent_finite_scales():
    smallest = np.nextafter(0.0, 1.0)
    vectors = np.array([[1e308, 1e-300, smallest], [1e308, 1e-300, smallest]])
    np.testing.assert_array_equal(
        centroids_from_labels(np.array([0, 0]), vectors), vectors[:1]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_stable_mean_preserves_independent_coordinates_and_readonly_inputs(
    dtype, weighted
):
    largest = np.finfo(dtype).max
    smallest = np.nextafter(dtype(0), dtype(1))
    # Cancellation, constant extreme/subnormal coordinates, ordinary values and
    # zero columns have independent analytic means. The view is non-contiguous.
    storage = np.zeros((4, 12), dtype=dtype)
    storage[:, ::2] = [
        [-largest, largest, smallest, 2, 0, -0.0],
        [largest, largest, smallest, 6, 0, -0.0],
        [-largest, largest, smallest, 2, 0, -0.0],
        [largest, largest, smallest, 6, 0, -0.0],
    ]
    vectors = storage[:, ::2]
    vectors.flags.writeable = False
    before = storage.tobytes()
    weights = np.array([1.0, 3.0, 1.0, 3.0]) if weighted else None
    if weights is not None:
        weights.flags.writeable = False
        weights_before = weights.tobytes()
    expected = [
        float(largest) * 0.5 if weighted else 0,
        largest,
        smallest,
        5 if weighted else 4,
        0,
        0,
    ]
    result = _mean_vector(vectors, weights)
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=0)
    assert np.isfinite(result).all()
    assert storage.tobytes() == before
    assert not vectors.flags.writeable
    if weights is not None:
        assert weights.tobytes() == weights_before
        assert not weights.flags.writeable
