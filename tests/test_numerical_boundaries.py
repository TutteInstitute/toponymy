"""Real numerical kernels at scaling and array ownership boundaries."""

import numpy as np
import pytest

from toponymy.clustering import PLSCANClusterer
from toponymy.feature_extraction import TextExemplarExtractor
from toponymy.clustering import PrecomputedClusterer
from toponymy.utility_functions import distance_to_vector


@pytest.mark.parametrize("scale", [1e-300, 1e-30, 1.0, 1e30, 1e300])
def test_cosine_distance_on_finite_float64_vectors_is_scale_invariant(scale):
    query = np.array([scale, 0.0], dtype=np.float64)
    candidates = np.array([[scale, 0.0], [0.0, scale], [-scale, 0.0]])
    distances = distance_to_vector(query, candidates)
    np.testing.assert_allclose(distances, [0.0, 1.0, 2.0], atol=1e-12, rtol=0)


def test_cosine_uses_independent_scales_and_preserves_zero_vector_policy():
    query = np.array([1e-300, -1e-300])
    candidates = np.array([[1e300, -1e300], [-1e300, 1e300], [0.0, 0.0]])
    np.testing.assert_allclose(distance_to_vector(query, candidates), [0, 2, 1])
    np.testing.assert_array_equal(
        distance_to_vector(np.zeros(2), candidates), [1, 1, 1]
    )


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_cosine_float32_inputs_do_not_overflow_before_accumulation(scale):
    query = np.array([scale, -scale], dtype=np.float32)
    np.testing.assert_allclose(
        distance_to_vector(query, query[None, :]), [0], atol=1e-12
    )


def test_real_reproducible_plscan_repeats_on_readonly_strided_rows():
    rng = np.random.default_rng(946)
    compact = np.vstack([rng.normal(-4, 0.08, (16, 2)), rng.normal(4, 0.08, (16, 2))])
    backing = np.zeros((32, 4))
    backing[:, ::2] = compact
    vectors = backing[:, ::2]
    vectors.flags.writeable = False
    first = PLSCANClusterer(reproducible=True).fit(vectors)
    first_labels = [layer.labels.tolist() for layer in first]
    first_tree = {key: tuple(children) for key, children in first.cluster_tree_.items()}
    first.fit(vectors)
    assert [layer.labels.tolist() for layer in first] == first_labels
    assert {
        key: tuple(children) for key, children in first.cluster_tree_.items()
    } == first_tree
    assert vectors.flags.writeable is False
    np.testing.assert_array_equal(vectors, compact)


def test_real_central_exemplars_keep_source_indices_for_strided_readonly_vectors():
    compact = np.array([[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0], [-1.0, -0.1]])
    backing = np.zeros((4, 4))
    backing[:, ::2] = compact
    vectors = backing[:, ::2]
    vectors.flags.writeable = False
    objects = ["cat one", "cat two", "star one", "star two"]
    clusterer = PrecomputedClusterer([[7, 7, 90, 90]]).fit(vectors)
    extractor = TextExemplarExtractor(n_exemplars=1, selection_method="central")
    features = extractor.fit_predict(objects, clusterer, embedding_vectors=vectors)
    assert len(features[0]) == 2
    assert extractor.indices_[0][0][0] in {0, 1}
    assert extractor.indices_[0][1][0] in {2, 3}
    for result, indices in zip(features[0], extractor.indices_[0]):
        assert result == [objects[index] for index in indices]
    np.testing.assert_array_equal(vectors, compact)
