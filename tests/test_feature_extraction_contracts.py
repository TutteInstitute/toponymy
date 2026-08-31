from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
import scipy.sparse as sp

from toponymy.feature_extraction import TextExemplarExtractor, TextKeyphraseExtractor, SubtopicExtractor


def layers(*labels):
    return [SimpleNamespace(labels=np.asarray(values, dtype=np.int64)) for values in labels]


def test_features_are_fit_results_and_refit_changes_objects():
    extractor = TextExemplarExtractor(
        "random", n_exemplars=1, object_to_text_function=lambda values: values
    )
    with pytest.raises(NotFittedError):
        extractor.predict()
    clustering = layers([11, -1, 200])
    assert extractor.fit_predict(["first", "noise", "last"], clustering) == [
        [["first"], ["last"]]
    ]
    assert extractor.features is extractor.predict()
    assert extractor.fit_predict(["new", "ignored", "other"], clustering) == [
        [["new"], ["other"]]
    ]
    assert not hasattr(clustering[0], "features")


@pytest.mark.parametrize("labels", [[], [-1, -1]])
@pytest.mark.parametrize("method", TextExemplarExtractor.supported_selection_methods)
def test_empty_clusterings_need_no_embeddings(labels, method):
    objects = ["ignored"] * len(labels)
    assert TextExemplarExtractor(method).fit_predict(objects, layers(labels)) == [[]]
    assert TextKeyphraseExtractor().fit_predict(objects, layers(labels)) == [[]]


def test_invalid_configuration_and_labels_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        TextExemplarExtractor("unknown").fit_predict(["a"], layers([0]))
    with pytest.raises(ValueError, match="positive"):
        TextExemplarExtractor(n_exemplars=0).fit_predict(["a"], layers([0]))
    with pytest.raises(ValueError, match="match"):
        TextExemplarExtractor().fit_predict(["a"], layers([0, 1]))
    with pytest.raises(ValueError, match="integers"):
        TextExemplarExtractor().fit_predict(["a"], [SimpleNamespace(labels=[0.5])])


def test_failed_refit_does_not_expose_previous_results():
    extractor = TextExemplarExtractor("random", object_to_text_function=list)
    extractor.fit_predict(["old"], layers([0]))
    with pytest.raises(ValueError):
        extractor.fit_predict(["new"], layers([0]), selection_method="unknown")
    with pytest.raises(NotFittedError):
        extractor.predict()


@pytest.mark.parametrize("method", TextExemplarExtractor.supported_selection_methods)
def test_real_exemplars_preserve_sparse_id_order_and_member_indices(method):
    objects = ["apple", "pear", "plum", "engine", "motor", "wheel", "noise"]
    storage = np.array([[3., 0., 0., 0.], [2., 0., 1., 0.], [3., 0., 1., 0.],
                        [0., 0., 3., 0.], [1., 0., 2., 0.], [1., 0., 3., 0.],
                        [4., 0., 4., 0.]])
    vectors = storage[:, ::2]
    vectors.flags.writeable = False
    extractor = TextExemplarExtractor(method, n_exemplars=2, random_state=23)
    values = extractor.fit_predict(objects, layers([9, 9, 9, 40, 40, 40, -1]),
                                   object_vectors=vectors)
    assert len(values) == 1 and list(map(len, values[0])) == [2, 2]
    for ordinal, expected_indices in enumerate(({0, 1, 2}, {3, 4, 5})):
        assert set(extractor.indices_[0][ordinal]) <= expected_indices
        assert values[0][ordinal] == [objects[i] for i in extractor.indices_[0][ordinal]]
    assert extractor.fit_predict(objects, layers([9, 9, 9, 40, 40, 40, -1]),
                                 object_vectors=vectors) == values


def test_random_exemplars_do_not_change_global_random_state():
    np.random.seed(42)
    expected = np.random.random(4)
    np.random.seed(42)
    TextExemplarExtractor("random").fit_predict(["a", "b", "c"], layers([0, 0, 0]))
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize("vectors", [np.ones((2, 2)), [[np.nan]], [[np.inf]], [[1j]]])
def test_exemplars_validate_semantic_vectors(vectors):
    with pytest.raises(ValueError, match="Vectors"):
        TextExemplarExtractor().fit_predict(["a"], layers([1]), object_vectors=vectors)


@pytest.mark.parametrize("method", TextKeyphraseExtractor.supported_selection_methods)
def test_real_keyphrase_strategies_are_optional_and_preserve_inputs(method):
    vocabulary = ["apple", "pear", "engine", "motor"]
    counts = sp.csr_matrix([[4, 2, 0, 0], [2, 4, 0, 0], [0, 0, 4, 2], [0, 0, 2, 4]])
    vectors = np.array([[3., 1.], [2., 1.], [1., 3.], [1., 2.]])
    original = vectors.copy()
    vectors.flags.writeable = False
    extractor = TextKeyphraseExtractor(method, n_keyphrases=2)
    values = extractor.fit_predict(["a", "b", "c", "d"], layers([7, 7, 98, 98]),
        object_x_keyphrase_matrix=counts, keyphrase_list=vocabulary,
        keyphrase_vectors=vectors)
    assert values[0][0] and set(values[0][0]) <= {"apple", "pear"}
    assert values[0][1] and set(values[0][1]) <= {"engine", "motor"}
    np.testing.assert_array_equal(vectors, original)
    np.testing.assert_array_equal(counts.toarray(), [[4, 2, 0, 0], [2, 4, 0, 0],
                                                   [0, 0, 4, 2], [0, 0, 2, 4]])


@pytest.mark.parametrize("method", TextKeyphraseExtractor.supported_selection_methods)
def test_singleton_keyphrase_dimension_and_empty_term_cluster(method):
    extractor = TextKeyphraseExtractor(method, n_keyphrases=1)
    result = extractor.fit_predict(["apple", "empty"], layers([4, 99]),
        object_x_keyphrase_matrix=sp.csr_matrix([[1], [0]]),
        keyphrase_list=["apple"], keyphrase_vectors=np.array([[1., 2.]]))
    assert len(result[0]) == 2
    assert result[0][0] == ["apple"]
    assert result[0][1] in ([], ["No notable keyphrases"])


def test_zero_vectors_and_empty_diversification_are_finite():
    from toponymy.utility_functions import distance_to_vector, diversify_max_alpha, centroids_from_labels
    np.testing.assert_allclose(distance_to_vector(np.array([1., 0.]),
                               np.array([[0., 0.], [1., 0.], [-1., 0.]])), [1., 0., 2.])
    assert diversify_max_alpha(np.zeros(2), np.empty((0, 2)), 3) == []
    assert centroids_from_labels(np.array([], dtype=np.int64), np.empty((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError, match="tolerance"):
        diversify_max_alpha(np.zeros(2), np.zeros((2, 2)), 1, tolerance=0)


def test_subtopics_follow_containment_tree_after_lower_topics_are_named():
    from toponymy.clustering import PrecomputedClusterer

    clustering = PrecomputedClusterer(
        [
            np.array([8, 8, 9, 12, 12, 12]),
            np.array([30, 30, 30, 90, 90, 90]),
            np.array([200, 200, 200, 200, 200, 200]),
        ]
    ).fit(np.ones((6, 2)))
    extractor = SubtopicExtractor(n_subtopics=2)
    assert extractor.layer_dependent
    assert extractor.extract_layer(0, {}, clustering) == [
        {"major": [], "minor": [], "misc": []} for _ in range(3)
    ]
    topics = {
        (0, 8): SimpleNamespace(name="Fruit"),
        (0, 9): SimpleNamespace(name="Seeds"),
        (0, 12): SimpleNamespace(name="Engines"),
    }
    assert extractor.extract_layer(1, topics, clustering) == [
        {"major": ["Fruit", "Seeds"], "minor": [], "misc": []},
        {"major": ["Engines"], "minor": [], "misc": []},
    ]
    with pytest.raises(ValueError, match="must be named"):
        extractor.extract_layer(2, topics, clustering)
    topics.update({(1, 30): {"name": "Plants"}, (1, 90): {"name": "Machines"}})
    assert extractor.extract_layer(2, topics, clustering) == [
        {"major": ["Plants", "Machines"], "minor": [], "misc": []}
    ]


def test_subtopics_use_skipped_layers_and_deduplicate_names():
    from toponymy.clustering import PrecomputedClusterer

    clustering = PrecomputedClusterer(
        [
            np.array([4, 4, -1, -1]),
            np.array([8, -1, 8, -1]),
            np.array([20, 20, 20, 20]),
        ]
    ).fit(np.ones((4, 2)))
    topics = {
        (0, 4): SimpleNamespace(name="Shared"),
        (1, 8): SimpleNamespace(name="Shared"),
    }
    assert SubtopicExtractor().extract_layer(2, topics, clustering) == [
        {"major": ["Shared"], "minor": [], "misc": []}
    ]
