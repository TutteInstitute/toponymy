import numpy as np
import pytest

from toponymy.new_feature_extractor import (
    FeatureExtractorBase,
    TextExemplarExtractor,
)


@pytest.fixture
def text_exemplar_extractor():
    return TextExemplarExtractor()


def test_cannot_create_abstract_feature_extractor():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class FeatureExtractorBase without an implementation for abstract methods 'feature_return_type', 'fit'",
    ):
        FeatureExtractorBase()


@pytest.mark.parametrize("n_exemplars", [4, 15])
def test_random_exemplar(
    n_exemplars,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_objects
):
    exemplars, indices = TextExemplarExtractor.random_exemplars(
        test_object_cluster_label_vector, all_topic_objects, n_exemplars=n_exemplars
    )
    assert len(exemplars) == test_object_cluster_label_vector.max() + 1

    for i in range(test_object_cluster_label_vector.max() + 1):
        assert len(exemplars[i]) == min(
            n_exemplars, len(topic_objects[i]["paragraphs"])
        )


@pytest.mark.parametrize("n_exemplars", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("sampling_method", ["centroid", "random"])
def test_diverse_exemplar_result_sizes(
    n_exemplars,
    diversify_alpha,
    sampling_method,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
):
    exemplar_results, exemplar_indices = TextExemplarExtractor.diverse_exemplars(
        cluster_label_vector=test_object_cluster_label_vector,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        n_exemplars=n_exemplars,
        diversify_alpha=diversify_alpha,
        sampling_method=sampling_method,
    )
    assert len(exemplar_results) == len(np.unique(test_object_cluster_label_vector)) - 1
    if diversify_alpha == 1.0:
        print(exemplar_results)
    print([len(x) for x in exemplar_results])
    assert all([len(x) == n_exemplars for x in exemplar_results]) and exemplar_results
    assert all([len(set(x)) == n_exemplars for x in exemplar_results])


@pytest.mark.parametrize("sampling_method", ["centroid", "random"])
def test_empty_cluster_diverse(
    sampling_method,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
):
    new_clustering = test_object_cluster_label_vector.copy()
    new_clustering[new_clustering == 0] = 9
    exemplar_results, exemplar_indices = TextExemplarExtractor.diverse_exemplars(
        cluster_label_vector=new_clustering,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        sampling_method=sampling_method,
    )
    assert len(exemplar_results[0]) == 0


@pytest.mark.parametrize(
    "submodular_function", ["facility_location", "saturated_coverage"]
)
def test_empty_cluster_diverse(
    submodular_function,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
):
    new_clustering = test_object_cluster_label_vector.copy()
    new_clustering[new_clustering == 0] = 9
    exemplar_results, exemplar_indices = TextExemplarExtractor.submodular_selection_exemplars(
        cluster_label_vector=new_clustering,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        submodular_function=submodular_function,
    )
    assert len(exemplar_results[0]) == 0


def test_bad_submodular_function(
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors
):
    with pytest.raises(ValueError):
        TextExemplarExtractor.submodular_selection_exemplars(
            cluster_label_vector=test_object_cluster_label_vector,
            objects=all_topic_objects,
            object_vectors=topic_vectors,
            submodular_function="bad_function",
        )


@pytest.mark.parametrize("n_exemplars", [4, 15])
def test_submodular_exemplar(
    n_exemplars,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    topic_objects,
):
    exemplar_results, exemplar_indices = TextExemplarExtractor.submodular_selection_exemplars(
        cluster_label_vector=test_object_cluster_label_vector,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        n_exemplars=n_exemplars,
    )
    assert len(exemplar_results) == test_object_cluster_label_vector.max() + 1

    for i in range(test_object_cluster_label_vector.max() + 1):
        assert len(exemplar_results[i]) == min(
            n_exemplars, len(topic_objects[i]["paragraphs"])
        )


def test_empty_cluster_random(
    test_object_cluster_label_vector,
    all_topic_objects
):
    new_clustering = test_object_cluster_label_vector.copy()
    new_clustering[new_clustering == 0] = 9
    exemplar_results, exemplar_indices = TextExemplarExtractor.random_exemplars(
        cluster_label_vector=new_clustering,
        objects=all_topic_objects,
    )
    assert len(exemplar_results[0]) == 0
