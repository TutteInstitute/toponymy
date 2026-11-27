from typing import Literal
import pytest
import json
import numpy as np
from toponymy.exemplar_texts import (
    calculate_gains_,
    calculate_gains_sparse_,
    random_exemplars,
    diverse_exemplars,
    submodular_selection_exemplars,
)
from pathlib import Path
from scipy.sparse import csr_matrix
import sentence_transformers

from toponymy.clustering import (
    centroids_from_labels,
)


def test_json_load(test_objects):
    assert len(test_objects) == 125


@pytest.mark.parametrize("n_exemplars", [4, 15])
def test_random_exemplar(
    n_exemplars, test_object_cluster_label_vector, all_topic_objects, topic_objects
):
    exemplars, indices = random_exemplars(
        test_object_cluster_label_vector, all_topic_objects, n_exemplars=n_exemplars
    )
    assert len(exemplars) == test_object_cluster_label_vector.max() + 1

    for i in range(test_object_cluster_label_vector.max() + 1):
        assert len(exemplars[i]) == min(
            n_exemplars, len(topic_objects[i]["paragraphs"])
        )


@pytest.mark.parametrize("n_exemplars", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("method", ["centroid", "random"])
def test_diverse_exemplar_result_sizes(
    n_exemplars,
    diversify_alpha,
    method,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    test_object_centroid_vectors,
):
    exemplar_results, exemplar_indices = diverse_exemplars(
        cluster_label_vector=test_object_cluster_label_vector,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        centroid_vectors=test_object_centroid_vectors,
        n_exemplars=n_exemplars,
        diversify_alpha=diversify_alpha,
        method=method,
    )
    assert len(exemplar_results) == len(np.unique(test_object_cluster_label_vector)) - 1
    if diversify_alpha == 1.0:
        print(exemplar_results)
    print([len(x) for x in exemplar_results])
    assert all([len(x) == n_exemplars for x in exemplar_results]) and exemplar_results
    assert all([len(set(x)) == n_exemplars for x in exemplar_results])


@pytest.mark.parametrize("method", ["centroid", "random"])
def test_empty_cluster_diverse(
    method,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    test_object_centroid_vectors,
):
    new_clustering = test_object_cluster_label_vector.copy()
    new_clustering[new_clustering == 0] = 9
    exemplar_results, exemplar_indices = diverse_exemplars(
        cluster_label_vector=new_clustering,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        centroid_vectors=test_object_centroid_vectors,
        method=method,
    )
    assert len(exemplar_results[0]) == 0


def test_diverse_exemplars_rejects_mismatched_lengths(
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    test_object_centroid_vectors,
):
    with pytest.raises(IndexError):
        diverse_exemplars(
            cluster_label_vector=test_object_cluster_label_vector,
            objects=all_topic_objects,
            object_vectors=topic_vectors[:-1],
            centroid_vectors=test_object_centroid_vectors,
        )


@pytest.mark.parametrize(
    "submodular_function", ["facility_location", "saturated_coverage"]
)
def test_empty_cluster_diverse(
    submodular_function,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    test_object_centroid_vectors,
):
    new_clustering = test_object_cluster_label_vector.copy()
    new_clustering[new_clustering == 0] = 9
    exemplar_results, exemplar_indices = submodular_selection_exemplars(
        cluster_label_vector=new_clustering,
        objects=all_topic_objects,
        object_vectors=topic_vectors,
        submodular_function=submodular_function,
    )
    assert len(exemplar_results[0]) == 0


def test_bad_submodular_function(
    test_object_cluster_label_vector, all_topic_objects, topic_vectors
):
    with pytest.raises(ValueError):
        submodular_selection_exemplars(
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
    exemplar_results, exemplar_indices = submodular_selection_exemplars(
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


def test_empty_cluster_random(test_object_cluster_label_vector, all_topic_objects):
    new_clustering = test_object_cluster_label_vector.copy()
    new_clustering[new_clustering == 0] = 9
    exemplar_results, exemplar_indices = random_exemplars(
        cluster_label_vector=new_clustering,
        objects=all_topic_objects,
    )
    assert len(exemplar_results[0]) == 0


def test_facility_location_calculate_gains():
    X_pairwise = np.random.rand(4, 4)
    X_pairwise = X_pairwise + X_pairwise.T
    gains = np.zeros(4, dtype=np.float64)
    idxs = np.array([0, 1, 2, 3])
    current_values = np.random.rand(4)
    calculate_gains_.py_func(
        X=X_pairwise,
        gains=gains,
        current_values=current_values,
        idxs=idxs,
    )
    assert gains.shape == (4,)
    assert np.all(gains == np.sum(np.maximum(X_pairwise, current_values), axis=1))


@pytest.mark.parametrize("data_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_facility_location_sparse_calculate_gains(data_dtype, index_dtype):
    similarities = np.array(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]],
        dtype=data_dtype,
    )
    graph = csr_matrix(similarities)
    graph.indices = graph.indices.astype(index_dtype)
    graph.indptr = graph.indptr.astype(index_dtype)
    current_values = np.array([0.75, 0.0, 0.5], dtype=np.float64)
    idxs = np.array([2, 0], dtype=np.int64)
    gains = np.zeros(len(idxs), dtype=np.float64)

    calculate_gains_sparse_(
        graph.data, graph.indices, graph.indptr, gains, current_values, idxs
    )

    expected = np.maximum(similarities[idxs] - current_values, 0).sum(axis=1)
    np.testing.assert_allclose(gains, expected)


def test_submodular_exemplar_large_cluster():
    n_objects = 4097
    vectors = np.random.default_rng(42).normal(size=(n_objects, 4))
    objects = [f"object {i}" for i in range(n_objects)]

    exemplars, indices = submodular_selection_exemplars(
        np.zeros(n_objects, dtype=np.int64), objects, vectors, n_exemplars=8
    )

    assert len(exemplars) == len(indices) == 1
    assert len(indices[0]) == len(set(indices[0])) == 8
    assert all(0 <= i < n_objects for i in indices[0])
    assert exemplars[0] == [objects[i] for i in indices[0]]
