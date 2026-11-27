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


# Expected indices are from the legacy implementation on main at 2538a49 (https://github.com/TutteInstitute/toponymy/commit/2538a497dd4180a75bc7c93f8b4ab3b4027b6aa3).
def test_diverse_exemplars_match_legacy(
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    test_object_centroid_vectors,
):
    _, indices = diverse_exemplars(
        test_object_cluster_label_vector,
        all_topic_objects,
        topic_vectors,
        test_object_centroid_vectors,
        n_exemplars=3,
    )
    assert [list(map(int, i)) for i in indices] == [
        [4, 9, 0],
        [12, 10, 16],
        [21, 26, 24],
        [30, 34, 38],
        [41, 47, 46],
        [54, 52, 50],
        [63, 69, 64],
        [76, 71, 73],
        [87, 85, 82],
        [90, 99, 97],
    ]


def test_submodular_exemplars_match_legacy(
    test_object_cluster_label_vector, all_topic_objects, topic_vectors
):
    _, indices = submodular_selection_exemplars(
        test_object_cluster_label_vector,
        all_topic_objects,
        topic_vectors,
        n_exemplars=3,
    )
    assert [list(map(int, i)) for i in indices] == [
        [4, 1, 3],
        [12, 10, 13],
        [21, 23, 25],
        [30, 34, 39],
        [47, 49, 46],
        [54, 56, 58],
        [63, 64, 66],
        [76, 72, 79],
        [87, 82, 83],
        [90, 96, 92],
    ]


def test_random_exemplars_match_legacy(
    test_object_cluster_label_vector, all_topic_objects
):
    np.random.seed(0)
    _, indices = random_exemplars(
        test_object_cluster_label_vector, all_topic_objects, n_exemplars=3
    )
    assert [list(map(int, i)) for i in indices] == [
        [2, 8, 4],
        [13, 15, 11],
        [22, 23, 28],
        [36, 31, 39],
        [45, 42, 47],
        [51, 58, 59],
        [63, 62, 61],
        [70, 79, 72],
        [87, 88, 89],
        [91, 92, 95],
    ]


def _single_layer_store(labels, objects, vectors, centroids):
    from toponymy.annotation import AnnotationTree, Annotation, AnnotationStore, NodeId

    n_clusters = labels.max() + 1
    tree = AnnotationTree({NodeId(1, 0): [NodeId(0, c) for c in range(n_clusters)]})
    annotations = {
        name: Annotation(name, tree)
        for name in (
            "cluster_objects",
            "cluster_object_vectors",
            "cluster_object_indices",
            "cluster_centroid",
        )
    }
    for c in range(n_clusters):
        mask = labels == c
        annotations["cluster_objects"][0, c] = [o for o, m in zip(objects, mask) if m]
        annotations["cluster_object_vectors"][0, c] = vectors[mask]
        annotations["cluster_object_indices"][0, c] = np.where(mask)[0]
        annotations["cluster_centroid"][0, c] = centroids[c]
    return AnnotationStore(tree, list(annotations.values()))


@pytest.mark.parametrize(
    "annotator_name, legacy_name",
    [
        ("DiverseExemplarAnnotator", "diverse_exemplars"),
        ("SubmodularExemplarAnnotator", "submodular_selection_exemplars"),
        ("RandomExemplarAnnotator", "random_exemplars"),
    ],
)
def test_exemplar_annotators_run_under_executor(
    annotator_name,
    legacy_name,
    test_object_cluster_label_vector,
    all_topic_objects,
    topic_vectors,
    test_object_centroid_vectors,
):
    import toponymy.exemplar_texts as exemplar_texts
    from toponymy.annotation import Executor

    labels = test_object_cluster_label_vector
    objects = all_topic_objects
    vectors = topic_vectors
    centroids = test_object_centroid_vectors
    null_topic = vectors.mean(axis=0)
    annotator_cls = getattr(exemplar_texts, annotator_name)
    kwargs = {"n_exemplars": 3}
    if annotator_name != "RandomExemplarAnnotator":
        kwargs["null_topic"] = null_topic

    store = _single_layer_store(labels, objects, vectors, centroids)
    np.random.seed(0)
    failures = Executor(store).run(annotator_cls(**kwargs))
    assert failures == {}

    legacy_args = {
        "diverse_exemplars": (labels, objects, vectors, centroids),
        "submodular_selection_exemplars": (labels, objects, vectors),
        "random_exemplars": (labels, objects),
    }[legacy_name]
    np.random.seed(0)
    legacy_exemplars, legacy_indices = getattr(exemplar_texts, legacy_name)(
        *legacy_args, n_exemplars=3
    )
    for c in range(labels.max() + 1):
        assert list(store["exemplars"][0, c]) == list(legacy_exemplars[c])
        assert list(map(int, store["exemplar_original_indices"][0, c])) == list(
            map(int, legacy_indices[c])
        )


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
