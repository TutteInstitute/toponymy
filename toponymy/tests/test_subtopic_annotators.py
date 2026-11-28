import numpy as np
import pytest

from toponymy.annotation import Annotation, AnnotationStore, AnnotationTree, Executor
from toponymy.clusterers import build_cluster_tree
from toponymy.subtopics import (
    CentralSubtopicAnnotator,
    SubmodularSubtopicAnnotator,
    central_subtopics,
    submodular_subtopics,
)

# Twelve layer 0 clusters of five objects each, nested four to a layer 1 cluster
LAYER_0 = np.repeat(np.arange(12), 5)
LAYER_1 = LAYER_0 // 4
NAMES = [f"topic {i}" for i in range(12)]


def make_store(layer_0, layer_1, names, vectors):
    tree = AnnotationTree(build_cluster_tree([layer_0, layer_1]))
    store = AnnotationStore(tree)
    topic_names = Annotation("topic_names", tree)
    embeddings = Annotation("topic_name_embeddings", tree)
    for node in tree.layer(0):
        topic_names[node] = names[node.cluster]
        embeddings[node] = vectors[node.cluster]
    store.add(topic_names)
    store.add(embeddings)
    return store


@pytest.fixture
def vectors():
    return np.random.default_rng(0).normal(size=(12, 8))


@pytest.fixture
def store(vectors):
    return make_store(LAYER_0, LAYER_1, NAMES, vectors)


def run_on_layer_1(store, annotator):
    failures = Executor(store).run(annotator, nodes=store.tree.layer(1))
    assert not failures
    return [store.subtopics[node] for node in store.tree.layer(1)]


@pytest.mark.parametrize("n_subtopics", [2, 3])
def test_central_annotator_matches_legacy_central_subtopics(
    store, vectors, n_subtopics
):
    annotator = CentralSubtopicAnnotator(
        n_subtopics=n_subtopics, diversify_alpha=0.5, null_topic=vectors.mean(axis=0)
    )

    expected = central_subtopics(
        LAYER_1,
        NAMES,
        LAYER_0,
        subtopic_vectors=vectors,
        n_subtopics=n_subtopics,
        diversify_alpha=0.5,
    )
    assert run_on_layer_1(store, annotator) == expected


@pytest.mark.parametrize("function", ["facility_location", "saturated_coverage"])
def test_submodular_annotator_matches_legacy_submodular_subtopics(
    store, vectors, function
):
    annotator = SubmodularSubtopicAnnotator(
        n_subtopics=2, null_topic=vectors.mean(axis=0), submodular_function=function
    )

    expected = submodular_subtopics(
        LAYER_1,
        NAMES,
        LAYER_0,
        subtopic_vectors=vectors,
        n_subtopics=2,
        submodular_function=function,
    )
    assert run_on_layer_1(store, annotator) == expected


def test_subtopics_are_layer_0_descendants_in_the_tree(vectors):
    # Layer 0 cluster 11 straddles layer 1 clusters 0 and 1, so the tree attaches
    # it to the root and it is a subtopic of neither
    layer_1 = LAYER_1.copy()
    layer_1[LAYER_0 == 11] = [0, 0, 1, 1, 1]
    store = make_store(LAYER_0, layer_1, NAMES, vectors)

    subtopics = run_on_layer_1(store, SubmodularSubtopicAnnotator(n_subtopics=64))

    assert subtopics == [NAMES[0:4], NAMES[4:8], NAMES[8:11]]


def test_a_node_without_named_descendants_gets_no_subtopics(store):
    annotator = CentralSubtopicAnnotator(n_subtopics=2)

    assert not Executor(store).run(annotator, nodes=store.tree.layer(0))
    assert all(store.subtopics[node] == [] for node in store.tree.layer(0))


def test_submodular_annotator_rejects_an_unknown_function():
    with pytest.raises(ValueError, match="not a valid selection"):
        SubmodularSubtopicAnnotator(submodular_function="nonsense")
