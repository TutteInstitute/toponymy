import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from toponymy.annotation import AnnotationStore, Executor, NodeId
from toponymy.keyphrases import (
    InformationWeightedKeyphraseAnnotator,
    KeyphraseBuilder,
)


@pytest.fixture(scope="module")
def fitted_clusterer(clusterer, clusterable_vectors, object_vectors):
    clusterer.fit(
        clusterable_vectors, object_vectors, show_progress_bar=False, verbose=False
    )
    return clusterer


@pytest.fixture
def store(fitted_clusterer):
    return AnnotationStore.from_clusterer(fitted_clusterer)


@pytest.fixture
def builder(embedder, all_sentences):
    return KeyphraseBuilder(n_jobs=1, embedder=embedder).fit(all_sentences)


@pytest.fixture
def builder_with_missing_vectors(builder):
    # An all-zero vector marks a keyphrase to embed when it is first needed
    builder.keyphrase_vectors_[::2] = 0.0
    return builder


@pytest.fixture
def builder_without_vectors(all_sentences):
    return KeyphraseBuilder(n_jobs=1).fit(all_sentences)


def legacy_keyphrases(layer, builder, vectors, embedder):
    # The legacy method writes embeddings back into the vectors it is given
    return layer.make_keyphrases(
        builder.keyphrase_list_,
        builder.object_x_keyphrase_matrix_,
        vectors.copy(),
        embedder,
        method="information_weighted",
    )


def test_annotator_matches_legacy_make_keyphrases(
    builder, embedder, store, fitted_clusterer
):
    annotator = InformationWeightedKeyphraseAnnotator(builder, embedder)

    assert Executor(store).run(annotator) == {}

    for layer_id, layer in enumerate(fitted_clusterer.cluster_layers_):
        expected = legacy_keyphrases(
            layer, builder, builder.keyphrase_vectors_, embedder
        )
        assert [store.keyphrases[n] for n in store.tree.layer(layer_id)] == expected


def test_annotator_matches_legacy_make_keyphrases_with_missing_vectors(
    builder_with_missing_vectors, embedder, store, fitted_clusterer
):
    builder = builder_with_missing_vectors
    annotator = InformationWeightedKeyphraseAnnotator(builder, embedder)

    Executor(store).run(annotator)

    layer = fitted_clusterer.cluster_layers_[0]
    expected = legacy_keyphrases(layer, builder, builder.keyphrase_vectors_, embedder)
    assert [store.keyphrases[n] for n in store.tree.layer(0)] == expected


def test_annotator_embeds_missing_vectors_into_its_own_copy(
    builder_with_missing_vectors, embedder, store
):
    builder = builder_with_missing_vectors
    before = builder.keyphrase_vectors_.copy()
    annotator = InformationWeightedKeyphraseAnnotator(builder, embedder)

    Executor(store).run(annotator)

    np.testing.assert_array_equal(builder.keyphrase_vectors_, before)
    assert not np.array_equal(annotator.keyphrase_vectors, before)


def test_annotator_selects_n_keyphrases_for_each_node(builder, embedder, store):
    annotator = InformationWeightedKeyphraseAnnotator(builder, embedder, n_keyphrases=3)

    Executor(store).run(annotator)

    assert max(len(keyphrases) for keyphrases in store.keyphrases.values()) == 3


def test_by_layer_groups_the_nodes_of_each_layer(builder):
    annotator = InformationWeightedKeyphraseAnnotator(builder)
    nodes = [NodeId(0, 0), NodeId(1, 0), NodeId(0, 1)]

    groups = annotator.by_layer(dict.fromkeys(nodes))

    assert groups == [(NodeId(0, 0), NodeId(0, 1)), (NodeId(1, 0),)]


def test_annotator_requires_a_fitted_builder():
    with pytest.raises(NotFittedError):
        InformationWeightedKeyphraseAnnotator(KeyphraseBuilder())


def test_annotator_needs_an_embedding_model_if_the_builder_has_no_vectors(
    builder_without_vectors,
):
    with pytest.raises(ValueError, match="no keyphrase vectors"):
        InformationWeightedKeyphraseAnnotator(builder_without_vectors)


def test_annotator_embeds_the_keyphrases_if_the_builder_has_no_vectors(
    builder_without_vectors, embedder
):
    annotator = InformationWeightedKeyphraseAnnotator(builder_without_vectors, embedder)

    n_keyphrases = len(builder_without_vectors.keyphrase_list_)
    assert annotator.keyphrase_vectors.shape[0] == n_keyphrases
