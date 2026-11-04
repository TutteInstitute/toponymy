import copy

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from toponymy.annotation import AnnotationStore, Executor, NodeId
from toponymy.keyphrases import (
    InformationWeightedKeyphraseAnnotator,
    KeyphraseBuilder,
    information_weighted_keyphrases,
)

# Chosen so that on this data, leaving any one at its default changes the keyphrases
PARAMS = dict(
    n_keyphrases=8,
    prior_strength=0.5,
    weight_power=1.0,
    max_alpha=0.8,
    min_alpha=0.4,
    alpha_tolerance=0.2,
)


@pytest.fixture(scope="module")
def fitted_clusterer(clusterer, clusterable_vectors, object_vectors):
    clusterer = copy.deepcopy(clusterer)
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


def legacy_keyphrases(layers, builder, embedder):
    # Legacy shares one vectors array across layers and embeds into it
    vectors = builder.keyphrase_vectors_.copy()
    return [
        information_weighted_keyphrases(
            layer.cluster_labels,
            builder.object_x_keyphrase_matrix_,
            builder.keyphrase_list_,
            vectors,
            embedder,
            **PARAMS,
        )
        for layer in layers
    ]


@pytest.mark.parametrize("builder_name", ["builder", "builder_with_missing_vectors"])
def test_annotator_matches_legacy_information_weighted_keyphrases(
    builder_name, request, embedder, store, fitted_clusterer
):
    builder = request.getfixturevalue(builder_name)
    annotator = InformationWeightedKeyphraseAnnotator(builder, embedder, **PARAMS)

    Executor(store).run(annotator)

    expected = legacy_keyphrases(fitted_clusterer.cluster_layers_, builder, embedder)
    actual = [
        [store.keyphrases[n] for n in store.tree.layer(layer_id)]
        for layer_id in range(len(expected))
    ]
    assert actual == expected


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


def test_annotator_requires_a_fitted_builder():
    with pytest.raises(NotFittedError):
        InformationWeightedKeyphraseAnnotator(KeyphraseBuilder())


def test_annotate_rejects_indices_beyond_the_keyphrase_matrix(builder):
    annotator = InformationWeightedKeyphraseAnnotator(builder)
    n_objects = builder.object_x_keyphrase_matrix_.shape[0]
    indices = {NodeId(0, 0): np.arange(5), NodeId(0, 1): np.array([n_objects])}

    with pytest.raises(ValueError, match="same corpus"):
        annotator.annotate(tuple(indices), cluster_object_indices=indices)


def test_annotate_needs_two_nodes_with_objects(builder):
    annotator = InformationWeightedKeyphraseAnnotator(builder)
    indices = {NodeId(0, 0): np.arange(5), NodeId(0, 1): np.array([], dtype=int)}

    with pytest.raises(ValueError, match="at least two nodes"):
        annotator.annotate(tuple(indices), cluster_object_indices=indices)


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


def test_annotator_rejects_a_partial_layer(builder, store):
    annotator = InformationWeightedKeyphraseAnnotator(builder)
    first = next(iter(store.tree.layer(0)))

    with pytest.raises(ValueError, match="only partly selected"):
        Executor(store).run(annotator, nodes=[first])
