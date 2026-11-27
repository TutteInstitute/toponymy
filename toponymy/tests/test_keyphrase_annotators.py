import copy

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from toponymy.annotation import AnnotationStore, Executor, NodeId
from toponymy.keyphrases import (
    BM25KeyphraseAnnotator,
    CentralKeyphraseAnnotator,
    InformationWeightedKeyphraseAnnotator,
    KeyphraseBuilder,
    SubmodularKeyphraseAnnotator,
    bm25_keyphrases,
    central_keyphrases,
    information_weighted_keyphrases,
    submodular_selection_information_keyphrases,
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
    assert_plain_keyphrase_lists(actual)
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


CENTRAL_PARAMS = dict(n_keyphrases=8, diversify_alpha=0.6)


def legacy_central_keyphrases(layers, builder, embedder):
    vectors = builder.keyphrase_vectors_.copy()
    return [
        central_keyphrases(
            layer.cluster_labels,
            builder.object_x_keyphrase_matrix_,
            builder.keyphrase_list_,
            vectors,
            embedder,
            **CENTRAL_PARAMS,
        )
        for layer in layers
    ]


@pytest.mark.parametrize("builder_name", ["builder", "builder_with_missing_vectors"])
def test_central_annotator_matches_legacy_central_keyphrases(
    builder_name, request, embedder, store, fitted_clusterer
):
    builder = request.getfixturevalue(builder_name)
    annotator = CentralKeyphraseAnnotator(builder, embedder, **CENTRAL_PARAMS)

    Executor(store).run(annotator)

    expected = legacy_central_keyphrases(
        fitted_clusterer.cluster_layers_, builder, embedder
    )
    actual = [
        [store.keyphrases[n] for n in store.tree.layer(layer_id)]
        for layer_id in range(len(expected))
    ]
    assert_plain_keyphrase_lists(actual)
    assert actual == expected


def test_central_annotator_embeds_missing_vectors_into_its_own_copy(
    builder_with_missing_vectors, embedder, store
):
    builder = builder_with_missing_vectors
    before = builder.keyphrase_vectors_.copy()
    annotator = CentralKeyphraseAnnotator(builder, embedder)

    Executor(store).run(annotator)

    np.testing.assert_array_equal(builder.keyphrase_vectors_, before)
    assert not np.array_equal(annotator.keyphrase_vectors, before)


def test_central_annotator_selects_n_keyphrases_for_each_node(builder, embedder, store):
    annotator = CentralKeyphraseAnnotator(builder, embedder, n_keyphrases=3)

    failures = Executor(store).run(annotator)

    assert failures == {}
    assert max(len(keyphrases) for keyphrases in store.keyphrases.values()) == 3


def test_central_annotator_requires_a_fitted_builder():
    with pytest.raises(NotFittedError):
        CentralKeyphraseAnnotator(KeyphraseBuilder())


def test_central_annotate_rejects_indices_beyond_the_keyphrase_matrix(builder):
    annotator = CentralKeyphraseAnnotator(builder)
    n_objects = builder.object_x_keyphrase_matrix_.shape[0]
    indices = {NodeId(0, 0): np.arange(5), NodeId(0, 1): np.array([n_objects])}

    with pytest.raises(ValueError, match="same corpus"):
        annotator.annotate(tuple(indices), cluster_object_indices=indices)


def test_central_annotate_rejects_a_node_without_objects(builder):
    annotator = CentralKeyphraseAnnotator(builder)
    indices = {NodeId(0, 0): np.arange(5), NodeId(0, 1): np.array([], dtype=int)}

    with pytest.raises(ValueError, match="no objects"):
        annotator.annotate(tuple(indices), cluster_object_indices=indices)


def test_central_annotator_needs_an_embedding_model_if_the_builder_has_no_vectors(
    builder_without_vectors,
):
    with pytest.raises(ValueError, match="no keyphrase vectors"):
        CentralKeyphraseAnnotator(builder_without_vectors)


def test_central_annotator_embeds_the_keyphrases_if_the_builder_has_no_vectors(
    builder_without_vectors, embedder
):
    annotator = CentralKeyphraseAnnotator(builder_without_vectors, embedder)

    n_keyphrases = len(builder_without_vectors.keyphrase_list_)
    assert annotator.keyphrase_vectors.shape[0] == n_keyphrases


def test_central_annotator_rejects_a_partial_layer(builder, store):
    annotator = CentralKeyphraseAnnotator(builder)
    first = next(iter(store.tree.layer(0)))

    with pytest.raises(ValueError, match="only partly selected"):
        Executor(store).run(annotator, nodes=[first])


BM25_PARAMS = dict(n_keyphrases=8, k1=1.0, b=0.3, diversify_alpha=0.6)


def legacy_bm25_keyphrases(layers, builder, embedder):
    vectors = builder.keyphrase_vectors_.copy()
    return [
        bm25_keyphrases(
            layer.cluster_labels,
            builder.object_x_keyphrase_matrix_,
            builder.keyphrase_list_,
            vectors,
            embedder,
            **BM25_PARAMS,
        )
        for layer in layers
    ]


@pytest.mark.parametrize("builder_name", ["builder", "builder_with_missing_vectors"])
def test_bm25_annotator_matches_legacy_bm25_keyphrases(
    builder_name, request, embedder, store, fitted_clusterer
):
    builder = request.getfixturevalue(builder_name)
    annotator = BM25KeyphraseAnnotator(builder, embedder, **BM25_PARAMS)

    Executor(store).run(annotator)

    expected = legacy_bm25_keyphrases(
        fitted_clusterer.cluster_layers_, builder, embedder
    )
    actual = [
        [store.keyphrases[n] for n in store.tree.layer(layer_id)]
        for layer_id in range(len(expected))
    ]
    assert_plain_keyphrase_lists(actual)
    assert actual == expected


def test_bm25_annotator_embeds_missing_vectors_into_its_own_copy(
    builder_with_missing_vectors, embedder, store
):
    builder = builder_with_missing_vectors
    before = builder.keyphrase_vectors_.copy()
    annotator = BM25KeyphraseAnnotator(builder, embedder)

    Executor(store).run(annotator)

    np.testing.assert_array_equal(builder.keyphrase_vectors_, before)
    assert not np.array_equal(annotator.keyphrase_vectors, before)


def test_bm25_annotator_selects_n_keyphrases_for_each_node(builder, embedder, store):
    annotator = BM25KeyphraseAnnotator(builder, embedder, n_keyphrases=3)

    failures = Executor(store).run(annotator)

    assert failures == {}
    assert max(len(keyphrases) for keyphrases in store.keyphrases.values()) == 3


def test_bm25_annotator_requires_a_fitted_builder():
    with pytest.raises(NotFittedError):
        BM25KeyphraseAnnotator(KeyphraseBuilder())


def test_bm25_annotate_rejects_indices_beyond_the_keyphrase_matrix(builder):
    annotator = BM25KeyphraseAnnotator(builder)
    n_objects = builder.object_x_keyphrase_matrix_.shape[0]
    indices = {NodeId(0, 0): np.arange(5), NodeId(0, 1): np.array([n_objects])}

    with pytest.raises(ValueError, match="same corpus"):
        annotator.annotate(tuple(indices), cluster_object_indices=indices)


def test_bm25_annotate_matches_legacy_on_a_node_without_objects(builder, embedder):
    annotator = BM25KeyphraseAnnotator(builder, embedder, **BM25_PARAMS)
    indices = {
        NodeId(0, 0): np.arange(20),
        NodeId(0, 1): np.array([], dtype=int),
        NodeId(0, 2): np.arange(20, 40),
    }
    labels = np.full(builder.object_x_keyphrase_matrix_.shape[0], -1)
    labels[:20] = 0
    labels[20:40] = 2

    result = annotator.annotate(tuple(indices), cluster_object_indices=indices)

    expected = bm25_keyphrases(
        labels,
        builder.object_x_keyphrase_matrix_,
        builder.keyphrase_list_,
        builder.keyphrase_vectors_.copy(),
        embedder,
        **BM25_PARAMS,
    )
    assert expected[1] == ["No notable keyphrases"]
    assert list(result["keyphrases"].values()) == expected


def test_bm25_annotator_needs_an_embedding_model_if_the_builder_has_no_vectors(
    builder_without_vectors,
):
    with pytest.raises(ValueError, match="no keyphrase vectors"):
        BM25KeyphraseAnnotator(builder_without_vectors)


def test_bm25_annotator_embeds_the_keyphrases_if_the_builder_has_no_vectors(
    builder_without_vectors, embedder
):
    annotator = BM25KeyphraseAnnotator(builder_without_vectors, embedder)

    n_keyphrases = len(builder_without_vectors.keyphrase_list_)
    assert annotator.keyphrase_vectors.shape[0] == n_keyphrases


def test_bm25_annotator_rejects_a_partial_layer(builder, store):
    annotator = BM25KeyphraseAnnotator(builder)
    first = next(iter(store.tree.layer(0)))

    with pytest.raises(ValueError, match="only partly selected"):
        Executor(store).run(annotator, nodes=[first])


SUBMODULAR_PARAMS = dict(n_keyphrases=8, prior_strength=0.5, weight_power=1.0)


def legacy_submodular_keyphrases(layers, builder, embedder, submodular_function):
    vectors = builder.keyphrase_vectors_.copy()
    return [
        submodular_selection_information_keyphrases(
            layer.cluster_labels,
            builder.object_x_keyphrase_matrix_,
            builder.keyphrase_list_,
            vectors,
            embedder,
            submodular_function=submodular_function,
            **SUBMODULAR_PARAMS,
        )
        for layer in layers
    ]


def assert_plain_keyphrase_lists(layers):
    for layer in layers:
        for phrases in layer:
            assert type(phrases) is list
            assert all(type(phrase) is str for phrase in phrases)


def as_lists(layers_keyphrases):
    return [[list(keyphrases) for keyphrases in layer] for layer in layers_keyphrases]


@pytest.mark.parametrize(
    "submodular_function", SubmodularKeyphraseAnnotator.submodular_functions
)
@pytest.mark.parametrize("builder_name", ["builder", "builder_with_missing_vectors"])
def test_submodular_annotator_matches_legacy_submodular_keyphrases(
    builder_name, submodular_function, request, embedder, store, fitted_clusterer
):
    builder = request.getfixturevalue(builder_name)
    annotator = SubmodularKeyphraseAnnotator(
        builder,
        embedder,
        submodular_function=submodular_function,
        **SUBMODULAR_PARAMS,
    )

    Executor(store).run(annotator)

    expected = legacy_submodular_keyphrases(
        fitted_clusterer.cluster_layers_, builder, embedder, submodular_function
    )
    actual = [
        [store.keyphrases[n] for n in store.tree.layer(layer_id)]
        for layer_id in range(len(expected))
    ]
    assert_plain_keyphrase_lists(actual)
    assert actual == as_lists(expected)


def test_submodular_annotator_embeds_missing_vectors_into_its_own_copy(
    builder_with_missing_vectors, embedder, store
):
    builder = builder_with_missing_vectors
    before = builder.keyphrase_vectors_.copy()
    annotator = SubmodularKeyphraseAnnotator(builder, embedder)

    Executor(store).run(annotator)

    np.testing.assert_array_equal(builder.keyphrase_vectors_, before)
    assert not np.array_equal(annotator.keyphrase_vectors, before)


def test_submodular_annotator_selects_n_keyphrases_for_each_node(
    builder, embedder, store
):
    annotator = SubmodularKeyphraseAnnotator(builder, embedder, n_keyphrases=3)

    failures = Executor(store).run(annotator)

    assert failures == {}
    assert max(len(keyphrases) for keyphrases in store.keyphrases.values()) == 3


def test_submodular_annotator_rejects_an_unknown_submodular_function(builder):
    with pytest.raises(ValueError, match="unknown submodular_function"):
        SubmodularKeyphraseAnnotator(builder, submodular_function="greedy")


def test_submodular_annotator_requires_a_fitted_builder():
    with pytest.raises(NotFittedError):
        SubmodularKeyphraseAnnotator(KeyphraseBuilder())


def test_submodular_annotate_rejects_indices_beyond_the_keyphrase_matrix(builder):
    annotator = SubmodularKeyphraseAnnotator(builder)
    n_objects = builder.object_x_keyphrase_matrix_.shape[0]
    indices = {NodeId(0, 0): np.arange(5), NodeId(0, 1): np.array([n_objects])}

    with pytest.raises(ValueError, match="same corpus"):
        annotator.annotate(tuple(indices), cluster_object_indices=indices)


def test_submodular_annotate_matches_legacy_on_a_layer_with_one_node(builder, embedder):
    annotator = SubmodularKeyphraseAnnotator(builder, embedder, **SUBMODULAR_PARAMS)
    indices = {NodeId(0, 0): np.arange(40)}
    labels = np.full(builder.object_x_keyphrase_matrix_.shape[0], -1)
    labels[:40] = 0

    result = annotator.annotate(tuple(indices), cluster_object_indices=indices)

    expected = submodular_selection_information_keyphrases(
        labels,
        builder.object_x_keyphrase_matrix_,
        builder.keyphrase_list_,
        builder.keyphrase_vectors_.copy(),
        embedder,
        **SUBMODULAR_PARAMS,
    )
    assert as_lists([list(result["keyphrases"].values())]) == as_lists([expected])


def test_submodular_annotator_needs_an_embedding_model_if_the_builder_has_no_vectors(
    builder_without_vectors,
):
    with pytest.raises(ValueError, match="no keyphrase vectors"):
        SubmodularKeyphraseAnnotator(builder_without_vectors)


def test_submodular_annotator_embeds_the_keyphrases_if_the_builder_has_no_vectors(
    builder_without_vectors, embedder
):
    annotator = SubmodularKeyphraseAnnotator(builder_without_vectors, embedder)

    n_keyphrases = len(builder_without_vectors.keyphrase_list_)
    assert annotator.keyphrase_vectors.shape[0] == n_keyphrases


def test_submodular_annotator_rejects_a_partial_layer(builder, store):
    annotator = SubmodularKeyphraseAnnotator(builder)
    first = next(iter(store.tree.layer(0)))

    with pytest.raises(ValueError, match="only partly selected"):
        Executor(store).run(annotator, nodes=[first])
