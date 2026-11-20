from types import SimpleNamespace

import numpy as np
import pytest

from toponymy._annotation_bridge import copy_layers_to_store
from toponymy.annotation import Annotation, AnnotationStore, AnnotationTree
from toponymy.cluster_layer import ClusterLayerText
from toponymy.clustering import build_cluster_tree, centroids_from_labels
from toponymy.exemplar_texts import DiverseExemplarAnnotator, diverse_exemplars
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.tests.helpers.dummy_wrappers import CountingNamer
from toponymy.toponymy import Toponymy, _with_null_topic


@pytest.fixture(scope="module")
def fitted_clusterer(clusterer, clusterable_vectors, object_vectors):
    clusterer.fit(
        clusterable_vectors, object_vectors, show_progress_bar=False, verbose=False
    )
    return clusterer


@pytest.fixture
def make_model(llm, embedder, fitted_clusterer):
    def make(**kwargs):
        return Toponymy(
            llm,
            embedder,
            fitted_clusterer,
            keyphrase_builder=KeyphraseBuilder(n_jobs=1, embedder=embedder),
            verbose=False,
            **kwargs,
        )

    return make


@pytest.fixture
def make_counting_model(embedder, fitted_clusterer):
    def make(namer, clusterer=fitted_clusterer):
        return Toponymy(
            namer,
            embedder,
            clusterer,
            keyphrase_builder=KeyphraseBuilder(n_jobs=1, embedder=embedder),
            verbose=False,
        )

    return make


@pytest.fixture
def make_prefit_clusterer(fitted_clusterer, object_vectors):
    """Make a prefit clusterer whose layer i joins 2**i layer 0 clusters of the shared one."""
    fine = fitted_clusterer.cluster_layers_[0].cluster_labels

    def make(n_layers):
        labels = [np.where(fine >= 0, fine // 2**i, -1) for i in range(n_layers)]
        return SimpleNamespace(
            cluster_layers_=[
                ClusterLayerText(
                    layer_labels, centroids_from_labels(layer_labels, object_vectors), i
                )
                for i, layer_labels in enumerate(labels)
            ],
            cluster_tree_=build_cluster_tree(labels),
        )

    return make


def test_copy_layers_to_store_fills_only_nodes_no_annotator_computed():
    tree = AnnotationTree({(1, 0): [(0, 0), (0, 1)]})
    store = AnnotationStore(tree)
    keyphrases = Annotation("keyphrases", tree)
    keyphrases[(0, 0)] = ["annotated"]
    store.add(keyphrases)
    layers = [SimpleNamespace(keyphrases=[["legacy a"], ["legacy b"]])]

    copy_layers_to_store(store, layers, "keyphrases", "keyphrases")

    assert store.keyphrases[(0, 0)] == ["annotated"]
    assert store.keyphrases[(0, 1)] == ["legacy b"]


@pytest.mark.parametrize(
    "method", ["central", "facility_location", "saturated_coverage"]
)
def test_fit_exemplars_match_the_layer_method(
    make_model, all_sentences, object_vectors, clusterable_vectors, method
):
    model = make_model().fit(
        all_sentences, object_vectors, clusterable_vectors, exemplar_method=method
    )

    for layer in model.cluster_layers_:
        expected = ClusterLayerText.make_exemplar_texts(
            layer, all_sentences, object_vectors, method=method
        )
        assert (layer.exemplars, layer.exemplar_indices) == expected


def test_fit_rejects_an_unknown_exemplar_method(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    with pytest.raises(ValueError, match="Unknown exemplar generation method"):
        make_model().fit(
            all_sentences,
            object_vectors,
            clusterable_vectors,
            exemplar_method="nonsense",
        )


def test_fit_reports_failed_exemplar_nodes(
    make_model, fitted_clusterer, all_sentences, object_vectors, clusterable_vectors
):
    layer = fitted_clusterer.cluster_layers_[1]
    layer.object_to_text_function = lambda objects: 1 / 0
    try:
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            make_model().fit(all_sentences, object_vectors, clusterable_vectors)
    finally:
        layer.object_to_text_function = None


def test_fit_leaves_custom_layer_classes_alone(
    make_model, fitted_clusterer, all_sentences, object_vectors, clusterable_vectors
):
    class CustomLayer(ClusterLayerText):
        def make_exemplar_texts(self, objects, object_vectors, method="central"):
            self.custom_method = method
            return super().make_exemplar_texts(objects, object_vectors, method=method)

    layer = fitted_clusterer.cluster_layers_[-1]
    layer.__class__ = CustomLayer
    try:
        model = make_model().fit(
            all_sentences, object_vectors, clusterable_vectors, exemplar_method="random"
        )
    finally:
        layer.__class__ = ClusterLayerText

    assert layer.custom_method == "random"
    nodes = model.store_.tree.layer(len(model.cluster_layers_) - 1)
    assert [model.store_.exemplars[node] for node in nodes] == layer.exemplars
    assert [
        model.store_.exemplar_original_indices[node] for node in nodes
    ] == layer.exemplar_indices


def test_fit_keyphrases_match_the_layer_method(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    model = make_model().fit(all_sentences, object_vectors, clusterable_vectors)

    for layer in model.cluster_layers_:
        fitted = layer.keyphrases
        expected = ClusterLayerText.make_keyphrases(
            layer,
            model.keyphrase_list_,
            model.object_x_keyphrase_matrix_,
            model.keyphrase_vectors_.copy(),
            model.embedding_model,
        )
        assert fitted == expected


def test_fit_annotates_information_weighted_keyphrases(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    model = make_model().fit(all_sentences, object_vectors, clusterable_vectors)

    for layer_id, layer in enumerate(model.cluster_layers_):
        nodes = model.store_.tree.layer(layer_id)
        assert [model.store_.keyphrases[node] for node in nodes] == layer.keyphrases


def test_fit_copies_other_keyphrase_methods_into_the_store(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    model = make_model().fit(
        all_sentences, object_vectors, clusterable_vectors, keyphrase_method="central"
    )

    for layer_id, layer in enumerate(model.cluster_layers_):
        nodes = model.store_.tree.layer(layer_id)
        assert [model.store_.keyphrases[node] for node in nodes] == layer.keyphrases


def test_fit_leaves_custom_keyphrase_layers_alone(
    make_model, fitted_clusterer, all_sentences, object_vectors, clusterable_vectors
):
    class CustomLayer(ClusterLayerText):
        def make_keyphrases(self, *args, **kwargs):
            self.custom_keyphrases = True
            return super().make_keyphrases(*args, **kwargs)

    layer = fitted_clusterer.cluster_layers_[-1]
    layer.__class__ = CustomLayer
    try:
        model = make_model().fit(all_sentences, object_vectors, clusterable_vectors)
    finally:
        layer.__class__ = ClusterLayerText

    assert layer.custom_keyphrases
    nodes = model.store_.tree.layer(len(model.cluster_layers_) - 1)
    assert [model.store_.keyphrases[node] for node in nodes] == layer.keyphrases


@pytest.mark.parametrize(
    "method", ["central", "facility_location", "saturated_coverage"]
)
def test_fit_subtopics_match_the_layer_method(
    make_counting_model,
    embedder,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    method,
):
    model = make_counting_model(CountingNamer()).fit(
        all_sentences, object_vectors, clusterable_vectors, subtopic_method=method
    )

    names = model.topic_names_[0]
    name_vectors = embedder.encode(names)
    for layer_id, layer in enumerate(model.cluster_layers_[1:], 1):
        fitted = layer.subtopics
        expected = ClusterLayerText.make_subtopics(
            layer,
            names,
            model.cluster_layers_[0].cluster_labels,
            name_vectors,
            method=method,
        )
        assert fitted == expected
        nodes = model.store_.tree.layer(layer_id)
        assert [model.store_.subtopics[node] for node in nodes] == expected


def test_fit_prompts_above_layer_1_list_the_subtopics(
    make_counting_model,
    make_prefit_clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
):
    # Subtopics reach the prompts from layer 2 up, and the shared clusterer has two layers
    namer = CountingNamer()
    make_counting_model(namer, make_prefit_clusterer(3)).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    assert any("miscellaneous detailed subtopics" in str(p) for p in namer.prompts)


def test_fit_gives_clashing_topics_distinct_names(
    make_counting_model,
    make_prefit_clusterer,
    embedder,
    all_sentences,
    object_vectors,
    clusterable_vectors,
):
    namer = CountingNamer(name="Same topic")
    model = make_counting_model(namer, make_prefit_clusterer(3)).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    store = model.store_
    names = model.topic_names_[0]
    assert len(set(names)) == len(names)
    replaced = store.replaced_topic_names.computed(store.tree.layer(0))
    assert set(replaced.values()) == {"Same topic"}
    assert any("Major subtopics" in p["user"] for p in namer.disambiguation_prompts)
    # The embeddings, which the subtopics read, are of the final names
    nodes = list(store.topic_name_embeddings)
    np.testing.assert_allclose(
        [store.topic_name_embeddings[node] for node in nodes],
        embedder.encode([store.topic_names[node] for node in nodes]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("rename, passes", [(True, 1), (False, 2)])
def test_fit_disambiguates_again_while_a_name_is_shared_by_more_than_two_topics(
    make_counting_model,
    make_prefit_clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    rename,
    passes,
):
    # Every topic of the single layer is in the one group
    namer = CountingNamer(name="Same topic", rename=rename)
    make_counting_model(namer, make_prefit_clusterer(1)).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    assert len(namer.disambiguation_prompts) == passes


def test_fit_leaves_unnamed_layer_0_topics_out_of_the_subtopics(
    make_counting_model, all_sentences, object_vectors, clusterable_vectors
):
    # fit names layer 0 before the other layers, so calls 0 and 1 are layer 0 topics
    model = make_counting_model(CountingNamer(fail_calls={0, 1})).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    assert model.topic_names_[0].count("") == 2
    tree = model.store_.tree
    for layer_id in range(1, len(model.cluster_layers_)):
        for node in tree.layer(layer_id):
            assert "" not in model.store_.subtopics[node]


def test_fit_rejects_an_unknown_subtopic_method(
    make_counting_model, all_sentences, object_vectors, clusterable_vectors
):
    namer = CountingNamer()
    with pytest.raises(ValueError, match="Unknown subtopic_method"):
        make_counting_model(namer).fit(
            all_sentences,
            object_vectors,
            clusterable_vectors,
            subtopic_method="nonsense",
        )
    assert not namer.prompts


def test_fit_runs_the_given_annotators(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    annotator = DiverseExemplarAnnotator(n_exemplars=3)
    model = make_model(annotators=[annotator]).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    for layer in model.cluster_layers_:
        expected = diverse_exemplars(
            layer.cluster_labels,
            all_sentences,
            object_vectors,
            layer.centroid_vectors,
            n_exemplars=3,
            object_to_text_function=None,
        )
        assert (layer.exemplars, layer.exemplar_indices) == expected
    assert annotator.null_topic is None


def test_fit_with_several_annotators_keeps_the_last_exemplars(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    annotators = [
        DiverseExemplarAnnotator(n_exemplars=2),
        DiverseExemplarAnnotator(n_exemplars=3),
    ]
    model = make_model(annotators=annotators).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    for layer_id, layer in enumerate(model.cluster_layers_):
        expected = diverse_exemplars(
            layer.cluster_labels,
            all_sentences,
            object_vectors,
            layer.centroid_vectors,
            n_exemplars=3,
            object_to_text_function=None,
        )
        nodes = model.store_.tree.layer(layer_id)
        in_store = (
            [model.store_.exemplars[node] for node in nodes],
            [model.store_.exemplar_original_indices[node] for node in nodes],
        )
        assert (layer.exemplars, layer.exemplar_indices) == expected
        assert in_store == expected


def test_with_null_topic_only_fills_in_unset_exemplar_annotators():
    mean = np.ones(3)

    class Unmarked:
        null_topic = None

    unmarked = Unmarked()
    exemplar = DiverseExemplarAnnotator()
    set_already = DiverseExemplarAnnotator(null_topic=np.zeros(3))

    filled = _with_null_topic(exemplar, mean)

    assert filled.null_topic is mean and exemplar.null_topic is None
    assert _with_null_topic(unmarked, mean) is unmarked
    assert _with_null_topic(set_already, mean) is set_already


def test_fit_non_text_annotators_make_the_layer_0_exemplars(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    annotator = DiverseExemplarAnnotator(
        n_exemplars=3, object_to_text_function=lambda x: x
    )
    model = make_model(annotators=[annotator]).fit(
        all_sentences, object_vectors, clusterable_vectors
    )

    layer = model.cluster_layers_[0]
    expected = diverse_exemplars(
        layer.cluster_labels,
        all_sentences,
        object_vectors,
        layer.centroid_vectors,
        n_exemplars=3,
    )
    assert (layer.exemplars, layer.exemplar_indices) == expected


def test_fit_rejects_exemplar_method_with_annotators(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    model = make_model(annotators=[DiverseExemplarAnnotator()])

    with pytest.raises(ValueError, match="cannot be combined"):
        model.fit(
            all_sentences,
            object_vectors,
            clusterable_vectors,
            exemplar_method="random",
        )


def test_fit_requires_an_annotator_that_makes_exemplars(
    make_model, all_sentences, object_vectors, clusterable_vectors
):
    class OtherAnnotator:
        outputs = ("something_else",)

    model = make_model(annotators=[OtherAnnotator()])

    with pytest.raises(ValueError, match="must include one that outputs"):
        model.fit(all_sentences, object_vectors, clusterable_vectors)


def test_fit_non_text_layer_0_exemplars_ignore_exemplar_method(
    llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
):
    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        object_to_text_function=lambda x: x,
        show_progress_bar=False,
        verbose=False,
    )
    model = Toponymy(
        llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1, embedder=embedder),
        verbose=False,
    )
    model.fit(
        all_sentences, object_vectors, clusterable_vectors, exemplar_method="random"
    )

    layer = model.cluster_layers_[0]
    expected = layer.make_exemplar_texts(
        all_sentences, object_vectors, method="facility_location"
    )
    assert (layer.exemplars, layer.exemplar_indices) == expected
