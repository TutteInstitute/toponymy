import pytest

from toponymy.cluster_layer import ClusterLayerText
from toponymy.exemplar_texts import DiverseExemplarAnnotator, diverse_exemplars
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.toponymy import Toponymy


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
        make_model().fit(
            all_sentences, object_vectors, clusterable_vectors, exemplar_method="random"
        )
    finally:
        layer.__class__ = ClusterLayerText

    assert layer.custom_method == "random"


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
