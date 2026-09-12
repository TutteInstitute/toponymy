"""Prepared pipeline ownership and extractor capability contracts."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from toponymy import PrecomputedClusterer, Toponymy
from toponymy.clustering import PLSCANClusterer
from toponymy.feature_extraction import FeatureExtractorBase, SubtopicExtractor


class Namer:
    def __init__(self):
        self.calls = 0

    def generate_topic_name(self, prompt, *, response_parser):
        self.calls += 1
        return f"name-{self.calls}"


def test_prepared_subtopics_use_the_prepared_hierarchy_after_shared_refit():
    labels_a = [[7, 7, 9, 9], [20, 20, 30, 30]]
    labels_b = [[7, 7, 9, 9], [30, 30, 20, 20]]
    shared = PrecomputedClusterer(labels_a)
    objects = ["a", "b", "c", "d"]
    vectors = np.arange(8.0).reshape(4, 2)
    first = Toponymy(
        Namer(),
        clusterer=shared,
        feature_extractors=[SubtopicExtractor()],
        disambiguate=False,
    ).prepare(objects, vectors)
    shared.set_params(labels=labels_b)
    Toponymy(
        Namer(),
        clusterer=shared,
        feature_extractors=[],
        disambiguate=False,
    ).prepare(objects, vectors)
    first.name_topics()
    assert first.topics_[(1, 20)].features["cluster_subtopics"]["major"] == ["name-1"]
    assert first.topics_[(1, 30)].features["cluster_subtopics"]["major"] == ["name-2"]


class DeferredMetadataExtractor(FeatureExtractorBase):
    feature_key = "cluster_sentences"
    layer_dependent = True

    def __init__(self, configuration=None):
        self.configuration = configuration

    def fit(self, objects, clusterer, **configuration):
        self.objects_ = tuple(objects)
        self.features_ = [[[] for _ in layer] for layer in clusterer.cluster_layers_]
        if self.configuration is not None:
            self.configuration["fits"].append(self.objects_[0])
        return self

    def extract_layer(self, layer_index, topics, clusterer):
        return [
            [self.objects_[int(cluster.members[0])]]
            for cluster in clusterer.cluster_layers_[layer_index]
        ]


def _prepare_deferred(extractor, prefix):
    return Toponymy(
        Namer(),
        clusterer=PrecomputedClusterer([[0, 0, 1, 1], [0, 0, 0, 0]]),
        feature_extractors=[extractor],
        disambiguate=False,
    ).prepare([f"{prefix}-{index}" for index in range(4)], np.eye(4))


def test_prepared_deferred_extractors_isolate_data_and_constructor_containers():
    configuration = {"fits": []}
    shared = DeferredMetadataExtractor(configuration)
    first = _prepare_deferred(shared, "first")
    second = _prepare_deferred(shared, "second")

    first.name_topics()
    second.name_topics()

    assert first.topics_[(1, 0)].features["cluster_sentences"] == ["first-0"]
    assert second.topics_[(1, 0)].features["cluster_sentences"] == ["second-0"]
    assert "first-0" in first.topics_[(1, 0)].prompt.user
    assert "second-0" not in first.topics_[(1, 0)].prompt.user
    assert configuration == {"fits": []}
    assert first.feature_extractors_[0].configuration == {"fits": ["first-0"]}
    assert second.feature_extractors_[0].configuration == {"fits": ["second-0"]}
    assert shared.features is None


class PrefittedDeferredExtractor(DeferredMetadataExtractor):
    def can_fit_from_objects(self):
        return False


def test_prefitted_deferred_extractor_snapshots_learned_state():
    shared = PrefittedDeferredExtractor()
    shared.objects_ = ["domain-first"] * 4
    shared.features_ = [[[], []], [[]]]
    first = _prepare_deferred(shared, "unused-first")
    shared.objects_[0] = "domain-second"
    second = _prepare_deferred(shared, "unused-second")

    first.name_topics()
    second.name_topics()

    assert first.topics_[(1, 0)].features["cluster_sentences"] == ["domain-first"]
    assert second.topics_[(1, 0)].features["cluster_sentences"] == ["domain-second"]


def test_uncopyable_deferred_extractor_fails_before_clustering():
    import threading

    extractor = PrefittedDeferredExtractor()
    extractor.resource = threading.Lock()
    clusterer = PrecomputedClusterer([[0, 0]])
    model = Toponymy(Namer(), clusterer=clusterer, feature_extractors=[extractor])
    with pytest.raises(TypeError, match="Layer-dependent extractors") as caught:
        model.prepare(["a", "b"], np.eye(2))
    assert isinstance(caught.value.__cause__, TypeError)
    assert not hasattr(clusterer, "cluster_layers_")
    assert model.llm_wrapper.calls == 0


class PrefittedMetadataExtractor(FeatureExtractorBase):
    feature_key = "metadata"

    def __init__(self):
        self.fit_calls = 0
        self.features_ = [[["domain metadata"]]]

    def can_fit_from_objects(self):
        return False

    def fit(self, metadata, clusterer, **configuration):
        self.fit_calls += 1
        raise AssertionError("Custom metadata must not be refitted from documents")

    def __deepcopy__(self, memo):
        raise AssertionError("Immediate extractors must not copy their resources")


def test_prefitted_custom_data_extractor_honors_its_capability_hook():
    extractor = PrefittedMetadataExtractor()
    pipeline = Toponymy(
        Namer(),
        clusterer=PrecomputedClusterer([[7, 7]]),
        feature_extractors=[extractor],
        disambiguate=False,
    ).prepare(["a", "b"], np.ones((2, 2)))
    assert extractor.fit_calls == 0
    assert pipeline.topics_[(0, 7)].features["metadata"] == ["domain metadata"]


class EmbedderRequiredExtractor(DeferredMetadataExtractor):
    requires_embedder = True

    def __init__(self, layer_dependent=False):
        self.layer_dependent = layer_dependent
        super().__init__()

    def fit(self, objects, clusterer, *, embedder, **configuration):
        self.received_embedder_ = embedder
        return super().fit(objects, clusterer, **configuration)


def test_declared_embedder_requirement_fails_at_construction():
    with pytest.raises(ValueError, match="cluster_sentences.*requires.*embedding"):
        Toponymy(Namer(), feature_extractors=[EmbedderRequiredExtractor()])


@pytest.mark.parametrize("layer_dependent", [False, True])
def test_declared_embedder_reaches_custom_fit(layer_dependent):
    embedder = object()
    pipeline = Toponymy(
        Namer(),
        text_embedding_model=embedder,
        clusterer=PrecomputedClusterer([[0, 0]]),
        feature_extractors=[EmbedderRequiredExtractor(layer_dependent)],
        disambiguate=False,
    ).prepare(["a", "b"], np.eye(2))
    assert pipeline.feature_extractors_[0].received_embedder_ is embedder
    assert pipeline.llm_wrapper.calls == 0


@pytest.mark.parametrize("features, error", [(None, NotFittedError), ([], ValueError)])
def test_custom_extractor_must_be_fitted_and_aligned_before_naming(features, error):
    extractor = PrefittedMetadataExtractor()
    extractor.features_ = features
    namer = Namer()
    pipeline = Toponymy(
        namer,
        clusterer=PrecomputedClusterer([[7, 7]]),
        feature_extractors=[extractor],
        disambiguate=False,
    )
    with pytest.raises(error):
        pipeline.prepare(["a", "b"], np.ones((2, 2)))
    assert extractor.fit_calls == 0
    assert namer.calls == 0


class FittedPLSCAN:
    def __init__(self, **options):
        pass

    def fit(self, vectors):
        self.cluster_layers_ = [np.array([7, 7, 7, 9, 9, 9])]
        self.membership_strength_layers_ = [np.full(6, 0.75)]
        self.layer_persistence_scores_ = [1.5]
        self.min_cluster_sizes_ = [3]
        return self


def test_density_empty_refit_does_not_retain_prior_learned_metadata(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "fast_hdbscan", SimpleNamespace(PLSCAN=FittedPLSCAN)
    )
    adapter = PLSCANClusterer(min_samples=2, base_min_cluster_size=2)
    adapter.fit(np.arange(12.0).reshape(6, 2))
    adapter.fit(np.empty((0, 2)))
    assert adapter.cluster_layers_ == []
    assert adapter.plscan_ is None
    assert not getattr(adapter, "cluster_probabilities_", [])
    assert not getattr(adapter, "cluster_persistence_scores_", [])
    assert getattr(adapter, "plscan_min_cluster_sizes_", None) in (None, [])


def test_density_tiny_refit_metadata_matches_current_observation_count(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "fast_hdbscan", SimpleNamespace(PLSCAN=FittedPLSCAN)
    )
    adapter = PLSCANClusterer(min_samples=2, base_min_cluster_size=2)
    adapter.fit(np.arange(12.0).reshape(6, 2))
    adapter.fit(np.ones((1, 2)))
    assert adapter.cluster_layers_[0].labels.tolist() == [-1]
    probabilities = getattr(adapter, "cluster_probabilities_", [])
    assert not probabilities or all(len(values) == 1 for values in probabilities)
