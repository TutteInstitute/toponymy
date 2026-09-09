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
