"""Explicit prepared hierarchy reuse and provider-independent renderings."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from toponymy import PrecomputedClusterer, Toponymy, TopicModel
from toponymy.templates import Prompt
from toponymy.llm_wrappers import validate_prompt
from test_pipeline_contracts import RecordingNamer
from test_provider_contracts import RecordingWrapper


def test_reuse_preserves_prefit_memberships_even_when_new_config_has_same_size():
    vectors = np.eye(4)
    clusterer = PrecomputedClusterer([[7, 7, 90, 90]]).fit(vectors)
    fitted_layer = clusterer.cluster_layers_[0]
    clusterer.set_params(labels=[[90, 90, 7, 7]])
    model = Toponymy(
        RecordingNamer(),
        clusterer=clusterer,
        feature_extractors=[],
        reuse_clusterer=True,
    ).prepare(list("abcd"), vectors)
    assert model.cluster_layers_[0] is fitted_layer
    assert model.topics_[(0, 7)].members.tolist() == [0, 1]
    assert model.topic_model_.metadata["clustering_reused"] is True
    ordinary = Toponymy(
        RecordingNamer(), clusterer=clusterer, feature_extractors=[]
    ).prepare(list("abcd"), vectors)
    assert ordinary.topics_[(0, 7)].members.tolist() == [2, 3]
    assert model.topics_[(0, 7)].members.tolist() == [0, 1]
    assert ordinary.topic_model_.metadata["clustering_reused"] is False


def test_reuse_never_calls_fit(monkeypatch):
    clusterer = PrecomputedClusterer([[7, 7]]).fit(np.eye(2))

    def forbidden(*args, **kwargs):
        raise AssertionError("Prefitted hierarchy must not be refit")

    monkeypatch.setattr(clusterer, "fit", forbidden)
    model = Toponymy(
        RecordingNamer(),
        clusterer=clusterer,
        feature_extractors=[],
        reuse_clusterer=True,
    )
    model.fit(["a", "b"], np.eye(2))
    assert model.topic_names_ == [{7: "Topic 1"}]


@pytest.mark.parametrize("bad", ["unfitted", "rows", "tree"])
def test_invalid_reused_hierarchy_fails_before_naming(bad):
    clusterer = PrecomputedClusterer([[7, 7]])
    if bad != "unfitted":
        clusterer.fit(np.eye(2))
    if bad == "tree":
        clusterer.cluster_tree_ = {(1, 0): [(0, 90)]}
    namer = RecordingNamer()
    model = Toponymy(
        namer, clusterer=clusterer, feature_extractors=[], reuse_clusterer=True
    )
    with pytest.raises(NotFittedError if bad == "unfitted" else ValueError):
        model.prepare(
            ["a"] if bad == "rows" else ["a", "b"],
            np.ones((1 if bad == "rows" else 2, 2)),
        )
    assert namer.calls == []


@pytest.mark.parametrize("value", [1, "yes", None])
def test_reuse_option_is_explicit_boolean(value):
    with pytest.raises(ValueError, match="reuse_clusterer"):
        Toponymy(RecordingNamer(), reuse_clusterer=value)


@pytest.mark.parametrize("combined", ["Distinct rendering", ""])
def test_canonical_combined_rendering_survives_provider_normalization(combined):
    schema = {"type": "object"}
    prompt = Prompt("System", "User", schema, combined=combined)
    schema["type"] = "array"
    assert prompt.json_schema == {"type": "object"}
    assert validate_prompt(prompt, False)["combined"] == combined
    assert validate_prompt(prompt, True)["system"] == "System"
    wrapper = RecordingWrapper()
    assert wrapper.generate_topic_name(prompt) == "Transit"
    assert wrapper.calls[0][0]["combined"] == combined


def test_default_prompt_keeps_legacy_shape_and_joined_rendering():
    prompt = Prompt("System", "User", {"type": "object"})
    assert set(prompt._asdict()) == {"system", "user", "json_schema"}
    assert validate_prompt(prompt, False)["combined"] == "System\n\nUser"


@pytest.mark.parametrize("format", ["zip", "lance"])
def test_combined_prompt_round_trip(format, tmp_path):
    model = (
        Toponymy(
            RecordingNamer(),
            clusterer=PrecomputedClusterer([[7, 7]]),
            feature_extractors=[],
        )
        .prepare(["a", "b"], np.eye(2))
        .topic_model_
    )
    model.topics[(0, 7)].prompt = Prompt("S", "U", combined="separate")
    path = tmp_path / format
    if format == "zip":
        model.to_file(path)
        loaded = TopicModel.from_file(path)
    else:
        model.to_lance(path)
        loaded = TopicModel.from_lance(path)
    assert loaded.topics[(0, 7)].prompt == model.topics[(0, 7)].prompt
