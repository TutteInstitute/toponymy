from toponymy.cluster_layer import ClusterLayerText, ClusterLayer
from toponymy.llm_wrappers import LLMWrapper
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.clustering import centroids_from_labels, ToponymyClusterer

from pathlib import Path

import numpy as np

import pytest
import json
import sentence_transformers
from dataclasses import dataclass
from abc import ABCMeta
from scipy.sparse import csr_matrix


def test_abstract_cluster_layer(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
):
    ClusterLayer.__abstractmethods__ = set()

    @dataclass
    class ConcreteClusterLayer(ClusterLayer):
        pass

    concrete_cluster_layer = ConcreteClusterLayer()
    exemplar_texts = concrete_cluster_layer.make_exemplar_texts(
        all_sentences, object_vectors
    )
    keyphrases = concrete_cluster_layer.make_keyphrases(
        all_sentences, csr_matrix(np.array([[1, 0], [0, 1]])), object_vectors, embedder
    )
    subtopics = concrete_cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    prompts = concrete_cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    topics = concrete_cluster_layer.name_topics(
        None,
        0.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "objects",
        "collection of objects",
        cluster_tree,
        embedder,
    )
    assert concrete_cluster_layer
    assert isinstance(concrete_cluster_layer, ClusterLayer)
    assert isinstance(ClusterLayer, ABCMeta)
    assert exemplar_texts is None
    assert keyphrases is None
    assert subtopics is None
    assert prompts is None
    assert topics is None


def test_layer_creation(cluster_label_vector, cluster_centroid_vectors, embedder):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    assert cluster_layer


def test_make_data(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, embedder)
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_make_data_alternative_methods1(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(
        keyphrases, matrix, keyphrase_vectors, embedder, method="central"
    )
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_make_data_alternative_methods2(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(
        keyphrases, matrix, keyphrase_vectors, embedder, method="bm25"
    )
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_cluster_layer_prompt_format_is_deprecated_and_ignored(
    embedder,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    with pytest.warns(DeprecationWarning, match="prompt_format is deprecated"):
        cluster_layer = ClusterLayerText(
            cluster_label_vector,
            cluster_centroid_vectors,
            1,
            embedder,
            prompt_format="system_user",
        )

    assert not hasattr(cluster_layer, "prompt_format")


def test_make_prompts_produces_every_rendering(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """Prompts reach the LLM wrapper carrying every rendering, whatever it supports."""
    cluster_layer = ClusterLayerText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, embedder)
    cluster_layer.make_subtopics(
        all_subtopics, subtopic_label_vector, subtopic_centroid_vectors
    )

    prompts = cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )

    assert prompts
    for prompt in prompts:
        assert set(prompt) == {"system", "user", "combined"}


def test_disambiguation_skips_empty_topic_names(monkeypatch):
    cluster_layer = ClusterLayerText(np.arange(7), np.eye(7), 0)
    cluster_layer.topic_names = ["", "", "", "", "Cats", "Cats", "Dogs"]
    cluster_layer.topic_name_embeddings = np.array(
        [[1.0, 0.0, 0.0]] * 3
        + [[0.0, 0.0, 1.0]]
        + [[0.0, 1.0, 0.0]] * 2
        + [[0.0, 0.0, 1.0]]
    )
    monkeypatch.setattr(
        "toponymy.cluster_layer.distinguish_topic_names_prompt",
        lambda topic_indices, *args, **kwargs: list(topic_indices),
    )

    cluster_layer._make_disambiguation_prompts(
        1.0, [cluster_layer.topic_names], "sentences", "about cats"
    )

    assert [list(indices) for indices in cluster_layer.dismbiguation_topic_indices] == [
        [4, 5]
    ]
    assert cluster_layer.disambiguation_prompts == [[4, 5]]


def test_failed_names_do_not_trigger_extra_disambiguation(monkeypatch):
    class FirstPassFailsLLM(LLMWrapper):
        def __init__(self):
            self.names = iter(["", "", "", "Cats", "Dogs", "Fish", "Birds"])

        def _call_llm(self, prompt, temperature, max_tokens):
            raise AssertionError("not used")

        def _call_llm_with_system_prompt(
            self, system_prompt, user_prompt, temperature, max_tokens
        ):
            raise AssertionError("not used")

        def generate_topic_name(self, prompt, *args, **kwargs):
            return next(self.names)

    cluster_layer = ClusterLayerText(np.arange(4), np.eye(4), 0)
    cluster_layer.prompts = ["prompt 0", "prompt 1", "prompt 2", "prompt 3"]
    disambiguation_passes = []
    monkeypatch.setattr(
        cluster_layer,
        "disambiguate_topics",
        lambda **kwargs: disambiguation_passes.append(list(cluster_layer.topic_names)),
    )

    cluster_layer.name_topics(FirstPassFailsLLM(), 1.0, [[]], "sentences", "about cats")

    assert disambiguation_passes == [["", "", "", "Cats"]]
