from toponymy.cluster_layer import ClusterLayerText, ClusterLayer
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

EMBEDDER = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

SUBTOPIC_OBJECTS = json.load(open(Path(__file__).parent / "subtopic_objects.json", "r"))
ALL_SENTENCES = sum(
    [x["sentences"] for subtopics in SUBTOPIC_OBJECTS for x in subtopics["subtopics"]],
    [],
)
CLUSTER_LABEL_VECTOR = np.arange(5).repeat(25)
SUBTOPIC_LABEL_VECTOR = np.arange(25).repeat(5)
OBJECT_VECTORS = EMBEDDER.encode(ALL_SENTENCES)
CLUSTER_CENTROID_VECTORS = centroids_from_labels(CLUSTER_LABEL_VECTOR, OBJECT_VECTORS)
SUBTOPIC_CENTROID_VECTORS = centroids_from_labels(SUBTOPIC_LABEL_VECTOR, OBJECT_VECTORS)
SUBTOPICS = [[x["subtopic"] for x in topic["subtopics"]] for topic in SUBTOPIC_OBJECTS]
ALL_SUBTOPICS = sum(SUBTOPICS, [])
SUBTOPIC_VECTORS = EMBEDDER.encode(ALL_SUBTOPICS)
CLUSTER_TREE = {(1, i): [(0, i * 5 + j) for j in range(5)] for i in range(5)}


def test_abstract_cluster_layer():
    ClusterLayer.__abstractmethods__ = set()

    @dataclass
    class ConcreteClusterLayer(ClusterLayer):
        pass

    concrete_cluster_layer = ConcreteClusterLayer()
    exemplar_texts = concrete_cluster_layer.make_exemplar_texts(
        ALL_SENTENCES, OBJECT_VECTORS
    )
    keyphrases = concrete_cluster_layer.make_keyphrases(
        ALL_SENTENCES, csr_matrix(np.array([[1, 0], [0, 1]])), OBJECT_VECTORS, EMBEDDER
    )
    subtopics = concrete_cluster_layer.make_subtopics(
        ALL_SUBTOPICS, SUBTOPIC_LABEL_VECTOR, SUBTOPIC_CENTROID_VECTORS
    )
    prompts = concrete_cluster_layer.make_prompts(
        1.0,
        [ALL_SUBTOPICS, []],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )
    topics = concrete_cluster_layer.name_topics(
        None,
        0.0,
        [ALL_SUBTOPICS, [x["topic"] for x in SUBTOPIC_OBJECTS]],
        "objects",
        "collection of objects",
        CLUSTER_TREE,
        EMBEDDER,
    )
    assert concrete_cluster_layer
    assert isinstance(concrete_cluster_layer, ClusterLayer)
    assert isinstance(ClusterLayer, ABCMeta)
    assert exemplar_texts is None
    assert keyphrases is None
    assert subtopics is None
    assert prompts is None
    assert topics is None


def test_layer_creation():
    cluster_layer = ClusterLayerText(
        CLUSTER_LABEL_VECTOR,
        CLUSTER_CENTROID_VECTORS,
        1,
        EMBEDDER,
    )
    assert cluster_layer


def test_make_data():
    cluster_layer = ClusterLayerText(
        CLUSTER_LABEL_VECTOR,
        CLUSTER_CENTROID_VECTORS,
        1,
        EMBEDDER,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(ALL_SENTENCES)
    keyphrase_vectors = EMBEDDER.encode(keyphrases)
    cluster_layer.make_exemplar_texts(ALL_SENTENCES, OBJECT_VECTORS)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, EMBEDDER)
    cluster_layer.make_subtopics(
        ALL_SUBTOPICS, SUBTOPIC_LABEL_VECTOR, SUBTOPIC_CENTROID_VECTORS
    )
    cluster_layer.make_prompts(
        1.0,
        [ALL_SUBTOPICS, []],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in SUBTOPIC_OBJECTS]
    cluster_layer.embed_topic_names(EMBEDDER)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [ALL_SUBTOPICS, [x["topic"] for x in SUBTOPIC_OBJECTS]],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )

def test_make_data_alternative_methods1():
    cluster_layer = ClusterLayerText(
        CLUSTER_LABEL_VECTOR,
        CLUSTER_CENTROID_VECTORS,
        1,
        EMBEDDER,
        prompt_format="system_user",
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(ALL_SENTENCES)
    keyphrase_vectors = EMBEDDER.encode(keyphrases)
    cluster_layer.make_exemplar_texts(ALL_SENTENCES, OBJECT_VECTORS)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, EMBEDDER, method="central")
    cluster_layer.make_subtopics(
        ALL_SUBTOPICS, SUBTOPIC_LABEL_VECTOR, SUBTOPIC_CENTROID_VECTORS
    )
    cluster_layer.make_prompts(
        1.0,
        [ALL_SUBTOPICS, []],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in SUBTOPIC_OBJECTS]
    cluster_layer.embed_topic_names(EMBEDDER)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [ALL_SUBTOPICS, [x["topic"] for x in SUBTOPIC_OBJECTS]],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )

def test_make_data_alternative_methods2():
    cluster_layer = ClusterLayerText(
        CLUSTER_LABEL_VECTOR,
        CLUSTER_CENTROID_VECTORS,
        1,
        EMBEDDER,
        prompt_format="combined",
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(ALL_SENTENCES)
    keyphrase_vectors = EMBEDDER.encode(keyphrases)
    cluster_layer.make_exemplar_texts(ALL_SENTENCES, OBJECT_VECTORS)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, EMBEDDER, method="bm25")
    cluster_layer.make_subtopics(
        ALL_SUBTOPICS, SUBTOPIC_LABEL_VECTOR, SUBTOPIC_CENTROID_VECTORS
    )
    cluster_layer.make_prompts(
        1.0,
        [ALL_SUBTOPICS, []],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )
    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in SUBTOPIC_OBJECTS]
    cluster_layer.embed_topic_names(EMBEDDER)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [ALL_SUBTOPICS, [x["topic"] for x in SUBTOPIC_OBJECTS]],
        "sentences",
        "about specific popular topics",
        CLUSTER_TREE,
    )
