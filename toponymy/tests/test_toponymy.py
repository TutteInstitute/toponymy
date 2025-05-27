from toponymy.toponymy import Toponymy
from toponymy.llm_wrappers import HuggingFace, AsyncHuggingFace
from toponymy.clustering import centroids_from_labels, ToponymyClusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.cluster_layer import ClusterLayerText
from sentence_transformers import SentenceTransformer

from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from pathlib import Path

import numpy as np
import pandas as pd
import json
import umap

import pytest

#LLM = HuggingFace("Qwen/Qwen2.5-0.5B-Instruct")
LLM = HuggingFace("Qwen/Qwen3-0.6B", llm_specific_instructions=" /no_think")
# LLM = HuggingFace("Qwen/Qwen3-0.6B-GPTQ-Int8", llm_specific_instructions=" /no_think")
ASYNC_LLM = AsyncHuggingFace(
    "Qwen/Qwen3-0.6B",
    llm_specific_instructions=" /no_think",
    max_concurrent_requests=4,
)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
SUBTOPIC_OBJECTS = json.load(open(Path(__file__).parent / "subtopic_objects.json", "r"))
ALL_SENTENCES = sum(
    [x["sentences"] for subtopics in SUBTOPIC_OBJECTS for x in subtopics["subtopics"]],
    [],
)
CLUSTER_LABEL_VECTOR = np.arange(5).repeat(25)
SUBTOPIC_LABEL_VECTOR = np.arange(25).repeat(5)
OBJECT_VECTORS = EMBEDDER.encode(ALL_SENTENCES)
CLUSTERABLE_VECTORS = umap.UMAP(
    metric="cosine", n_neighbors=15, n_components=2, min_dist=0.001, random_state=42
).fit_transform(OBJECT_VECTORS)
CLUSTERER = ToponymyClusterer(
    min_samples=5,
    base_min_cluster_size=4,
    next_cluster_size_quantile=1.0,
    min_clusters=4,
    verbose=True,
)


def test_toponymy():
    model = Toponymy(
        LLM,
        EMBEDDER,
        CLUSTERER,
        keyphrase_builder = KeyphraseBuilder(n_jobs=1),
        object_description = "sentences",
        corpus_description = "collection of sentences",
        lowest_detail_level = 0.8,
        highest_detail_level = 1.0,
        show_progress_bars=True,
    )
    model.fit(ALL_SENTENCES, OBJECT_VECTORS, CLUSTERABLE_VECTORS)
    embedded_topic_names = EMBEDDER.encode(model.topic_names_[1])
    print(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        EMBEDDER.encode([topic["topic"] for topic in SUBTOPIC_OBJECTS]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.5
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == CLUSTER_LABEL_VECTOR
    )

def test_toponymy_alternative_options():
    CLUSTERER.fit(CLUSTERABLE_VECTORS, OBJECT_VECTORS, prompt_format="combined", object_to_text_function=lambda x: x)
    model = Toponymy(
        ASYNC_LLM,
        EMBEDDER,
        CLUSTERER,
        keyphrase_builder = KeyphraseBuilder(n_jobs=1, verbose=True, embedder=EMBEDDER),
        object_description = "sentences",
        corpus_description = "collection of sentences",
        lowest_detail_level = 0.8,
        highest_detail_level = 1.0,
        show_progress_bars=True,
    )
    topic_name_vectors = model.fit_predict(ALL_SENTENCES, OBJECT_VECTORS, CLUSTERABLE_VECTORS, keyphrase_method="bm25", subtopic_method="information_weighted")
    embedded_topic_names = EMBEDDER.encode(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        EMBEDDER.encode([topic["topic"] for topic in SUBTOPIC_OBJECTS]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.5
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == CLUSTER_LABEL_VECTOR
    )
    assert len(str(model.topic_tree_)) > 10

def test_toponymy_alternative_options_2():
    CLUSTERER.fit(CLUSTERABLE_VECTORS, OBJECT_VECTORS, prompt_format="system_user", object_to_text_function=lambda x: x)
    model = Toponymy(
        ASYNC_LLM,
        EMBEDDER,
        CLUSTERER,
        keyphrase_builder = KeyphraseBuilder(n_jobs=1, verbose=True, embedder=EMBEDDER),
        object_description = "sentences",
        corpus_description = "collection of sentences",
        lowest_detail_level = 0.8,
        highest_detail_level = 1.0,
        show_progress_bars=True,
    )
    topic_name_vectors = model.fit_predict(ALL_SENTENCES, OBJECT_VECTORS, CLUSTERABLE_VECTORS, keyphrase_method="bm25", subtopic_method="information_weighted")
    embedded_topic_names = EMBEDDER.encode(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        EMBEDDER.encode([topic["topic"] for topic in SUBTOPIC_OBJECTS]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.5
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == CLUSTER_LABEL_VECTOR
    )
    assert len(str(model.topic_tree_)) > 10