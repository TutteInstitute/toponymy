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


def test_toponymy(
    llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
):
    model = Toponymy(
        llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        show_progress_bars=True,
    )
    model.fit(all_sentences, object_vectors, clusterable_vectors)
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    print(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.7
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )


def test_toponymy_alternative_options(
    async_llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    subtopic_objects,
    cluster_label_vector,
):
    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        prompt_format="combined",
        object_to_text_function=lambda x: x,
    )
    model = Toponymy(
        async_llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1, verbose=True, embedder=embedder),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        show_progress_bars=True,
    )
    topic_name_vectors = model.fit_predict(
        all_sentences,
        object_vectors,
        clusterable_vectors,
        keyphrase_method="bm25",
        subtopic_method="information_weighted",
    )
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.7
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )
    assert len(str(model.topic_tree_)) > 10


def test_toponymy_alternative_options_2(
    async_llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    subtopic_objects,
    cluster_label_vector,
):
    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        prompt_format="system_user",
        object_to_text_function=lambda x: x,
    )
    model = Toponymy(
        async_llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1, verbose=True, embedder=embedder),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        show_progress_bars=True,
    )
    topic_name_vectors = model.fit_predict(
        all_sentences,
        object_vectors,
        clusterable_vectors,
        keyphrase_method="bm25",
        subtopic_method="information_weighted",
    )
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.7
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )
    assert len(str(model.topic_tree_)) > 10
