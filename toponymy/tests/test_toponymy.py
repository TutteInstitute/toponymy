from unittest.mock import MagicMock
from toponymy.cluster_layer import (
    ClusterLayerSummaryText,
    ClusterLayerText,
)
from toponymy.templates import PROMPT_TEMPLATES, SUMMARY_PROMPT_TEMPLATES

from toponymy.toponymy import Toponymy
from toponymy.llm_wrappers import (
    OllamaNamer,
    AsyncOllamaNamer,
    LLMWrapper,
)
from toponymy.tools.notebook_test_helpers import get_test_ollama_model
from toponymy.tests.conftest import ollama_has_model
from toponymy.keyphrases import KeyphraseBuilder

import itertools
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from pathlib import Path

import numpy as np
import pandas as pd

import pytest
import logging

logger = logging.getLogger(__name__)


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
        verbose=True,
    )
    model.fit(all_sentences, object_vectors, clusterable_vectors)
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    logger.info(f"Topic names: {model.topic_names_[1]}")
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.5
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )


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
        verbose=True,
    )
    model.fit(all_sentences, object_vectors, clusterable_vectors)
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    logger.info(f"Topic names: {model.topic_names_[1]}")
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.5
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )


def test_toponymy_resyncs_runtime_layer_config_for_prefit_clusterer(
    llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
):
    # Pre-fit clusterer with old settings
    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        exemplar_delimiters=["<<OLD>>", "<</OLD>>"],
        show_progress_bar=False,
        verbose=False,
    )

    new_exemplar_delimiters = ["<EXAMPLE>", "</EXAMPLE>"]

    model = Toponymy(
        llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        exemplar_delimiters=new_exemplar_delimiters,
        verbose=True,
    )

    model.fit(all_sentences, object_vectors, clusterable_vectors)

    # runtime config should be updated
    assert all(
        layer.exemplar_delimiters == new_exemplar_delimiters
        for layer in model.cluster_layers_
    )

    assert all(
        layer.show_progress_bar == model.show_progress_bars
        for layer in model.cluster_layers_
    )

    assert all(layer.verbose == model.verbose for layer in model.cluster_layers_)


# @pytest.mark.skip(reason="Lowering runtime for CI")
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
        verbose=True,
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


# @pytest.mark.skip(reason="Lowering runtime for CI")
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
        verbose=True,
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


# @pytest.mark.skip(reason="Lowering runtime for CI")
def test_toponymy_with_ollama(
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
    ollama_running,
):
    """Integration test using Ollama for local LLM inference.

    Uses a very small model suitable for CI environments with limited resources.
    The test will check system resources and skip if insufficient.
    """

    model_name = get_test_ollama_model()
    if not ollama_running:
        pytest.skip("Ollama service is not available or failed to start")

    if not ollama_has_model(model_name):
        pytest.skip(f"Ollama model {model_name} is not available")

    ollama_llm = OllamaNamer(
        model=model_name,  # Very small model for CI
        api_base="http://localhost:11434",
    )

    # Set a timeout for the entire model fitting process
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Ollama test timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # 2 minute timeout for model fitting

    try:
        model = Toponymy(
            ollama_llm,
            embedder,
            clusterer,
            keyphrase_builder=KeyphraseBuilder(n_jobs=1),
            object_description="sentences",
            corpus_description="collection of sentences",
            lowest_detail_level=0.8,
            highest_detail_level=1.0,
            verbose=True,
        )

        model.fit(all_sentences, object_vectors, clusterable_vectors)

        # Verify that topic names were generated
        n_topic_names = len(model.topic_names_[1])
        logger.info(f"Number of sync Ollama generated topic names: {n_topic_names}")
        logger.info(f"Ollama generated topic names: {model.topic_names_[1]}")
        assert n_topic_names > 0

        # Check that topic names are mostly reasonable strings
        non_empty = [t for t in model.topic_names_[1] if t.strip()]
        assert len(non_empty) > 0, "No topic names were generated at all"
        assert (
            len(non_empty) / n_topic_names >= 0.8
        ), f"Too many empty topic names: {len(non_empty)}/{n_topic_names} non-empty"

        # Verify cluster structure is maintained
        assert len(model.cluster_layers_[1].cluster_labels) == len(cluster_label_vector)

    except TimeoutError:
        pytest.skip("Ollama test timed out - likely too slow for CI environment")
    finally:
        signal.alarm(0)  # Cancel the alarm


# @pytest.mark.skip(reason="Lowering runtime for CI")
def test_toponymy_async_ollama(
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
    ollama_running,
):
    """Integration test using AsyncOllama for concurrent local LLM inference.

    Uses a very small model suitable for CI environments with limited resources.
    The test will check system resources and skip if insufficient.
    """

    model_name = get_test_ollama_model()
    if not ollama_running:
        pytest.skip("Ollama service is not available or failed to start")

    if not ollama_has_model(model_name):
        pytest.skip(f"Ollama model {model_name} is not available")

    # Use async Ollama with very limited concurrency for small models
    async_ollama_llm = AsyncOllamaNamer(
        model=model_name,  # Very small model for CI
        api_base="http://localhost:11434",
        max_concurrent_requests=1,  # Very conservative for small models and CI
    )

    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        object_to_text_function=lambda x: x,
    )

    # Set a timeout for the entire async model fitting process
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("AsyncOllama test timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)  # 3 minute timeout for async model fitting

    try:
        model = Toponymy(
            async_ollama_llm,
            embedder,
            clusterer,
            keyphrase_builder=KeyphraseBuilder(
                n_jobs=1, verbose=True, embedder=embedder
            ),
            object_description="sentences",
            corpus_description="collection of sentences",
            lowest_detail_level=0.8,
            highest_detail_level=1.0,
            verbose=True,
        )

        topic_name_vectors = model.fit_predict(
            all_sentences,
            object_vectors,
            clusterable_vectors,
            keyphrase_method="bm25",
            subtopic_method="information_weighted",
        )

        # Verify that topic names were generated
        n_topic_names = len(model.topic_names_[1])
        logger.info(f"Number of async Ollama generated topic names: {n_topic_names}")
        logger.info(f"Async Ollama generated topic names: {model.topic_names_[1]}")

        assert n_topic_names > 0

        # Check that topic names are mostly reasonable strings
        non_empty = [t for t in model.topic_names_[1] if t.strip()]
        assert len(non_empty) > 0, "No topic names were generated at all"
        assert (
            len(non_empty) / n_topic_names >= 0.8
        ), f"Too many empty topic names: {len(non_empty)}/{n_topic_names} non-empty"

        # Verify cluster structure is maintained
        assert len(model.cluster_layers_[1].cluster_labels) == len(cluster_label_vector)

        # Verify topic tree was generated
        assert len(str(model.topic_tree_)) > 10

    except TimeoutError:
        pytest.skip("AsyncOllama test timed out - likely too slow for CI environment")
    finally:
        signal.alarm(0)  # Cancel the alarm


class MockNamer(LLMWrapper):
    """In-memory namer returning canned JSON. Distinguishes single-topic
    naming (max_tokens=128) from cluster naming (max_tokens=1024) similarly
    to the original Ollama mock."""

    def __init__(self, single_responses, cluster_responses):
        self.model = "mock-model"
        self._single = itertools.cycle(single_responses)
        self._cluster = itertools.cycle(cluster_responses)

    def _next(self, max_tokens):
        return next(self._cluster if max_tokens > 500 else self._single)

    def _call_llm(self, prompt, temperature, max_tokens):
        return self._next(max_tokens)

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        return self._next(max_tokens)


def test_toponymy_with_mock_namer(
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
):
    single_topic_responses = [
        '{"topic_name": "Technology and Innovation", "topic_specificity": 0.8}',
        '{"topic_name": "Data Science Methods", "topic_specificity": 0.7}',
        '{"topic_name": "Machine Learning Applications", "topic_specificity": 0.9}',
        '{"topic_name": "AI Research", "topic_specificity": 0.6}',
        '{"topic_name": "Computing Technologies", "topic_specificity": 0.5}',
    ]

    cluster_topic_responses = [
        '{"new_topic_name_mapping": ["Advanced Technology", "Data Analytics", "ML Systems", "AI Studies", "Tech Infrastructure"], "topic_specificities": [0.8, 0.7, 0.9, 0.6, 0.5]}',
        '{"new_topic_name_mapping": ["Innovation Hub", "Science Methods", "Learning Apps", "Research Areas", "Computing Tech"], "topic_specificities": [0.75, 0.65, 0.85, 0.55, 0.45]}',
    ]

    namer = MockNamer(single_topic_responses, cluster_topic_responses)

    model = Toponymy(
        namer,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        verbose=True,
    )
    model.fit(all_sentences, object_vectors, clusterable_vectors)

    # Verify topic names
    assert len(model.topic_names_[1]) > 0
    for topic_name in model.topic_names_[1]:
        assert isinstance(topic_name, str)
        assert len(topic_name.strip()) > 0
    # Verify cluster structure
    assert len(model.cluster_layers_[1].cluster_labels) == len(cluster_label_vector)


def test_toponymy_uses_summary_templates_for_summary_layer_class():
    model = Toponymy(
        MagicMock(),
        MagicMock(),
        layer_class=ClusterLayerSummaryText,
        prompt_template=PROMPT_TEMPLATES,
    )

    assert model.prompt_template == SUMMARY_PROMPT_TEMPLATES


def test_toponymy_uses_default_templates_for_regular_layer_class():
    model = Toponymy(
        MagicMock(),
        MagicMock(),
        layer_class=ClusterLayerText,
    )

    assert model.prompt_template == PROMPT_TEMPLATES


def test_toponymy_does_not_override_custom_prompt_templates():
    custom_template = {"custom": "template"}

    model = Toponymy(
        MagicMock(),
        MagicMock(),
        layer_class=ClusterLayerSummaryText,
        prompt_template=custom_template,
    )

    assert model.prompt_template is custom_template
