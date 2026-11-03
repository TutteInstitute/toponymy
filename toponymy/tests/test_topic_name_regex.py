"""The get_topic_name(s)_regex arguments must reach the parser."""

import json
from unittest.mock import Mock

import numpy as np

from toponymy.cluster_layer import ClusterLayerSummaryText
from toponymy.llm_wrappers import AsyncLLMWrapper, LLMWrapper
from toponymy.tests.helpers.llm_test_config import make_prompt

SUMMARY_RESPONSE = json.dumps(
    {
        "topic_analysis": "analysis",
        "topic_summary": "summary",
        "topic_name": "Named topic",
        "topic_specificity": 0.5,
    }
)


class CustomFormatWrapper(LLMWrapper):
    model = "dummy-model"

    def _call_llm(self, prompt, temperature, max_tokens):
        return json.dumps({"names": ["A", "B", "C"]})

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        return self._call_llm(prompt, temperature, max_tokens)


class SummaryBatchWrapper(AsyncLLMWrapper):
    model = "dummy-model"

    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return [SUMMARY_RESPONSE] * len(prompts)

    async def _call_llm_with_system_prompt_batch(
        self, prompts, temperature, max_tokens
    ):
        return [SUMMARY_RESPONSE] * len(prompts)


def test_generate_topic_cluster_names_uses_the_given_regex():
    names = CustomFormatWrapper().generate_topic_cluster_names(
        make_prompt(),
        ["a", "b", "c"],
        extract_topic_names_function=lambda info, old_names, raw: info["names"],
        get_topic_names_regex=r"\{.*\}",
    )
    assert names == ["A", "B", "C"]


def test_async_summary_layer_names_topics_without_a_template(monkeypatch):
    n_clusters = 3
    layer = ClusterLayerSummaryText(
        np.arange(n_clusters).repeat(4),
        np.eye(n_clusters),
        0,
        Mock(),
        show_progress_bar=False,
    )
    layer.prompts = [make_prompt(i) for i in range(n_clusters)]
    monkeypatch.setattr(layer, "disambiguate_topics", lambda **kwargs: None)

    layer.name_topics(
        SummaryBatchWrapper(),
        detail_level=0.5,
        all_topic_names=[[]],
        object_description="documents",
        corpus_description="a corpus",
        all_topic_summaries=[[]],
        all_topic_explanations=[[]],
    )

    assert layer.topic_names == ["Named topic"] * n_clusters
    assert layer.topic_summaries == ["summary"] * n_clusters
