"""The get_topic_name(s)_regex arguments must reach the parser."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from toponymy.cluster_layer import ClusterLayerSummaryText
from toponymy.llm_wrappers import AsyncLLMWrapper, LLMWrapper, llm_output_to_result
from toponymy.templates import (
    MULTILINGUAL_EN_FR_PROMPT_TEMPLATES,
    PROMPT_TEMPLATES,
    SUMMARY_PROMPT_TEMPLATES,
)
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


TEMPLATE_FIELDS = {
    "default": (PROMPT_TEMPLATES, {"topic_name": "Goalie masks"}),
    "summary": (
        SUMMARY_PROMPT_TEMPLATES,
        {
            "topic_analysis": "Masks worn by goalies.",
            "topic_summary": "Posts about goalie masks.",
            "topic_name": "Goalie masks",
        },
    ),
    "multilingual": (
        MULTILINGUAL_EN_FR_PROMPT_TEMPLATES,
        {
            "english_topic_name": "Goalie masks",
            "nom_du_sujet_en_français": "Masques de gardien",
        },
    ),
}


@pytest.mark.parametrize("with_specificity", [True, False])
@pytest.mark.parametrize("template_id", TEMPLATE_FIELDS)
def test_topic_name_regex_makes_specificity_optional(template_id, with_specificity):
    templates, fields = TEMPLATE_FIELDS[template_id]
    reply = dict(fields, topic_specificity=0.8) if with_specificity else fields
    raw = "Here you go:\n" + json.dumps(reply, ensure_ascii=False) + "\nDone."

    parsed = llm_output_to_result(raw, templates["layer"]["get_topic_name_regex"])

    assert parsed == reply


@pytest.mark.parametrize("template_id", TEMPLATE_FIELDS)
def test_topic_name_regex_without_specificity_keeps_a_brace_in_a_name(template_id):
    templates, fields = TEMPLATE_FIELDS[template_id]
    reply = {key: value + " {v2}" for key, value in fields.items()}
    raw = json.dumps(reply, ensure_ascii=False)

    parsed = llm_output_to_result(raw, templates["layer"]["get_topic_name_regex"])

    assert parsed == reply


def test_summary_reply_without_specificity_from_ci_parses():
    # qwen2.5:0.5b's reply in CI, Nov 2026
    raw = (
        '{"topic_analysis": "Mask and Goalie Masks", "topic_summary": "The group '
        "discusses mask and goalie masks, including their custom-fit design, usage, "
        "and the specific mask used by John Vanbiesbrouck for the Philadelphia "
        "Flyers. The summary is 100 words. The name is 'Mask and Goalie Masks' with "
        'a score of 8.0.", "topic_name": "Mask and Goalie Masks"}'
    )
    layer = SUMMARY_PROMPT_TEMPLATES["layer"]

    parsed = llm_output_to_result(raw, layer["get_topic_name_regex"])

    assert parsed["topic_name"] == "Mask and Goalie Masks"
    assert layer["extract_topic_name"](parsed)[0] == "Mask and Goalie Masks"


MAPPING = {"1. Old A": "New {A}", "2. Old B": "New B"}


@pytest.mark.parametrize("with_specificities", [True, False])
@pytest.mark.parametrize(
    "templates",
    [PROMPT_TEMPLATES, SUMMARY_PROMPT_TEMPLATES, MULTILINGUAL_EN_FR_PROMPT_TEMPLATES],
    ids=["default", "summary", "multilingual"],
)
def test_cluster_names_regex_makes_specificities_optional(
    templates, with_specificities
):
    reply = {"new_topic_name_mapping": MAPPING}
    if with_specificities:
        reply["topic_specificities"] = [0.9, 0.7]
    raw = "Sure:\n" + json.dumps(reply) + "\n"
    disambiguate = templates["disambiguate_topics"]

    parsed = llm_output_to_result(raw, disambiguate["get_topic_names_regex"])

    assert parsed == reply
    names = disambiguate["extract_topic_names"](parsed, ["Old A", "Old B"], raw)
    assert names == ["New {A}", "New B"]
