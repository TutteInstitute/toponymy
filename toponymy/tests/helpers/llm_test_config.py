import json
from typing import List
import os
import pytest
from toponymy.llm_wrappers import (
    AnthropicNamer,
    AsyncAnthropicNamer,
    AsyncAzureAINamer,
    AsyncCohereNamer,
    AsyncGoogleGeminiNamer,
    AsyncHuggingFaceNamer,
    AsyncLiteLLMNamer,
    AsyncOllamaNamer,
    AsyncOpenAINamer,
    AsyncTogether,
    AsyncVLLMNamer,
    AzureAINamer,
    BatchAnthropicNamer,
    BatchAzureAINamer,
    CohereBatchNamer,
    CohereNamer,
    GoogleGeminiNamer,
    HuggingFaceNamer,
    LiteLLMNamer,
    LlamaCppNamer,
    OllamaNamer,
    OpenAINamer,
    ReplicateNamer,
    TogetherNamer,
    VLLMNamer,
    AsyncOpenAINamer,
    AsyncLiteLLMNamer,
    GoogleGeminiNamer,
    HuggingFaceNamer,
    LiteLLMNamer,
    LlamaCppNamer,
    OpenAINamer,
)

# Mock responses for different scenarios
VALID_TOPIC_NAME_RESPONSE = {"topic_name": "Machine Learning", "topic_specificity": 0.6}

VALID_CLUSTER_NAMES_RESPONSE = {
    "new_topic_name_mapping": {
        "1. data": "Data Science",
        "2. ml": "Machine Learning\\ML",
        "3. ai": "Artificial Intelligence",
    },
    "topic_specificities": [
        0.6,
        0.8,
        0.7,
    ],
}

MALFORMED_JSON_RESPONSE = "{"  # Incomplete JSON
RECOVERABLE_MALFORMED_JSON_RESPONSE = """
the topic name is Machine Learning
```json
{"topic_name": "Machine Learning", "topic_specificity": 0.6}
```
postamble.
"""
EMPTY_MAPPING_RESPONSE = {"new_topic_name_mapping": {}}
MALFORMED_MAPPING_RESPONSE = """{"new_topic_name_mapping": {"data science": "Data Science", "data science": "Machine Learning\\ML", "data science": "Artificial Intelligence"} , "topic_specificities": [0.6, 0.8, 0.7]}"""


def make_mock_data():
    return {
        "valid_topic_name": json.dumps(VALID_TOPIC_NAME_RESPONSE),
        "valid_cluster_names": json.dumps(VALID_CLUSTER_NAMES_RESPONSE),
        "old_names": ["data", "ml", "ai"],
        "old_names_list": [["data", "ml", "ai"], ["x", "y", "z"]],
        "malformed_mapping": MALFORMED_MAPPING_RESPONSE,
        "malformed_json": MALFORMED_JSON_RESPONSE,
        "recoverable_malformed_json": RECOVERABLE_MALFORMED_JSON_RESPONSE,
        "empty_mapping_response": json.dumps(EMPTY_MAPPING_RESPONSE),
    }


def make_prompt(label: object = "test") -> dict:
    """
    Build a prompt of the shape toponymy.prompt_construction produces.

    Prompts carry every rendering, and the LLM wrapper picks one at call time, so a
    test prompt needs all of them regardless of which one the wrapper under test uses.
    """
    return {
        "system": f"system prompt {label}",
        "user": f"user prompt {label}",
        "combined": f"combined prompt {label}",
    }


# Helper functions for validation
def validate_topic_name(result: str):
    assert result == "Machine Learning"


def validate_cluster_names(result: List[str]):
    expected = ["Data Science", "Machine Learning\\ML", "Artificial Intelligence"]
    assert result == expected


LITELLM_PROVIDER_CASES = [
    pytest.param(
        {
            "provider_name": "OpenAI",
            "model": "openai/gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
        id="openai",
    ),
    pytest.param(
        {
            "provider_name": "Anthropic",
            "model": "anthropic/claude-haiku-4-5-20251001",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        id="anthropic",
    ),
]

SUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS = [
    (AnthropicNamer, {"api_key": "dummy"}),
    (LiteLLMNamer, {"api_key": "dummy"}),
    (OpenAINamer, {"api_key": "dummy"}),
    (CohereNamer, {"api_key": "dummy"}),
    (TogetherNamer, {"api_key": "dummy"}),
    (AzureAINamer, {"api_key": "dummy", "model": "dummy"}),
    (GoogleGeminiNamer, {"api_key": "dummy"}),
    (OllamaNamer, {"api_key": "dummy"}),
    (ReplicateNamer, {"api_key": "dummy"}),
]
UNSUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS = [
    (HuggingFaceNamer, {"model": "hf-internal-testing/tiny-random-gpt2"}),
    # exclude namers needing mocking to get around setup
    # (VLLMNamer, {}),
    # (LlamaCppNamer, {"model_path": "dummy/path/to/model.gguf"}),
]
SUPPORTED_ASYNC_DEBUG_CALLBACK_NAMERS = [
    (AsyncAnthropicNamer, {"api_key": "dummy"}),
    (AsyncLiteLLMNamer, {"api_key": "dummy"}),
    (AsyncOpenAINamer, {"api_key": "dummy"}),
    (AsyncCohereNamer, {"api_key": "dummy"}),
    (AsyncTogether, {"api_key": "dummy"}),
    (AsyncAzureAINamer, {"api_key": "dummy", "model": "dummy"}),
    (AsyncGoogleGeminiNamer, {"api_key": "dummy"}),
    (AsyncOllamaNamer, {"api_key": "dummy"}),
]
UNSUPPORTED_ASYNC_DEBUG_CALLBACK_NAMERS = [
    (AsyncHuggingFaceNamer, {"model": "hf-internal-testing/tiny-random-gpt2"}),
    (CohereBatchNamer, {"api_key": "dummy"}),
    (BatchAnthropicNamer, {"api_key": "dummy"}),
    (
        BatchAzureAINamer,
        {"api_key": "dummy", "endpoint": "dummy", "model": "dummy/model"},
    ),
    # exclude namers needing mocking to get around setup
    # (AsyncVLLMNamer, {}),
    #
]

SYNC_LITELLM_NAMERS = [
    (AnthropicNamer, {"api_key": "dummy"}),
    (LiteLLMNamer, {"api_key": "dummy"}),
    (OpenAINamer, {"api_key": "dummy"}),
    (CohereNamer, {"api_key": "dummy"}),
    (TogetherNamer, {"api_key": "dummy"}),
    (AzureAINamer, {"api_key": "dummy", "model": "dummy"}),
    (GoogleGeminiNamer, {"api_key": "dummy"}),
    (OllamaNamer, {"api_key": "dummy"}),
    (ReplicateNamer, {"api_key": "dummy"}),
]

ASYNC_LITELLM_NAMERS = [
    (AsyncAnthropicNamer, {"api_key": "dummy"}),
    (AsyncLiteLLMNamer, {"api_key": "dummy"}),
    (AsyncOpenAINamer, {"api_key": "dummy"}),
    (AsyncCohereNamer, {"api_key": "dummy"}),
    (AsyncTogether, {"api_key": "dummy"}),
    (AsyncAzureAINamer, {"api_key": "dummy", "model": "dummy"}),
    (AsyncGoogleGeminiNamer, {"api_key": "dummy"}),
    (AsyncOllamaNamer, {"api_key": "dummy"}),
]
