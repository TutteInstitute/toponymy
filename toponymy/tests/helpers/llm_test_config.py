import json
from typing import List
import os
import pytest


# Mock responses for different scenarios
VALID_TOPIC_NAME_RESPONSE = {
    "topic_name": "Machine Learning",
    "topic_specificity": 0.6
}

VALID_CLUSTER_NAMES_RESPONSE = {
    "new_topic_name_mapping": {
        "1. data": "Data Science",
        "2. ml": "Machine Learning\\ML",
        "3. ai": "Artificial Intelligence"
    },
    "topic_specificities": [
        0.6,
        0.8,
        0.7,
    ]
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