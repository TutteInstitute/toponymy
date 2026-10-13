import json
import os
import re

from prompt_toolkit import prompt
import pytest

from toponymy.llm_wrappers import AsyncOllamaNamer, OllamaNamer
from toponymy.prompt_construction import render_prompt
from toponymy.templates import PROMPT_TEMPLATES
from toponymy.tests.conftest import ollama_has_model
from toponymy.tests.helpers.llm_test_config import SYNC_SUPPORTED_NAMERS
from toponymy.tools.notebook_test_helpers import get_test_ollama_model
from toponymy.tools.load_prompt_examples import load_prompt_params


def load_json(filename):
    """Read a json file from disk"""
    with open(filename) as f:
        obj = json.load(f)
    return obj


def matches_expected_patterns(result, patterns):
    """Check if the result matches any of the expected patterns"""
    text = " ".join(result) if isinstance(result, tuple) else str(result)
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return any(re.search(p, normalized) for p in patterns)


def test_ollama_sync_smoke_test(ollama_running):
    model = get_test_ollama_model()
    if not ollama_running:
        pytest.skip("Ollama service is not available or failed to start")
    if not ollama_has_model(model):
        pytest.skip(f"{model} not available in local Ollama")
    namer = OllamaNamer(model=model)
    result = namer.connectivity_status()

    render_params = load_prompt_params("hockey_masks")
    prompt = render_prompt(PROMPT_TEMPLATES["layer"], render_params)

    # Note: do not match on hockey since it's too broad
    accepted_patterns = [
        r"\bgoalie\b",
        r"\bgoaltend",  # matches goaltender/goaltending
        r"\bmask",  # matches mask/masks
    ]

    result = namer.generate_topic_name(prompt)
    assert type(result) == str
    assert len(result) > 0
    assert matches_expected_patterns(result, accepted_patterns)


@pytest.mark.external
@pytest.mark.parametrize(
    "namer_class, api_key_name, namer_kwargs", SYNC_SUPPORTED_NAMERS
)
def test_supported_models_sync_smoke_test(namer_class, api_key_name, namer_kwargs):
    if os.getenv(api_key_name) is None:
        pytest.skip(f"{api_key_name} is not set in the environment")

    namer = namer_class(**namer_kwargs)
    result = namer.connectivity_status()

    render_params = load_prompt_params("hockey_masks")
    prompt = render_prompt(PROMPT_TEMPLATES["layer"], render_params)

    # Note: do not match on hockey since it's too broad
    accepted_patterns = [
        r"\bgoalie\b",
        r"\bgoaltend",  # matches goaltender/goaltending
        r"\bmask",  # matches mask/masks
    ]

    result = namer.generate_topic_name(prompt)
    assert type(result) == str
    assert len(result) > 0
    assert matches_expected_patterns(result, accepted_patterns)
