import json
import os
import re

from unittest.mock import AsyncMock
from prompt_toolkit import prompt
import pytest

from toponymy.llm_wrappers import AsyncOllamaNamer, OllamaNamer
from toponymy.prompt_construction import render_prompt
from toponymy.templates import PROMPT_TEMPLATES
from toponymy.tests.conftest import ollama_has_model
from toponymy.tests.helpers.llm_test_config import (
    ASYNC_SUPPORTED_NAMERS,
    SYNC_SUPPORTED_NAMERS,
)
from toponymy.tools.notebook_test_helpers import get_test_ollama_model
from toponymy.tools.load_prompt_examples import load_prompt_params
from toponymy.tests.test_llm_wrapper_base import DummySingleWrapper
from toponymy.tests.test_async_llm_wrappers_base import (
    DummySingleWrapper as DummyAsyncSingleWrapper,
)


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
    namer = OllamaNamer(model=model, temperature_override=0.0)
    assert namer.connectivity_status()["success"] == True

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
    assert namer.connectivity_status()["success"] == True

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


## Test user facing run_example api
def test_ollama_run_prompt_example_topic_name_smoke(ollama_running):
    model = get_test_ollama_model()
    if not ollama_running:
        pytest.skip("Ollama service is not available or failed to start")
    if not ollama_has_model(model):
        pytest.skip(f"{model} not available in local Ollama")

    namer = OllamaNamer(model=model, temperature_override=0.0)

    accepted_patterns = [
        r"\bgoalie\b",
        r"\bgoaltend",
        r"\bmask",
    ]

    result = namer.run_prompt_example(
        example_id="hockey_masks",
        verbose=True,
    )

    assert result["success"] is True, result
    assert result["prompt_kind"] == "topic_name"
    assert result["prompt"] is not None
    assert isinstance(result["topic_names"], str)
    assert len(result["topic_names"]) > 0
    assert matches_expected_patterns(result["topic_names"], accepted_patterns)


def test_run_prompt_example_invalid_example_id_raises():
    namer = DummySingleWrapper()
    with pytest.raises(ValueError, match="example_id must be in"):
        namer.run_prompt_example(example_id="not_a_real_example")


def test_run_prompt_example_invalid_prompt_kind_raises():
    namer = DummySingleWrapper()
    with pytest.raises(ValueError, match="prompt_kind must be"):
        namer.run_prompt_example(example_id="hockey_masks", prompt_kind="bad_kind")


def test_run_prompt_example_connectivity_failure_returns_error(monkeypatch):
    namer = DummySingleWrapper()

    monkeypatch.setattr(
        namer,
        "connectivity_status",
        lambda: {
            "success": False,
            "error_type": "RuntimeError",
            "error_message": "boom",
            "original_exception": RuntimeError("boom"),
        },
    )

    result = namer.run_prompt_example(example_id="hockey_masks")

    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert isinstance(result["original_exception"], RuntimeError)


# Async
@pytest.mark.asyncio
async def test_ollama_async_smoke_test(ollama_running):
    model = get_test_ollama_model()
    if not ollama_running:
        pytest.skip("Ollama service is not available or failed to start")
    if not ollama_has_model(model):
        pytest.skip(f"{model} not available in local Ollama")
    namer = AsyncOllamaNamer(model=model, temperature_override=0.0)
    result = await namer.connectivity_status()
    assert result["success"] is True

    render_params = load_prompt_params("hockey_masks")
    prompt = render_prompt(PROMPT_TEMPLATES["layer"], render_params)

    # Note: do not match on hockey since it's too broad
    accepted_patterns = [
        r"\bgoalie\b",
        r"\bgoaltend",  # matches goaltender/goaltending
        r"\bmask",  # matches mask/masks
    ]

    result_list = await namer.generate_topic_names([prompt])
    result = result_list[0]
    assert type(result) == str
    assert len(result) > 0
    assert matches_expected_patterns(result, accepted_patterns)


@pytest.mark.asyncio
@pytest.mark.external
@pytest.mark.parametrize(
    "namer_class, api_key_name, namer_kwargs", ASYNC_SUPPORTED_NAMERS
)
async def test_supported_models_async_smoke_test(
    namer_class, api_key_name, namer_kwargs
):
    if os.getenv(api_key_name) is None:
        pytest.skip(f"{api_key_name} is not set in the environment")

    namer = namer_class(**namer_kwargs)
    result = await namer.connectivity_status()
    assert result["success"] is True

    render_params = load_prompt_params("hockey_masks")
    prompt = render_prompt(PROMPT_TEMPLATES["layer"], render_params)

    # Note: do not match on hockey since it's too broad
    accepted_patterns = [
        r"\bgoalie\b",
        r"\bgoaltend",  # matches goaltender/goaltending
        r"\bmask",  # matches mask/masks
    ]

    result_list = await namer.generate_topic_names([prompt])
    result = result_list[0]
    assert type(result) == str
    assert len(result) > 0
    assert matches_expected_patterns(result, accepted_patterns)


## Test user facing run_example api
@pytest.mark.asyncio
async def test_async_ollama_run_prompt_example_topic_name_smoke(ollama_running):
    model = get_test_ollama_model()
    if not ollama_running:
        pytest.skip("Ollama service is not available or failed to start")
    if not ollama_has_model(model):
        pytest.skip(f"{model} not available in local Ollama")

    namer = AsyncOllamaNamer(model=model, temperature_override=0.0)

    accepted_patterns = [
        r"\bgoalie\b",
        r"\bgoaltend",
        r"\bmask",
    ]

    result = await namer.run_prompt_example(
        example_id="hockey_masks",
        verbose=True,
    )

    assert result["success"] is True, result
    assert result["prompt_kind"] == "topic_name"
    assert result["prompt"] is not None
    assert isinstance(result["topic_names"], str)
    assert len(result["topic_names"]) > 0
    assert matches_expected_patterns(result["topic_names"], accepted_patterns)


@pytest.mark.asyncio
async def test_async_run_prompt_example_invalid_example_id_raises():
    namer = DummyAsyncSingleWrapper()
    with pytest.raises(ValueError, match="example_id must be in"):
        await namer.run_prompt_example(example_id="not_a_real_example")


@pytest.mark.asyncio
async def test_async_run_prompt_example_invalid_prompt_kind_raises():
    namer = DummyAsyncSingleWrapper()
    with pytest.raises(ValueError, match="prompt_kind must be"):
        await namer.run_prompt_example(
            example_id="hockey_masks", prompt_kind="bad_kind"
        )


@pytest.mark.asyncio
async def test_async_run_prompt_example_connectivity_failure_returns_error(
    monkeypatch,
):
    namer = DummyAsyncSingleWrapper()

    monkeypatch.setattr(
        namer,
        "connectivity_status",
        AsyncMock(
            return_value={
                "success": False,
                "error_type": "RuntimeError",
                "error_message": "boom",
                "original_exception": RuntimeError("boom"),
            }
        ),
    )

    result = await namer.run_prompt_example(example_id="hockey_masks")

    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert isinstance(result["original_exception"], RuntimeError)
