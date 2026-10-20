import asyncio
import json
import os
import re

import pytest
from unittest.mock import AsyncMock

from toponymy.llm_wrappers import AsyncOllamaNamer, OllamaNamer, _SyncBridge
from toponymy.prompt_construction import render_prompt
from toponymy.tools.load_prompt_examples import load_prompt_params
from toponymy.templates import PROMPT_TEMPLATES
from toponymy.tests.helpers.llm_test_config import (
    EXTERNAL_SUPPORTED_NAMERS,
)
from toponymy.tests.test_llm_wrapper_base import DummySingleWrapper
from toponymy.tests.test_async_llm_wrappers_base import (
    DummySingleWrapper as DummyAsyncSingleWrapper,
)

ACCEPTED_PATTERNS = {
    "hockey_masks": [
        r"\bgoalie\b",
        r"\bgoaltend",
        r"\bmask",
    ]  # Note: do not match on hockey since it's too broad
}


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


def test_ollama_sync_smoke_test(ollama_test_model):
    namer = OllamaNamer(model=ollama_test_model, temperature_override=0.0)
    status = namer.connectivity_status()
    assert status["success"] == True, status

    example_id = "hockey_masks"
    result = namer.run_prompt_example(
        example_id=example_id,
        verbose=True,
    )

    assert result["success"] is True, result
    assert isinstance(result["topic_result"], str), result
    assert len(result["topic_result"]) > 0, result
    assert matches_expected_patterns(
        result["topic_result"], ACCEPTED_PATTERNS[example_id]
    ), result


def test_ollama_run_prompt_example_topic_name_smoke(ollama_test_model):
    """Test user facing run_prompt_example API"""
    namer = OllamaNamer(model=ollama_test_model, temperature_override=0.0)

    example_id = "hockey_masks"

    result = namer.run_prompt_example(
        example_id=example_id,
        verbose=True,
    )

    assert result["success"] is True, result
    assert result["prompt_kind"] == "topic_name", result
    assert result["prompt"] is not None, result
    assert isinstance(result["topic_result"], str), result
    assert len(result["topic_result"]) > 0, result
    assert matches_expected_patterns(
        result["topic_result"], ACCEPTED_PATTERNS[example_id]
    ), result


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

    assert result["success"] is False, result
    assert result["error_type"] == "RuntimeError", result
    assert result["error_message"] == "boom", result
    assert isinstance(result["original_exception"], RuntimeError), result


# Async
@pytest.mark.asyncio
async def test_ollama_async_smoke_test(ollama_test_model):
    namer = AsyncOllamaNamer(model=ollama_test_model, temperature_override=0.0)
    example_id = "hockey_masks"

    try:
        status = await namer.aconnectivity_status()
        assert status["success"] is True, status

        result = await namer.arun_prompt_example(example_id=example_id)
    finally:
        await namer.close()

    assert result["success"] is True, result
    assert isinstance(result["topic_result"], str), result
    assert len(result["topic_result"]) > 0, result
    assert matches_expected_patterns(
        result["topic_result"], ACCEPTED_PATTERNS[example_id]
    ), result


def test_async_ollama_run_prompt_example_topic_name_smoke(ollama_test_model):
    """Test user facing run_prompt_example API for Async Ollama Namer"""
    namer = AsyncOllamaNamer(model=ollama_test_model, temperature_override=0.0)
    example_id = "hockey_masks"
    result = namer.run_prompt_example(example_id=example_id, verbose=True)

    assert result["success"] is True, result
    assert result["prompt_kind"] == "topic_name", result
    assert result["prompt"] is not None, result
    assert isinstance(result["topic_result"], str), result
    assert len(result["topic_result"]) > 0, result
    assert matches_expected_patterns(
        result["topic_result"], ACCEPTED_PATTERNS[example_id]
    ), result


@pytest.mark.asyncio
async def test_async_run_prompt_example_invalid_example_id_raises():
    namer = DummyAsyncSingleWrapper()
    with pytest.raises(ValueError, match="example_id must be in"):
        await namer.arun_prompt_example(example_id="not_a_real_example")


@pytest.mark.asyncio
async def test_async_run_prompt_example_invalid_prompt_kind_raises():
    namer = DummyAsyncSingleWrapper()
    with pytest.raises(ValueError, match="prompt_kind must be"):
        await namer.arun_prompt_example(
            example_id="hockey_masks", prompt_kind="bad_kind"
        )


@pytest.mark.asyncio
async def test_async_run_prompt_example_connectivity_failure_returns_error(
    monkeypatch,
):
    namer = DummyAsyncSingleWrapper()

    monkeypatch.setattr(
        namer,
        "aconnectivity_status",
        AsyncMock(
            return_value={
                "success": False,
                "error_type": "RuntimeError",
                "error_message": "boom",
                "original_exception": RuntimeError("boom"),
            }
        ),
    )

    result = await namer.arun_prompt_example(example_id="hockey_masks")

    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert isinstance(result["original_exception"], RuntimeError)


@pytest.mark.external
@pytest.mark.parametrize(
    "namer_class, api_key_name, namer_kwargs",
    EXTERNAL_SUPPORTED_NAMERS,
)
def test_supported_models_smoke_test(namer_class, api_key_name, namer_kwargs):
    if os.getenv(api_key_name) is None:
        pytest.skip(f"{api_key_name} is not set in the environment")

    namer = namer_class(**namer_kwargs)
    result = namer.connectivity_status()
    assert namer.connectivity_status()["success"] == True

    example_id = "hockey_masks"

    result = namer.run_prompt_example(
        example_id=example_id,
        verbose=True,
    )

    assert result["success"] is True, result
    assert isinstance(result["topic_result"], str), result
    assert len(result["topic_result"]) > 0, result
    assert matches_expected_patterns(
        result["topic_result"], ACCEPTED_PATTERNS[example_id]
    ), result


def test_sync_and_async_run_prompt_example_return_same_result_verbose():
    namer = DummyAsyncSingleWrapper()
    sync_result = namer.run_prompt_example(example_id="hockey_masks", verbose=True)
    async_result = asyncio.run(
        namer.arun_prompt_example(example_id="hockey_masks", verbose=True)
    )
    assert sync_result == async_result


def test_sync_and_async_run_prompt_example_return_same_result_non_verbose():
    namer = DummyAsyncSingleWrapper()
    sync_result = namer.run_prompt_example(example_id="hockey_masks", verbose=False)
    async_result = asyncio.run(
        namer.arun_prompt_example(example_id="hockey_masks", verbose=False)
    )
    assert sync_result == async_result


def test_sync_run_prompt_example_on_async_wrapper_propagates_exceptions():
    namer = DummyAsyncSingleWrapper()
    with pytest.raises(ValueError, match="prompt_kind must be"):
        namer.run_prompt_example(example_id="hockey_masks", prompt_kind="bad_kind")


@pytest.mark.asyncio
async def test_sync_run_prompt_example_works_inside_running_loop():
    """
    Tests that the synchronous run_prompt_example method works correctly
    when called inside an already running asyncio event loop.
    This simulates the behavior in environments like Jupyter notebooks.
    """
    asyncio.get_running_loop()  # Ensures we are inside a running event loop
    namer = DummyAsyncSingleWrapper()
    result = namer.run_prompt_example(example_id="hockey_masks")
    assert result["success"]


def test_sync_method_called_from_bridge_loop_raises():
    namer = DummyAsyncSingleWrapper()

    async def call_sync_from_inside_bridge():
        return namer.connectivity_status()

    with pytest.raises(RuntimeError, match="sync bridge loop"):
        _SyncBridge.run(call_sync_from_inside_bridge())


def test_sync_bridge_reuses_one_loop():
    loops = []

    async def record_loop():
        loops.append(asyncio.get_running_loop())

    _SyncBridge.run(record_loop())
    _SyncBridge.run(record_loop())
    assert loops[0] is loops[1]
