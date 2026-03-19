import pytest
from unittest.mock import AsyncMock
from toponymy.llm_wrappers import AsyncLLMWrapper, CallResult, FailFastLLMError, InvalidLLMInputError

from toponymy.tests.helpers.make_llm_data import (
    validate_topic_name,
    validate_cluster_names,
)

class DummyAsyncProviderError(Exception):
    pass

class DummyAsyncFailFastWrapper(AsyncLLMWrapper):
    model = "dummy-model"
    FAIL_FAST_EXCEPTIONS = (DummyAsyncProviderError,)

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        raise DummyAsyncProviderError("bad config")

    async def _call_single_llm_with_system(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        raise DummyAsyncProviderError("bad config")

class DummySingleWrapper(AsyncLLMWrapper):
    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return "single-ok"

    async def _call_single_llm_with_system(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        return "single-system-ok"


class DummyBatchWrapper(AsyncLLMWrapper):
    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return ["batch-ok"]

    async def _call_llm_with_system_prompt_batch(
        self, system_prompts, user_prompts, temperature, max_tokens
    ):
        return ["batch-system-ok"]


class DummyBatchCallResultWrapper(AsyncLLMWrapper):
    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return [CallResult(value="batch-ok")]


class DummyBatchErrorWrapper(AsyncLLMWrapper):
    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return [CallResult(error=RuntimeError("batch failed"))]


@pytest.mark.asyncio
async def test_async_connectivity_status_uses_single_call():
    wrapper = DummySingleWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == "single-ok"
    assert result["wrapper"] == "DummySingleWrapper"
    assert result["model"] == "dummy-model"

@pytest.mark.asyncio
async def test_async_connectivity_status_uses_single_call_with_system():
    wrapper = DummySingleWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status(
        prompt="user prompt",
        system_prompt="system prompt",
    )

    assert result["success"] is True
    assert result["response"] == "single-system-ok"
    assert result["wrapper"] == "DummySingleWrapper"
    assert result["model"] == "dummy-model"

@pytest.mark.asyncio
async def test_async_connectivity_status_falls_back_to_batch():
    wrapper = DummyBatchWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == "batch-ok"
    assert result["wrapper"] == "DummyBatchWrapper"
    assert result["model"] == "dummy-model"

@pytest.mark.asyncio
async def test_async_connectivity_status_falls_back_to_system_batch():
    wrapper = DummyBatchWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status(
        prompt="user prompt",
        system_prompt="system prompt",
    )

    assert result["success"] is True
    assert result["response"] == "batch-system-ok"
    assert result["wrapper"] == "DummyBatchWrapper"
    assert result["model"] == "dummy-model"

@pytest.mark.asyncio
async def test_async_connectivity_status_unwraps_call_result_from_batch():
    wrapper = DummyBatchCallResultWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == "batch-ok"
    assert result["wrapper"] == "DummyBatchCallResultWrapper"
    assert result["model"] == "dummy-model"

@pytest.mark.asyncio
async def test_async_connectivity_status_batch_call_result_error():
    wrapper = DummyBatchErrorWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is False
    assert result["response"] is None
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "batch failed"
    assert isinstance(result["original_exception"], RuntimeError)
    assert result["wrapper"] == "DummyBatchErrorWrapper"
    assert result["model"] == "dummy-model"

@pytest.mark.asyncio
async def test_safe_call_with_retry_result_success():
    wrapper = DummySingleWrapper()

    async def ok_fn():
        return "ok"

    result = await wrapper._safe_call_with_retry_result(ok_fn)

    assert isinstance(result, CallResult)
    assert result.ok is True
    assert result.value == "ok"
    assert result.error is None

@pytest.mark.asyncio
async def test_safe_call_with_retry_result_exhausted_retries():
    wrapper = DummySingleWrapper()

    async def failing_fn():
        raise RuntimeError("temporary failure")

    result = await wrapper._safe_call_with_retry_result(failing_fn)

    assert isinstance(result, CallResult)
    assert result.ok is False
    assert result.value is None
    assert isinstance(result.error, RuntimeError)
    assert str(result.error) == "temporary failure"

@pytest.mark.asyncio
async def test_safe_call_with_retry_result_raises_fail_fast():
    wrapper = DummyAsyncFailFastWrapper()

    async def fail_fast_fn():
        raise DummyAsyncProviderError("bad config")

    with pytest.raises(FailFastLLMError, match="dummy-model"):
        await wrapper._safe_call_with_retry_result(fail_fast_fn)

@pytest.mark.asyncio
async def test_safe_call_with_retry_result_raises_invalid_input():
    wrapper = DummySingleWrapper()

    async def invalid_fn():
        raise InvalidLLMInputError("bad input")

    with pytest.raises(InvalidLLMInputError, match="bad input"):
        await wrapper._safe_call_with_retry_result(invalid_fn)


@pytest.mark.asyncio
async def test_async_generate_topic_names_empty_input_returns_empty_list():
    wrapper = DummySingleWrapper()

    result = await wrapper.generate_topic_names([])

    assert result == []

@pytest.mark.asyncio
async def test_async_generate_topic_names_invalid_prompt_type_raises():
    wrapper = DummySingleWrapper()

    with pytest.raises(InvalidLLMInputError):
        await wrapper.generate_topic_names([123])


@pytest.mark.asyncio
async def test_async_generate_topic_cluster_names_empty_input_returns_empty_list():
    wrapper = DummySingleWrapper()

    result = await wrapper.generate_topic_cluster_names([], [])

    assert result == []


@pytest.mark.asyncio
async def test_async_generate_topic_cluster_names_length_mismatch_raises():
    wrapper = DummySingleWrapper()

    with pytest.raises(ValueError, match="Number of prompts must match"):
        await wrapper.generate_topic_cluster_names(
            ["prompt 1", "prompt 2"],
            [["old1", "old2"]],
        )

@pytest.mark.asyncio
async def test_async_generate_topic_names_routes_string_prompts_to_call_llm_batch():
    wrapper = DummySingleWrapper()
    wrapper._call_llm_batch = AsyncMock(return_value=[])

    await wrapper.generate_topic_names(["test prompt"])

    wrapper._call_llm_batch.assert_awaited_once_with(
        ["test prompt"],
        0.4,
        max_tokens=128,
    )

@pytest.mark.asyncio
async def test_async_generate_topic_names_routes_dict_prompts_to_call_with_system_prompt_batch():
    wrapper = DummySingleWrapper()
    wrapper._call_llm_with_system_prompt_batch = AsyncMock(return_value=[])

    prompts = [{"system": "system prompt", "user": "test prompt"}]
    await wrapper.generate_topic_names(prompts)

    wrapper._call_llm_with_system_prompt_batch.assert_awaited_once_with(
        ["system prompt"],
        ["test prompt"],
        0.4,
        max_tokens=128,
    )

@pytest.mark.asyncio
async def test_async_generate_topic_names_partial_success_preserves_successful_items(
    mock_data,
):
    wrapper = DummySingleWrapper()
    wrapper._call_llm_batch = AsyncMock(
        return_value=[
            CallResult(value=mock_data["valid_topic_name"]),
            CallResult(error=RuntimeError("temporary failure")),
            CallResult(value=mock_data["valid_topic_name"]),
        ]
    )

    result = await wrapper.generate_topic_names(["p1", "p2", "p3"])

    validate_topic_name(result[0])
    assert result[1] == ""
    validate_topic_name(result[2])


@pytest.mark.asyncio
async def test_async_generate_topic_cluster_names_partial_success_preserves_successful_items(
    mock_data,
):
    wrapper = DummySingleWrapper()
    wrapper._call_llm_batch = AsyncMock(
        return_value=[
            CallResult(value=mock_data["valid_cluster_names"]),
            CallResult(error=RuntimeError("temporary failure")),
            CallResult(value=mock_data["valid_cluster_names"]),
        ]
    )

    fallback_old_names = ["old_x", "old_y", "old_z"]
    result = await wrapper.generate_topic_cluster_names(
        ["p1", "p2", "p3"],
        [
            mock_data["old_names"],
            fallback_old_names,
            mock_data["old_names"],
        ],
    )

    validate_cluster_names(result[0])
    assert result[1] == fallback_old_names
    validate_cluster_names(result[2])