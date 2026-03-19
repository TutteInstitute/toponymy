import pytest
from toponymy.llm_wrappers import AsyncLLMWrapper, CallResult

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


@pytest.mark.asyncio
async def test_async_connectivity_status_falls_back_to_batch():
    wrapper = DummyBatchWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == "batch-ok"


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


@pytest.mark.asyncio
async def test_async_connectivity_status_unwraps_call_result_from_batch():
    wrapper = DummyBatchCallResultWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == "batch-ok"


@pytest.mark.asyncio
async def test_async_connectivity_status_batch_call_result_error():
    wrapper = DummyBatchErrorWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.connectivity_status()

    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "batch failed"
