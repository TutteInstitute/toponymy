import warnings

import pytest
from unittest.mock import AsyncMock
from toponymy.llm_wrappers import (
    AsyncLLMWrapper,
    CallResult,
    FailFastLLMError,
    InvalidLLMInputError,
)

from toponymy.tests.helpers.dummy_wrappers import (
    DummyAsyncSingleWrapper,
    DummyAsyncProviderError,
    DummyAsyncFailFastWrapper,
    SINGLE_RESPONSE,
    SYSTEM_RESPONSE,
)
from toponymy.tests.helpers.llm_test_config import (
    make_prompt,
    validate_topic_name,
    validate_cluster_names,
)

BATCH_SINGLE_RESPONSE = '{"topic_name": "batch-ok", "topic_specificity": 0.9}'
BATCH_SYSTEM_RESPONSE = '{"topic_name": "batch-system-ok", "topic_specificity": 0.9}'
CALL_VALUE = "value"


async def call_fn(**kwargs):
    return CALL_VALUE


class MockCallResult:
    @staticmethod
    def success(value: str) -> CallResult[str]:
        return CallResult(value=value)

    @staticmethod
    def failure(error: Exception | None = None) -> CallResult[str]:
        return CallResult(error=error or RuntimeError("temporary failure"))


class DummyBatchWrapper(AsyncLLMWrapper):
    _supports_debug_callback = True

    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return [BATCH_SINGLE_RESPONSE]

    async def _call_llm_with_system_prompt_batch(
        self, prompts, temperature, max_tokens
    ):
        return [BATCH_SYSTEM_RESPONSE]


class DummyCombinedOnlyWrapper(DummyAsyncSingleWrapper):
    """A wrapper whose provider has no system role, so it takes the combined rendering."""

    @property
    def supports_system_prompts(self) -> bool:
        return False


class DummyBatchCallResultWrapper(AsyncLLMWrapper):
    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return [MockCallResult.success(BATCH_SINGLE_RESPONSE)]


class DummyBatchErrorWrapper(AsyncLLMWrapper):
    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        return [MockCallResult.failure(RuntimeError("batch failed"))]


@pytest.mark.asyncio
async def test_async_connectivity_status_uses_single_call():
    wrapper = DummyAsyncSingleWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.aconnectivity_status()

    assert result["success"] is True
    assert result["response"] == SINGLE_RESPONSE

    assert result["wrapper"] == "DummyAsyncSingleWrapper"
    assert result["model"] == "dummy-model"


@pytest.mark.asyncio
async def test_async_connectivity_status_uses_single_call_with_system():
    wrapper = DummyAsyncSingleWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.aconnectivity_status(
        prompt="user prompt",
        system_prompt="system prompt",
    )

    assert result["success"] is True
    assert result["response"] == SYSTEM_RESPONSE
    assert result["wrapper"] == "DummyAsyncSingleWrapper"
    assert result["model"] == "dummy-model"


@pytest.mark.asyncio
async def test_async_connectivity_status_falls_back_to_batch():
    wrapper = DummyBatchWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.aconnectivity_status()

    assert result["success"] is True
    assert result["response"] == BATCH_SINGLE_RESPONSE
    assert result["wrapper"] == "DummyBatchWrapper"
    assert result["model"] == "dummy-model"


@pytest.mark.asyncio
async def test_async_connectivity_status_falls_back_to_system_batch():
    wrapper = DummyBatchWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.aconnectivity_status(
        prompt="user prompt",
        system_prompt="system prompt",
    )

    assert result["success"] is True
    assert result["response"] == BATCH_SYSTEM_RESPONSE
    assert result["wrapper"] == "DummyBatchWrapper"
    assert result["model"] == "dummy-model"


@pytest.mark.asyncio
async def test_async_connectivity_status_unwraps_call_result_from_batch():
    wrapper = DummyBatchCallResultWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.aconnectivity_status()

    assert result["success"] is True
    assert result["response"] == BATCH_SINGLE_RESPONSE
    assert result["wrapper"] == "DummyBatchCallResultWrapper"
    assert result["model"] == "dummy-model"


@pytest.mark.asyncio
async def test_async_connectivity_status_batch_call_result_error():
    wrapper = DummyBatchErrorWrapper()
    wrapper.model = "dummy-model"

    result = await wrapper.aconnectivity_status()

    assert result["success"] is False
    assert result["response"] is None
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "batch failed"
    assert isinstance(result["original_exception"], RuntimeError)
    assert result["wrapper"] == "DummyBatchErrorWrapper"
    assert result["model"] == "dummy-model"


@pytest.mark.asyncio
async def test_safe_call_with_retry_result_success():
    wrapper = DummyAsyncSingleWrapper()

    async def ok_fn():
        return "ok"

    result = await wrapper._safe_call_with_retry_result(ok_fn)

    assert isinstance(result, CallResult)
    assert result.ok is True
    assert result.value == "ok"
    assert result.error is None


@pytest.mark.asyncio
async def test_safe_call_with_retry_result_exhausted_retries():
    wrapper = DummyAsyncSingleWrapper()

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

    with pytest.raises(FailFastLLMError, match="bad config"):
        await wrapper._safe_call_with_retry_result(fail_fast_fn)


@pytest.mark.asyncio
async def test_safe_call_with_retry_result_raises_invalid_input():
    wrapper = DummyAsyncSingleWrapper()

    async def invalid_fn():
        raise InvalidLLMInputError("bad input")

    with pytest.raises(InvalidLLMInputError, match="bad input"):
        await wrapper._safe_call_with_retry_result(invalid_fn)


@pytest.mark.asyncio
async def test_async_generate_topic_names_empty_input_returns_empty_list():
    wrapper = DummyAsyncSingleWrapper()

    result = await wrapper.generate_topic_names([])

    assert result == []


@pytest.mark.asyncio
async def test_async_generate_topic_names_invalid_prompt_type_raises():
    wrapper = DummyAsyncSingleWrapper()

    with pytest.raises(InvalidLLMInputError):
        await wrapper.generate_topic_names([123])


@pytest.mark.asyncio
async def test_async_generate_topic_cluster_names_empty_input_returns_empty_list():
    wrapper = DummyAsyncSingleWrapper()

    result = await wrapper.generate_topic_cluster_names([], [])

    assert result == []


@pytest.mark.asyncio
async def test_async_generate_topic_cluster_names_length_mismatch_raises():
    wrapper = DummyAsyncSingleWrapper()

    with pytest.raises(ValueError, match="Number of prompts must match"):
        await wrapper.generate_topic_cluster_names(
            ["prompt 1", "prompt 2"],
            [["old1", "old2"]],
        )


@pytest.mark.asyncio
async def test_async_generate_topic_names_routes_to_batch_without_system_prompts():
    wrapper = DummyCombinedOnlyWrapper()
    wrapper._call_llm_batch = AsyncMock(return_value=[])

    prompts = [make_prompt(1)]
    await wrapper.generate_topic_names(prompts)

    # The whole prompt is passed through, not just the rendering that gets used.
    wrapper._call_llm_batch.assert_awaited_once_with(
        prompts,
        0.4,
        max_tokens=128,
    )


@pytest.mark.asyncio
async def test_async_generate_topic_names_routes_to_system_prompt_batch():
    wrapper = DummyAsyncSingleWrapper()
    wrapper._call_llm_with_system_prompt_batch = AsyncMock(return_value=[])

    prompts = [make_prompt(1)]
    await wrapper.generate_topic_names(prompts)

    wrapper._call_llm_with_system_prompt_batch.assert_awaited_once_with(
        prompts,
        0.4,
        max_tokens=128,
    )


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_generate_topic_names_partial_success(
    mock_data,
):
    wrapper = DummyAsyncSingleWrapper()
    wrapper._call_llm_with_system_prompt_batch = AsyncMock(
        return_value=[
            MockCallResult.success(mock_data["valid_topic_name"]),
            MockCallResult.failure(),
            MockCallResult.success(mock_data["valid_topic_name"]),
        ]
    )

    result = await wrapper.generate_topic_names(
        [make_prompt(1), make_prompt(2), make_prompt(3)]
    )

    validate_topic_name(result[0])
    assert result[1] == ""
    validate_topic_name(result[2])


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_generate_topic_cluster_names_partial_success(
    mock_data,
):
    wrapper = DummyAsyncSingleWrapper()
    wrapper._call_llm_with_system_prompt_batch = AsyncMock(
        return_value=[
            MockCallResult.success(mock_data["valid_cluster_names"]),
            MockCallResult.failure(),
            MockCallResult.success(mock_data["valid_cluster_names"]),
        ]
    )

    fallback_old_names = ["old_x", "old_y", "old_z"]
    old_names = [
        mock_data["old_names"],
        fallback_old_names,
        mock_data["old_names"],
    ]
    result = await wrapper.generate_topic_cluster_names(
        [make_prompt(1), make_prompt(2), make_prompt(3)],
        old_names,
    )

    validate_cluster_names(result[0])
    assert result[1] == fallback_old_names
    validate_cluster_names(result[2])


#
# ASYNC TESTS
#


@pytest.mark.asyncio
async def test_safe_call_with_retry_result_emits_debug_callback_on_success():
    events = []

    wrapper = DummyAsyncSingleWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    async def fn(prompt, temperature, max_tokens, **kwargs):
        return await wrapper._call_single_llm(prompt, temperature, max_tokens)

    result = await wrapper._safe_call_with_retry_result(
        fn,
        prompt="test prompt",
        temperature=0.4,
        max_tokens=128,
    )

    assert result.ok
    assert result.value == SINGLE_RESPONSE
    assert len(events) == 1

    event = events[0]
    assert event["event"] == "llm_call_success"
    assert event["prompt"] == "test prompt"
    assert event["raw_response"] == SINGLE_RESPONSE
    assert event["model"] == "dummy-model"
    assert event["wrapper"] == "DummyAsyncSingleWrapper"


@pytest.mark.asyncio
async def test_safe_call_with_retry_result_fail_fast_raises_and_no_callback():
    events = []

    wrapper = DummyAsyncFailFastWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    async def fn(prompt, temperature, max_tokens, **kwargs):
        return await wrapper._call_single_llm(prompt, temperature, max_tokens)

    with pytest.raises(FailFastLLMError, match="bad config"):
        await wrapper._safe_call_with_retry_result(
            fn,
            prompt="test prompt",
            temperature=0.4,
            max_tokens=128,
        )

    assert len(events) == 0


def test_async_wrapper_test_llm_connectivity_success():
    wrapper = DummyAsyncSingleWrapper()

    result = wrapper.test_llm_connectivity()

    assert result == SINGLE_RESPONSE


def test_async_wrapper_test_llm_connectivity_failure():
    wrapper = DummyAsyncFailFastWrapper()

    result = wrapper.test_llm_connectivity()

    assert result == "<error>"


@pytest.mark.asyncio
async def test_async_atest_llm_connectivity_success():
    wrapper = DummyAsyncSingleWrapper()

    result = await wrapper.atest_llm_connectivity()

    assert result == SINGLE_RESPONSE


@pytest.mark.asyncio
async def test_async_atest_llm_connectivity_failure():
    wrapper = DummyAsyncFailFastWrapper()

    result = await wrapper.atest_llm_connectivity()

    assert result == "<error>"


class TestAsafeCallLLM:
    @pytest.mark.asyncio
    async def test_asafe_call_llm_success_emits_callback_and_returns_result(self):
        events = []
        wrapper = DummyAsyncSingleWrapper()
        wrapper.callback = lambda payload: events.append(payload)

        result = await wrapper._asafe_call_llm(
            call_fn,
            prompt="test prompt",
            temperature=0.4,
            max_tokens=128,
            routine="my_routine",
        )

        assert isinstance(result, CallResult)
        assert result.ok is True
        assert result.value == CALL_VALUE
        assert result.error is None
        assert len(events) == 1

        event = events[0]
        assert event["wrapper"] == "DummyAsyncSingleWrapper"
        assert event["model"] == "dummy-model"
        assert event["event"] == "llm_call_success"
        assert event["routine"] == "my_routine"
        assert event["prompt"] == "test prompt"
        assert event["raw_response"] == CALL_VALUE

    @pytest.mark.asyncio
    async def test_asafe_call_llm_success_no_callback(self):
        wrapper = DummyAsyncSingleWrapper()
        wrapper.callback = None

        result = await wrapper._asafe_call_llm(call_fn)
        assert result.value == CALL_VALUE

    @pytest.mark.asyncio
    async def test_asafe_call_llm_error_emits_callback_and_raises(self):
        events = []
        wrapper = DummyAsyncSingleWrapper()
        wrapper.callback = lambda payload: events.append(payload)

        async def fn(**kwargs):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await wrapper._asafe_call_llm(fn, prompt="p", routine="r")

        assert len(events) == 1
        event = events[0]
        assert event["event"] == "llm_call_error"
        assert event["routine"] == "r"
        assert event["prompt"] == "p"
        assert event["error"]["type"] == "RuntimeError"
        assert event["error"]["message"] == "boom"

    @pytest.mark.asyncio
    async def test_asafe_call_llm_invalid_input_raises_without_callback(self):
        events = []
        wrapper = DummyAsyncSingleWrapper()
        wrapper.callback = lambda payload: events.append(payload)

        async def fn(**kwargs):
            raise InvalidLLMInputError("bad")

        with pytest.raises(InvalidLLMInputError, match="bad"):
            await wrapper._asafe_call_llm(fn, prompt="p")

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_asafe_call_llm_fail_fast_raises_without_callback(self):
        events = []
        wrapper = DummyAsyncFailFastWrapper()
        wrapper.callback = lambda payload: events.append(payload)

        async def fn(**kwargs):
            raise DummyAsyncProviderError("bad config")

        with pytest.raises(FailFastLLMError, match="dummy-model"):
            await wrapper._asafe_call_llm(fn, prompt="p")

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_asafe_call_llm_callback_exception_is_swallowed(self):
        wrapper = DummyAsyncSingleWrapper()
        # callback raises
        wrapper.callback = lambda payload: 1 / 0

        # Should not propagate callback error
        result = await wrapper._asafe_call_llm(call_fn, prompt="p")
        assert result.value == CALL_VALUE

    @pytest.mark.asyncio
    async def test_asafe_call_llm_routine_is_popped_from_kwargs(self):
        received_kwargs = {}

        async def fn(**kwargs):
            received_kwargs.update(kwargs)
            return "ok"

        wrapper = DummyAsyncSingleWrapper()
        wrapper.callback = lambda p: None

        await wrapper._asafe_call_llm(fn, routine="r", prompt="p")
        # routine should not be passed to fn
        assert "routine" not in received_kwargs
        assert received_kwargs.get("prompt") == "p"

    @pytest.mark.asyncio
    async def test_asafe_call_llm_passes_args_and_kwargs_through(self):
        async def fn(a, b, kw=None, prompt=None, routine=None):
            return a + b + (kw or 0)

        wrapper = DummyAsyncSingleWrapper()
        result = await wrapper._asafe_call_llm(fn, 1, 2, kw=3, prompt="p")
        assert result.value == 6
