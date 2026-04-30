from toponymy.llm_wrappers import (
    AsyncLLMWrapper,
    LLMWrapper,
    _should_retry,
    FailFastLLMError,
    InvalidLLMInputError,
)
import pytest


class DummySingleWrapper(LLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    def _call_llm(self, prompt, temperature, max_tokens):
        return "single-ok"

    def _call_llm_with_system_prompt(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        return "single-system-ok"


class DummyFailureWrapper(LLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    def _call_llm(self, prompt, temperature, max_tokens):
        raise RuntimeError("error")

    def _call_llm_with_system_prompt(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        raise RuntimeError("error")


class DummyFailFastProviderError(Exception):
    pass


class DummyFailFastWrapper(LLMWrapper):
    model = "dummy-model"
    FAIL_FAST_EXCEPTIONS = (DummyFailFastProviderError,)

    def _call_llm(self, prompt, temperature, max_tokens):
        raise DummyFailFastProviderError("bad config")

    def _call_llm_with_system_prompt(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        raise DummyFailFastProviderError("bad config")


class DummyAsyncSingleWrapper(AsyncLLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return "async-single-ok"

    async def _call_single_llm_with_system_prompt(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        return "async-system-ok"


class DummyAsyncFailureWrapper(AsyncLLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        raise RuntimeError("error")

    async def _call_single_llm_with_system_prompt(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        raise RuntimeError("error")


class DummyAsyncFailFastProviderError(Exception):
    pass


class DummyAsyncFailFastWrapper(AsyncLLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True
    FAIL_FAST_EXCEPTIONS = (DummyAsyncFailFastProviderError,)

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        raise DummyAsyncFailFastProviderError("bad config")

    async def _call_single_llm_with_system_prompt(
        self, system_prompt, user_prompt, temperature, max_tokens
    ):
        raise DummyAsyncFailFastProviderError("bad config")


def test_sync_connectivity_status_uses_plain_call():
    wrapper = DummySingleWrapper()

    result = wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == "single-ok"
    assert result["wrapper"] == "DummySingleWrapper"
    assert result["model"] == "dummy-model"


# Connectivity Tests
def test_sync_connectivity_status_uses_system_call():
    wrapper = DummySingleWrapper()

    result = wrapper.connectivity_status(
        prompt="user prompt",
        system_prompt="system prompt",
    )

    assert result["success"] is True
    assert result["response"] == "single-system-ok"
    assert result["wrapper"] == "DummySingleWrapper"
    assert result["model"] == "dummy-model"


def test_sync_connectivity_status_failure():
    wrapper = DummyFailureWrapper()

    result = wrapper.connectivity_status()

    assert result["success"] is False
    assert result["response"] is None
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "error"
    assert isinstance(result["original_exception"], RuntimeError)


def test_sync_test_llm_connectivity_success():
    wrapper = DummySingleWrapper()

    result = wrapper.test_llm_connectivity()

    assert result == "single-ok"


def test_sync_test_llm_connectivity_failure():
    wrapper = DummyFailureWrapper()

    result = wrapper.test_llm_connectivity()

    assert result == "<error>"


# Test the retry-policy
def test_should_retry_invalid_input_returns_false():
    assert _should_retry(InvalidLLMInputError("bad input")) is False


def test_should_retry_fail_fast_returns_false():
    assert _should_retry(FailFastLLMError("fail fast")) is False


def test_should_retry_generic_exception_returns_true():
    assert _should_retry(RuntimeError("retry me")) is True


# Test safe calling
def test_safe_call_llm_wraps_fail_fast_exception():
    wrapper = DummyFailFastWrapper()

    with pytest.raises(FailFastLLMError, match="dummy-model"):
        wrapper._safe_call_llm("prompt", temperature=0.4, max_tokens=128)


def test_safe_call_llm_with_system_prompt_wraps_fail_fast_exception():
    wrapper = DummyFailFastWrapper()

    with pytest.raises(FailFastLLMError, match="dummy-model"):
        wrapper._safe_call_llm_with_system_prompt(
            "system prompt",
            "user prompt",
            temperature=0.4,
            max_tokens=128,
        )


def test_generate_topic_name_invalid_prompt_type_raises():
    wrapper = DummySingleWrapper()

    with pytest.raises(InvalidLLMInputError):
        wrapper.generate_topic_name(["not", "a", "valid", "prompt"])


def test_generate_topic_cluster_names_invalid_prompt_type_raises():
    wrapper = DummySingleWrapper()

    with pytest.raises(InvalidLLMInputError):
        wrapper.generate_topic_cluster_names(
            ["not", "a", "valid", "prompt"],
            ["old1", "old2"],
        )


def test_supports_system_prompts_defaults_true():
    wrapper = DummySingleWrapper()
    assert wrapper.supports_system_prompts is True


def test_handle_exception_reraises_retryable_exception():
    wrapper = DummySingleWrapper()

    with pytest.raises(RuntimeError, match="retry me"):
        wrapper._handle_exception(RuntimeError("retry me"))


def test_safe_call_llm_emits_debug_callback_on_success():
    events = []

    wrapper = DummySingleWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    result = wrapper._safe_call_llm(
        "test prompt",
        temperature=0.4,
        max_tokens=128,
        prompt_index=2,
    )

    assert result == "single-ok"
    assert len(events) == 1
    assert events[0]["event"] == "llm_call_success"
    assert events[0]["prompt"] == "test prompt"
    assert events[0]["prompt_index"] == 2
    assert events[0]["raw_response"] == "single-ok"
    assert events[0]["model"] == "dummy-model"
    assert events[0]["wrapper"] == "DummySingleWrapper"


def test_safe_call_llm_with_system_prompt_emits_debug_callback_on_success():

    events = []

    wrapper = DummySingleWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    result = wrapper._safe_call_llm_with_system_prompt(
        "system prompt",
        "user prompt",
        temperature=0.4,
        max_tokens=128,
        prompt_index=3,
    )

    assert result == "single-system-ok"
    assert len(events) == 1
    assert events[0]["event"] == "llm_call_success"
    assert events[0]["prompt"] == {
        "system": "system prompt",
        "user": "user prompt",
    }
    assert events[0]["prompt_index"] == 3
    assert events[0]["raw_response"] == "single-system-ok"


def test_safe_call_llm_emits_debug_callback_on_error():
    events = []

    wrapper = DummyFailureWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    with pytest.raises(RuntimeError, match="error"):
        wrapper._safe_call_llm(
            "test prompt",
            temperature=0.4,
            max_tokens=128,
            prompt_index=4,
        )

    assert len(events) == 1
    assert events[0]["event"] == "llm_call_error"
    assert events[0]["prompt"] == "test prompt"
    assert events[0]["prompt_index"] == 4
    assert events[0]["error"]["type"] == "RuntimeError"
    assert events[0]["error"]["message"] == "error"


def test_safe_call_llm_with_system_prompt_emits_debug_callback_on_error():
    events = []

    wrapper = DummyFailureWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    with pytest.raises(RuntimeError, match="error"):
        wrapper._safe_call_llm_with_system_prompt(
            "system prompt",
            "user prompt",
            temperature=0.4,
            max_tokens=128,
            prompt_index=5,
        )

    assert len(events) == 1
    assert events[0]["event"] == "llm_call_error"
    assert events[0]["prompt"] == {
        "system": "system prompt",
        "user": "user prompt",
    }
    assert events[0]["prompt_index"] == 5
    assert events[0]["error"]["type"] == "RuntimeError"
    assert events[0]["error"]["message"] == "error"
