import asyncio
import pytest
from unittest.mock import Mock

from toponymy.llm_wrappers import (
    AsyncLLMWrapper,
    LLMWrapper,
    _should_retry,
    FailFastLLMError,
    InvalidLLMInputError,
    repair_json_string_backslashes,
)
from toponymy.tests.helpers.dummy_wrappers import (
    DummySingleWrapper,
    SINGLE_RESPONSE,
    SYSTEM_RESPONSE,
)
from toponymy.tests.helpers.llm_test_config import (
    make_prompt,
    validate_topic_name,
)


class DummyCombinedOnlyWrapper(DummySingleWrapper):
    """A wrapper whose provider has no system role, so it takes the combined rendering."""

    @property
    def supports_system_prompts(self) -> bool:
        return False


class DummyFailureWrapper(LLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    def _call_llm(self, prompt, temperature, max_tokens):
        raise RuntimeError("error")

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        raise RuntimeError("error")


class DummyFailFastProviderError(Exception):
    pass


class DummyFailFastWrapper(LLMWrapper):
    model = "dummy-model"
    FAIL_FAST_EXCEPTIONS = (DummyFailFastProviderError,)

    def _call_llm(self, prompt, temperature, max_tokens):
        raise DummyFailFastProviderError("bad config")

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        raise DummyFailFastProviderError("bad config")


def test_sync_connectivity_status_uses_plain_call():
    wrapper = DummySingleWrapper()

    result = wrapper.connectivity_status()

    assert result["success"] is True
    assert result["response"] == SINGLE_RESPONSE
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
    assert result["response"] == SYSTEM_RESPONSE
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

    assert result == SINGLE_RESPONSE


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
        wrapper._safe_call_llm({"combined": "prompt"}, temperature=0.4, max_tokens=128)


def test_safe_call_llm_with_system_prompt_wraps_fail_fast_exception():
    wrapper = DummyFailFastWrapper()

    with pytest.raises(FailFastLLMError, match="dummy-model"):
        wrapper._safe_call_llm_with_system_prompt(
            {"system": "system prompt", "user": "user prompt"},
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


def test_generate_topic_name_missing_rendering_raises():
    wrapper = DummySingleWrapper()

    with pytest.raises(InvalidLLMInputError, match="missing the system rendering"):
        wrapper.generate_topic_name({"user": "user prompt", "combined": "combined"})


def test_generate_topic_name_uses_system_rendering_when_supported(mock_data):
    wrapper = DummySingleWrapper()
    wrapper._call_llm_with_system_prompt = Mock(
        return_value=mock_data["valid_topic_name"]
    )
    wrapper._call_llm = Mock()

    prompt = make_prompt()
    validate_topic_name(wrapper.generate_topic_name(prompt))

    wrapper._call_llm.assert_not_called()
    # The whole prompt is passed through, not just the rendering that gets used.
    assert wrapper._call_llm_with_system_prompt.call_args.args[0] == prompt


def test_generate_topic_name_uses_combined_rendering_without_system_prompts(mock_data):
    wrapper = DummyCombinedOnlyWrapper()
    wrapper._call_llm = Mock(return_value=mock_data["valid_topic_name"])
    wrapper._call_llm_with_system_prompt = Mock()

    prompt = make_prompt()
    validate_topic_name(wrapper.generate_topic_name(prompt))

    wrapper._call_llm_with_system_prompt.assert_not_called()
    assert wrapper._call_llm.call_args.args[0] == prompt


def test_generate_topic_cluster_names_uses_combined_rendering_without_system_prompts(
    mock_data,
):
    wrapper = DummyCombinedOnlyWrapper()
    wrapper._call_llm = Mock(return_value=mock_data["valid_cluster_names"])
    wrapper._call_llm_with_system_prompt = Mock()

    prompt = make_prompt()
    wrapper.generate_topic_cluster_names(prompt, mock_data["old_names"])

    wrapper._call_llm_with_system_prompt.assert_not_called()
    assert wrapper._call_llm.call_args.args[0] == prompt


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
        {"combined": "test prompt"},
        temperature=0.4,
        max_tokens=128,
    )

    assert result == SINGLE_RESPONSE
    assert len(events) == 1
    assert events[0]["event"] == "llm_call_success"
    assert events[0]["prompt"] == {"combined": "test prompt"}
    assert events[0]["raw_response"] == SINGLE_RESPONSE
    assert events[0]["model"] == "dummy-model"
    assert events[0]["wrapper"] == "DummySingleWrapper"


def test_safe_call_llm_with_system_prompt_emits_debug_callback_on_success():

    events = []

    wrapper = DummySingleWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    result = wrapper._safe_call_llm_with_system_prompt(
        {"system": "system prompt", "user": "user prompt"},
        temperature=0.4,
        max_tokens=128,
    )

    assert result == SYSTEM_RESPONSE
    assert len(events) == 1
    assert events[0]["event"] == "llm_call_success"
    assert events[0]["prompt"] == {
        "system": "system prompt",
        "user": "user prompt",
    }
    assert events[0]["raw_response"] == SYSTEM_RESPONSE


def test_safe_call_llm_emits_debug_callback_on_error():
    events = []

    wrapper = DummyFailureWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    with pytest.raises(RuntimeError, match="error"):
        wrapper._safe_call_llm(
            {"combined": "test prompt"},
            temperature=0.4,
            max_tokens=128,
        )

    assert len(events) == 1
    assert events[0]["event"] == "llm_call_error"
    assert events[0]["prompt"] == {"combined": "test prompt"}
    assert events[0]["error"]["type"] == "RuntimeError"
    assert events[0]["error"]["message"] == "error"


def test_safe_call_llm_with_system_prompt_emits_debug_callback_on_error():
    events = []

    wrapper = DummyFailureWrapper()
    wrapper.callback = lambda payload: events.append(payload)

    with pytest.raises(RuntimeError, match="error"):
        wrapper._safe_call_llm_with_system_prompt(
            {"system": "system prompt", "user": "user prompt"},
            temperature=0.4,
            max_tokens=128,
        )

    assert len(events) == 1
    assert events[0]["event"] == "llm_call_error"
    assert events[0]["prompt"] == {
        "system": "system prompt",
        "user": "user prompt",
    }
    assert events[0]["error"]["type"] == "RuntimeError"
    assert events[0]["error"]["message"] == "error"


# Test JSON repair
def test_repair_json_string_backslashes_already_valid():
    """Test that valid JSON strings are not modified."""

    valid_json = '{"key": "value with \\"quotes\\" inside"}'
    result = repair_json_string_backslashes(valid_json)
    assert result == valid_json


def test_repair_json_string_backslashes_unescaped():
    """Test repairing strings with unescaped backslashes."""

    invalid_json = '{"topic_name": "Machine Learning\\ML"}'
    expected = '{"topic_name": "Machine Learning\\\\ML"}'
    result = repair_json_string_backslashes(invalid_json)
    assert result == expected


def test_repair_json_string_backslashes_mixed():
    """Test repairing strings with both properly escaped and unescaped backslashes."""

    mixed_json = '{"path": "C:\\Users\\username", "quoted": "with \\"quotes\\""}'
    expected = '{"path": "C:\\\\Users\\\\username", "quoted": "with \\"quotes\\""}'
    result = repair_json_string_backslashes(mixed_json)
    assert result == expected


def test_repair_json_string_backslashes_all_escape_sequences():
    """Test that all valid escape sequences are preserved."""

    json_with_escapes = '{"special": "\\n\\r\\t\\b\\f\\/\\\\", "invalid": "\\x"}'
    expected = '{"special": "\\n\\r\\t\\b\\f\\/\\\\", "invalid": "\\\\x"}'
    result = repair_json_string_backslashes(json_with_escapes)
    assert result == expected


def test_repair_json_string_backslashes_empty():
    """Test with empty string."""

    empty = ""
    result = repair_json_string_backslashes(empty)
    assert result == empty


def test_sync_and_async_connectivity_status_return_same_result(dummy_namer):
    sync_status = dummy_namer.connectivity_status()
    async_status = asyncio.run(dummy_namer.aconnectivity_status())

    assert sync_status["success"] is True, sync_status
    assert sync_status == async_status
