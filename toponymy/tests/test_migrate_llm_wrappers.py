## This is a temporary file for checking that the old tests still work (once updated appropriately)
## This can be deleted once we check it works and the coverage matches

import warnings

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import litellm

litellm.set_verbose = False

from toponymy.llm_wrappers import (
    AnthropicNamer,
    AsyncAnthropicNamer,
    AsyncAzureAINamer,
    AsyncCohereNamer,
    AsyncGoogleGeminiNamer,
    AsyncLiteLLMNamer,
    AsyncOllamaNamer,
    AsyncOpenAINamer,
    AsyncTogether,
    AzureAINamer,
    BatchAnthropicNamer,
    CallResult,
    CohereNamer,
    FailFastLLMError,
    GoogleGeminiNamer,
    GoogleGeminiNamer,
    HuggingFaceNamer,
    LiteLLMNamer,
    LlamaCppNamer,
    OllamaNamer,
    OllamaNamer,
    OpenAINamer,
    repair_json_string_backslashes,
    ReplicateNamer,
    TogetherNamer,
)

from toponymy.tests.helpers.llm_test_config import (
    validate_cluster_names,
    validate_topic_name,
    LITELLM_PROVIDER_CASES,
    SUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS,
    UNSUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS,
    SUPPORTED_ASYNC_DEBUG_CALLBACK_NAMERS,
    UNSUPPORTED_ASYNC_DEBUG_CALLBACK_NAMERS,
)
from toponymy.tests.helpers.errors import (
    make_openai_error,
    LITELLM_FAIL_FAST,
    LITELLM_RETRYABLE,
    ANTHROPIC_FAIL_FAST,
    ANTHROPIC_RETRYABLE,
    make_anthropic_error,
    LITELLM_FAIL_FAST,
    LITELLM_RETRYABLE,
    make_litellm_error,
)
import logging

logger = logging.getLogger(__name__)

LITELLM_ASYNC_LOGGING_WARNING_FILTER = (
    "ignore:.*Logging\\.async_success_handler.*was never awaited.*:RuntimeWarning"
)


class MockLLMResponse:
    """Mock response object that mimics different LLM service response structures"""

    @staticmethod
    def create_anthropic_response(content: str):
        class Content:
            def __init__(self, text):
                self.text = text

        class Response:
            def __init__(self, content):
                self.content = [Content(content)]

        return Response(content)

    @staticmethod
    def create_chat_response(content: str):
        class Choice:
            def __init__(self, content):
                self.message = Mock(content=content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(content)

    @staticmethod
    def create_cohere_response(content: str):
        return Mock(text=content)

    @staticmethod
    def create_cohere_response_v2(content: str):
        class Content:
            def __init__(self, text):
                self.text = text

        class Message:
            def __init__(self, content):
                self.content = [Content(content)]

        class Response:
            def __init__(self, content):
                self.message = Message(content)

        return Response(content)

    @staticmethod
    def create_huggingface_response(content: str):
        return [{"generated_text": content}]

    @staticmethod
    def create_llama_response(content: str):
        return {"choices": [{"text": content}]}

    @staticmethod
    def create_azureai_response(content: str):
        class Choice:
            def __init__(self, content):
                self.message = Mock(content=content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(content)

    @staticmethod
    def create_ollama_response(content: str):
        return {"response": content}

    @staticmethod
    def create_google_gemini_response(content: str):
        class MockText:
            def __init__(self, text):
                self.text = text

        class MockResponse:
            def __init__(self, text):
                self.model = Mock()
                self.model.generate_content = Mock(return_value=MockText(text))

        return MockResponse(content)


# Helper for async tests
async def async_return(value):
    return value


class MockAsyncResponse:
    """Mock async response objects for different LLM services"""

    @staticmethod
    def create_anthropic_response(content: str):
        class Content:
            def __init__(self, text):
                self.text = text

        class Response:
            def __init__(self, content):
                self.content = [Content(content)]

        return Response(content)

    @staticmethod
    def create_chat_response(content: str):
        class Choice:
            def __init__(self, content):
                self.message = Mock(content=content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(content)

    @staticmethod
    def create_cohere_response(content: str):
        class Content:
            def __init__(self, text):
                self.text = text

        class Message:
            def __init__(self, content):
                self.content = [Content(content)]

        class Response:
            def __init__(self, content):
                self.message = Message(content)

        return Response(content)


##============================================================
# OpenAI Tests
##============================================================


@pytest.fixture
def openai_wrapper():
    with patch("openai.OpenAI"):
        wrapper = OpenAINamer(api_key="dummy")
        return wrapper


@pytest.mark.parametrize("error", LITELLM_FAIL_FAST)
def test_openai_topic_name_fail_fast_error(openai_wrapper, error):
    with patch(
        "litellm.completion",
        side_effect=make_litellm_error(error),
    ):
        with pytest.raises(FailFastLLMError):
            result = openai_wrapper.generate_topic_name("test prompt")
            logger.error(f"No exception raised! Got result: {result!r}")


@pytest.mark.parametrize("error", LITELLM_FAIL_FAST)
def test_openai_topic_cluster_names_fail_fast_error(openai_wrapper, error, mock_data):
    with patch(
        "litellm.completion",
        side_effect=make_litellm_error(error),
    ):
        with pytest.raises(FailFastLLMError):
            result = openai_wrapper.generate_topic_cluster_names(
                "test prompt", mock_data["old_names"]
            )
            logger.error(f"No exception raised! Got result: {result!r}")


def test_openai_generate_topic_name_success(openai_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=response):
        result = openai_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)


def test_openai_generate_topic_name_success_system_prompt(openai_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=response):
        result = openai_wrapper.generate_topic_name(
            {"system": "system prompt", "user": "test prompt"}
        )
    validate_topic_name(result)


def test_openai_generate_cluster_names_success(openai_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_cluster_names"])
    with patch("litellm.completion", return_value=response):
        result = openai_wrapper.generate_topic_cluster_names(
            "test prompt", mock_data["old_names"]
        )
    validate_cluster_names(result)


def test_openai_generate_cluster_names_success_system_prompt(openai_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_cluster_names"])
    with patch("litellm.completion", return_value=response):
        result = openai_wrapper.generate_topic_cluster_names(
            {"system": "system prompt", "user": "test prompt"}, mock_data["old_names"]
        )
    validate_cluster_names(result)


def test_openai_generate_cluster_names_success_on_malformed_mapping(
    openai_wrapper, mock_data
):
    response = MockLLMResponse.create_chat_response(mock_data["malformed_mapping"])
    with patch("litellm.completion", return_value=response):
        result = openai_wrapper.generate_topic_cluster_names(
            "test prompt", mock_data["old_names"]
        )
    validate_cluster_names(result)


@pytest.mark.parametrize("error_class", LITELLM_RETRYABLE)
@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_openai_generate_topic_name_retry_exhausted_returns_empty(
    openai_wrapper, error_class
):
    with patch(
        "litellm.completion",
        side_effect=[make_litellm_error(error_class) for _ in range(3)],
    ) as mock_completion:
        result = openai_wrapper.generate_topic_name("test prompt")

    assert result == ""
    assert mock_completion.call_count == 3


def test_openai_generate_topic_name_failure_malformed_json(openai_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["malformed_json"])
    with patch("litellm.completion", return_value=response):
        result = openai_wrapper.generate_topic_name("test prompt")
    assert result == ""


@pytest.mark.parametrize("error_class", LITELLM_RETRYABLE)
@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_openai_generate_cluster_names_retry_exhausted_returns_old_names(
    openai_wrapper, mock_data, error_class
):
    with patch(
        "litellm.completion",
        side_effect=[make_litellm_error(error_class) for _ in range(3)],
    ) as mock_completion:
        result = openai_wrapper.generate_topic_cluster_names(
            "test prompt",
            mock_data["old_names"],
        )

    assert result == mock_data["old_names"]
    assert mock_completion.call_count == 3


## OpenAI Async Tests
@pytest_asyncio.fixture
async def async_openai_wrapper():
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        wrapper = AsyncOpenAINamer(api_key="dummy")
        try:
            yield wrapper
        finally:
            await wrapper.close()


@pytest.mark.asyncio
async def test_async_openai_generate_topic_names_success(
    async_openai_wrapper, mock_data
):
    response = MockAsyncResponse.create_chat_response(mock_data["valid_topic_name"])
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await async_openai_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_names_system_prompt(
    async_openai_wrapper, mock_data
):
    response = MockAsyncResponse.create_chat_response(mock_data["valid_topic_name"])
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await async_openai_wrapper.generate_topic_names(
            [{"system": "system prompt", "user": "test prompt"}]
        )
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_success(
    async_openai_wrapper, mock_data
):
    response = MockAsyncResponse.create_chat_response(mock_data["valid_cluster_names"])
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await async_openai_wrapper.generate_topic_cluster_names(
            ["test prompt"], [mock_data["old_names"]]
        )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_system_prompt(
    async_openai_wrapper, mock_data
):
    response = MockAsyncResponse.create_chat_response(mock_data["valid_cluster_names"])
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await async_openai_wrapper.generate_topic_cluster_names(
            [{"system": "system prompt", "user": "test prompt"}],
            [mock_data["old_names"]],
        )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_malformed_mapping(
    async_openai_wrapper, mock_data
):
    response = MockAsyncResponse.create_chat_response(mock_data["malformed_mapping"])
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await async_openai_wrapper.generate_topic_cluster_names(
            ["test prompt"], [mock_data["old_names"]]
        )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", LITELLM_FAIL_FAST)
async def test_async_openai_generate_topic_names_fail_fast_raises(
    async_openai_wrapper, error_class
):
    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=make_litellm_error(error_class)),
    ):

        with pytest.raises(FailFastLLMError):
            await async_openai_wrapper.generate_topic_names(["test prompt"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", LITELLM_RETRYABLE)
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_openai_generate_topic_names_retryable_returns_empty(
    async_openai_wrapper,
    error_class,
):
    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[make_litellm_error(error_class) for _ in range(3)]),
    ) as mock_acompletion:
        result = await async_openai_wrapper.generate_topic_names(["test prompt"])

    assert result == [""]
    assert mock_acompletion.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", LITELLM_RETRYABLE)
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_openai_generate_topic_cluster_names_retryable_returns_old_names(
    async_openai_wrapper,
    mock_data,
    error_class,
):
    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[make_litellm_error(error_class) for _ in range(3)]),
    ):

        result = await async_openai_wrapper.generate_topic_cluster_names(
            ["test prompt"],
            [mock_data["old_names"]],
        )

    assert result == [mock_data["old_names"]]


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_openai_retries_per_item_not_whole_batch(
    async_openai_wrapper, mock_data
):
    good_response = MockAsyncResponse.create_chat_response(
        mock_data["valid_topic_name"]
    )
    error_class = LITELLM_RETRYABLE[0]
    call_counts = {"prompt1": 0, "prompt2": 0}

    async def mock_acompletion(**kwargs):
        prompt_text = kwargs["messages"][0]["content"]
        call_counts[prompt_text] += 1

        if prompt_text == "prompt1":
            return good_response
        if prompt_text == "prompt2":
            raise make_litellm_error(error_class)

        raise AssertionError(f"Unexpected prompt: {prompt_text}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=mock_acompletion)):
        result = await async_openai_wrapper.generate_topic_names(["prompt1", "prompt2"])

    assert len(result) == 2
    validate_topic_name(result[0])
    assert result[1] == ""
    assert call_counts["prompt1"] == 1
    assert call_counts["prompt2"] == 3


@pytest.mark.asyncio
async def test_async_openai_generate_topic_names_retry_exhausted_warns(
    async_openai_wrapper,
):
    error_class = LITELLM_RETRYABLE[0]

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[make_litellm_error(error_class) for _ in range(3)]),
    ):

        with pytest.warns(UserWarning, match="Failed to generate topic name"):
            result = await async_openai_wrapper.generate_topic_names(["test prompt"])

    assert result == [""]


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_retry_exhausted_warns(
    async_openai_wrapper, mock_data
):
    error_class = LITELLM_RETRYABLE[0]

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[make_litellm_error(error_class) for _ in range(3)]),
    ):

        with pytest.warns(UserWarning):
            result = await async_openai_wrapper.generate_topic_cluster_names(
                ["test prompt"],
                [mock_data["old_names"]],
            )

    assert result == [mock_data["old_names"]]


# Azure AI
