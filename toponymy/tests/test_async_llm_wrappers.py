import os

import pytest
from unittest.mock import Mock, patch, AsyncMock

#import asyncio
import pytest_asyncio

from toponymy.llm_wrappers import (
    AsyncCohereNamer, AsyncAnthropicNamer, BatchAnthropicNamer, AsyncOpenAINamer, AsyncAzureAINamer,
    AsyncOllamaNamer, AsyncGoogleGeminiNamer, AsyncTogether, FailFastLLMError, CallResult
)
from toponymy.tests.helpers.make_llm_data import (
    validate_topic_name,
    validate_cluster_names
)
from toponymy.tests.helpers.errors import (
    ANTHROPIC_FAIL_FAST,
    ANTHROPIC_RETRYABLE,
    make_anthropic_error,
    OPENAI_FAIL_FAST,
    OPENAI_RETRYABLE,
    make_openai_error
)



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
    def create_openai_response(content: str):
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


    @staticmethod
    def create_azureai_response(content: str):
        class Choice:
            def __init__(self, content):
                self.message = Mock(content=content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(content)


# AsyncCohere Tests
@pytest_asyncio.fixture
async def async_cohere_wrapper():
    with patch('cohere.AsyncClientV2'):
        wrapper = AsyncCohereNamer(api_key="dummy")
        yield wrapper
        # Clean up any resources
        try:
            await wrapper.close()
        except:
            pass


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_names_success(async_cohere_wrapper, mock_data):
    response = MockAsyncResponse.create_cohere_response(mock_data["valid_topic_name"])
    async_cohere_wrapper.llm.chat = AsyncMock(return_value=response)

    result = await async_cohere_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_names_system_prompt(async_cohere_wrapper, mock_data):
    response = MockAsyncResponse.create_cohere_response(mock_data["valid_topic_name"])
    async_cohere_wrapper.llm.chat = AsyncMock(return_value=response)

    result = await async_cohere_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_cluster_names_success(async_cohere_wrapper, mock_data):
    response = MockAsyncResponse.create_cohere_response(mock_data["valid_cluster_names"])
    async_cohere_wrapper.llm.chat = AsyncMock(return_value=response)

    result = await async_cohere_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_cluster_names_system_prompt(async_cohere_wrapper, mock_data):
    response = MockAsyncResponse.create_cohere_response(mock_data["valid_cluster_names"])
    async_cohere_wrapper.llm.chat = AsyncMock(return_value=response)

    result = await async_cohere_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_cluster_names_malformed_mapping(async_cohere_wrapper, mock_data):
    response = MockAsyncResponse.create_cohere_response(mock_data["malformed_mapping"])
    async_cohere_wrapper.llm.chat = AsyncMock(return_value=response)

    result = await async_cohere_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_names_failure(async_cohere_wrapper):
    async_cohere_wrapper.llm.chat = AsyncMock(side_effect=Exception("API Error"))
    result = await async_cohere_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    assert result[0] == ""


@pytest.mark.asyncio
async def test_async_cohere_generate_topic_cluster_names_failure(async_cohere_wrapper, mock_data):
    async_cohere_wrapper.llm.chat = AsyncMock(side_effect=Exception("API Error"))
    result = await async_cohere_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    assert result[0] == mock_data["old_names"]


@pytest.mark.asyncio
async def test_async_cohere_batch_processing(async_cohere_wrapper, mock_data):
    response = MockAsyncResponse.create_cohere_response(mock_data["valid_topic_name"])
    async_cohere_wrapper.llm.chat = AsyncMock(return_value=response)

    # Test batch processing with multiple prompts
    result = await async_cohere_wrapper.generate_topic_names(["prompt1", "prompt2", "prompt3"])
    assert len(result) == 3
    assert all(name == "Machine Learning" for name in result)

# AsyncAnthropic Tests
@pytest_asyncio.fixture
async def async_anthropic_wrapper():
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        wrapper = AsyncAnthropicNamer(api_key="dummy")
        try:
            yield wrapper
        finally:
            await wrapper.close()



@pytest.mark.external
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_connectivity_async_plain_canary():
    namer = AsyncAnthropicNamer(api_key=os.getenv("ANTHROPIC_API_KEY"))

    result = await namer.connectivity_status()

    assert result["success"], (
        f"Async plain canary failed for Anthropic:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


@pytest.mark.external
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_connectivity_async_system_canary():
    namer = AsyncAnthropicNamer(api_key=os.getenv("ANTHROPIC_API_KEY"))

    result = await namer.connectivity_status(
        prompt="Return a short JSON object describing your role.",
        system_prompt="You are a topic naming assistant.",
    )

    assert result["success"], (
        f"Async system canary failed for Anthropic:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_names_success(async_anthropic_wrapper, mock_data):
    response = MockAsyncResponse.create_anthropic_response(mock_data["valid_topic_name"])
    async_anthropic_wrapper.client.messages.create = AsyncMock(return_value=response)

    result = await async_anthropic_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_names_system_prompt(async_anthropic_wrapper, mock_data):
    response = MockAsyncResponse.create_anthropic_response(mock_data["valid_topic_name"])
    async_anthropic_wrapper.client.messages.create = AsyncMock(return_value=response)

    result = await async_anthropic_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_cluster_names_success(async_anthropic_wrapper, mock_data):
    response = MockAsyncResponse.create_anthropic_response(mock_data["valid_cluster_names"])
    async_anthropic_wrapper.client.messages.create = AsyncMock(return_value=response)

    result = await async_anthropic_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_cluster_names_system_prompt(async_anthropic_wrapper, mock_data):
    response = MockAsyncResponse.create_anthropic_response(mock_data["valid_cluster_names"])
    async_anthropic_wrapper.client.messages.create = AsyncMock(return_value=response)

    result = await async_anthropic_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_cluster_names_malformed_mapping(async_anthropic_wrapper, mock_data):
    response = MockAsyncResponse.create_anthropic_response(mock_data["malformed_mapping"])
    async_anthropic_wrapper.client.messages.create = AsyncMock(return_value=response)

    result = await async_anthropic_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", ANTHROPIC_FAIL_FAST)
async def test_async_anthropic_generate_topic_names_fail_fast_raises(async_anthropic_wrapper, error_class):
    async_anthropic_wrapper.client.messages.create = AsyncMock(
        side_effect=make_anthropic_error(error_class)
    )

    with pytest.raises(FailFastLLMError):
        await async_anthropic_wrapper.generate_topic_names(
            ["test prompt"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", ANTHROPIC_RETRYABLE)
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_anthropic_generate_topic_names_retryable_returns_empty(
    async_anthropic_wrapper,
    error_class,
):
    async_anthropic_wrapper.client.messages.create = AsyncMock(
        side_effect=[
            make_anthropic_error(error_class) for _ in range(3)
        ]
    )

    result = await async_anthropic_wrapper.generate_topic_names(
        ["test prompt"]
    )

    assert result == [""]
    assert async_anthropic_wrapper.client.messages.create.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", ANTHROPIC_RETRYABLE)
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_anthropic_generate_topic_cluster_names_retryable_returns_old_names(
    async_anthropic_wrapper,
    mock_data,
    error_class,
):
    async_anthropic_wrapper.client.messages.create = AsyncMock(
        side_effect=[
            make_anthropic_error(error_class) for _ in range(3)
        ]
    )

    result = await async_anthropic_wrapper.generate_topic_cluster_names(
        ["test prompt"],
        [mock_data["old_names"]],
    )

    assert result == [mock_data["old_names"]]


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_anthropic_retries_per_item_not_whole_batch(
    async_anthropic_wrapper, mock_data
):
    good_response = MockAsyncResponse.create_anthropic_response(
        mock_data["valid_topic_name"]
    )
    error_class = ANTHROPIC_RETRYABLE[0]

    call_counts = {"prompt1": 0, "prompt2": 0}

    async def mock_create(*args, **kwargs):
        prompt_text = kwargs["messages"][0]["content"]
        call_counts[prompt_text] += 1

        if prompt_text == "prompt1":
            return good_response
        if prompt_text == "prompt2":
            raise make_anthropic_error(error_class)

        raise AssertionError(f"Unexpected prompt: {prompt_text}")

    async_anthropic_wrapper.client.messages.create = AsyncMock(
        side_effect=mock_create
    )

    result = await async_anthropic_wrapper.generate_topic_names(
        ["prompt1", "prompt2"]
    )

    assert len(result) == 2
    validate_topic_name(result[0])
    assert result[1] == ""

    assert call_counts["prompt1"] == 1
    assert call_counts["prompt2"] == 3


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_names_retry_exhausted_warns(
    async_anthropic_wrapper,
):
    error_class = ANTHROPIC_RETRYABLE[0]

    async_anthropic_wrapper.client.messages.create = AsyncMock(
        side_effect=[
            make_anthropic_error(error_class) for _ in range(3)
        ]
    )

    with pytest.warns(
        UserWarning,
        match="Failed to generate topic name",
    ):
        result = await async_anthropic_wrapper.generate_topic_names(
            ["test prompt"]
        )

    assert result == [""]


@pytest.mark.asyncio
async def test_async_anthropic_generate_topic_cluster_names_retry_exhausted_warns(
    async_anthropic_wrapper, mock_data
):
    error_class = ANTHROPIC_RETRYABLE[0]

    async_anthropic_wrapper.client.messages.create = AsyncMock(
        side_effect=[
            make_anthropic_error(error_class) for _ in range(3)
        ]
    )

    with pytest.warns(UserWarning):
        result = await async_anthropic_wrapper.generate_topic_cluster_names(
            ["test prompt"],
            [mock_data["old_names"]],
        )

    assert result == [mock_data["old_names"]]

# BatchAnthropic Tests
@pytest_asyncio.fixture
async def batch_anthropic_wrapper():
    with patch('anthropic.Anthropic'):
        wrapper = BatchAnthropicNamer(api_key="dummy")
        yield wrapper


@pytest.mark.asyncio
async def test_batch_anthropic_submit_batch(batch_anthropic_wrapper, mock_data):
    # Mock the batch creation
    batch_id = "batch_123456"
    batch_anthropic_wrapper.client.beta.messages.batches.create = Mock(
        return_value=Mock(id=batch_id)
    )

    # Test the submit_batch method
    result = batch_anthropic_wrapper.submit_batch(
        ["prompt1", "prompt2"], 0.4, 128
    )
    assert result == batch_id

    # Verify the client was called with the expected parameters
    batch_anthropic_wrapper.client.beta.messages.batches.create.assert_called_once()


@pytest.mark.asyncio
async def test_batch_anthropic_get_batch_status(batch_anthropic_wrapper):
    # Mock the batch retrieval
    status = "processing"
    batch_anthropic_wrapper.client.beta.messages.batches.retrieve = Mock(
        return_value=Mock(processing_status=status)
    )

    # Test the get_batch_status method
    result = batch_anthropic_wrapper.get_batch_status("batch_123456")
    assert result == status


@pytest.mark.asyncio
async def test_batch_anthropic_cancel_batch(batch_anthropic_wrapper):
    # Mock the batch cancellation
    batch_anthropic_wrapper.client.beta.messages.batches.cancel = Mock()

    # Test the cancel_batch method
    batch_anthropic_wrapper.cancel_batch("batch_123456")

    # Verify the client was called with the expected parameters
    batch_anthropic_wrapper.client.beta.messages.batches.cancel.assert_called_once_with("batch_123456")


@pytest.mark.asyncio
async def test_batch_anthropic_wait_for_completion(batch_anthropic_wrapper):
    # Mock the batch retrieval to simulate a completed batch
    batch_anthropic_wrapper.client.beta.messages.batches.retrieve = Mock(
        return_value=Mock(processing_status="ended")
    )

    # Test the _wait_for_completion_async method with a very short timeout
    batch_anthropic_wrapper.polling_interval = 0.01  # Use a short polling interval for the test
    result = await batch_anthropic_wrapper._wait_for_completion_async("batch_123456")
    assert result is True


# AsyncOpenAI Tests
@pytest_asyncio.fixture
async def async_openai_wrapper():
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    with patch('openai.AsyncOpenAI', return_value=mock_client):
        wrapper = AsyncOpenAINamer(api_key="dummy")
        try:
            yield wrapper
        finally:
            await wrapper.close()

@pytest.mark.external
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_connectivity_async_plain_canary():
    """
    Canary test verifying live async connectivity to the OpenAI API
    using the plain prompt path.
    """
    namer = AsyncOpenAINamer(api_key=os.getenv("OPENAI_API_KEY"))

    result = await namer.connectivity_status()

    assert result["success"], (
        f"Async plain canary failed for OpenAI:\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


@pytest.mark.external
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_connectivity_async_system_canary():
    """
    Canary test verifying live async connectivity to the OpenAI API
    using the system prompt path.
    """
    namer = AsyncOpenAINamer(api_key=os.getenv("OPENAI_API_KEY"))

    result = await namer.connectivity_status(
        prompt="Return a short JSON object describing your role.",
        system_prompt="You are a topic naming assistant.",
    )

    assert result["success"], (
        f"Async system canary failed for OpenAI:\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


@pytest.mark.asyncio
async def test_async_openai_generate_topic_names_success(async_openai_wrapper, mock_data):
    response = MockAsyncResponse.create_openai_response(mock_data["valid_topic_name"])
    async_openai_wrapper.client.chat.completions.create = AsyncMock(return_value=response)

    result = await async_openai_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_names_system_prompt(async_openai_wrapper, mock_data):
    response = MockAsyncResponse.create_openai_response(mock_data["valid_topic_name"])
    async_openai_wrapper.client.chat.completions.create = AsyncMock(return_value=response)

    result = await async_openai_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_success(async_openai_wrapper, mock_data):
    response = MockAsyncResponse.create_openai_response(mock_data["valid_cluster_names"])
    async_openai_wrapper.client.chat.completions.create = AsyncMock(return_value=response)

    result = await async_openai_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_system_prompt(async_openai_wrapper, mock_data):
    response = MockAsyncResponse.create_openai_response(mock_data["valid_cluster_names"])
    async_openai_wrapper.client.chat.completions.create = AsyncMock(return_value=response)

    result = await async_openai_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_malformed_mapping(async_openai_wrapper, mock_data):
    response = MockAsyncResponse.create_openai_response(mock_data["malformed_mapping"])
    async_openai_wrapper.client.chat.completions.create = AsyncMock(return_value=response)

    result = await async_openai_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])

@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", OPENAI_FAIL_FAST)
async def test_async_openai_generate_topic_names_fail_fast_raises(async_openai_wrapper, error_class):
    async_openai_wrapper.client.chat.completions.create = AsyncMock(
        side_effect=make_openai_error(error_class)
    )

    with pytest.raises(FailFastLLMError):
        await async_openai_wrapper.generate_topic_names(["test prompt"])

@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", OPENAI_RETRYABLE)
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_openai_generate_topic_names_retryable_returns_empty(
    async_openai_wrapper,
    error_class,
):
    async_openai_wrapper.client.chat.completions.create = AsyncMock(
        side_effect=[make_openai_error(error_class) for _ in range(3)]
    )

    result = await async_openai_wrapper.generate_topic_names(["test prompt"])

    assert result == [""]
    assert async_openai_wrapper.client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", OPENAI_RETRYABLE)
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_openai_generate_topic_cluster_names_retryable_returns_old_names(
    async_openai_wrapper,
    mock_data,
    error_class,
):
    async_openai_wrapper.client.chat.completions.create = AsyncMock(
        side_effect=[make_openai_error(error_class) for _ in range(3)]
    )

    result = await async_openai_wrapper.generate_topic_cluster_names(
        ["test prompt"],
        [mock_data["old_names"]],
    )

    assert result == [mock_data["old_names"]]

@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Failed to generate")
async def test_async_openai_retries_per_item_not_whole_batch( async_openai_wrapper, mock_data):
    good_response = MockAsyncResponse.create_openai_response( mock_data["valid_topic_name"] )
    error_class = OPENAI_RETRYABLE[0]
    call_counts = {"prompt1": 0, "prompt2": 0}
    async def mock_create(*args, **kwargs):
        prompt_text = kwargs["messages"][0]["content"]
        call_counts[prompt_text] += 1
        if prompt_text == "prompt1":
            return good_response
        elif prompt_text == "prompt2":
            raise make_openai_error(error_class)
        raise AssertionError(f"Unexpected prompt: {prompt_text}")

    async_openai_wrapper.client.chat.completions.create = AsyncMock(
        side_effect=mock_create
    )
    result = await async_openai_wrapper.generate_topic_names(
        ["prompt1", "prompt2"]
    )

    assert len(result) == 2
    validate_topic_name(result[0])
    assert result[1] == ""
    assert call_counts["prompt1"] == 1
    assert call_counts["prompt2"] == 3

@pytest.mark.asyncio
async def test_async_openai_generate_topic_names_retry_exhausted_warns(async_openai_wrapper):
    error_class = OPENAI_RETRYABLE[0]

    async_openai_wrapper.client.chat.completions.create = AsyncMock(
        side_effect=[make_openai_error(error_class) for _ in range(3)]
    )

    with pytest.warns(UserWarning, match="Failed to generate topic name"):
        result = await async_openai_wrapper.generate_topic_names(["test prompt"])

    assert result == [""]

@pytest.mark.asyncio
async def test_async_openai_generate_topic_cluster_names_retry_exhausted_warns(async_openai_wrapper, mock_data):
    error_class = OPENAI_RETRYABLE[0]

    async_openai_wrapper.client.chat.completions.create = AsyncMock(
        side_effect=[make_openai_error(error_class) for _ in range(3)]
    )

    with pytest.warns(UserWarning):
        result = await async_openai_wrapper.generate_topic_cluster_names(
            ["test prompt"],
            [mock_data["old_names"]],
        )

    assert result == [mock_data["old_names"]]

# AsyncAzureAI Tests
@pytest_asyncio.fixture
async def async_azureai_wrapper():
    with patch('azure.ai.inference.aio.ChatCompletionsClient'):
        wrapper = AsyncAzureAINamer(api_key="dummy", endpoint="https://dummy.services.ai.azure.com/models", model="dummy")
        yield wrapper
        # Clean up any resources
        try:
            await wrapper.close()
        except:
            pass


@pytest.mark.asyncio
async def test_async_azureai_generate_topic_names_success(async_azureai_wrapper, mock_data):
    response = MockAsyncResponse.create_azureai_response(mock_data["valid_topic_name"])
    async_azureai_wrapper.client.complete = AsyncMock(return_value=response)

    result = await async_azureai_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_azureai_generate_topic_names_system_prompt(async_azureai_wrapper, mock_data):
    response = MockAsyncResponse.create_azureai_response(mock_data["valid_topic_name"])
    async_azureai_wrapper.client.complete = AsyncMock(return_value=response)

    result = await async_azureai_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_azureai_generate_topic_cluster_names_success(async_azureai_wrapper, mock_data):
    response = MockAsyncResponse.create_azureai_response(mock_data["valid_cluster_names"])
    async_azureai_wrapper.client.complete = AsyncMock(return_value=response)

    result = await async_azureai_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_azureai_generate_topic_cluster_names_system_prompt(async_azureai_wrapper, mock_data):
    response = MockAsyncResponse.create_azureai_response(mock_data["valid_cluster_names"])
    async_azureai_wrapper.client.complete = AsyncMock(return_value=response)

    result = await async_azureai_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_azureai_generate_topic_names_multiple(async_azureai_wrapper, mock_data):
    """Test processing multiple prompts in a single call."""
    response = MockAsyncResponse.create_azureai_response(mock_data["valid_topic_name"])
    async_azureai_wrapper.client.complete = AsyncMock(return_value=response)

    # Test with 3 prompts
    result = await async_azureai_wrapper.generate_topic_names(["prompt1", "prompt2", "prompt3"])
    assert len(result) == 3
    for name in result:
        assert name == "Machine Learning"


@pytest.mark.asyncio
async def test_async_azureai_generate_topic_cluster_names_multiple(async_azureai_wrapper, mock_data):
    """Test processing multiple prompt/old_names pairs in a single call."""
    response = MockAsyncResponse.create_azureai_response(mock_data["valid_cluster_names"])
    async_azureai_wrapper.client.complete = AsyncMock(return_value=response)

    old_names_list = [["data", "ml", "ai"], ["x", "y", "z"]]
    prompts = ["prompt1", "prompt2"]

    result = await async_azureai_wrapper.generate_topic_cluster_names(prompts, old_names_list)
    assert len(result) == 2
    validate_cluster_names(result[0])


# AsyncOllama Tests
@pytest_asyncio.fixture
async def async_ollama_wrapper():
    with patch('ollama.AsyncClient'):
        wrapper = AsyncOllamaNamer(model="llama3.2", host="http://localhost:11434")
        yield wrapper


@pytest.mark.asyncio
async def test_async_ollama_generate_topic_names_success(async_ollama_wrapper, mock_data):
    async_ollama_wrapper.client.generate = AsyncMock(return_value={'response': mock_data["valid_topic_name"]})

    result = await async_ollama_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_ollama_generate_topic_names_system_prompt(async_ollama_wrapper, mock_data):
    async_ollama_wrapper.client.chat = AsyncMock(return_value={'message': {'content': mock_data["valid_topic_name"]}})

    result = await async_ollama_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_ollama_generate_topic_cluster_names_success(async_ollama_wrapper, mock_data):
    async_ollama_wrapper.client.generate = AsyncMock(return_value={'response': mock_data["valid_cluster_names"]})

    result = await async_ollama_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_ollama_generate_topic_cluster_names_system_prompt(async_ollama_wrapper, mock_data):
    async_ollama_wrapper.client.chat = AsyncMock(return_value={'message': {'content': mock_data["valid_cluster_names"]}})

    result = await async_ollama_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_ollama_generate_topic_names_failure(async_ollama_wrapper):
    async_ollama_wrapper.client.generate = AsyncMock(side_effect=Exception("API Error"))
    result = await async_ollama_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    assert result[0] == ""


@pytest.mark.asyncio
async def test_async_ollama_generate_topic_cluster_names_failure(async_ollama_wrapper, mock_data):
    async_ollama_wrapper.client.generate = AsyncMock(side_effect=Exception("API Error"))
    result = await async_ollama_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    assert result[0] == mock_data["old_names"]


@pytest.mark.asyncio
async def test_async_ollama_batch_processing(async_ollama_wrapper, mock_data):
    async_ollama_wrapper.client.generate = AsyncMock(return_value={'response': mock_data["valid_topic_name"]})

    # Test batch processing with multiple prompts
    result = await async_ollama_wrapper.generate_topic_names(["prompt1", "prompt2", "prompt3"])
    assert len(result) == 3
    assert all(name == "Machine Learning" for name in result)


# AsyncGoogleGemini Tests
@pytest_asyncio.fixture
async def async_google_gemini_wrapper():
    with patch('google.generativeai.configure'), patch('google.generativeai.GenerativeModel'):
        wrapper = AsyncGoogleGeminiNamer(api_key="dummy", model="gemini-1.5-flash")
        yield wrapper


@pytest.mark.asyncio
async def test_async_google_gemini_generate_topic_names_success(async_google_gemini_wrapper, mock_data):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(return_value=Mock(text=mock_data["valid_topic_name"]))

    result = await async_google_gemini_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_google_gemini_generate_topic_names_system_prompt(async_google_gemini_wrapper, mock_data):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(return_value=Mock(text=mock_data["valid_topic_name"]))

    result = await async_google_gemini_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_google_gemini_generate_topic_cluster_names_success(async_google_gemini_wrapper, mock_data):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(return_value=Mock(text=mock_data["valid_cluster_names"]))

    result = await async_google_gemini_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_google_gemini_generate_topic_cluster_names_system_prompt(async_google_gemini_wrapper, mock_data):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(return_value=Mock(text=mock_data["valid_cluster_names"]))

    result = await async_google_gemini_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_google_gemini_generate_topic_names_failure(async_google_gemini_wrapper):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(side_effect=Exception("API Error"))
    result = await async_google_gemini_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    assert result[0] == ""


@pytest.mark.asyncio
async def test_async_google_gemini_generate_topic_cluster_names_failure(async_google_gemini_wrapper, mock_data):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(side_effect=Exception("API Error"))
    result = await async_google_gemini_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    assert result[0] == mock_data["old_names"]


@pytest.mark.asyncio
async def test_async_google_gemini_batch_processing(async_google_gemini_wrapper, mock_data):
    async_google_gemini_wrapper.model.generate_content_async = AsyncMock(return_value=Mock(text=mock_data["valid_topic_name"]))

    # Test batch processing with multiple prompts
    result = await async_google_gemini_wrapper.generate_topic_names(["prompt1", "prompt2", "prompt3"])
    assert len(result) == 3
    assert all(name == "Machine Learning" for name in result)


# AsyncTogether Tests
@pytest_asyncio.fixture
async def async_together_wrapper():
    with patch('together.AsyncTogether'):
        wrapper = AsyncTogether(api_key="dummy", model="meta-llama/Llama-3-8b-chat-hf")
        yield wrapper
        # Clean up any resources
        try:
            await wrapper.close()
        except:
            pass


@pytest.mark.asyncio
async def test_async_together_generate_topic_names_success(async_together_wrapper, mock_data):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = mock_data["valid_topic_name"]
    async_together_wrapper.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await async_together_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_together_generate_topic_names_system_prompt(async_together_wrapper, mock_data):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = mock_data["valid_topic_name"]
    async_together_wrapper.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await async_together_wrapper.generate_topic_names([{"system": "system prompt", "user": "test prompt"}])
    assert len(result) == 1
    validate_topic_name(result[0])


@pytest.mark.asyncio
async def test_async_together_generate_topic_cluster_names_success(async_together_wrapper, mock_data):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = mock_data["valid_cluster_names"]
    async_together_wrapper.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await async_together_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_together_generate_topic_cluster_names_system_prompt(async_together_wrapper, mock_data):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = mock_data["valid_cluster_names"]
    async_together_wrapper.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await async_together_wrapper.generate_topic_cluster_names(
        [{"system": "system prompt", "user": "test prompt"}],
        [mock_data["old_names"]]
    )
    assert len(result) == 1
    validate_cluster_names(result[0])


@pytest.mark.asyncio
async def test_async_together_generate_topic_names_failure(async_together_wrapper):
    async_together_wrapper.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    result = await async_together_wrapper.generate_topic_names(["test prompt"])
    assert len(result) == 1
    assert result[0] == ""


@pytest.mark.asyncio
async def test_async_together_generate_topic_cluster_names_failure(async_together_wrapper, mock_data):
    async_together_wrapper.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    result = await async_together_wrapper.generate_topic_cluster_names(
        ["test prompt"], [mock_data["old_names"]]
    )
    assert len(result) == 1
    assert result[0] == mock_data["old_names"]


@pytest.mark.asyncio
async def test_async_together_batch_processing(async_together_wrapper, mock_data):
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = mock_data["valid_topic_name"]
    async_together_wrapper.client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test batch processing with multiple prompts
    result = await async_together_wrapper.generate_topic_names(["prompt1", "prompt2", "prompt3"])
    assert len(result) == 3
    assert all(name == "Machine Learning" for name in result)


# Test for AsyncLLMWrapper base class error cases
@pytest.mark.asyncio
async def test_async_wrapper_invalid_input(async_openai_wrapper):
    """Test handling of invalid input types."""
    with pytest.raises(ValueError):
        await async_openai_wrapper.generate_topic_names([123])  # Not a string or dict

    with pytest.raises(ValueError):
        await async_openai_wrapper.generate_topic_cluster_names(
            ["prompt1", "prompt2"],
            [["name1", "name2"]]
        )  # Mismatched lengths