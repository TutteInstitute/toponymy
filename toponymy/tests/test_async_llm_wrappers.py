import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from typing import List

import asyncio
import pytest_asyncio

from toponymy.llm_wrappers import (
    AsyncCohere, AsyncAnthropic, BatchAnthropic, AsyncOpenAI, AsyncAzureAI
)
from toponymy.tests.test_llm_wrappers import (
    MockLLMResponse, VALID_TOPIC_NAME_RESPONSE, VALID_CLUSTER_NAMES_RESPONSE,
    MALFORMED_JSON_RESPONSE, MALFORMED_MAPPING_RESPONSE, validate_topic_name,
    validate_cluster_names
)


@pytest.fixture
def mock_data():
    return {
        "valid_topic_name": json.dumps(VALID_TOPIC_NAME_RESPONSE),
        "valid_cluster_names": json.dumps(VALID_CLUSTER_NAMES_RESPONSE),
        "old_names": ["data", "ml", "ai"],
        "old_names_list": [["data", "ml", "ai"], ["x", "y", "z"]],
        "malformed_mapping": MALFORMED_MAPPING_RESPONSE,
        "malformed_json": MALFORMED_JSON_RESPONSE,
    }


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
        wrapper = AsyncCohere(api_key="dummy")
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
    with patch('anthropic.AsyncAnthropic'):
        wrapper = AsyncAnthropic(api_key="dummy")
        yield wrapper


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


# BatchAnthropic Tests
@pytest_asyncio.fixture
async def batch_anthropic_wrapper():
    with patch('anthropic.Anthropic'):
        wrapper = BatchAnthropic(api_key="dummy")
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
    with patch('openai.AsyncOpenAI'):
        wrapper = AsyncOpenAI(api_key="dummy")
        yield wrapper
        # Clean up any resources
        try:
            await wrapper.close()
        except:
            pass


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


# AsyncAzureAI Tests
@pytest_asyncio.fixture
async def async_azureai_wrapper():
    with patch('azure.ai.inference.aio.ChatCompletionsClient'):
        wrapper = AsyncAzureAI(api_key="dummy", endpoint="https://dummy.services.ai.azure.com/models", model="dummy")
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
    
    old_names_list = [["data1", "ml1", "ai1"], ["data2", "ml2", "ai2"]]
    prompts = ["prompt1", "prompt2"]
    
    result = await async_azureai_wrapper.generate_topic_cluster_names(prompts, old_names_list)
    assert len(result) == 2
    for names in result:
        validate_cluster_names(names)


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