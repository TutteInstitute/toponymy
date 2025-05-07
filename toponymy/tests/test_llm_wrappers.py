import pytest
from unittest.mock import Mock, patch
import json
from typing import List

from toponymy.llm_wrappers import Anthropic, OpenAI, Cohere, HuggingFace, AzureAI #, LlamaCpp

# Mock responses for different scenarios
VALID_TOPIC_NAME_RESPONSE = {
    "topic_name": "Machine Learning",
    "topic_specificity": 0.6
}

VALID_CLUSTER_NAMES_RESPONSE = {
    "new_topic_name_mapping": {
        "1. data": "Data Science",
        "2. ml": "Machine Learning",
        "3. ai": "Artificial Intelligence"
    },
    "topic_specificities": [
        0.6,
        0.8,
        0.7,
    ]
}

MALFORMED_JSON_RESPONSE = "{"  # Incomplete JSON
RECOVERABLE_MALFORMED_JSON_RESPONSE = """
the topic name is Machine Learning
```json
{"topic_name": "Machine Learning", "topic_specificity": 0.6}
```
postamble.
"""
EMPTY_MAPPING_RESPONSE = {"new_topic_name_mapping": {}}
MALFORMED_MAPPING_RESPONSE = """{"new_topic_name_mapping": {"data science": "Data Science", "data science": "Machine Learning", "data science": "Artificial Intelligence"} , "topic_specificities": [0.6, 0.8, 0.7]}"""

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

# Helper functions for validation
def validate_topic_name(result: str):
    assert result == "Machine Learning"

def validate_cluster_names(result: List[str]):
    expected = ["Data Science", "Machine Learning", "Artificial Intelligence"]
    assert result == expected

@pytest.fixture
def mock_data():
    return {
        "valid_topic_name": json.dumps(VALID_TOPIC_NAME_RESPONSE),
        "valid_cluster_names": json.dumps(VALID_CLUSTER_NAMES_RESPONSE),
        "old_names": ["data", "ml", "ai"],
        "malformed_mapping": MALFORMED_MAPPING_RESPONSE,
        "malformed_json": MALFORMED_JSON_RESPONSE,
        "recoverable_malformed_json": RECOVERABLE_MALFORMED_JSON_RESPONSE,
    }

# # LlamaCpp Tests
# @pytest.fixture
# def llamacpp_wrapper():
#     with patch('llama_cpp.Llama'):
#         wrapper = LlamaCpp(model_path="dummy")
#         return wrapper

# def test_llamacpp_generate_topic_name_success(llamacpp_wrapper, mock_data):
#     response = MockLLMResponse.create_llama_response(mock_data["valid_topic_name"])
#     llamacpp_wrapper.llm = Mock(return_value=response)
    
#     result = llamacpp_wrapper.generate_topic_name("test prompt")
#     validate_topic_name(result)

# def test_llamacpp_generate_cluster_names_success(llamacpp_wrapper, mock_data):
#     response = MockLLMResponse.create_llama_response(mock_data["valid_cluster_names"])
#     llamacpp_wrapper.llm = Mock(return_value=response)
    
#     result = llamacpp_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
#     validate_cluster_names(result)

# def test_llamacpp_generate_cluster_names_success_on_malformed_mapping(llamacpp_wrapper, mock_data):
#     response = MockLLMResponse.create_llama_response(mock_data["malformed_mapping"])
#     llamacpp_wrapper.llm = Mock(return_value=response)
    
#     result = llamacpp_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
#     validate_cluster_names(result)

# def test_llamacpp_generate_topic_name_failure(llamacpp_wrapper):
#     llamacpp_wrapper.llm = Mock(side_effect=Exception("API Error"))
#     result = llamacpp_wrapper.generate_topic_name("test prompt")
#     assert result == ""

# def test_llamacpp_generate_topic_name_failure_malformed_json(llamacpp_wrapper, mock_data):
#     llamacpp_wrapper.llm = Mock(mock_data["malformed_json"])
#     result = llamacpp_wrapper.generate_topic_name("test prompt")
#     assert result == ""

# def test_llamacpp_generate_cluster_names_failure(llamacpp_wrapper, mock_data):
#     llamacpp_wrapper.llm = Mock(side_effect=Exception("API Error"))
#     result = llamacpp_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
#     assert result == mock_data["old_names"]


# Huggingface Tests
@pytest.fixture
def huggingface_wrapper():
    with patch('transformers.pipeline'):
        wrapper = HuggingFace(model="dummy")
        return wrapper

def test_huggingface_generate_topic_name_success(huggingface_wrapper, mock_data):
    response = MockLLMResponse.create_huggingface_response(mock_data["valid_topic_name"])
    huggingface_wrapper.llm = Mock(return_value=response)
    
    result = huggingface_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)

def test_huggingface_generate_cluster_names_success(huggingface_wrapper, mock_data):
    response = MockLLMResponse.create_huggingface_response(mock_data["valid_cluster_names"])
    huggingface_wrapper.llm = Mock(return_value=response)
    
    result = huggingface_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_huggingface_generate_cluster_names_success_on_malformed_mapping(huggingface_wrapper, mock_data):
    response = MockLLMResponse.create_huggingface_response(mock_data["malformed_mapping"])
    huggingface_wrapper.llm = Mock(return_value=response)
    
    result = huggingface_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_huggingface_generate_topic_name_failure(huggingface_wrapper):
    huggingface_wrapper.llm = Mock(side_effect=Exception("API Error"))
    result = huggingface_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_huggingface_generate_topic_name_failure_malformed_json(huggingface_wrapper, mock_data):
    huggingface_wrapper.llm = Mock(mock_data["malformed_json"])
    result = huggingface_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_huggingface_generate_cluster_names_failure(huggingface_wrapper, mock_data):
    huggingface_wrapper.llm = Mock(side_effect=Exception("API Error"))
    result = huggingface_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    assert result == mock_data["old_names"]


# Anthropic Tests
@pytest.fixture
def anthropic_wrapper():
    with patch('anthropic.Anthropic'):
        wrapper = Anthropic(API_KEY="dummy")
        return wrapper

def test_anthropic_generate_topic_name_success(anthropic_wrapper, mock_data):
    response = MockLLMResponse.create_anthropic_response(mock_data["valid_topic_name"])
    anthropic_wrapper.llm.messages.create = Mock(return_value=response)
    
    result = anthropic_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)

def test_anthropic_generate_cluster_names_success(anthropic_wrapper, mock_data):
    response = MockLLMResponse.create_anthropic_response(mock_data["valid_cluster_names"])
    anthropic_wrapper.llm.messages.create = Mock(return_value=response)
    
    result = anthropic_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_anthropic_generate_cluster_names_success_on_malformed_mapping(anthropic_wrapper, mock_data):
    response = MockLLMResponse.create_anthropic_response(mock_data["malformed_mapping"])
    anthropic_wrapper.llm.messages.create = Mock(return_value=response)
    
    result = anthropic_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_anthropic_generate_topic_name_failure(anthropic_wrapper):
    anthropic_wrapper.llm.messages.create = Mock(side_effect=Exception("API Error"))
    result = anthropic_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_anthropic_generate_topic_name_failure_malformed_json(anthropic_wrapper, mock_data):
    anthropic_wrapper.llm.messages.create = Mock(mock_data["malformed_json"])
    result = anthropic_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_anthropic_generate_cluster_names_failure(anthropic_wrapper, mock_data):
    anthropic_wrapper.llm.messages.create = Mock(side_effect=Exception("API Error"))
    result = anthropic_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    assert result == mock_data["old_names"]

# OpenAI Tests
@pytest.fixture
def openai_wrapper():
    with patch('openai.OpenAI'):
        wrapper = OpenAI(API_KEY="dummy")
        return wrapper

def test_openai_generate_topic_name_success(openai_wrapper, mock_data):
    response = MockLLMResponse.create_openai_response(mock_data["valid_topic_name"])
    openai_wrapper.llm.chat.completions.create = Mock(return_value=response)
    
    result = openai_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)

def test_openai_generate_cluster_names_success(openai_wrapper, mock_data):
    response = MockLLMResponse.create_openai_response(mock_data["valid_cluster_names"])
    openai_wrapper.llm.chat.completions.create = Mock(return_value=response)
    
    result = openai_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_openai_generate_cluster_names_success_on_malformed_mapping(openai_wrapper, mock_data):
    response = MockLLMResponse.create_openai_response(mock_data["malformed_mapping"])
    openai_wrapper.llm.chat.completions.create = Mock(return_value=response)
    
    result = openai_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_openai_generate_topic_name_failure(openai_wrapper):
    openai_wrapper.llm.messages.create = Mock(side_effect=Exception("API Error"))
    result = openai_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_openai_generate_topic_name_failure_malformed_json(openai_wrapper, mock_data):
    openai_wrapper.llm.messages.create = Mock(mock_data["malformed_json"])
    result = openai_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_openai_generate_cluster_names_failure(openai_wrapper, mock_data):
    openai_wrapper.llm.messages.create = Mock(side_effect=Exception("API Error"))
    result = openai_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    assert result == mock_data["old_names"]

# Cohere Tests
@pytest.fixture
def cohere_wrapper():
    with patch('cohere.ClientV2') as mock_client:
        # Mock the models.get method to prevent UnauthorizedError
        mock_client.return_value.models = Mock()
        mock_client.return_value.models.get = Mock()
        wrapper = Cohere(API_KEY="dummy")
        return wrapper

def test_cohere_generate_topic_name_success(cohere_wrapper, mock_data):
    response = MockLLMResponse.create_cohere_response_v2(mock_data["valid_topic_name"])
    cohere_wrapper.llm.chat = Mock(return_value=response)
    
    result = cohere_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)

def test_cohere_generate_cluster_names_success(cohere_wrapper, mock_data):
    response = MockLLMResponse.create_cohere_response_v2(mock_data["valid_cluster_names"])
    cohere_wrapper.llm.chat = Mock(return_value=response)
    
    result = cohere_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_cohere_generate_cluster_names_success_on_malformed_mapping(cohere_wrapper, mock_data):
    response = MockLLMResponse.create_cohere_response_v2(mock_data["malformed_mapping"])
    cohere_wrapper.llm.chat = Mock(return_value=response)
    
    result = cohere_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_cohere_generate_topic_name_failure(cohere_wrapper):
    cohere_wrapper.llm.chat = Mock(side_effect=Exception("API Error"))
    result = cohere_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_cohere_generate_topic_name_failure_malformed_json(cohere_wrapper, mock_data):
    cohere_wrapper.llm.chat = Mock(mock_data["malformed_json"])
    result = cohere_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_cohere_generate_cluster_names_failure(cohere_wrapper, mock_data):
    cohere_wrapper.llm.chat = Mock(side_effect=Exception("API Error"))
    result = cohere_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    assert result == mock_data["old_names"]

# AzureAI Tests
@pytest.fixture
def azureai_wrapper():
    with patch('azure.ai.inference.ChatCompletionsClient'):
        wrapper = AzureAI(API_KEY="dummy", endpoint="https://dummy.services.ai.azure.com/models", model="dummy")
        return wrapper
    
def test_azureai_generate_topic_name_success(azureai_wrapper, mock_data):
    response = MockLLMResponse.create_openai_response(mock_data["valid_topic_name"])
    azureai_wrapper.llm.complete = Mock(return_value=response)
    
    result = azureai_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)

def test_azureai_generate_cluster_names_success(azureai_wrapper, mock_data):
    response = MockLLMResponse.create_openai_response(mock_data["valid_cluster_names"])
    azureai_wrapper.llm.complete = Mock(return_value=response)
    
    result = azureai_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_azureai_generate_cluster_names_success_on_malformed_mapping(azureai_wrapper, mock_data):
    response = MockLLMResponse.create_azureai_response(mock_data["malformed_mapping"])
    azureai_wrapper.llm.complete = Mock(return_value=response)
    
    result = azureai_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_azureai_generate_topic_name_failure(azureai_wrapper):
    azureai_wrapper.llm.complete = Mock(side_effect=Exception("API Error"))
    result = azureai_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_azureai_generate_topic_name_failure_malformed_json(azureai_wrapper, mock_data):
    azureai_wrapper.llm.complete = Mock(mock_data["malformed_json"])
    result = azureai_wrapper.generate_topic_name("test prompt")
    assert result == ""

def test_azureai_generate_cluster_names_failure(azureai_wrapper, mock_data):
    azureai_wrapper.llm.complete = Mock(side_effect=Exception("API Error"))
    result = azureai_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    assert result == mock_data["old_names"]

# @pytest.skip("Not quite sure this is right yet")
# def test_importerror_handling():
#     """Test that import errors are properly handled"""
#     with patch.dict('sys.modules', {'anthropic': None}):
#         from importlib import reload
#         import sys
        
#         # Force reload of the module to trigger ImportError handling
#         if 'toponymy.llm_wrappers' in sys.modules:
#             reload(sys.modules['toponymy.llm_wrappers'])
        
#         # Verify that the Anthropic class is not available
#         assert 'Anthropic' not in globals()