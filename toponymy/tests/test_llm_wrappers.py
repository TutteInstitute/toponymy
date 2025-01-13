import pytest
from unittest.mock import Mock, patch
import json
from typing import List

from toponymy.llm_wrappers import Anthropic, OpenAI, Cohere, HuggingFace, LlamaCpp

# Mock responses for different scenarios
VALID_TOPIC_NAME_RESPONSE = {
    "topic_name": "Machine Learning"
}

VALID_CLUSTER_NAMES_RESPONSE = {
    "new_topic_name_mapping": {
        "0. data": "Data Science",
        "1. ml": "Machine Learning",
        "2. ai": "Artificial Intelligence"
    }
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
MALFORMED_MAPPING_RESPONSE = """{"new_topic_name_mapping": {"data science": "Data Science", "data science": "Machine Learning", "data science": "Artificial Intelligence"}}"""

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
    def create_huggingface_response(content: str):
        return [{"generated_text": [{"content": content}]}]
    
    @staticmethod
    def create_llama_response(content: str):
        return {"choices": [{"text": content}]}

# Helper functions for validation
def validate_topic_name(result: str):
    assert result == "Machine Learning"

def validate_cluster_names(result: List[str]):
    expected = ["Data Science", "Machine Learning", "Artificial Intelligence"]
    assert result == expected

# Anthropic Tests
@pytest.fixture
def anthropic_wrapper():
    with patch('anthropic.Anthropic'):
        wrapper = Anthropic(API_KEY="dummy")
        return wrapper

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
    with patch('cohere.Client'):
        wrapper = Cohere(API_KEY="dummy")
        return wrapper

def test_cohere_generate_topic_name_success(cohere_wrapper, mock_data):
    response = MockLLMResponse.create_cohere_response(mock_data["valid_topic_name"])
    cohere_wrapper.llm.chat = Mock(return_value=response)
    
    result = cohere_wrapper.generate_topic_name("test prompt")
    validate_topic_name(result)

def test_cohere_generate_cluster_names_success(cohere_wrapper, mock_data):
    response = MockLLMResponse.create_cohere_response(mock_data["valid_cluster_names"])
    cohere_wrapper.llm.chat = Mock(return_value=response)
    
    result = cohere_wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
    validate_cluster_names(result)

def test_cohere_generate_cluster_names_success_on_malformed_mapping(cohere_wrapper, mock_data):
    response = MockLLMResponse.create_cohere_response(mock_data["malformed_mapping"])
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