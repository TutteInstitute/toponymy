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
EMPTY_MAPPING_RESPONSE = {"new_topic_name_mapping": {}}

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
        "old_names": ["data", "ml", "ai"]
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

def test_anthropic_generate_topic_name_failure(anthropic_wrapper):
    anthropic_wrapper.llm.messages.create = Mock(side_effect=Exception("API Error"))
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

def test_importerror_handling():
    """Test that import errors are properly handled"""
    with patch.dict('sys.modules', {'anthropic': None}):
        from importlib import reload
        import sys
        
        # Force reload of the module to trigger ImportError handling
        if 'toponymy.llm_wrappers' in sys.modules:
            reload(sys.modules['toponymy.llm_wrappers'])
        
        # Verify that the Anthropic class is not available
        assert 'Anthropic' not in globals()