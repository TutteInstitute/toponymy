import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# Import your module (assuming it's called embedders.py)
# If your module has a different name, adjust this import
sys.path.append('.')
import toponymy.embedding_wrappers as embedders

class TestCohereEmbedder:
    @patch('cohere.ClientV2')
    def test_encode(self, mock_client_v2):
        # Set up mock response
        mock_client = MagicMock()
        mock_client_v2.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.embeddings.float_ = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_response
        
        # Create the embedder and call encode
        embedder = embedders.CohereEmbedder(api_key="fake_key")
        texts = ["sample text 1", "sample text 2"]
        result = embedder.encode(texts)
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        # Verify API was called correctly
        mock_client.embed.assert_called_once_with(
            texts=texts, 
            model="embed-multilingual-v3.0", 
            input_type="search_query",
            embedding_types=['float']
        )

class TestOpenAIEmbedder:
    @patch('openai.OpenAI')
    def test_encode(self, mock_openai):
        # Set up mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_data_item1 = MagicMock()
        mock_data_item1.embedding = [0.1, 0.2]
        mock_data_item2 = MagicMock()
        mock_data_item2.embedding = [0.3, 0.4]
        
        mock_response = MagicMock()
        mock_response.data = [mock_data_item1, mock_data_item2]
        mock_client.embeddings.create.return_value = mock_response
        
        # Create the embedder and call encode
        embedder = embedders.OpenAIEmbedder(api_key="fake_key")
        texts = ["sample text 1", "sample text 2"]
        result = embedder.encode(texts)
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        # Verify API was called correctly
        mock_client.embeddings.create.assert_called_once()

class TestAnthropicEmbedder:
    @patch('anthropic.Anthropic')
    def test_encode(self, mock_anthropic):
        # Set up mock response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response1 = MagicMock()
        mock_response1.embedding = [0.1, 0.2]
        mock_response2 = MagicMock()
        mock_response2.embedding = [0.3, 0.4]
        
        mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]
        
        # Create the embedder and call encode
        embedder = embedders.AnthropicEmbedder(api_key="fake_key")
        texts = ["sample text 1", "sample text 2"]
        result = embedder.encode(texts)
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        # Verify API was called correctly
        assert mock_client.embeddings.create.call_count == 2

class TestAzureAIEmbedder:
    @patch('azure.ai.inference.EmbeddingsClient')
    def test_encode(self, mock_ai_client):
        # Set up mock response
        mock_client = MagicMock()
        mock_ai_client.return_value = mock_client
        
        # Create mock embedding results
        mock_embedding_result1 = MagicMock()
        mock_embedding_result1.embedding = [0.1, 0.2]
        mock_embedding_result2 = MagicMock()
        mock_embedding_result2.embedding = [0.3, 0.4]
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_result1, mock_embedding_result2]
        
        # Configure the mock client to return our mock response
        mock_client.embed.return_value = mock_response
        
        # Create the embedder and call encode
        embedder = embedders.AzureAIEmbedder(
            api_key="fake_key", 
            endpoint="https://fake-endpoint.azure.com/", 
            model="text-embedding"
        )
        texts = ["sample text 1", "sample text 2"]
        result = embedder.encode(texts)
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        # Verify API was called correctly
        mock_client.embed.assert_called_once()
        # Check that the deployment name was passed correctly
        args, kwargs = mock_client.embed.call_args
        assert kwargs["model"] == "text-embedding"
        # Check that BatchInput was created with correct number of items
        assert len(kwargs["input"]) == 2

class TestMistralEmbedder:
    @patch('mistralai.client.MistralClient')
    def test_encode(self, mock_mistral_client):
        # Set up mock response
        mock_client = MagicMock()
        mock_mistral_client.return_value = mock_client
        
        mock_data_item1 = MagicMock()
        mock_data_item1.embedding = [0.1, 0.2]
        mock_data_item2 = MagicMock()
        mock_data_item2.embedding = [0.3, 0.4]
        
        mock_response = MagicMock()
        mock_response.data = [mock_data_item1, mock_data_item2]
        mock_client.embeddings.return_value = mock_response
        
        # Create the embedder and call encode
        embedder = embedders.MistralEmbedder(api_key="fake_key")
        texts = ["sample text 1", "sample text 2"]
        result = embedder.encode(texts)
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        # Verify API was called correctly
        mock_client.embeddings.assert_called_once_with(
            model="mistral-embed",
            inputs=texts
        )

class TestVoyageAIEmbedder:
    @patch('requests.post')
    def test_encode(self, mock_post):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]}
            ]
        }
        mock_post.return_value = mock_response
        
        # Create the embedder and call encode
        embedder = embedders.VoyageAIEmbedder(api_key="fake_key")
        texts = ["sample text 1", "sample text 2"]
        result = embedder.encode(texts)
        
        # Verify the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "voyage-2"
        assert kwargs["json"]["input"] == texts