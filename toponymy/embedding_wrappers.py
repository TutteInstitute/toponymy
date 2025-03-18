import numpy as np
from tqdm.auto import tqdm
import httpx

from typing import Optional, List

# Cohere
try:
    import cohere

    class CohereEmbedder:
        def __init__(self, api_key, model: str = "embed-multilingual-v3.0", base_url: str = None, httpx_client: Optional[httpx.Client] = None):
            self.co = cohere.ClientV2(api_key=api_key)
            self.model = model
            self.base_url = base_url
            self.httpx_client = httpx_client
            self.input_type="search_query" # We will be embedding keyphrases and subtopic names to match against documents
            self.embedding_types=['float']

        def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
            result = []
            for i in tqdm(range(0, len(texts), 96), desc="embedding texts", disable=(not show_progress_bar)):
                response = self.co.embed(texts=texts[i:i+96], model=self.model, input_type=self.input_type, embedding_types=self.embedding_types)
                result.append(np.asarray(response.embeddings.float_))

            return np.vstack(result)
    
except ImportError:
    pass

# OpenAI
try:
    import openai

    class OpenAIEmbedder:

        def __init__(self, api_key, model: str = "text-embedding-3-small", base_url: str = None):
            self.api_key = api_key
            self.model = model
            self.base_url = base_url
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
            result = []
            for i in tqdm(range(0, len(texts), 96), desc="embedding texts", disable=(not show_progress_bar)):
                response = self.client.embeddings.create(texts=texts[i:i+96], model=self.model, encoding_format="float")
                result.append(np.asarray([item.embedding for item in response.data]))       

            return np.vstack(result)
        
except ImportError:
    pass

# Anthropic
try:
    import anthropic

    class AnthropicEmbedder:
        def __init__(self, api_key, model: str = "claude-3-haiku-20240307", base_url: str = None, httpx_client: Optional[httpx.Client] = None):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.base_url = base_url
            self.httpx_client = httpx_client

        def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
            result = []
            for i in tqdm(range(0, len(texts), 96), desc="embedding texts", disable=(not show_progress_bar)):
                batch = texts[i:i+96]
                # Anthropic embeddings are done one at a time in the current API
                batch_embeddings = []
                for text in tqdm(batch, desc="embedding batch", disable=(not show_progress_bar), leave=False):
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=text,
                    )
                    batch_embeddings.append(response.embedding)
                result.append(np.array(batch_embeddings))

            return np.vstack(result)
except ImportError:
    pass

# Microsoft Azure
try:
    import azure.ai.inference
    from azure.core.credentials import AzureKeyCredential

    class AzureAIEmbedder:
        def __init__(self, api_key: str, endpoint: str, model: str):
            self.credentials = AzureKeyCredential(api_key)
            self.client = azure.ai.inference.EmbeddingsClient(endpoint=endpoint, credential=self.credentials)
            self.model = model

        def encode(self, texts: list, show_progress_bar: bool = False) -> np.ndarray:
            result = []
            
            for i in tqdm(range(0, len(texts), 96), desc="embedding texts", disable=(not show_progress_bar)):
                batch = texts[i:i+96]
                
                # Call the Azure AI Inference API
                response = self.client.embed(
                    model=self.model,
                    input=batch
                )
                
                # Extract embeddings from the response
                embeddings = []
                for embedding_result in response.data:
                    embeddings.append(embedding_result.embedding)
                
                result.append(np.array(embeddings))
            
            return np.vstack(result)

except ImportError as e:
    pass

# Mistral
try:
    import mistralai.client

    class MistralEmbedder:
        def __init__(self, api_key: str, model: str = "mistral-embed"):
            self.client = mistralai.client.MistralClient(api_key=api_key)
            self.model = model

        def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
            result = []
            for i in tqdm(range(0, len(texts), 96), desc="embedding texts", disable=(not show_progress_bar)):
                response = self.client.embeddings(
                    model=self.model,
                    inputs=texts[i:i+96]
                )
                result.append(np.array([item.embedding for item in response.data]))

            return np.vstack(result)
except ImportError:
    pass

# Voyage AI
try:
    import requests

    class VoyageAIEmbedder:
        def __init__(self, api_key: str, model: str = "voyage-2"):
            self.api_key = api_key
            self.model = model
            self.base_url = "https://api.voyageai.com/v1/embeddings"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

        def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
            result = []
            for i in tqdm(range(0, len(texts), 96), desc="embedding texts", disable=(not show_progress_bar)):
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "input": texts[i:i+96],
                        "encoding_format": "float"
                    }
                )
                response.raise_for_status()
                data = response.json()
                result.append(np.array([item["embedding"] for item in data["data"]]))

            return np.vstack(result)
except ImportError:
    pass