import os
import numpy as np
from tqdm.auto import tqdm
import httpx


from typing import Optional, List
from toponymy._utils import handle_verbose_params, resolve_api_key
from toponymy.types import TextEmbedderProtocol


def _embedding_matrix(vectors, count):
    result = np.asarray(vectors)
    if (
        result.ndim != 2
        or result.shape[0] != count
        or result.shape[1] == 0
        or result.dtype.kind not in "fiu"
        or not np.isfinite(result).all()
    ):
        raise ValueError("Embedding response must contain one finite vector per input")
    return result.astype(np.float64, copy=False)


def _ordered_embeddings(rows, count):
    missing = object()
    vectors = [missing] * count
    for row in rows:
        index = (
            row.get("index") if isinstance(row, dict) else getattr(row, "index", None)
        )
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < count
            or vectors[index] is not missing
        ):
            raise ValueError(
                "Embedding response has an invalid or duplicate input index"
            )
        vectors[index] = (
            row.get("embedding")
            if isinstance(row, dict)
            else getattr(row, "embedding", None)
        )
    if any(vector is missing for vector in vectors):
        raise ValueError("Embedding response is missing an input index")
    return _embedding_matrix(vectors, count)


# Cohere
try:
    import cohere

    class CohereEmbedder:
        def __init__(
            self,
            api_key: str = None,
            model: str = "embed-multilingual-v3.0",
            base_url: str = None,
            httpx_client: Optional[httpx.Client] = None,
        ):
            api_key = resolve_api_key(
                api_key, env_new="COHERE_API_KEY", env_legacy="CO_API_KEY"
            )
            options = {}
            if base_url is not None:
                options["base_url"] = base_url
            if httpx_client is not None:
                options["httpx_client"] = httpx_client
            self.co = cohere.ClientV2(api_key=api_key, **options)
            self.model = model
            self.base_url = base_url
            self.httpx_client = httpx_client
            self.input_type = "search_query"  # We will be embedding keyphrases and subtopic names to match against documents
            self.embedding_types = ["float"]

        def encode(
            self, texts: List[str], verbose: bool = None, show_progress_bar: bool = None
        ) -> np.ndarray:
            # Handle verbose parameters
            show_progress_bar_val, _ = handle_verbose_params(
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                default_verbose=False,
            )

            result = []
            for i in tqdm(
                range(0, len(texts), 96),
                desc="embedding texts",
                disable=(not show_progress_bar_val),
            ):
                response = self.co.embed(
                    texts=texts[i : i + 96],
                    model=self.model,
                    input_type=self.input_type,
                    embedding_types=self.embedding_types,
                )
                result.append(
                    _embedding_matrix(
                        response.embeddings.float_, len(texts[i : i + 96])
                    )
                )

            return np.vstack(result) if result else np.empty((0, 0), dtype=float)

except ImportError:
    pass

# OpenAI
try:
    import openai

    class OpenAIEmbedder:

        def __init__(
            self,
            api_key: str = None,
            model: str = "text-embedding-3-small",
            base_url: str = None,
            http_client: Optional[httpx.Client] = None,
        ):
            api_key = resolve_api_key(api_key, env_new="OPENAI_API_KEY")
            self.api_key = api_key
            self.model = model
            self.base_url = base_url
            self.http_client = http_client

            self.client = openai.OpenAI(
                api_key=api_key, base_url=base_url, http_client=http_client
            )

        def encode(
            self, texts: List[str], verbose: bool = None, show_progress_bar: bool = None
        ) -> np.ndarray:
            # Handle verbose parameters
            show_progress_bar_val, _ = handle_verbose_params(
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                default_verbose=False,
            )

            result = []
            for i in tqdm(
                range(0, len(texts), 96),
                desc="embedding texts",
                disable=(not show_progress_bar_val),
            ):
                response = self.client.embeddings.create(
                    input=texts[i : i + 96], model=self.model, encoding_format="float"
                )
                result.append(
                    _ordered_embeddings(response.data, len(texts[i : i + 96]))
                )

            return np.vstack(result) if result else np.empty((0, 0), dtype=float)

except ImportError:
    pass


class AnthropicEmbedder:
    """Retired embedding adapter; Anthropic has no native embeddings endpoint."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-haiku-4-5-20251001",
        base_url: str = None,
        httpx_client: Optional[httpx.Client] = None,
    ):
        raise NotImplementedError(
            "Anthropic has no native embeddings API. Supply a text embedder such as "
            "VoyageAIEmbedder or OpenAIEmbedder; Anthropic naming remains supported."
        )


# Microsoft Azure
try:
    import azure.ai.inference
    from azure.core.credentials import AzureKeyCredential

    class AzureAIEmbedder:
        def __init__(
            self, api_key: str = None, endpoint: str = None, model: str = None
        ):
            api_key = resolve_api_key(api_key, env_new="AZURE_API_KEY")
            if endpoint is None:
                endpoint = os.getenv("AZURE_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "No Azure endpoint provided. Set AZURE_ENDPOINT environment variable "
                    "or pass endpoint parameter to AzureAIEmbedder."
                )
            self.credentials = AzureKeyCredential(api_key)
            self.client = azure.ai.inference.EmbeddingsClient(
                endpoint=endpoint, credential=self.credentials
            )
            self.model = model
            if self.model is None:
                raise ValueError(
                    "No Azure AI model specified. Pass the model parameter to AzureAIEmbedder "
                    "(e.g., model='text-embedding-3-small')."
                )

        def _encode_batch(self, texts: list) -> np.ndarray:
            # Call the Azure AI Inference API
            response = self.client.embed(
                model=self.model,
                input=[str(x) if len(x) > 0 else "[NO_TEXT]" for x in texts],
            )
            return _ordered_embeddings(response.data, len(texts))

        def encode(
            self, texts: list, verbose: bool = None, show_progress_bar: bool = None
        ) -> np.ndarray:
            # Handle verbose parameters
            show_progress_bar_val, _ = handle_verbose_params(
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                default_verbose=False,
            )

            result = []

            for i in tqdm(
                range(0, len(texts), 96),
                desc="embedding texts",
                disable=(not show_progress_bar_val),
            ):
                embeddings = self._encode_batch(texts[i : i + 96])
                result.append(embeddings)

            return np.vstack(result) if result else np.empty((0, 0), dtype=float)

except ImportError as e:
    pass

# Mistral
try:
    import mistralai.client

    class MistralEmbedder:
        def __init__(self, api_key: str = None, model: str = "mistral-embed"):
            api_key = resolve_api_key(api_key, env_new="MISTRAL_API_KEY")
            self.client = mistralai.client.Mistral(api_key=api_key)
            self.model = model

        def encode(
            self, texts: List[str], verbose: bool = None, show_progress_bar: bool = None
        ) -> np.ndarray:
            # Handle verbose parameters
            show_progress_bar_val, _ = handle_verbose_params(
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                default_verbose=False,
            )

            result = []
            for i in tqdm(
                range(0, len(texts), 96),
                desc="embedding texts",
                disable=(not show_progress_bar_val),
            ):
                response = self.client.embeddings.create(
                    model=self.model, inputs=texts[i : i + 96]
                )
                result.append(
                    _ordered_embeddings(response.data, len(texts[i : i + 96]))
                )

            return np.vstack(result) if result else np.empty((0, 0), dtype=float)

except ImportError:
    pass

# Voyage AI
try:
    import requests

    class VoyageAIEmbedder:
        def __init__(self, api_key: str = None, model: str = "voyage-2"):
            api_key = resolve_api_key(api_key, env_new="VOYAGEAI_API_KEY")
            self.api_key = api_key
            self.model = model
            self.base_url = "https://api.voyageai.com/v1/embeddings"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

        def encode(
            self, texts: List[str], verbose: bool = None, show_progress_bar: bool = None
        ) -> np.ndarray:
            # Handle verbose parameters
            show_progress_bar_val, _ = handle_verbose_params(
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                default_verbose=False,
            )

            result = []
            for i in tqdm(
                range(0, len(texts), 96),
                desc="embedding texts",
                disable=(not show_progress_bar_val),
            ):
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "input": texts[i : i + 96],
                        "encoding_format": "float",
                    },
                    timeout=(10, 30),
                )
                response.raise_for_status()
                data = response.json()
                result.append(_ordered_embeddings(data["data"], len(texts[i : i + 96])))

            return np.vstack(result) if result else np.empty((0, 0), dtype=float)

except ImportError:
    pass

try:
    import vllm

    class VLLMEmbedder:
        def __init__(self, model: str = "all-MiniLM-L6-v2", kwargs: dict = None):
            self.llm = vllm.LLM(model=model, task="embed", **(kwargs or {}))

        def encode(
            self, texts: List[str], verbose: bool = None, show_progress_bar: bool = None
        ) -> np.ndarray:
            # Handle verbose parameters
            show_progress_bar_val, _ = handle_verbose_params(
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                default_verbose=False,
            )

            outputs = self.llm.embed(texts, use_tqdm=show_progress_bar_val)
            embeddings = np.vstack([o.outputs.embedding for o in outputs])
            return embeddings

except ImportError:
    pass
