import numpy as np
from tqdm.auto import tqdm
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed


from typing import Optional, List, Protocol, Sequence
from toponymy._utils import handle_verbose_params


class TextEmbedderProtocol(Protocol):
    """Protocol defining the minimal interface required by a text embedder."""

    def encode(
        self,
        texts: Sequence[str],
        show_progress_bar: Optional[bool],
        *args,
        **kwargs,
    ) -> np.typing.NDArray[np.floating]: ...


# Cohere
try:
    import cohere

    class CohereEmbedder:
        def __init__(
            self,
            api_key,
            model: str = "embed-multilingual-v3.0",
            base_url: str = None,
            httpx_client: Optional[httpx.Client] = None,
        ):
            self.model = model
            self.base_url = base_url
            self.httpx_client = httpx_client
            self.input_type = "search_query"  # We will be embedding keyphrases and subtopic names to match against documents
            self.embedding_types = ["float"]
            self.co = cohere.ClientV2(
                api_key=api_key,
                base_url=base_url,
                httpx_client=httpx_client,
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
                response = self.co.embed(
                    texts=list(texts[i : i + 96]),
                    model=self.model,
                    input_type=self.input_type,
                    embedding_types=self.embedding_types,
                )
                result.append(np.asarray(response.embeddings.float_))

            return np.vstack(result)

except ImportError:
    pass

# OpenAI
try:
    import openai

    class OpenAIEmbedder:

        def __init__(
            self,
            api_key,
            model: str = "text-embedding-3-small",
            base_url: str = None,
            http_client: Optional[httpx.Client] = None,
        ):
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
                result.append(np.asarray([item.embedding for item in response.data]))

            return np.vstack(result)

except ImportError:
    pass

# Anthropic
try:
    import anthropic

    class AnthropicEmbedder:
        def __init__(
            self,
            api_key,
            model: str = "claude-3-haiku-20240307",
            base_url: str = None,
            http_client: Optional[httpx.Client] = None,
        ):
            self.model = model
            self.base_url = base_url
            self.http_client = http_client
            self.client = anthropic.Anthropic(
                api_key=api_key, 
                base_url=base_url, 
                http_client=http_client,
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
                batch = texts[i : i + 96]
                # Anthropic embeddings are done one at a time in the current API
                batch_embeddings = []
                for text in tqdm(
                    batch,
                    desc="embedding batch",
                    disable=(not show_progress_bar),
                    leave=False,
                ):
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
            self.client = azure.ai.inference.EmbeddingsClient(
                endpoint=endpoint, credential=self.credentials
            )
            self.model = model

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        def _encode_batch(self, texts: list) -> np.ndarray:
            # Call the Azure AI Inference API
            response = self.client.embed(
                model=self.model,
                input=[str(x) if len(x) > 0 else "[NO_TEXT]" for x in texts],
            )
            # Extract embeddings from the response
            embeddings = [item.embedding for item in response.data]
            if len(embeddings) != len(texts):
                print(
                    f"Warning: Expected {len(texts)} embeddings, but got {len(embeddings)}."
                )
                print(f"Texts: {texts}")
            assert len(embeddings) == len(texts)
            return np.array(embeddings)

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
                response = self.client.embeddings(
                    model=self.model, inputs=texts[i : i + 96]
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
                )
                response.raise_for_status()
                data = response.json()
                result.append(np.array([item["embedding"] for item in data["data"]]))

            return np.vstack(result)

except ImportError:
    pass

try:
    import vllm

    class VLLMEmbedder:
        def __init__(self, model: str = "all-MiniLM-L6-v2", kwargs: dict = {}):
            self.llm = vllm.LLM(model=model, task="embed", **kwargs)

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
