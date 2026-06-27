import warnings

import json
import pytest
from unittest.mock import Mock, patch
import os


from toponymy.llm_wrappers import (
    LiteLLMNamer,
    FailFastLLMError,
    repair_json_string_backslashes,
)
from toponymy.llm_wrappers import (
    AnthropicNamer,
    OpenAINamer,
    CohereNamer,
    HuggingFaceNamer,
    AzureAINamer,
    LlamaCppNamer,
    OllamaNamer,
    GoogleGeminiNamer,
    TogetherNamer,
    ReplicateNamer,
    GoogleGeminiNamer,
)
from toponymy.new_templates import Prompt, TextTemplate
from toponymy.templates import default_extract_topic_names
from conftest import is_ollama_model_available

from toponymy.tests.helpers.llm_test_config import (
    validate_cluster_names,
    validate_topic_name,
    LITELLM_PROVIDER_CASES,
    SUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS,
    UNSUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS,
)
from toponymy.tests.helpers.errors import (
    LITELLM_FAIL_FAST,
    LITELLM_RETRYABLE,
    make_litellm_error,
)
import logging

logger = logging.getLogger(__name__)


def extract_cluster_names(old_names):
    return lambda response: default_extract_topic_names(
        json.loads(repair_json_string_backslashes(response)), old_names, response
    )


class MockLLMResponse:
    """Mock response object that mimics different LLM service response structures"""

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
    def create_huggingface_response(content: str):
        return [{"generated_text": content}]

    @staticmethod
    def create_llama_response(content: str):
        return {"choices": [{"message": {"content": content}}]}


@pytest.mark.parametrize("namer_cls, kwargs", SUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS)
def test_supported_namers_do_not_warn_on_callback(namer_cls, kwargs):
    callback = lambda payload: None

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        namer = namer_cls(callback=callback, **kwargs)

    debug_warnings = [w for w in record if "debug callback" in str(w.message)]

    assert len(debug_warnings) == 0
    assert namer._supports_debug_callback is True


@pytest.mark.parametrize("namer_cls, kwargs", UNSUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS)
def test_unsupported_namers_warn_on_callback(namer_cls, kwargs):
    callback = lambda payload: None

    with pytest.warns(UserWarning, match="debug callback") as record:
        namer = namer_cls(callback=callback, **kwargs)
    assert namer._supports_debug_callback is False


@pytest.mark.parametrize(
    "namer_cls, kwargs",
    SUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS + UNSUPPORTED_SYNC_DEBUG_CALLBACK_NAMERS,
)
def test_no_warning_when_no_callback(namer_cls, kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        namer_cls(callback=None, **kwargs)
    debug_warnings = [w for w in record if "debug callback" in str(w.message)]

    assert len(debug_warnings) == 0


# LlamaCpp Tests
@pytest.fixture
def llamacpp_wrapper():
    pytest.importorskip("llama_cpp")
    with patch("llama_cpp.Llama"):
        wrapper = LlamaCppNamer(
            model_path="dummy", n_ctx=4096, n_batch=512, n_threads=4
        )
        return wrapper


def test_llamacpp_generate_topic_name_success(llamacpp_wrapper, mock_data):
    response = MockLLMResponse.create_llama_response(mock_data["valid_topic_name"])
    llamacpp_wrapper.llm = Mock()
    llamacpp_wrapper.llm.create_chat_completion.return_value = response

    result = llamacpp_wrapper.generate_topic_name(
        Prompt("", "test prompt"),
        TextTemplate.extract_name,
    )
    validate_topic_name(result)


def test_llamacpp_generate_cluster_names_success(llamacpp_wrapper, mock_data):
    response = MockLLMResponse.create_llama_response(mock_data["valid_cluster_names"])
    llamacpp_wrapper.llm = Mock()
    llamacpp_wrapper.llm.create_chat_completion.return_value = response

    result = llamacpp_wrapper.generate_topic_cluster_names(
        Prompt("", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    validate_cluster_names(result)


def test_llamacpp_generate_cluster_names_success_on_malformed_mapping(
    llamacpp_wrapper, mock_data
):
    response = MockLLMResponse.create_llama_response(mock_data["malformed_mapping"])
    llamacpp_wrapper.llm = Mock()
    llamacpp_wrapper.llm.create_chat_completion.return_value = response

    result = llamacpp_wrapper.generate_topic_cluster_names(
        Prompt("", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    validate_cluster_names(result)


@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_llamacpp_generate_topic_name_failure(llamacpp_wrapper):
    llamacpp_wrapper.llm = Mock()
    llamacpp_wrapper.llm.create_chat_completion.side_effect = Exception("API Error")
    result = llamacpp_wrapper.generate_topic_name(
        Prompt("", "test prompt"),
        TextTemplate.extract_name,
    )
    assert result == ""


@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_llamacpp_generate_topic_name_failure_malformed_json(
    llamacpp_wrapper, mock_data
):
    response = MockLLMResponse.create_llama_response(mock_data["malformed_json"])
    llamacpp_wrapper.llm = Mock()
    llamacpp_wrapper.llm.create_chat_completion.return_value = response
    result = llamacpp_wrapper.generate_topic_name(
        Prompt("", "test prompt"),
        TextTemplate.extract_name,
    )
    assert result == ""


@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_llamacpp_generate_cluster_names_failure(llamacpp_wrapper, mock_data):
    llamacpp_wrapper.llm = Mock()
    llamacpp_wrapper.llm.create_chat_completion.side_effect = Exception("API Error")
    result = llamacpp_wrapper.generate_topic_cluster_names(
        Prompt("", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    assert result == mock_data["old_names"]


# Huggingface Tests
@pytest.fixture
def huggingface_wrapper():
    with patch("transformers.pipeline"):
        wrapper = HuggingFaceNamer(model="dummy")
        return wrapper


def test_huggingface_generate_topic_name_success(huggingface_wrapper, mock_data):
    response = MockLLMResponse.create_huggingface_response(
        mock_data["valid_topic_name"]
    )
    huggingface_wrapper.llm = Mock(return_value=response)

    result = huggingface_wrapper.generate_topic_name(
        Prompt("", "test prompt"),
        TextTemplate.extract_name,
    )
    validate_topic_name(result)


def test_huggingface_generate_topic_name_success_system_prompt(
    huggingface_wrapper, mock_data
):
    response = MockLLMResponse.create_huggingface_response(
        mock_data["valid_topic_name"]
    )
    huggingface_wrapper.llm = Mock(return_value=response)

    result = huggingface_wrapper.generate_topic_name(
        Prompt("system prompt", "test prompt"),
        TextTemplate.extract_name,
    )
    validate_topic_name(result)


def test_huggingface_generate_cluster_names_success(huggingface_wrapper, mock_data):
    response = MockLLMResponse.create_huggingface_response(
        mock_data["valid_cluster_names"]
    )
    huggingface_wrapper.llm = Mock(return_value=response)

    result = huggingface_wrapper.generate_topic_cluster_names(
        Prompt("", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    validate_cluster_names(result)


def test_huggingface_generate_cluster_names_success_system_prompt(
    huggingface_wrapper, mock_data
):
    response = MockLLMResponse.create_huggingface_response(
        mock_data["valid_cluster_names"]
    )
    huggingface_wrapper.llm = Mock(return_value=response)

    result = huggingface_wrapper.generate_topic_cluster_names(
        Prompt("system prompt", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    validate_cluster_names(result)


def test_huggingface_generate_cluster_names_success_on_malformed_mapping(
    huggingface_wrapper, mock_data
):
    response = MockLLMResponse.create_huggingface_response(
        mock_data["malformed_mapping"]
    )
    huggingface_wrapper.llm = Mock(return_value=response)

    result = huggingface_wrapper.generate_topic_cluster_names(
        Prompt("", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    validate_cluster_names(result)


@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_huggingface_generate_topic_name_failure(huggingface_wrapper):
    huggingface_wrapper.llm = Mock(side_effect=Exception("API Error"))
    result = huggingface_wrapper.generate_topic_name(
        Prompt("", "test prompt"),
        TextTemplate.extract_name,
    )
    assert result == ""


@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_huggingface_generate_topic_name_failure_malformed_json(
    huggingface_wrapper, mock_data
):
    response = MockLLMResponse.create_huggingface_response(mock_data["malformed_json"])
    huggingface_wrapper.llm = Mock(return_value=response)
    result = huggingface_wrapper.generate_topic_name(
        Prompt("", "test prompt"),
        TextTemplate.extract_name,
    )
    assert result == ""


@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_huggingface_generate_cluster_names_failure(huggingface_wrapper, mock_data):
    huggingface_wrapper.llm = Mock(side_effect=Exception("API Error"))
    result = huggingface_wrapper.generate_topic_cluster_names(
        Prompt("", "test prompt"),
        mock_data["old_names"],
        extract_cluster_names(mock_data["old_names"]),
    )
    assert result == mock_data["old_names"]


# Anthropic Tests
@pytest.mark.external
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_connectivity_sync_system_canary():
    namer = AnthropicNamer()

    result = namer.connectivity_status(
        prompt=Prompt(
            "You are a topic naming assistant.",
            "Return a short JSON object describing your role.",
        ),
    )

    assert result["success"], (
        f"Sync system canary failed:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


def test_anthropic_namer_returns_litellm_namer():
    namer = AnthropicNamer()

    assert isinstance(namer, LiteLLMNamer)


def test_anthropic_namer_default():
    namer = AnthropicNamer()

    assert namer.model == "anthropic/claude-haiku-4-5-20251001"
    assert namer.use_json_object is True
    assert namer.supports_system_prompts is True


def test_anthropic_namer_provider_kwargs_passthrough():
    namer = AnthropicNamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


# OpenAI Tests
@pytest.mark.external
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_connectivity_sync_system_canary():
    """
    Canary test to verify live sync connectivity to the OpenAI API
    using the system prompt path.
    """
    namer = OpenAINamer()

    result = namer.connectivity_status(
        prompt=Prompt(
            "You are a topic naming assistant.",
            "Return a short JSON object describing your role.",
        ),
    )

    assert result["success"], (
        f"Sync system canary failed for OpenAI:\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


def test_openai_namer_returns_litellm_namer():
    namer = OpenAINamer()

    assert isinstance(namer, LiteLLMNamer)


def test_openai_namer_default():
    namer = OpenAINamer()

    assert namer.model == "openai/gpt-4o-mini"
    assert namer.use_json_object is True
    assert namer.supports_system_prompts is True


def test_openai_namer_provider_kwargs_passthrough():
    namer = OpenAINamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


def test_openai_namer_base_url_maps_to_api_base():
    """Remove once deprecation of base_url complete"""
    with pytest.warns(FutureWarning):
        namer = OpenAINamer(base_url="http://localhost")

    assert namer.api_base == "http://localhost"


def test_openai_namer_http_client_maps_to_provider_kwargs():
    """Remove once deprecation of http_client complete"""
    with pytest.warns(FutureWarning):
        namer = OpenAINamer(http_client="httpx.Client(timeout=123)")

    assert namer.provider_kwargs["http_client"] == "httpx.Client(timeout=123)"


# Cohere Tests
@pytest.mark.external
@pytest.mark.skipif(not os.getenv("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
def test_cohere_connectivity_sync_system_canary():
    """
    Canary test to verify live sync connectivity to the Cohere API
    using the system prompt path.
    """
    namer = CohereNamer()
    result = namer.connectivity_status(
        prompt=Prompt(
            "You are a topic naming assistant.",
            "Return a short JSON object describing your role.",
        ),
    )

    assert result["success"], (
        f"Sync system canary failed for Cohere:\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


def test_cohere_namer_returns_litellm_namer():
    namer = CohereNamer()

    assert isinstance(namer, LiteLLMNamer)


def test_cohere_namer_default():
    namer = CohereNamer()

    assert namer.model == "cohere/command-r-08-2024"
    assert namer.use_json_object is False  # until prompting is stricter
    assert namer.supports_system_prompts is True


def test_cohere_namer_provider_kwargs_passthrough():
    namer = CohereNamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


def test_cohere_namer_base_url_maps_to_api_base():
    """Remove once deprecation of base_url complete"""
    with pytest.warns(FutureWarning):
        namer = CohereNamer(base_url="http://localhost")

    assert namer.api_base == "http://localhost"


def test_cohere_namer_env_co_api_base_maps_to_api_base(monkeypatch):
    monkeypatch.delenv("COHERE_API_BASE", raising=False)
    monkeypatch.setenv("CO_API_URL", "dummy")
    with pytest.warns(FutureWarning):
        namer = CohereNamer()

    assert namer.api_base == "dummy"


def test_cohere_namer_httpx_client_maps_to_provider_kwargs():
    """Remove once deprecation of http_client complete"""
    with pytest.warns(FutureWarning):
        namer = CohereNamer(httpx_client="httpx.Client(timeout=123)")

    assert namer.provider_kwargs["httpx_client"] == "httpx.Client(timeout=123)"


def test_cohere_namer_env_co_api_url_maps_to_api_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "dummy")
    with pytest.warns(FutureWarning):
        namer = CohereNamer()

    assert namer.api_key == "dummy"


# AzureAI Tests
@pytest.mark.external
@pytest.mark.skipif(
    not os.getenv("AZURE_AI_API_KEY"), reason="AZURE_AI_API_KEY not set"
)
@pytest.mark.skipif(
    not os.getenv("AZURE_AI_API_BASE"), reason="AZURE_AI_API_BASE not set"
)
def test_azureai_connectivity_sync_plain_canary():
    """
    Canary test to verify live connectivity to Azure AI API. Tests the plain prompt path.
    """
    namer = AzureAINamer(model="gpt-4o")
    result = namer.connectivity_status()
    assert result["success"], (
        f"Sync plain canary test failed for Azure AI:\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


def test_azureai_namer_default():
    namer = AzureAINamer(model="dummy")

    assert namer.model == "azure_ai/dummy"


def test_azureai_namer_provider_kwargs_passthrough():
    namer = AzureAINamer(model="dummy", provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


def test_azureai_namer_endpoint_maps_to_api_base():
    namer = AzureAINamer(model="dummy", endpoint="http://localhost")

    assert namer.api_base == "http://localhost"


def test_azureai_namer_old_env_var_maps_to_api_key(monkeypatch):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "dummy")
    with pytest.warns(FutureWarning):
        namer = AzureAINamer(model="dummy")

    assert namer.api_key == "dummy"


# Ollama Tests
def test_ollama_connectivity_plain_sync_canary():
    model = "llama3.2"
    if not is_ollama_model_available(model):
        pytest.skip(f"{model} not available in local Ollama")
    namer = OllamaNamer(model=model)
    result = namer.connectivity_status()

    assert result["success"], (
        f"Sync plain canary test failed for Ollama:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


def test_ollama_namer_returns_litellm_namer():
    namer = OllamaNamer()

    assert isinstance(namer, LiteLLMNamer)


def test_ollama_namer_default():
    namer = OllamaNamer()
    assert namer.model == "ollama_chat/llama3.2"
    assert namer.api_base == "http://localhost:11434"


def test_ollama_namer_provider_kwargs_passthrough():
    namer = OllamaNamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


def test_ollama_namer_host_maps_to_api_base():
    """Remove once deprecation of host is complete"""
    with pytest.warns(FutureWarning):
        namer = OllamaNamer(host="http://localhost")

    assert namer.api_base == "http://localhost"


# Google Gemini Tests
@pytest.mark.external
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
@pytest.mark.filterwarnings("ignore:GoogleGeminiNamer is deprecated")
def test_gemini_connectivity_sync_system_canary():
    namer = GoogleGeminiNamer()

    result = namer.connectivity_status(
        prompt=Prompt(
            "You are a topic naming assistant.",
            "Return a short JSON object describing your role.",
        ),
    )

    assert result["success"], (
        f"Sync system canary failed:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


def test_gemini_namer_returns_litellm_namer():
    with pytest.warns(FutureWarning):
        namer = GoogleGeminiNamer()

    assert isinstance(namer, LiteLLMNamer)


@pytest.mark.filterwarnings("ignore:GoogleGeminiNamer is deprecated")
def test_gemini_namer_default():
    namer = GoogleGeminiNamer()

    assert namer.model == "gemini/gemini-2.5-flash-lite"
    assert namer.use_json_object is True
    assert namer.supports_system_prompts is True


@pytest.mark.filterwarnings("ignore:GoogleGeminiNamer is deprecated")
def test_gemini_namer_provider_kwargs_passthrough():
    namer = GoogleGeminiNamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


def test_gemini_name_old_env_var_maps_to_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    with pytest.warns(FutureWarning):
        namer = GoogleGeminiNamer()

    assert namer.api_key == "dummy"


# Together Tests
@pytest.mark.external
@pytest.mark.skipif(
    not os.getenv("TOGETHERAI_API_KEY"), reason="TOGETHERAI_API_KEY not set"
)
@pytest.mark.filterwarnings("ignore:TogetherNamer is deprecated")
def test_together_connectivity_plain_sync_canary():
    namer = TogetherNamer()
    result = namer.connectivity_status()

    assert result["success"], (
        f"Sync plain canary test failed for Together:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


def test_together_namer_returns_litellm_namer():
    with pytest.warns(FutureWarning):
        namer = TogetherNamer()

    assert isinstance(namer, LiteLLMNamer)


@pytest.mark.filterwarnings("ignore:TogetherNamer is deprecated")
def test_together_namer_default():
    namer = TogetherNamer()

    assert namer.model == "together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite"


@pytest.mark.filterwarnings("ignore:TogetherNamer is deprecated")
def test_together_namer_provider_kwargs_passthrough():
    namer = TogetherNamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


# Replicate Tests
@pytest.mark.external
@pytest.mark.skipif(
    not os.getenv("REPLICATE_API_KEY"), reason="REPLICATE_API_KEY not set"
)
@pytest.mark.filterwarnings("ignore:ReplicateNamer is deprecated")
def test_replicate_connectivity_plain_sync_canary():
    namer = ReplicateNamer()
    result = namer.connectivity_status()

    assert result["success"], (
        f"Sync plain canary test failed for Replicate:\n"
        f"{result['error_type']}: {result['error_message']}"
    )


def test_replicate_namer_returns_litellm_namer():
    with pytest.warns(FutureWarning):
        namer = ReplicateNamer()

    assert isinstance(namer, LiteLLMNamer)


@pytest.mark.filterwarnings("ignore:ReplicateNamer is deprecated")
def test_replicate_namer_default():
    namer = ReplicateNamer()

    assert namer.model == "replicate/meta/llama-2-70b-chat"
    assert namer.use_json_object is False


@pytest.mark.filterwarnings("ignore:ReplicateNamer is deprecated")
def test_replicate_namer_provider_kwargs_passthrough():
    namer = ReplicateNamer(provider_kwargs={"timeout": 123})

    assert namer.provider_kwargs["timeout"] == 123


@pytest.mark.filterwarnings("ignore:ReplicateNamer is deprecated")
def test_replicate_namer_api_token_maps_to_api_key(monkeypatch):
    monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
    with pytest.warns(FutureWarning):
        namer = ReplicateNamer(api_token="dummy")

    assert namer.api_key == "dummy"


@pytest.mark.filterwarnings("ignore:ReplicateNamer is deprecated")
def test_replicate_namer_env_api_token_maps_to_api_key(monkeypatch):
    monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
    monkeypatch.setenv("REPLICATE_API_TOKEN", "dummy")
    with pytest.warns(FutureWarning):
        namer = ReplicateNamer()

    assert namer.api_key == "dummy"


# LiteLLM Tests
@pytest.fixture
def litellm_wrapper():
    return LiteLLMNamer(
        api_key="dummy",
        model="openai/gpt-4o-mini",
    )


@pytest.mark.external
@pytest.mark.parametrize("provider_cfg", LITELLM_PROVIDER_CASES)
def test_litellm_connectivity_canary_sync_plain(provider_cfg):
    if not os.getenv(provider_cfg["api_key_env"]):
        pytest.skip(f"{provider_cfg['api_key_env']} not set")

    namer = LiteLLMNamer(
        model=provider_cfg["model"],
    )

    result = namer.connectivity_status()

    assert result["success"], (
        f"Sync plain canary test failed for LiteLLM ({provider_cfg['provider_name']}):\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


@pytest.mark.external
@pytest.mark.parametrize("provider_cfg", LITELLM_PROVIDER_CASES)
def test_litellm_connectivity_canary_sync_system(provider_cfg):
    if not os.getenv(provider_cfg["api_key_env"]):
        print(
            f"Skipping test for {provider_cfg['provider_name']} because {provider_cfg['api_key_env']} is not set."
        )
        pytest.skip(f"{provider_cfg['api_key_env']} not set")

    namer = LiteLLMNamer(
        model=provider_cfg["model"],
    )
    result = namer.connectivity_status(
        prompt=Prompt(
            "You are a topic naming assistant.",
            "Return a short JSON object describing your role.",
        ),
    )

    assert result["success"], (
        f"Sync system canary failed for LiteLLM ({provider_cfg['provider_name']}):\n"
        f"  Error: {result['error_type']}: {result['error_message']}"
    )


def test_litellm_generate_topic_name_success(litellm_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=response):
        result = litellm_wrapper.generate_topic_name(
            Prompt("", "test prompt"),
            TextTemplate.extract_name,
        )

    validate_topic_name(result)


def test_litellm_generate_topic_name_success_system_prompt(litellm_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=response):
        result = litellm_wrapper.generate_topic_name(
            Prompt("system prompt", "test prompt"),
            TextTemplate.extract_name,
        )

    validate_topic_name(result)


def test_litellm_generate_cluster_names_success(litellm_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["valid_cluster_names"])

    with patch("litellm.completion", return_value=response):
        result = litellm_wrapper.generate_topic_cluster_names(
            Prompt("", "test prompt"),
            mock_data["old_names"],
            extract_cluster_names(mock_data["old_names"]),
        )

    validate_cluster_names(result)


def test_litellm_generate_cluster_names_success_system_prompt(
    litellm_wrapper, mock_data
):
    response = MockLLMResponse.create_chat_response(mock_data["valid_cluster_names"])

    with patch("litellm.completion", return_value=response):
        result = litellm_wrapper.generate_topic_cluster_names(
            Prompt("system prompt", "test prompt"),
            mock_data["old_names"],
            extract_cluster_names(mock_data["old_names"]),
        )

    validate_cluster_names(result)


def test_litellm_generate_cluster_names_success_on_malformed_mapping(
    litellm_wrapper,
    mock_data,
):
    response = MockLLMResponse.create_chat_response(mock_data["malformed_mapping"])

    with patch("litellm.completion", return_value=response):
        result = litellm_wrapper.generate_topic_cluster_names(
            Prompt("", "test prompt"),
            mock_data["old_names"],
            extract_cluster_names(mock_data["old_names"]),
        )

    validate_cluster_names(result)


def test_litellm_generate_topic_name_failure_malformed_json(litellm_wrapper, mock_data):
    response = MockLLMResponse.create_chat_response(mock_data["malformed_json"])

    with patch("litellm.completion", return_value=response):
        result = litellm_wrapper.generate_topic_name(
            Prompt("", "test prompt"),
            TextTemplate.extract_name,
        )

    assert result == ""


@pytest.mark.parametrize("error_class", LITELLM_FAIL_FAST)
def test_litellm_topic_name_fail_fast_error(litellm_wrapper, error_class):
    with patch(
        "litellm.completion",
        side_effect=make_litellm_error(error_class),
    ):
        with pytest.raises(FailFastLLMError):
            litellm_wrapper.generate_topic_name(
                Prompt("", "test prompt"),
                TextTemplate.extract_name,
            )


@pytest.mark.parametrize("error_class", LITELLM_FAIL_FAST)
def test_litellm_topic_cluster_names_fail_fast_error(
    litellm_wrapper,
    error_class,
    mock_data,
):
    with patch(
        "litellm.completion",
        side_effect=make_litellm_error(error_class),
    ):
        with pytest.raises(FailFastLLMError):
            litellm_wrapper.generate_topic_cluster_names(
                Prompt("", "test prompt"),
                mock_data["old_names"],
                extract_cluster_names(mock_data["old_names"]),
            )


@pytest.mark.parametrize("error_class", LITELLM_RETRYABLE)
@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_litellm_generate_topic_name_retry_exhausted_returns_empty(
    litellm_wrapper,
    error_class,
):
    with patch(
        "litellm.completion",
        side_effect=[make_litellm_error(error_class) for _ in range(3)],
    ) as mock_completion:
        result = litellm_wrapper.generate_topic_name(
            Prompt("", "test prompt"),
            TextTemplate.extract_name,
        )

    assert result == ""
    assert mock_completion.call_count == 3


@pytest.mark.parametrize("error_class", LITELLM_RETRYABLE)
@pytest.mark.filterwarnings("ignore:All retries exhausted")
def test_litellm_generate_cluster_names_retry_exhausted_returns_old_names(
    litellm_wrapper,
    mock_data,
    error_class,
):
    with patch(
        "litellm.completion",
        side_effect=[make_litellm_error(error_class) for _ in range(3)],
    ) as mock_completion:
        result = litellm_wrapper.generate_topic_cluster_names(
            Prompt("", "test prompt"),
            mock_data["old_names"],
            extract_cluster_names(mock_data["old_names"]),
        )

    assert result == mock_data["old_names"]
    assert mock_completion.call_count == 3


@pytest.mark.parametrize(
    "use_json_object,detected_support,expected",
    [
        (True, False, True),
        (False, True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_litellm_should_use_json_object(
    use_json_object,
    detected_support,
    expected,
):
    wrapper = LiteLLMNamer(
        model="openai/gpt-4o-mini",
        use_json_object=use_json_object,
    )

    with patch.object(
        wrapper,
        "_detect_json_object_support",
        return_value=detected_support,
    ) as mock_detect:
        first = wrapper._should_use_json_object()
        second = wrapper._should_use_json_object()

    assert first is expected
    assert second is expected

    if use_json_object is None:
        assert mock_detect.call_count == 1
    else:
        mock_detect.assert_not_called()


def test_litellm_system_prompt_probe_falls_back_and_caches(litellm_wrapper, mock_data):
    unsupported_error = Exception("system messages are not supported")
    good_response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch(
        "litellm.completion",
        side_effect=[unsupported_error, good_response],
    ) as mock_completion:
        result = litellm_wrapper._call_llm(
            Prompt("system", "user"),
            temperature=0.4,
            max_tokens=20,
        )

    assert result == mock_data["valid_topic_name"]
    assert litellm_wrapper._system_prompt_capability is False
    assert mock_completion.call_count == 2


def test_litellm_system_prompt_cached_false_flattens_immediately(
    litellm_wrapper,
    mock_data,
):
    litellm_wrapper._system_prompt_capability = False
    good_response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=good_response) as mock_completion:
        litellm_wrapper._call_llm(
            Prompt("system", "user"),
            temperature=0.4,
            max_tokens=20,
        )

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "user", "content": "System: system\n\nUser: user"}
    ]


def test_litellm_system_prompt_probe_success_caches_true(
    litellm_wrapper,
    mock_data,
):
    good_response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=good_response):
        result = litellm_wrapper._call_llm(
            Prompt("system", "user"),
            temperature=0.4,
            max_tokens=20,
        )

    assert result == mock_data["valid_topic_name"]
    assert litellm_wrapper._system_prompt_capability is True


# Test max_tokens configuration


def test_litellm_namer_default_max_tokens():
    """Test that LiteLLMNamer has correct default max_tokens values"""
    namer = LiteLLMNamer(model="openai/gpt-4o-mini")
    assert namer.max_tokens_topic_name == 128
    assert namer.max_tokens_cluster_names == 1024


def test_litellm_namer_custom_max_tokens():
    """Test that LiteLLMNamer accepts custom max_tokens values"""
    namer = LiteLLMNamer(
        model="openai/gpt-4o-mini",
        max_tokens_topic_name=256,
        max_tokens_cluster_names=2048,
    )
    assert namer.max_tokens_topic_name == 256
    assert namer.max_tokens_cluster_names == 2048


def test_openai_namer_default_max_tokens():
    """Test that OpenAINamer has correct default max_tokens values"""
    namer = OpenAINamer()
    assert namer.max_tokens_topic_name == 128
    assert namer.max_tokens_cluster_names == 1024


def test_openai_namer_custom_max_tokens():
    """Test that OpenAINamer accepts and passes through custom max_tokens values"""
    namer = OpenAINamer(max_tokens_topic_name=512, max_tokens_cluster_names=2048)
    assert namer.max_tokens_topic_name == 512
    assert namer.max_tokens_cluster_names == 2048


def test_anthropic_namer_custom_max_tokens():
    """Test that AnthropicNamer accepts and passes through custom max_tokens values"""
    namer = AnthropicNamer(max_tokens_topic_name=200, max_tokens_cluster_names=1500)
    assert namer.max_tokens_topic_name == 200
    assert namer.max_tokens_cluster_names == 1500


def test_cohere_namer_custom_max_tokens():
    """Test that CohereNamer accepts and passes through custom max_tokens values"""
    namer = CohereNamer(max_tokens_topic_name=300, max_tokens_cluster_names=1200)
    assert namer.max_tokens_topic_name == 300
    assert namer.max_tokens_cluster_names == 1200


def test_azure_namer_custom_max_tokens():
    """Test that AzureAINamer accepts and passes through custom max_tokens values"""
    namer = AzureAINamer(
        model="gpt-4o",
        max_tokens_topic_name=150,
        max_tokens_cluster_names=1100,
    )
    assert namer.max_tokens_topic_name == 150
    assert namer.max_tokens_cluster_names == 1100


def test_ollama_namer_custom_max_tokens():
    """Test that OllamaNamer accepts and passes through custom max_tokens values"""
    namer = OllamaNamer(max_tokens_topic_name=175, max_tokens_cluster_names=1300)
    assert namer.max_tokens_topic_name == 175
    assert namer.max_tokens_cluster_names == 1300


def test_litellm_namer_generate_topic_name_uses_instance_default(
    litellm_wrapper, mock_data
):
    """Test that generate_topic_name uses instance max_tokens_topic_name when max_tokens=None"""
    # Set custom instance defaults
    litellm_wrapper.max_tokens_topic_name = 256
    litellm_wrapper.max_tokens_cluster_names = 2048

    good_response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=good_response) as mock_completion:
        litellm_wrapper.generate_topic_name(
            Prompt("", "test prompt"),
            TextTemplate.extract_name,
            # max_tokens not specified, should use instance default
        )

    # Check that the instance default was used
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 256


def test_litellm_namer_generate_topic_name_override_with_explicit_max_tokens(
    litellm_wrapper, mock_data
):
    """Test that generate_topic_name respects explicit max_tokens over instance default"""
    # Set custom instance defaults
    litellm_wrapper.max_tokens_topic_name = 256
    litellm_wrapper.max_tokens_cluster_names = 2048

    good_response = MockLLMResponse.create_chat_response(mock_data["valid_topic_name"])

    with patch("litellm.completion", return_value=good_response) as mock_completion:
        litellm_wrapper.generate_topic_name(
            Prompt("", "test prompt"),
            TextTemplate.extract_name,
            max_tokens=100,  # explicit override
        )

    # Check that the explicit value was used, not the instance default
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 100


def test_litellm_namer_generate_cluster_names_uses_instance_default(
    litellm_wrapper, mock_data
):
    """Test that generate_topic_cluster_names uses instance max_tokens_cluster_names when max_tokens=None"""
    # Set custom instance defaults
    litellm_wrapper.max_tokens_topic_name = 256
    litellm_wrapper.max_tokens_cluster_names = 2048

    good_response = MockLLMResponse.create_chat_response(
        mock_data["valid_cluster_names"]
    )

    with patch("litellm.completion", return_value=good_response) as mock_completion:
        litellm_wrapper.generate_topic_cluster_names(
            Prompt("", "test prompt"),
            mock_data["old_names"],
            extract_cluster_names(mock_data["old_names"]),
            # max_tokens not specified, should use instance default
        )

    # Check that the instance default was used
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 2048


def test_litellm_namer_generate_cluster_names_override_with_explicit_max_tokens(
    litellm_wrapper, mock_data
):
    """Test that generate_topic_cluster_names respects explicit max_tokens over instance default"""
    # Set custom instance defaults
    litellm_wrapper.max_tokens_topic_name = 256
    litellm_wrapper.max_tokens_cluster_names = 2048

    good_response = MockLLMResponse.create_chat_response(
        mock_data["valid_cluster_names"]
    )

    with patch("litellm.completion", return_value=good_response) as mock_completion:
        litellm_wrapper.generate_topic_cluster_names(
            Prompt("", "test prompt"),
            mock_data["old_names"],
            extract_cluster_names(mock_data["old_names"]),
            max_tokens=512,  # explicit override
        )

    # Check that the explicit value was used, not the instance default
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["max_tokens"] == 512
