"""Preserved provider factory, transport, and diagnostic contracts from v0.5."""

import asyncio
from contextlib import nullcontext
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from tenacity import wait_none

from toponymy import llm_wrappers as wrappers
from toponymy.templates import Prompt

FACTORIES = [
    ("OpenAINamer", "openai/gpt-4o-mini", False),
    ("AsyncOpenAINamer", "openai/gpt-4o-mini", False),
    ("AnthropicNamer", "anthropic/claude-haiku-4-5-20251001", False),
    ("AsyncAnthropicNamer", "anthropic/claude-haiku-4-5-20251001", False),
    ("CohereNamer", "cohere/command-r-08-2024", False),
    ("AsyncCohereNamer", "cohere/command-r-08-2024", False),
    ("OllamaNamer", "ollama_chat/llama3.2", False),
    ("AsyncOllamaNamer", "ollama_chat/llama3.2", False),
    ("AzureAINamer", "azure_ai/deployment", False),
    ("AsyncAzureAINamer", "azure_ai/deployment", False),
    ("GoogleGeminiNamer", "gemini/gemini-2.5-flash-lite", True),
    ("AsyncGoogleGeminiNamer", "gemini/gemini-2.5-flash-lite", True),
    ("TogetherNamer", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", True),
    ("AsyncTogether", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", True),
    ("ReplicateNamer", "replicate/meta/llama-2-70b-chat", True),
]


def construct(name, **kwargs):
    if "Azure" in name:
        kwargs.setdefault("model", "deployment")
    context = (
        pytest.warns(FutureWarning)
        if name in {row[0] for row in FACTORIES if row[2]}
        else nullcontext()
    )
    with context:
        return getattr(wrappers, name)(**kwargs)


@pytest.mark.parametrize("name,expected_model,deprecated", FACTORIES)
def test_factory_defaults_and_provider_options(name, expected_model, deprecated):
    options = {"timeout": 123}
    callback = lambda payload: None
    namer = construct(
        name, api_key="fixture-key", provider_kwargs=options, callback=callback
    )
    expected_type = (
        wrappers.AsyncLiteLLMNamer
        if name.startswith("Async")
        else wrappers.LiteLLMNamer
    )
    assert isinstance(namer, expected_type)
    assert namer.model == expected_model
    assert namer.api_key == "fixture-key"
    assert namer.provider_kwargs == {"timeout": 123}
    assert namer.provider_kwargs is not options
    assert namer.callback is callback
    assert namer._supports_debug_callback
    assert namer.supports_system_prompts
    assert namer.use_json_object is None
    assert namer.use_json_schema is None
    if "Ollama" in name:
        assert namer.api_base == "http://localhost:11434"


@pytest.mark.parametrize("name,expected_model,deprecated", FACTORIES)
def test_factory_structured_output_options_pass_through(
    name, expected_model, deprecated
):
    namer = construct(name, api_key="fixture", use_json_schema=True)
    assert namer.use_json_schema is True
    with pytest.raises(wrappers.InvalidLLMInputError):
        construct(name, api_key="fixture", use_json_schema=True, use_json_object=True)


@pytest.mark.parametrize(
    "name",
    [
        "OpenAINamer",
        "AsyncOpenAINamer",
        "AnthropicNamer",
        "AsyncAnthropicNamer",
        "CohereNamer",
        "AsyncCohereNamer",
        "AzureAINamer",
        "AsyncAzureAINamer",
        "OllamaNamer",
        "AsyncOllamaNamer",
    ],
)
def test_factory_token_limits_and_temperature_preserved(name):
    kwargs = {"max_tokens_topic_name": 317, "max_tokens_cluster_names": 913}
    if "Ollama" not in name:
        kwargs["temperature_override"] = 0.15
    namer = construct(name, api_key="fixture", **kwargs)
    assert namer.max_tokens_topic_name == 317
    assert namer.max_tokens_cluster_names == 913
    if "Ollama" not in name:
        assert namer.temperature_override == 0.15


@pytest.mark.parametrize(
    "name,alias,attribute,key",
    [
        ("OpenAINamer", "base_url", "api_base", None),
        ("AsyncOpenAINamer", "base_url", "api_base", None),
        ("OpenAINamer", "http_client", "provider_kwargs", "http_client"),
        ("AsyncOpenAINamer", "organization", "provider_kwargs", "organization"),
        ("CohereNamer", "base_url", "api_base", None),
        ("AsyncCohereNamer", "base_url", "api_base", None),
        ("CohereNamer", "httpx_client", "provider_kwargs", "httpx_client"),
        ("AsyncCohereNamer", "httpx_client", "provider_kwargs", "httpx_client"),
        ("OllamaNamer", "host", "api_base", None),
        ("AsyncOllamaNamer", "host", "api_base", None),
        ("AzureAINamer", "endpoint", "api_base", None),
        ("AsyncAzureAINamer", "endpoint", "api_base", None),
    ],
)
def test_deprecated_aliases_preserve_values(name, alias, attribute, key):
    value = object() if alias.endswith("client") else "https://fixture.invalid"
    with nullcontext() if alias == "endpoint" else pytest.warns(FutureWarning):
        namer = construct(name, api_key="fixture", **{alias: value})
    actual = getattr(namer, attribute)
    assert (actual[key] if key else actual) is value


@pytest.mark.parametrize(
    "name,new_key,legacy_key",
    [
        ("CohereNamer", "COHERE_API_KEY", "CO_API_KEY"),
        ("AsyncCohereNamer", "COHERE_API_KEY", "CO_API_KEY"),
        ("AzureAINamer", "AZURE_AI_API_KEY", "AZURE_API_KEY"),
        ("AsyncAzureAINamer", "AZURE_AI_API_KEY", "AZURE_API_KEY"),
        ("GoogleGeminiNamer", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        ("AsyncGoogleGeminiNamer", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        ("ReplicateNamer", "REPLICATE_API_KEY", "REPLICATE_API_TOKEN"),
    ],
)
def test_legacy_keys_and_explicit_key_precedence(
    monkeypatch, name, new_key, legacy_key
):
    monkeypatch.delenv(new_key, raising=False)
    monkeypatch.setenv(legacy_key, "legacy-fixture")
    if "Gemini" in name or "Replicate" in name:
        namer = construct(name)
    else:
        with pytest.warns(FutureWarning):
            namer = construct(name)
    assert namer.api_key == "legacy-fixture"
    monkeypatch.setenv(new_key, "modern-fixture")
    assert construct(name).api_key == "modern-fixture"
    assert construct(name, api_key="explicit-fixture").api_key == "explicit-fixture"


@pytest.mark.parametrize("name", ["CohereNamer", "AsyncCohereNamer"])
def test_cohere_legacy_endpoint_precedence(monkeypatch, name):
    monkeypatch.delenv("COHERE_API_BASE", raising=False)
    monkeypatch.setenv("CO_API_URL", "https://legacy.invalid")
    with pytest.warns(FutureWarning):
        assert construct(name, api_key="fixture").api_base == "https://legacy.invalid"
    assert (
        construct(name, api_key="fixture", api_base="https://explicit.invalid").api_base
        == "https://explicit.invalid"
    )


class CompletionProvider:
    def __init__(self, responses=None):
        self.responses = iter(responses or ['{"topic_name":"Transit"}'] * 20)
        self.calls = []

    def get_supported_openai_params(self, model):
        return []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=value))]
        )

    async def acompletion(self, **kwargs):
        await asyncio.sleep(0)
        return self.completion(**kwargs)


@pytest.fixture
def immediate_retries(monkeypatch):
    monkeypatch.setattr(
        wrappers, "wait_random_exponential", lambda **kwargs: wait_none()
    )
    monkeypatch.setattr(
        wrappers.LLMWrapper.generate_topic_name.retry, "wait", wait_none()
    )


@pytest.mark.parametrize("system", [False, True])
def test_sync_request_limits_renderings_and_diagnostic_status(monkeypatch, system):
    provider = CompletionProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.LiteLLMNamer(
        disable_system_prompts=not system,
        max_tokens_topic_name=317,
        temperature_override=0.15,
    )
    assert (
        namer.generate_topic_name(Prompt("sys", "user"), temperature=0.8) == "Transit"
    )
    assert provider.calls[0]["max_tokens"] == 317
    assert provider.calls[0]["temperature"] == 0.15
    assert provider.calls[0]["messages"][0] == (
        {"role": "system", "content": "sys"}
        if system
        else {"role": "user", "content": "sys\n\nuser"}
    )
    namer.generate_topic_name(Prompt("sys", "user"), max_tokens=211)
    assert provider.calls[1]["max_tokens"] == 211
    assert namer.connectivity_status()["success"]
    assert len(provider.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [False, True])
async def test_async_request_limits_renderings_and_diagnostic_status(
    monkeypatch, system
):
    provider = CompletionProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.AsyncLiteLLMNamer(
        disable_system_prompts=not system,
        max_tokens_topic_name=317,
        temperature_override=0.15,
    )
    assert await namer.generate_topic_names([Prompt("sys", "user")]) == ["Transit"]
    assert provider.calls[0]["max_tokens"] == 317
    assert provider.calls[0]["temperature"] == 0.15
    assert (await namer.connectivity_status())["success"]
    assert len(provider.calls) == 2
    await namer.close()


def test_role_fallback_uses_shared_retry_budget_and_preserves_temperature(
    monkeypatch, immediate_retries
):
    class UnsupportedRole(Exception):
        status_code = 400

    provider = CompletionProvider(
        [
            UnsupportedRole("does not support system role"),
            TimeoutError(),
            '{"topic_name":"Transit"}',
            '{"topic_name":"Cached"}',
        ]
    )
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.LiteLLMNamer(temperature_override=0.1)
    assert namer.generate_topic_name(Prompt("sys", "user")) == "Transit"
    assert len(provider.calls) == 3
    assert provider.calls[0]["messages"][0]["role"] == "system"
    assert all(
        request["messages"][0]["role"] == "user" for request in provider.calls[1:]
    )
    assert all(request["temperature"] == 0.1 for request in provider.calls)
    assert namer.generate_topic_name(Prompt("sys", "next")) == "Cached"
    assert len(provider.calls) == 4


def test_unexpected_error_mentioning_system_role_does_not_probe(
    monkeypatch, immediate_retries
):
    provider = CompletionProvider([RuntimeError("bug in system role processing")])
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    with pytest.raises(RuntimeError):
        wrappers.LiteLLMNamer().generate_topic_name(Prompt("sys", "user"))
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "status,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (302, False),
    ],
)
def test_provider_error_status_policy(status, retryable):
    error = RuntimeError("provider response")
    error.status_code = status
    assert wrappers._should_retry(error) is retryable


def test_http_client_passthrough_does_not_copy_resource(monkeypatch):
    provider = CompletionProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    client = threading.Lock()
    namer = wrappers.LiteLLMNamer(provider_kwargs={"http_client": client})
    namer.generate_topic_name("user")
    assert provider.calls[0]["http_client"] is client


def test_sync_disambiguation_mapping_and_token_limits(monkeypatch):
    provider = CompletionProvider(
        [
            '{"new_topic_name_mapping":{"2. Beta":"New Beta","1. Alpha":"New Alpha"},"topic_specificities":[0.5,0.5]}'
        ]
        * 2
    )
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.LiteLLMNamer(max_tokens_cluster_names=913)
    assert namer.generate_topic_cluster_names(
        Prompt("sys", "user"), ["Alpha", "Beta"]
    ) == ["New Alpha", "New Beta"]
    assert provider.calls[0]["max_tokens"] == 913
    namer.generate_topic_cluster_names(
        Prompt("sys", "user"), ["Alpha", "Beta"], max_tokens=717
    )
    assert provider.calls[1]["max_tokens"] == 717
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_async_disambiguation_and_empty_inputs(monkeypatch):
    provider = CompletionProvider(
        [
            '{"new_topic_name_mapping":{"2":"New Beta","1":"New Alpha"},"topic_specificities":[0.5,0.5]}'
        ]
    )
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.AsyncLiteLLMNamer(max_tokens_cluster_names=913)
    assert await namer.generate_topic_names([]) == []
    assert await namer.generate_topic_cluster_names([], []) == []
    assert provider.calls == []
    with pytest.raises(wrappers.InvalidLLMInputError):
        await namer.generate_topic_cluster_names([Prompt("s", "u")], [])
    assert provider.calls == []
    assert await namer.generate_topic_cluster_names(
        [Prompt("s", "u")], [["Alpha", "Beta"]]
    ) == [["New Alpha", "New Beta"]]
    assert provider.calls[0]["max_tokens"] == 913


def test_diagnostic_failures_retain_original_exception(monkeypatch):
    error = RuntimeError("provider programming error")
    provider = CompletionProvider([error])
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    result = wrappers.LiteLLMNamer().connectivity_status()
    assert result["success"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["original_exception"] is error
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_anthropic_batch_sdk_lifecycle_and_alignment(monkeypatch):
    calls = {"create": [], "retrieve": [], "results": [], "cancel": []}

    def create(**kwargs):
        calls["create"].append(kwargs)
        return SimpleNamespace(id="batch-fixture")

    def retrieve(batch_id):
        calls["retrieve"].append(batch_id)
        return SimpleNamespace(processing_status="ended")

    def results(batch_id):
        calls["results"].append(batch_id)
        return iter(
            SimpleNamespace(
                custom_id=str(index),
                result=SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                type="text", text='{"topic_name":"' + name + '"}'
                            )
                        ]
                    ),
                ),
            )
            for index, name in reversed(
                list(
                    enumerate(
                        ["First", "Second"][: len(calls["create"][-1]["requests"])]
                    )
                )
            )
        )

    def cancel(batch_id):
        calls["cancel"].append(batch_id)

    def client(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(
                    create=create, retrieve=retrieve, results=results, cancel=cancel
                )
            )
        )

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(
            Anthropic=client, APIError=Exception, APIConnectionError=ConnectionError
        ),
    )
    namer = wrappers.BatchAnthropicNamer("fixture", polling_interval=0.01, timeout=1)
    assert await namer.generate_topic_names(
        [Prompt("sys", "first"), Prompt("sys", "second")]
    ) == ["First", "Second"]
    assert len(calls["create"]) == len(calls["retrieve"]) == len(calls["results"]) == 1
    assert calls["create"][0]["requests"][0]["params"]["system"] == "sys"
    assert namer.get_batch_status("batch-fixture") == "ended"
    namer.cancel_batch("batch-fixture")
    assert calls["cancel"] == ["batch-fixture"]
    diagnostic = await namer.connectivity_status()
    assert diagnostic["success"] is True
    assert diagnostic["response"] == '{"topic_name":"First"}'
    assert len(calls["create"]) == 2
    assert calls["create"][-1]["requests"][0]["params"]["system"] == ""
    assert (
        "Identify yourself"
        in calls["create"][-1]["requests"][0]["params"]["messages"][0]["content"]
    )


def test_llama_cpp_real_argument_contract_at_fake_boundary(monkeypatch):
    calls = []

    class Llama:
        def __init__(self, **kwargs):
            assert kwargs == {"model_path": "fixture.gguf", "n_ctx": 512}

        def __call__(self, prompt, *, max_tokens, temperature):
            calls.append((prompt, max_tokens, temperature))
            return {"choices": [{"text": '{"topic_name":"Local"}'}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Llama))
    namer = wrappers.LlamaCppNamer("fixture.gguf", n_ctx=512)
    assert namer.generate_topic_name(Prompt("sys", "user"), max_tokens=211) == "Local"
    assert calls == [("sys\n\nuser", 211, 0.4)]
    diagnostic = namer.connectivity_status()
    assert diagnostic["success"] is True
    assert diagnostic["model"] == "fixture.gguf"
    assert diagnostic["response"] == '{"topic_name":"Local"}'
    assert len(calls) == 2


@pytest.mark.parametrize("system", [False, True])
def test_huggingface_transport_uses_pipeline_contract(monkeypatch, system):
    calls = []

    def pipeline(task, model, **kwargs):
        assert (task, model) == ("text-generation", "fixture-model")

        def generate(messages, **options):
            calls.append((messages, options))
            return [{"generated_text": '{"topic_name":"Local"}'}]

        generate.tokenizer = SimpleNamespace(eos_token_id=7)
        return generate

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    namer = wrappers.HuggingFaceNamer("fixture-model")
    prompt = wrappers.validate_prompt(Prompt("sys", "user"), system)
    method = namer._call_llm_with_system_prompt if system else namer._call_llm
    assert method(prompt, 0.2, 211) == '{"topic_name":"Local"}'
    assert calls[0][1]["max_new_tokens"] == 211
    assert calls[0][1]["pad_token_id"] == 7


@pytest.mark.asyncio
async def test_async_local_pipeline_inference_is_off_event_loop(monkeypatch):
    caller_thread = threading.get_ident()
    calls = []

    def pipeline(*args, **kwargs):
        def generate(messages, **options):
            assert threading.get_ident() != caller_thread
            calls.append(messages)
            return [{"generated_text": '{"topic_name":"Local"}'}]

        generate.tokenizer = SimpleNamespace(eos_token_id=7)
        return generate

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    namer = wrappers.AsyncHuggingFaceNamer("fixture-model")
    assert await namer.generate_topic_names(
        [Prompt("sys", "one"), Prompt("sys", "two")]
    ) == ["Local", "Local"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_vllm_batch_is_off_loop_and_preserves_order(monkeypatch):
    caller_thread = threading.get_ident()

    class EngineDeadError(RuntimeError):
        pass

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "fixture-model"

        def chat(self, messages, sampling_params):
            assert threading.get_ident() != caller_thread
            assert sampling_params.max_tokens == 211
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text='{"topic_name":"' + item[-1]["content"] + '"}'
                        )
                    ]
                )
                for item in messages
            ]

    module = ModuleType("vllm")
    module.LLM = Engine
    module.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.exceptions",
        SimpleNamespace(EngineDeadError=EngineDeadError),
    )
    namer = wrappers.AsyncVLLMNamer("fixture-model")
    assert await namer.generate_topic_names(
        [Prompt("sys", "First"), Prompt("sys", "Second")], max_tokens=211
    ) == ["First", "Second"]
