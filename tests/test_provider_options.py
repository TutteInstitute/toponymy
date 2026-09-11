"""Provider option ownership and real local-backend configuration validation."""

from types import SimpleNamespace

import pytest

from toponymy import llm_wrappers as wrappers
from toponymy.debug_logging import BasicDebugLogger
from toponymy.templates import Prompt

FACTORY_ALIASES = [
    ("CohereNamer", "httpx_client"),
    ("AsyncCohereNamer", "httpx_client"),
    ("OpenAINamer", "http_client"),
    ("AsyncOpenAINamer", "organization"),
]


@pytest.mark.parametrize("factory_name,alias", FACTORY_ALIASES)
def test_legacy_alias_wins_without_mutating_the_callers_options(factory_name, alias):
    previous = "previous-organization" if alias == "organization" else object()
    requested = "requested-organization" if alias == "organization" else object()
    options = {"timeout": 7, alias: previous}
    before = dict(options)
    with pytest.warns(FutureWarning, match="deprecated"):
        wrapper = getattr(wrappers, factory_name)(
            api_key="fixture-key", provider_kwargs=options, **{alias: requested}
        )
    assert wrapper.provider_kwargs[alias] is requested
    assert wrapper.provider_kwargs["timeout"] == 7
    assert wrapper.provider_kwargs is not options
    assert options == before, "Constructing a wrapper must not rewrite caller options"
    second = getattr(wrappers, factory_name)(
        api_key="fixture-key", provider_kwargs=options
    )
    assert second.provider_kwargs[alias] is previous


@pytest.mark.parametrize("factory_name,alias", FACTORY_ALIASES)
def test_options_without_legacy_alias_are_owned_and_preserve_values(
    factory_name, alias
):
    value = "configured-organization" if alias == "organization" else object()
    options = {"timeout": 7, alias: value}
    wrapper = getattr(wrappers, factory_name)(
        api_key="fixture-key", provider_kwargs=options
    )
    assert wrapper.provider_kwargs == options
    assert wrapper.provider_kwargs is not options
    wrapper.provider_kwargs["timeout"] = 19
    assert options["timeout"] == 7


@pytest.fixture
def validating_pipeline(monkeypatch):
    import transformers
    from transformers import GenerationConfig
    from transformers.generation.logits_process import TemperatureLogitsWarper

    calls = []

    def pipeline(task, model, **kwargs):
        assert task == "text-generation"
        assert model == "private-fixture-no-model-loaded"
        assert kwargs == {}

        def generate(messages, **options):
            assert options["return_full_text"] is False
            generation_options = {
                key: value
                for key, value in options.items()
                if key != "return_full_text"
            }
            # This is the installed library's actual configuration validation.
            # Its runtime sampling validator additionally rejects zero temperature.
            config = GenerationConfig(**generation_options)
            config.validate()
            if config.do_sample and config.temperature not in (None, 1.0):
                TemperatureLogitsWarper(config.temperature)
            calls.append((messages, options, config))
            return [{"generated_text": '{"topic_name":"Local fixture"}'}]

        generate.tokenizer = SimpleNamespace(eos_token_id=7)
        return generate

    # Keep the real Transformers package/configuration classes. Installing this
    # one attribute avoids importing or constructing its model pipeline machinery.
    monkeypatch.setitem(transformers.__dict__, "pipeline", pipeline)
    return calls


def configured_backend(async_mode, system):
    backend = (
        wrappers.AsyncHuggingFaceNamer if async_mode else wrappers.HuggingFaceNamer
    )

    class RouteWrapper(backend):
        @property
        def supports_system_prompts(self):
            return system

    return RouteWrapper("private-fixture-no-model-loaded")


def assert_generation_request(calls, system, temperature):
    assert len(calls) == 1
    messages, options, config = calls[0]
    expected_messages = (
        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
        if system
        else [{"role": "user", "content": "system\n\nuser"}]
    )
    assert messages == expected_messages
    assert config.max_new_tokens == 13
    assert config.pad_token_id == 7
    assert config.do_sample is (temperature > 0)
    if temperature > 0:
        assert options["temperature"] == temperature
    else:
        assert options.get("temperature") in (None, 1.0)


@pytest.mark.parametrize("system", [False, True], ids=["combined", "system"])
@pytest.mark.parametrize("temperature", [0.0, 0.4], ids=["greedy", "sampling"])
def test_huggingface_sync_routes_use_valid_installed_generation_options(
    validating_pipeline, system, temperature
):
    wrapper = configured_backend(False, system)
    assert (
        wrapper.generate_topic_name(
            Prompt("system", "user"), temperature=temperature, max_tokens=13
        )
        == "Local fixture"
    )
    assert_generation_request(validating_pipeline, system, temperature)


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [False, True], ids=["combined", "system"])
@pytest.mark.parametrize("temperature", [0.0, 0.4], ids=["greedy", "sampling"])
async def test_huggingface_async_routes_use_valid_installed_generation_options(
    validating_pipeline, system, temperature
):
    wrapper = configured_backend(True, system)
    assert await wrapper.generate_topic_names(
        [Prompt("system", "user")], temperature=temperature, max_tokens=13
    ) == ["Local fixture"]
    assert_generation_request(validating_pipeline, system, temperature)


def test_debug_logger_write_failure_warns_and_preserves_caller_event(tmp_path):
    logger = BasicDebugLogger(tmp_path)
    event = {"event": "fixture", "prompt": "local text"}
    before = dict(event)
    with pytest.warns(UserWarning, match="could not write"):
        result = logger(event)
    assert result is None
    assert event == before
