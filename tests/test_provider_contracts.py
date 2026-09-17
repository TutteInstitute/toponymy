import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from toponymy import llm_wrappers as wrappers
from toponymy.llm_wrappers import LLMWrapper, validate_prompt
from toponymy.templates import Prompt, TextTemplate
from toponymy.response_parsing import ResponseParseError


class RecordingWrapper(LLMWrapper):
    def __init__(self):
        self.calls = []

    def _call_llm(self, prompt, temperature, max_tokens):
        self.calls.append((prompt, temperature, max_tokens))
        return '{"topic_name":"Transit","topic_specificity":0.8}'

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        return self._call_llm(prompt, temperature, max_tokens)


def test_sync_request_characterization():
    wrapper = RecordingWrapper()
    prompt = {"system": "system", "user": "user", "combined": "both"}
    assert wrapper.generate_topic_name(prompt, max_tokens=321) == "Transit"
    assert wrapper.calls == [(prompt, 0.4, 321)]


def test_legacy_rendering_characterization():
    prompt = {"system": "system", "user": "user", "combined": "both"}
    assert validate_prompt(prompt, True) == prompt
    assert validate_prompt(prompt, False) == prompt


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(
        wrappers, "wait_random_exponential", lambda **kwargs: wait_none()
    )
    for method in (
        LLMWrapper.generate_topic_name,
        LLMWrapper.generate_topic_cluster_names,
    ):
        monkeypatch.setattr(method.retry, "wait", wait_none())


@pytest.mark.parametrize(
    "prompt", [Prompt("system", "user"), {"system": "system", "user": "user"}]
)
def test_canonical_prompt_has_combined_boundary_rendering(prompt):
    assert validate_prompt(prompt, False)["combined"] == "system\n\nuser"


@pytest.mark.parametrize(
    "prompt", [12, [], Prompt(12, "user"), {"system": "a"}, Prompt("s", "u", [])]
)
def test_prompt_validation_precedes_request(prompt):
    wrapper = RecordingWrapper()
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrapper.generate_topic_name(prompt)
    assert wrapper.calls == []


class FakeProvider:
    def __init__(self, schema=True, objects=True):
        self.schema = schema
        self.objects = objects
        self.calls = []
        self.active = self.peak = 0

    def supports_response_schema(self, model):
        return self.schema

    def get_supported_openai_params(self, model):
        return ["response_format"] if self.objects else []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"topic_name":"Transit"}')
                )
            ]
        )

    async def acompletion(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001)
            return self.completion(**kwargs)
        finally:
            self.active -= 1


@pytest.mark.parametrize(
    "schema,objects,schema_option,object_option,expected",
    [
        (True, True, None, None, "json_schema"),
        (False, True, None, None, "json_object"),
        (False, False, None, None, None),
        (True, True, False, None, "json_object"),
        (True, True, None, True, "json_object"),
        (True, True, False, False, None),
        (True, True, True, None, "json_schema"),
    ],
)
def test_schema_policy_and_existing_request_path(
    monkeypatch, schema, objects, schema_option, object_option, expected
):
    provider = FakeProvider(schema, objects)
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.LiteLLMNamer(
        use_json_schema=schema_option, use_json_object=object_option
    )
    template = TextTemplate("article", "city")
    prompt = template.cluster_prompt({}, "specific")
    assert (
        namer.generate_topic_name(prompt, response_parser=template.extract_name)
        == "Transit"
    )
    assert len(provider.calls) == 1
    request = provider.calls[0]
    assert request["messages"] == [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    assert request.get("response_format", {}).get("type") == expected
    if expected == "json_schema":
        assert request["response_format"]["json_schema"]["schema"] == prompt.json_schema
        assert (
            request["response_format"]["json_schema"]["schema"]
            is not prompt.json_schema
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_json_schema": True, "use_json_object": True},
        {
            "use_json_schema": False,
            "provider_kwargs": {"response_format": {"type": "json_object"}},
        },
        {
            "use_json_object": True,
            "provider_kwargs": {"response_format": {"type": "json_schema"}},
        },
        {"provider_kwargs": {"messages": []}},
        {"use_json_schema": "yes"},
    ],
)
def test_conflicting_options_are_rejected_at_construction(kwargs):
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrappers.LiteLLMNamer(**kwargs)


@pytest.mark.parametrize(
    "prompt,supported",
    [(Prompt("s", "u"), True), (Prompt("s", "u", {"type": "object"}), False)],
)
def test_required_schema_never_downgrades_or_calls(monkeypatch, prompt, supported):
    provider = FakeProvider(schema=supported)
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.LiteLLMNamer(use_json_schema=True)
    with pytest.raises(wrappers.InvalidLLMInputError):
        namer.generate_topic_name(prompt)
    assert provider.calls == []


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "nonsense"},
        {"properties": {"name": 12}},
        {"type": "object", "required": "name"},
        {"$ref": "https://example.invalid/schema"},
        {
            "type": "object",
            "$defs": {"external": {"$ref": "https://example.invalid/schema"}},
        },
        {"$ref": "#/$defs/missing"},
        {"$dynamicRef": "#missing"},
        {"$recursiveRef": "https://example.invalid/schema"},
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "dependencies": {
                "first": ["other"],
                "second": {"$ref": "https://example.invalid/schema"},
            },
        },
        {"contentSchema": {"$ref": "https://example.invalid/schema"}},
    ],
)
def test_invalid_schema_is_rejected_before_provider(monkeypatch, schema):
    provider = FakeProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrappers.LiteLLMNamer(use_json_schema=True).generate_topic_name(
            Prompt("s", "u", schema)
        )
    assert provider.calls == []


def test_local_schema_references_and_ref_named_properties_are_valid(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    schema = {
        "type": "object",
        "$defs": {"name": {"type": "string"}},
        "properties": {
            "topic_name": {"$ref": "#/$defs/name"},
            "$ref": {"type": "string"},
        },
    }
    assert (
        wrappers.LiteLLMNamer(use_json_schema=True).generate_topic_name(
            Prompt("s", "u", schema)
        )
        == "Transit"
    )


@pytest.mark.parametrize(
    "schema",
    [
        {
            "$defs": {"name": {"$anchor": "name", "type": "string"}},
            "properties": {"topic_name": {"$ref": "#name"}},
        },
        {
            "$defs": {"name/slash": {"type": "string"}},
            "properties": {"topic_name": {"$ref": "#/$defs/name~1slash"}},
        },
        {
            "$id": "https://example.invalid/root",
            "$defs": {
                "local": {
                    "$id": "child",
                    "$defs": {"name": {"type": "string"}},
                    "properties": {"topic_name": {"$ref": "#/$defs/name"}},
                }
            },
        },
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": {"name": {"type": "string"}},
            "dependencies": {
                "first": ["other"],
                "second": {"$ref": "#/definitions/name"},
            },
        },
    ],
)
def test_local_schema_anchors_escaped_pointers_and_nested_scope(monkeypatch, schema):
    provider = FakeProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    assert (
        wrappers.LiteLLMNamer(use_json_schema=True).generate_topic_name(
            Prompt("s", "u", schema)
        )
        == "Transit"
    )
    assert len(provider.calls) == 1


def test_explicit_response_format_is_preserved(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    options = {"response_format": {"type": "json_object"}, "timeout": 10}
    namer = wrappers.LiteLLMNamer(provider_kwargs=options)
    namer.generate_topic_name(Prompt("s", "u"))
    assert provider.calls[0]["response_format"] == options["response_format"]
    assert options == {"response_format": {"type": "json_object"}, "timeout": 10}
    assert provider.calls[0]["num_retries"] == 0


def test_prompt_is_inspectable_before_provider_call():
    namer = RecordingWrapper()
    events = []

    def callback(payload):
        if payload["event"] == "llm_call_start":
            assert namer.calls == []
            assert payload["prompt"]["user"] == "user"
        events.append(payload["event"])

    namer.callback = callback
    namer.generate_topic_name(Prompt("system", "user"))
    assert events == ["llm_call_start", "llm_call_success"]


class SequenceWrapper(RecordingWrapper):
    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)

    def _call_llm(self, prompt, temperature, max_tokens):
        self.calls.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_sync_parse_and_transient_failures_share_one_budget():
    namer = SequenceWrapper(
        [TimeoutError("retry"), '{"topic_name": null}', '{"topic_name":"done"}']
    )
    assert namer.generate_topic_name(Prompt("s", "u")) == "done"
    assert len(namer.calls) == 3
    namer = SequenceWrapper([TimeoutError("retry")] * 4)
    with pytest.raises(TimeoutError):
        namer.generate_topic_name(Prompt("s", "u"))
    assert len(namer.calls) == 3


@pytest.mark.parametrize(
    "error",
    [TypeError("programming"), ValueError("configuration"), RuntimeError("unexpected")],
)
def test_sync_unexpected_errors_propagate_without_retry(error):
    namer = SequenceWrapper([error])
    with pytest.raises(type(error)):
        namer.generate_topic_name(Prompt("s", "u"))
    assert len(namer.calls) == 1


def test_auth_failure_is_not_retried():
    class AuthError(Exception):
        status_code = 401

    namer = SequenceWrapper([AuthError("bad key")])
    with pytest.raises(wrappers.FailFastLLMError):
        namer.generate_topic_name(Prompt("s", "u"))
    assert len(namer.calls) == 1


class AsyncSequence(wrappers.AsyncLLMWrapper):
    def __init__(self, sequences):
        self.sequences = {key: iter(value) for key, value in sequences.items()}
        self.calls = Counter()

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        key = prompt["user"]
        self.calls[key] += 1
        await asyncio.sleep(0)
        response = next(self.sequences[key])
        if isinstance(response, BaseException):
            raise response
        return response

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens):
        return await self._call_single_llm(prompt, temperature, max_tokens)


@pytest.mark.asyncio
async def test_async_order_partial_results_and_independent_budgets():
    namer = AsyncSequence(
        {
            "a": [TimeoutError(), '{"topic_name":null}', '{"topic_name":"A"}'],
            "b": ['{"topic_name":"B"}'],
            "c": [TimeoutError()] * 4,
        }
    )
    results = await namer.generate_topic_names(
        [Prompt("s", key) for key in ("a", "b", "c")], return_results=True
    )
    assert [result.value for result in results] == ["A", "B", None]
    assert isinstance(results[2].error, TimeoutError)
    assert namer.calls == {"a": 3, "b": 1, "c": 3}


@pytest.mark.asyncio
async def test_async_programming_errors_and_cancellation_propagate():
    for error in (TypeError("bug"), asyncio.CancelledError()):
        namer = AsyncSequence({"a": [error]})
        with pytest.raises(type(error)):
            await namer.generate_topic_names([Prompt("s", "a")])
        assert namer.calls == {"a": 1}


@pytest.mark.asyncio
async def test_async_declared_fail_fast_overrides_transient_retry_policy():
    error = TimeoutError("declared non-retryable by this provider")
    namer = AsyncSequence({"a": [error] * 3})
    namer.FAIL_FAST_EXCEPTIONS = (TimeoutError,)
    with pytest.raises(wrappers.FailFastLLMError) as raised:
        await namer.generate_topic_names([Prompt("s", "a")], return_results=True)
    assert raised.value.original_exception is error
    assert namer.calls == {"a": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [None, TimeoutError("temporarily unavailable"), TypeError("provider bug")]
)
async def test_sync_async_callbacks_report_raw_transport_and_matching_error_payloads(
    error,
):
    raw = '{"topic_name":"Transit","topic_specificity":0.8}'
    responses = [error] * 3 if error else [raw]
    sync = SequenceWrapper(list(responses))
    asynchronous = AsyncSequence({"u": list(responses)})
    sync_events, async_events = [], []
    sync.callback = sync_events.append
    asynchronous.callback = async_events.append
    prompt = Prompt("s", "u")
    if error:
        with pytest.raises(type(error)):
            sync.generate_topic_name(prompt)
        with pytest.raises(type(error)):
            await asynchronous.generate_topic_names([prompt])
    else:
        assert sync.generate_topic_name(prompt) == "Transit"
        assert await asynchronous.generate_topic_names([prompt]) == ["Transit"]
    expected_attempts = 3 if isinstance(error, TimeoutError) else 1
    expected_event = "llm_call_error" if error else "llm_call_success"
    assert [item["event"] for item in sync_events] == [
        "llm_call_start",
        expected_event,
    ] * expected_attempts
    assert [item["event"] for item in async_events] == [
        "llm_call_start",
        expected_event,
    ] * expected_attempts
    for events in (sync_events, async_events):
        for item in events:
            assert item["prompt"]["system"] == "s"
            assert item["prompt"]["user"] == "u"
            assert item["prompt"]["combined"] == "s\n\nu"
            if item["event"] == "llm_call_success":
                assert item["raw_response"] == raw
                assert item["prompt_type"] == "system"
            elif item["event"] == "llm_call_error":
                assert item["error"] == {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                assert item["prompt_type"] == "system"


@pytest.mark.asyncio
async def test_async_concurrency_and_schema_parity(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: provider)
    namer = wrappers.AsyncLiteLLMNamer(max_concurrent_requests=2)
    prompt = TextTemplate("article", "city").cluster_prompt({}, "specific")
    assert await namer.generate_topic_names([prompt] * 6) == ["Transit"] * 6
    assert len(provider.calls) == 6
    assert provider.peak == 2
    assert all(
        request["response_format"]["type"] == "json_schema"
        for request in provider.calls
    )


class ManagedBatch(wrappers.AsyncLLMWrapper):
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def _call_llm_with_system_prompt_batch(
        self, prompts, temperature, max_tokens
    ):
        self.calls += 1
        return self.responses


@pytest.mark.asyncio
async def test_batch_alignment_and_parse_failure_does_not_resubmit():
    namer = ManagedBatch(['{"topic_name":"A"}', '{"topic_name":null}'])
    results = await namer.generate_topic_names(
        [Prompt("s", "a"), Prompt("s", "b")], return_results=True
    )
    assert results[0].value == "A"
    assert isinstance(results[1].error, ResponseParseError)
    assert namer.calls == 1
    namer = ManagedBatch(['{"topic_name":"A"}'])
    with pytest.raises(wrappers.InvalidLLMInputError):
        await namer.generate_topic_names([Prompt("s", "a"), Prompt("s", "b")])
    assert namer.calls == 1


def test_anthropic_results_align_by_id_and_expose_partial_failure():
    success = SimpleNamespace(
        type="succeeded",
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"topic_name":"A"}')]
        ),
    )
    failure = SimpleNamespace(
        type="errored",
        error=SimpleNamespace(type="rate_limit_error", message="slow down"),
    )
    records = [
        SimpleNamespace(custom_id="1", result=failure),
        SimpleNamespace(custom_id="0", result=success),
    ]
    results = wrappers._ordered_anthropic_results(records)
    assert results[0].value == '{"topic_name":"A"}'
    assert isinstance(results[1].error, wrappers.LLMBatchItemError)
    for bad in (
        [records[0]],
        [records[1], records[1]],
        [SimpleNamespace(custom_id="01", result=success)],
    ):
        with pytest.raises(wrappers.InvalidLLMInputError):
            wrappers._ordered_anthropic_results(bad)
