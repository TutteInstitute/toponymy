"""Provider preflight, structural parsing and cancellation boundaries."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import io
import json
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest
from tenacity import wait_none

from toponymy import llm_wrappers as wrappers
from toponymy.response_parsing import ResponseParseError, extract_response
from toponymy.templates import Prompt, TextTemplate

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"topic_name": {"type": "string"}},
    "required": ["topic_name"],
    "additionalProperties": False,
}


def name_schema(**constraints):
    schema = deepcopy(SIMPLE_SCHEMA)
    schema["properties"]["topic_name"].update(constraints)
    return schema


@pytest.fixture(autouse=True)
def no_retry_sleeps(monkeypatch):
    monkeypatch.setattr(
        wrappers, "wait_random_exponential", lambda **kwargs: wait_none()
    )
    monkeypatch.setattr(
        wrappers.LLMWrapper.generate_topic_name.retry, "wait", wait_none()
    )
    monkeypatch.setattr(
        wrappers.LLMWrapper.generate_topic_cluster_names.retry, "wait", wait_none()
    )


@pytest.fixture
def fake_sdk_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "cohere.core",
        SimpleNamespace(ApiError=type("ApiError", (Exception,), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(APIError=type("APIError", (Exception,), {})),
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"ignored":' + "[" * 1500 + "0" + "]" * 1500 + "}",
        '{"ignored":' + "7" * 5000 + "}",
    ],
)
def test_decoder_resource_failures_are_parse_errors(raw):
    with pytest.raises(ResponseParseError):
        TextTemplate.extract_name(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"ignored":' + "[" * 1500 + "0" + "]" * 1500 + "}",
        '{"ignored":' + "7" * 5000 + "}",
    ],
)
def test_unusable_complete_candidate_does_not_hide_later_valid_answer(raw):
    assert (
        TextTemplate.extract_name(raw + '{"topic_name":"later valid answer"}')
        == "later valid answer"
    )


def test_overlong_mapping_index_is_a_parse_failure():
    with pytest.raises(ResponseParseError):
        TextTemplate.extract_disambiguated_names(
            '{"new_topic_name_mapping":{"' + "7" * 5000 + '":"name"}}'
        )


def test_irrelevant_objects_do_not_repeat_search_to_eof():
    class SearchWorkString(str):
        searched = 0

        def find(self, sub, start=0, end=None):
            result = super().find(sub, start, len(self) if end is None else end)
            self.searched += (result + len(sub) if result >= 0 else len(self)) - start
            return result

    raw = SearchWorkString("{}" * 512 + '{"topic_name":"valid"}')
    assert TextTemplate.extract_name(raw) == "valid"
    # Observe requested search ranges, not elapsed time or implementation text.
    assert raw.searched <= 8 * len(raw)


class CohereSDK:
    """Fake actual SDK boundary; validation and create ownership remain production."""

    def __init__(self, markers=("one",), *, ready=True):
        self.uploaded = []
        self.created = []
        self.cancelled = []
        self.entered = {marker: threading.Event() for marker in markers}
        self.ready = {marker: threading.Event() for marker in markers}
        if ready:
            for event in self.ready.values():
                event.set()
        self.create_started = threading.Event()
        self.create_release = None
        self.datasets = SimpleNamespace(create=self.upload, get=self.dataset)
        self.batches = SimpleNamespace(
            create=self.create, retrieve=self.retrieve, cancel=self.cancel
        )

    def upload(self, *, data, **kwargs):
        rows = [json.loads(line) for line in data.getvalue().decode().splitlines()]
        self.uploaded.append(rows)
        marker = rows[0]["body"]["messages"][-1]["content"]
        return SimpleNamespace(id="input-" + marker)

    def dataset(self, *, id):
        if id.startswith("output-"):
            marker = id.removeprefix("output-")
            return SimpleNamespace(
                dataset=SimpleNamespace(
                    dataset_parts=[
                        SimpleNamespace(url=f"https://fake.invalid/{marker}.avro")
                    ]
                )
            )
        marker = id.removeprefix("input-")
        self.entered[marker].set()
        return SimpleNamespace(
            dataset=SimpleNamespace(
                validation_status=(
                    "validated" if self.ready[marker].is_set() else "pending"
                )
            )
        )

    def create(self, *, request):
        marker = request["input_dataset_id"].removeprefix("input-")
        self.created.append(marker)
        self.create_started.set()
        if self.create_release is not None and not self.create_release.wait(2):
            raise AssertionError("Bounded SDK create gate was not released")
        return SimpleNamespace(batch=SimpleNamespace(id="job-" + marker))

    def retrieve(self, batch_id):
        marker = batch_id.removeprefix("job-")
        return SimpleNamespace(
            batch=SimpleNamespace(
                id=batch_id,
                status="BATCH_STATUS_COMPLETED",
                input_dataset_id="input-" + marker,
                output_dataset_id="output-" + marker,
            )
        )

    def cancel(self, batch_id):
        self.cancelled.append(batch_id)


class AzureSDK:
    def __init__(self):
        self.uploaded = []
        self.files = SimpleNamespace(create=self.upload)
        self.batches = SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(id="azure-job")
        )

    def upload(self, *, file, purpose):
        self.uploaded.append(
            [json.loads(line) for line in file.getvalue().decode().splitlines()]
        )
        return SimpleNamespace(id="input-one")


def batch_wrapper(provider, client, **options):
    if provider == "cohere":
        return wrappers.CohereBatchNamer("dummy", client=client, **options)
    return wrappers.BatchAzureAINamer(
        "dummy", "https://fake.invalid", "gpt-4o", client=client, **options
    )


def test_cohere_mutated_required_schema_fails_before_upload(fake_sdk_imports):
    sdk = CohereSDK()
    wrapper = batch_wrapper("cohere", sdk)
    wrapper.use_json_schema = True
    with pytest.raises(ValueError, match="unsupported|JSON Schema"):
        wrapper.submit_batch([Prompt("s", "one", name_schema(minLength=5))], 0.4, 128)
    assert sdk.uploaded == []


@pytest.mark.parametrize("provider", ["cohere", "azure"])
def test_mutated_disabled_schema_controls_actual_batch_body(provider, fake_sdk_imports):
    sdk = CohereSDK() if provider == "cohere" else AzureSDK()
    wrapper = batch_wrapper(provider, sdk, use_json_schema=True)
    wrapper.use_json_schema = False
    wrapper.submit_batch([Prompt("s", "one", SIMPLE_SCHEMA)], 0.4, 128)
    assert "response_format" not in sdk.uploaded[0][0]["body"]


@pytest.mark.parametrize("provider", ["cohere", "azure"])
def test_mutated_object_mode_controls_actual_batch_body(provider, fake_sdk_imports):
    sdk = CohereSDK() if provider == "cohere" else AzureSDK()
    wrapper = batch_wrapper(provider, sdk)
    wrapper.use_json_object = True
    wrapper.submit_batch([Prompt("s", "one", SIMPLE_SCHEMA)], 0.4, 128)
    assert sdk.uploaded[0][0]["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("provider", ["cohere", "azure"])
def test_mutated_conflicting_modes_fail_before_upload(provider, fake_sdk_imports):
    sdk = CohereSDK() if provider == "cohere" else AzureSDK()
    wrapper = batch_wrapper(provider, sdk)
    wrapper.use_json_schema = True
    wrapper.use_json_object = True
    with pytest.raises(ValueError):
        wrapper.submit_batch([Prompt("s", "one", SIMPLE_SCHEMA)], 0.4, 128)
    assert sdk.uploaded == []


async def entered(event):
    for _ in range(400):
        if event.is_set():
            return
        await asyncio.sleep(0.002)
    raise AssertionError("Fake SDK boundary was not entered within bounded wait")


@pytest.mark.asyncio
async def test_cancel_during_cohere_validation_does_not_create_job(fake_sdk_imports):
    sdk = CohereSDK(ready=False)
    wrapper = batch_wrapper("cohere", sdk, polling_interval=0.05, timeout=0.5)
    task = asyncio.create_task(wrapper.generate_topic_names([Prompt("s", "one")]))
    try:
        await entered(sdk.entered["one"])
        task.cancel("cancel before creating remote batch")
        # The old code creates after this release; cooperative cancellation
        # should already have stopped preparation. No SDK call itself blocks.
        asyncio.get_running_loop().call_later(0.02, sdk.ready["one"].set)
        with pytest.raises(asyncio.CancelledError, match="cancel before creating"):
            await asyncio.wait_for(task, 2)
        assert sdk.created == []
        assert sdk.cancelled == []
    finally:
        sdk.ready["one"].set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_during_actual_cohere_create_keeps_late_id_ownership(
    fake_sdk_imports,
):
    sdk = CohereSDK()
    sdk.create_release = threading.Event()
    wrapper = batch_wrapper("cohere", sdk, polling_interval=0.02, timeout=0.5)
    task = asyncio.create_task(wrapper.generate_topic_names([Prompt("s", "one")]))
    try:
        await entered(sdk.create_started)
        task.cancel("cancel in-flight create")
        await asyncio.sleep(0)
        sdk.create_release.set()
        with pytest.raises(asyncio.CancelledError, match="in-flight create"):
            await asyncio.wait_for(task, 2)
        assert sdk.created == ["one"]
        assert sdk.cancelled == ["job-one"]
    finally:
        sdk.create_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def cohere_output_http(monkeypatch):
    # Real AVRO read and transport retrieval; only HTTP is replaced.
    writer = pytest.importorskip("fastavro").writer

    avro_schema = {
        "type": "record",
        "name": "Output",
        "fields": [
            {"name": "custom_id", "type": "string"},
            {"name": "response", "type": "string"},
        ],
    }

    def response(request):
        marker = request.url.path.removesuffix(".avro").lstrip("/")
        payload = io.BytesIO()
        writer(
            payload,
            avro_schema,
            [
                {
                    "custom_id": "0",
                    "response": json.dumps(
                        {
                            "body": {
                                "finish_reason": "COMPLETE",
                                "message": {
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": json.dumps(
                                                {"topic_name": "name " + marker}
                                            ),
                                        }
                                    ]
                                },
                            }
                        }
                    ),
                }
            ],
        )
        return httpx.Response(200, content=payload.getvalue())

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        yield


@pytest.mark.asyncio
async def test_concurrent_same_wrapper_cancellation_is_isolated(
    fake_sdk_imports, cohere_output_http
):
    sdk = CohereSDK(markers=("cancel", "keep"), ready=False)
    wrapper = batch_wrapper("cohere", sdk, polling_interval=0.05, timeout=0.5)
    cancelled = asyncio.create_task(
        wrapper.generate_topic_names([Prompt("s", "cancel")])
    )
    successful = asyncio.create_task(
        wrapper.generate_topic_names([Prompt("s", "keep")])
    )
    try:
        await entered(sdk.entered["cancel"])
        await entered(sdk.entered["keep"])
        cancelled.cancel("only cancel this submission")
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, sdk.ready["cancel"].set)
        loop.call_later(0.02, sdk.ready["keep"].set)
        with pytest.raises(asyncio.CancelledError, match="only cancel this"):
            await asyncio.wait_for(cancelled, 2)
        assert await asyncio.wait_for(successful, 2) == ["name keep"]
        assert sdk.created == ["keep"]
        assert sdk.cancelled == []
    finally:
        for event in sdk.ready.values():
            event.set()
        for task in (cancelled, successful):
            if not task.done():
                task.cancel()
        await asyncio.gather(cancelled, successful, return_exceptions=True)


class CompletionSDK:
    """The old suite's dependency boundary; sufficient for local preflight tests.

    Actual LiteLLM conversion is separately covered below, not simulated here.
    """

    def __init__(self):
        self.calls = []

    def supports_response_schema(self, model, **kwargs):
        return True

    def get_supported_openai_params(self, model, **kwargs):
        return ["response_format"]

    def get_llm_provider(
        self, model, custom_llm_provider=None, api_base=None, **kwargs
    ):
        provider = custom_llm_provider or (
            model.split("/", 1)[0] if "/" in model else "anthropic"
        )
        return model.removeprefix(provider + "/"), provider, None, api_base

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"topic_name":"valid name"}')
                )
            ]
        )

    async def acompletion(self, **kwargs):
        return self.completion(**kwargs)


def litellm_wrapper(monkeypatch, asynchronous, **options):
    sdk = CompletionSDK()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: sdk)
    cls = wrappers.AsyncLiteLLMNamer if asynchronous else wrappers.LiteLLMNamer
    return cls(api_key="dummy", model="anthropic/claude-sonnet-4-5", **options), sdk


async def name_with(wrapper, asynchronous, prompt):
    if asynchronous:
        return (await wrapper.generate_topic_names([prompt]))[0]
    return wrapper.generate_topic_name(prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "explicit_override", [False, True], ids=["prompt-schema", "explicit-schema"]
)
async def test_required_anthropic_schema_fails_before_completion(
    monkeypatch, asynchronous, explicit_override
):
    required = name_schema(minLength=5)
    options = (
        {
            "provider_kwargs": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "explicit",
                        "strict": True,
                        "schema": required,
                    },
                }
            }
        }
        if explicit_override
        else {"use_json_schema": True}
    )
    wrapper, sdk = litellm_wrapper(monkeypatch, asynchronous, **options)
    prompt = Prompt("s", "u", SIMPLE_SCHEMA if explicit_override else required)
    with pytest.raises(wrappers.InvalidLLMInputError):
        await name_with(wrapper, asynchronous, prompt)
    assert sdk.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "explicit_override", [False, True], ids=["prompt-schema", "explicit-schema"]
)
async def test_required_supported_schema_and_explicit_precedence(
    monkeypatch, asynchronous, explicit_override
):
    explicit = {
        "type": "json_schema",
        "json_schema": {
            "name": "selected",
            "strict": True,
            "schema": deepcopy(SIMPLE_SCHEMA),
        },
    }
    options = (
        {"provider_kwargs": {"response_format": explicit}}
        if explicit_override
        else {"use_json_schema": True}
    )
    wrapper, sdk = litellm_wrapper(monkeypatch, asynchronous, **options)
    # Unsupported *unselected* prompt constraint must not veto explicit schema.
    prompt = Prompt(
        "s", "u", name_schema(minLength=5) if explicit_override else SIMPLE_SCHEMA
    )
    before = deepcopy(explicit)
    assert await name_with(wrapper, asynchronous, prompt) == "valid name"
    assert len(sdk.calls) == 1
    assert sdk.calls[0]["response_format"]["json_schema"]["schema"] == SIMPLE_SCHEMA
    sdk.calls[0]["response_format"]["json_schema"]["schema"]["properties"].clear()
    assert explicit == before
    assert prompt.json_schema["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_required_litellm_oneof_cannot_be_relaxed_to_anyof(
    monkeypatch, asynchronous
):
    schema = name_schema(oneOf=[{"type": "string"}, {"enum": ["valid name"]}])
    wrapper, sdk = litellm_wrapper(monkeypatch, asynchronous, use_json_schema=True)
    with pytest.raises(wrappers.InvalidLLMInputError):
        await name_with(wrapper, asynchronous, Prompt("s", "u", schema))
    assert sdk.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "model,provider_options",
    [
        ("anthropic/claude-sonnet-4-5", {}),
        ("claude-sonnet-4-5", {}),
        ("claude-sonnet-4-5", {"custom_llm_provider": "anthropic"}),
        ("anthropic/claude-sonnet-4-5", {"custom_llm_provider": ""}),
    ],
    ids=["prefixed", "bare", "explicit-provider", "empty-provider-falls-back"],
)
async def test_required_schema_route_resolution(
    monkeypatch, asynchronous, model, provider_options
):
    sdk = CompletionSDK()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: sdk)
    cls = wrappers.AsyncLiteLLMNamer if asynchronous else wrappers.LiteLLMNamer
    wrapper = cls(
        model=model,
        api_key="dummy",
        provider_kwargs=provider_options,
        use_json_schema=True,
    )
    prompt = Prompt("s", "u", name_schema(minLength=5))
    with pytest.raises(wrappers.InvalidLLMInputError):
        if asynchronous:
            await wrapper.generate_topic_names([prompt])
        else:
            wrapper.generate_topic_name(prompt)
    assert sdk.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("constraint", ["minItems", "prefixItems"])
async def test_litellm_only_array_constraints_are_rejected_before_completion(
    monkeypatch, asynchronous, constraint
):
    schema = deepcopy(SIMPLE_SCHEMA)
    schema["properties"]["extra"] = {"type": "array", "items": {"type": "string"}}
    schema["properties"]["extra"][constraint] = (
        1 if constraint == "minItems" else [{"type": "string"}]
    )
    schema["required"].append("extra")
    sdk = CompletionSDK()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: sdk)
    cls = wrappers.AsyncLiteLLMNamer if asynchronous else wrappers.LiteLLMNamer
    wrapper = cls(
        model="anthropic/claude-sonnet-4-5", api_key="dummy", use_json_schema=True
    )
    with pytest.raises(wrappers.InvalidLLMInputError):
        if asynchronous:
            await wrapper.generate_topic_names([Prompt("s", "u", schema)])
        else:
            wrapper.generate_topic_name(Prompt("s", "u", schema))
    assert sdk.calls == []


@pytest.mark.parametrize("error_type", [ValueError, RecursionError])
def test_decoder_error_normalization_does_not_swallow_validator_errors(error_type):
    error = error_type("application validator sentinel")

    def validator(candidate):
        raise error

    with pytest.raises(error_type) as raised:
        extract_response('{"topic_name":"valid"}', validator)
    assert raised.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error_type", [ValueError, RecursionError])
async def test_application_response_parser_errors_are_not_retried(
    monkeypatch, asynchronous, error_type
):
    sdk = CompletionSDK()
    monkeypatch.setattr(wrappers, "_get_litellm", lambda: sdk)
    cls = wrappers.AsyncLiteLLMNamer if asynchronous else wrappers.LiteLLMNamer
    wrapper = cls(model="anthropic/claude-sonnet-4-5", api_key="dummy")
    error = error_type("application response parser sentinel")

    def parser(raw):
        raise error

    with pytest.raises(error_type) as raised:
        if asynchronous:
            await wrapper.generate_topic_names(
                [Prompt("s", "u")], response_parser=parser
            )
        else:
            wrapper.generate_topic_name(Prompt("s", "u"), response_parser=parser)
    assert raised.value is error
    assert len(sdk.calls) == 1
