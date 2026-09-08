import asyncio
import copy
import io
import json
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

import toponymy.provider_batches as provider_batches
from toponymy.provider_batches import (
    AzureBatchTransport,
    BatchItemError,
    BatchProtocolError,
    CohereBatchTransport,
)

PROMPTS = [
    {"system": "Name the topic.", "user": "Café trees 🌳", "combined": "unused"},
    {"system": "", "user": "Oceans", "combined": "unused"},
    {"system": "Name the topic.", "user": "Mountains", "combined": "unused"},
]

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_name": {"type": "string", "description": "Café 🌳"},
        "topic_specificity": {"type": "number"},
    },
    "required": ["topic_name", "topic_specificity"],
    "additionalProperties": False,
}


def azure_row(key, text="A topic", *, error=None, finish_reason="stop"):
    return {
        "custom_id": key,
        "error": error,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {"finish_reason": finish_reason, "message": {"content": text}}
                ]
            },
        },
    }


def cohere_row(key, text="A topic", *, error=None, finish_reason="COMPLETE"):
    return {
        "custom_id": key,
        "error": error,
        "response": {
            "status_code": 200,
            "body": {
                "finish_reason": finish_reason,
                "message": {"content": [{"type": "text", "text": text}]},
            },
        },
    }


class AzureClient:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.error_rows = []
        self.statuses = deque(["completed"])
        self.input_rows = []
        self.calls = []
        self.files = SimpleNamespace(create=self.upload, content=self.content)
        self.batches = SimpleNamespace(
            create=self.create, retrieve=self.retrieve, cancel=self.cancel
        )

    def upload(self, *, file, purpose):
        self.calls.append(("upload", purpose, file.name))
        self.input_rows = [
            json.loads(line) for line in file.read().decode("utf-8").splitlines()
        ]
        return SimpleNamespace(id="input-file")

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return SimpleNamespace(id="azure-batch")

    def retrieve(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        status = self.statuses.popleft() if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, Exception):
            raise status
        return SimpleNamespace(
            id=batch_id,
            status=status,
            input_file_id="input-file",
            output_file_id="output-file",
            error_file_id="error-file" if self.error_rows else None,
        )

    def content(self, file_id):
        self.calls.append(("content", file_id))
        rows = {
            "input-file": self.input_rows,
            "output-file": self.rows,
            "error-file": self.error_rows,
        }[file_id]
        return SimpleNamespace(
            text="\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        )

    def cancel(self, batch_id):
        self.calls.append(("cancel", batch_id))
        return SimpleNamespace(id=batch_id, status="cancelling")


class CohereClient:
    def __init__(self):
        self.calls = []
        self.input_rows = []
        self.validation = deque(["validated"])
        self.statuses = deque(["BATCH_STATUS_COMPLETED"])
        self.datasets = SimpleNamespace(create=self.upload, get=self.dataset)
        self.batches = SimpleNamespace(
            create=self.create, retrieve=self.retrieve, cancel=self.cancel
        )

    def upload(self, **kwargs):
        data = kwargs.pop("data")
        self.calls.append(("upload", kwargs, data.name))
        self.input_rows = [
            json.loads(line) for line in data.read().decode("utf-8").splitlines()
        ]
        return SimpleNamespace(id="input-dataset")

    def dataset(self, *, id):
        self.calls.append(("dataset", id))
        status = (
            self.validation.popleft()
            if len(self.validation) > 1
            else self.validation[0]
        )
        if isinstance(status, Exception):
            raise status
        return SimpleNamespace(
            dataset=SimpleNamespace(
                validation_status=status,
                validation_error="invalid dataset" if status == "failed" else None,
                dataset_parts=[
                    SimpleNamespace(url=f"https://datasets.example.invalid/{id}.avro")
                ],
            )
        )

    def create(self, *, request):
        self.calls.append(("create", request))
        return SimpleNamespace(batch=SimpleNamespace(id="cohere-batch"))

    def retrieve(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        status = self.statuses.popleft() if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, Exception):
            raise status
        return SimpleNamespace(
            batch=SimpleNamespace(
                id=batch_id,
                status=status,
                input_dataset_id="input-dataset",
                output_dataset_id="output-dataset",
            )
        )

    def cancel(self, batch_id):
        self.calls.append(("cancel", batch_id))
        return SimpleNamespace(batch=SimpleNamespace(status="BATCH_STATUS_CANCELING"))


class SDKTimeoutError(asyncio.TimeoutError):
    pass


def azure_transport(client, **kwargs):
    return AzureBatchTransport(
        "fixture",
        "https://resource.openai.azure.com",
        "topic-deployment",
        client=client,
        polling_interval=0.001,
        **kwargs,
    )


def cohere_transport(client, **kwargs):
    return CohereBatchTransport(
        "fixture", client=client, polling_interval=0.001, **kwargs
    )


@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize("use_json_schema", [None, True])
def test_batch_schema_reaches_native_body_unchanged(provider, use_json_schema):
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    transport = factory(
        client, use_json_schema=use_json_schema, supports_json_schema=True
    )
    prompts = [dict(PROMPTS[0], json_schema=copy.deepcopy(TOPIC_SCHEMA))]
    original = copy.deepcopy(prompts)
    transport.submit_batch(prompts, 0.2, 180)
    body = client.input_rows[0]["body"]
    if provider == "azure":
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "toponymy_response",
                "strict": True,
                "schema": TOPIC_SCHEMA,
            },
        }
    else:
        assert body["response_format"] == {
            "type": "json_object",
            "json_schema": TOPIC_SCHEMA,
        }
    assert prompts == original
    prompts[0]["json_schema"]["properties"]["topic_name"]["description"] = "changed"
    assert "changed" not in json.dumps(body["response_format"])


@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize("missing_schema", [False, True])
def test_required_schema_fails_before_any_batch_upload(provider, missing_schema):
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    transport = factory(
        client, use_json_schema=True, supports_json_schema=missing_schema
    )
    prompts = [dict(PROMPTS[0], json_schema=TOPIC_SCHEMA)]
    if missing_schema:
        # A valid first row must not cause an upload before later rows validate.
        prompts.append(PROMPTS[1])
    with pytest.raises(ValueError, match="json_schema|does not support"):
        transport.submit_batch(prompts, 0.2, 180)
    assert client.calls == []


@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({"use_json_schema": False, "supports_json_schema": True}, None),
        ({"supports_json_schema": False}, None),
        (
            {"use_json_object": True, "supports_json_schema": True},
            {"type": "json_object"},
        ),
        (
            {"use_json_object": True, "supports_json_schema": False},
            {"type": "json_object"},
        ),
        ({"use_json_schema": False, "use_json_object": False}, None),
    ],
)
def test_batch_schema_auto_and_explicit_object_policy(provider, options, expected):
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    transport = factory(client, **options)
    transport.submit_batch([dict(PROMPTS[0], json_schema=TOPIC_SCHEMA)], 0.2, 180)
    assert client.input_rows[0]["body"].get("response_format") == expected


@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize(
    "options",
    [
        {"use_json_schema": True, "use_json_object": True},
        {"use_json_schema": "auto"},
        {"use_json_object": 1},
        {"supports_json_schema": "yes"},
    ],
)
def test_batch_schema_option_errors_precede_client_calls(provider, options):
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    with pytest.raises(ValueError):
        factory(client, **options)
    assert client.calls == []


@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize(
    "schema", [[], {"type": "imaginary"}, {"enum": [float("nan")]}]
)
def test_invalid_selected_batch_schema_precedes_upload(provider, schema):
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    transport = factory(client, use_json_schema=True, supports_json_schema=True)
    with pytest.raises(ValueError):
        transport.submit_batch([dict(PROMPTS[0], json_schema=schema)], 0.2, 180)
    assert client.calls == []


def test_opaque_azure_deployment_requires_capability_declaration():
    client = AzureClient()
    transport = azure_transport(client, use_json_schema=True)
    assert transport.supports_json_schema is False
    with pytest.raises(ValueError, match="does not support"):
        transport.submit_batch([dict(PROMPTS[0], json_schema=TOPIC_SCHEMA)], 0.2, 180)
    assert client.calls == []
    known = AzureBatchTransport(
        "fixture",
        "https://resource.openai.azure.com",
        "gpt-4o-2024-08-06",
        client=client,
    )
    assert known.supports_json_schema is True
    disabled = AzureBatchTransport(
        "fixture",
        "https://resource.openai.azure.com",
        "gpt-4o-2024-08-06",
        client=client,
        supports_json_schema=False,
    )
    assert disabled.supports_json_schema is False


def test_cohere_schema_capability_uses_exact_model_identifiers():
    assert cohere_transport(CohereClient()).supports_json_schema is True
    assert (
        cohere_transport(
            CohereClient(), model="custom-command-r-model"
        ).supports_json_schema
        is False
    )


def test_cohere_current_sdk_matches_batch_schema_field():
    sdk = pytest.importorskip("cohere.types.response_format_v2")
    client = CohereClient()
    transport = cohere_transport(client)
    transport.submit_batch([dict(PROMPTS[0], json_schema=TOPIC_SCHEMA)], 0.2, 180)
    response_format = client.input_rows[0]["body"]["response_format"]
    sdk_format = sdk.JsonObjectResponseFormatV2(**response_format)
    assert sdk_format.json_schema == TOPIC_SCHEMA
    assert "json_schema" in response_format
    assert "schema" not in response_format


@pytest.mark.parametrize(
    "constraint",
    [
        {"minimum": 0},
        {"maximum": 1},
        {"allOf": []},
        {"minItems": 1},
        {"pattern": "^name$"},
    ],
)
def test_cohere_unsupported_constraints_fail_required_and_preserve_auto_prompt(
    constraint,
):
    schema = copy.deepcopy(TOPIC_SCHEMA)
    # Use a valid allOf schema so the provider-limit check is the failing boundary.
    if "allOf" in constraint:
        constraint = {"allOf": [{"type": "number"}]}
    schema["properties"]["topic_specificity"].update(constraint)
    prompts = [dict(PROMPTS[0], json_schema=schema)]
    original = copy.deepcopy(prompts)
    client = CohereClient()
    with pytest.raises(ValueError, match="Cohere batch JSON Schema is unsupported"):
        cohere_transport(client, use_json_schema=True).submit_batch(prompts, 0.2, 180)
    assert client.calls == []
    cohere_transport(client).submit_batch(prompts, 0.2, 180)
    assert "response_format" not in client.input_rows[0]["body"]
    assert prompts == original


def test_cohere_schema_property_names_are_not_treated_as_keywords():
    schema = {
        "type": "object",
        "properties": {"minimum": {"type": "string"}},
        "required": ["minimum"],
        "additionalProperties": False,
    }
    client = CohereClient()
    cohere_transport(client, use_json_schema=True).submit_batch(
        [dict(PROMPTS[0], json_schema=schema)], 0.2, 180
    )
    assert client.input_rows[0]["body"]["response_format"]["json_schema"] == schema


@pytest.fixture
def avro_download(monkeypatch):
    writer = pytest.importorskip("fastavro").writer
    downloads = {}
    calls = []

    def add(dataset_id, rows):
        schema = {
            "type": "record",
            "name": "BatchRecord",
            "fields": [
                {"name": "custom_id", "type": "string"},
                {"name": "response", "type": ["null", "string"]},
                {"name": "error", "type": ["null", "string"]},
            ],
        }
        encoded = [
            {
                "custom_id": row["custom_id"],
                "response": (
                    json.dumps(row["response"])
                    if row.get("response") is not None
                    else None
                ),
                "error": (
                    json.dumps(row["error"]) if row.get("error") is not None else None
                ),
            }
            for row in rows
        ]
        output = io.BytesIO()
        writer(output, schema, encoded)
        downloads[f"https://datasets.example.invalid/{dataset_id}.avro"] = (
            output.getvalue()
        )

    @contextmanager
    def stream(method, url, *, timeout):
        calls.append((method, url, timeout))
        response = httpx.Response(
            200, content=downloads[url], request=httpx.Request(method, url)
        )
        yield response
        response.close()

    monkeypatch.setattr(httpx, "stream", stream)
    return add, calls


def test_azure_submission_preserves_system_unicode_and_deployment_without_mutation():
    original = copy.deepcopy(PROMPTS)
    client = AzureClient()
    transport = azure_transport(client)
    assert transport.submit_batch(PROMPTS, 0.2, 180) == "azure-batch"
    assert PROMPTS == original
    assert client.input_rows[0] == {
        "custom_id": "0",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": "topic-deployment",
            "temperature": 0.2,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": "Name the topic."},
                {"role": "user", "content": "Café trees 🌳"},
            ],
        },
    }
    assert client.input_rows[1]["body"]["messages"] == [
        {"role": "user", "content": "Oceans"}
    ]
    assert client.calls == [
        ("upload", "batch", "toponymy-batch.jsonl"),
        (
            "create",
            {
                "input_file_id": "input-file",
                "endpoint": "/chat/completions",
                "completion_window": "24h",
            },
        ),
    ]


def test_cohere_submission_uses_validated_dataset_and_real_batch_request():
    original = copy.deepcopy(PROMPTS)
    client = CohereClient()
    client.validation = deque(["validating", "validated"])
    transport = cohere_transport(client)
    assert transport.submit_batch(PROMPTS, 0.3, 90) == "cohere-batch"
    assert PROMPTS == original
    assert client.input_rows[0] == {
        "custom_id": "0",
        "body": {
            "temperature": 0.3,
            "max_tokens": 90,
            "messages": [
                {"role": "system", "content": "Name the topic."},
                {"role": "user", "content": "Café trees 🌳"},
            ],
        },
    }
    assert client.calls[0][1] == {
        "name": "toponymy-chat-input",
        "type": "batch-chat-v2-input",
        "keep_fields": ["custom_id"],
        "skip_malformed_input": False,
    }
    assert client.calls[-1] == (
        "create",
        {
            "name": "toponymy-chat-batch",
            "input_dataset_id": "input-dataset",
            "model": "command-r-08-2024",
        },
    )
    assert sum(call[0] == "dataset" for call in client.calls) == 2


@pytest.mark.parametrize(
    "transport_factory,client_factory",
    [(azure_transport, AzureClient), (cohere_transport, CohereClient)],
)
@pytest.mark.parametrize(
    "prompts,temperature,tokens",
    [
        ([], 0.2, 100),
        (PROMPTS, float("nan"), 100),
        (PROMPTS, 0.2, 0),
        ([{"system": [], "user": "x"}], 0.2, 100),
    ],
)
def test_invalid_inputs_make_no_provider_requests(
    transport_factory, client_factory, prompts, temperature, tokens
):
    client = client_factory()
    transport = transport_factory(client)
    with pytest.raises(ValueError):
        transport.submit_batch(prompts, temperature, tokens)
    assert client.calls == []


def test_cohere_failed_dataset_validation_does_not_create_batch():
    client = CohereClient()
    client.validation = deque(["failed"])
    with pytest.raises(BatchProtocolError, match="validation"):
        cohere_transport(client).submit_batch(PROMPTS, 0.2, 100)
    assert all(call[0] != "create" for call in client.calls)


def test_cohere_dataset_poll_retries_read_without_reupload():
    client = CohereClient()
    client.validation = deque([http_failure(503), "validated"])
    assert cohere_transport(client).submit_batch(PROMPTS, 0.2, 100) == "cohere-batch"
    assert [call[0] for call in client.calls] == [
        "upload",
        "dataset",
        "dataset",
        "create",
    ]


@pytest.mark.asyncio
async def test_azure_results_join_success_and_error_files_in_original_order():
    client = AzureClient([azure_row("2", "Mountains"), azure_row("0", "Café 🌳")])
    client.error_rows = [
        {
            "custom_id": "1",
            "response": None,
            "error": {"code": "invalid_request", "message": "failed item"},
        }
    ]
    transport = azure_transport(client)
    batch_id = transport.submit_batch(PROMPTS, 0.2, 100)
    results = await transport.retrieve_batch_text_results(batch_id)
    assert results[0] == "Café 🌳"
    assert isinstance(results[1], BatchItemError)
    assert results[1].custom_id == "1"
    assert results[1].status_code == 400
    assert results[2] == "Mountains"
    assert ("content", "input-file") not in client.calls
    assert sum(call[0] == "create" for call in client.calls) == 1


@pytest.mark.asyncio
async def test_cohere_downloads_avro_and_aligns_errors(avro_download):
    add, calls = avro_download
    add(
        "output-dataset",
        [
            cohere_row("2", "Mountain"),
            cohere_row("0", "Café 🌳"),
            cohere_row("1", error={"code": "bad_request"}),
        ],
    )
    client = CohereClient()
    transport = cohere_transport(client)
    batch_id = transport.submit_batch(PROMPTS, 0.2, 100)
    results = await transport.retrieve_batch_text_results(batch_id)
    assert results[0] == "Café 🌳"
    assert isinstance(results[1], BatchItemError)
    assert results[2] == "Mountain"
    assert calls == [
        ("GET", "https://datasets.example.invalid/output-dataset.avro", 30)
    ]


@pytest.mark.asyncio
async def test_missing_and_duplicate_results_remain_explicit_per_item_errors():
    client = AzureClient([azure_row("0", "first"), azure_row("0", "duplicate")])
    transport = azure_transport(client)
    batch_id = transport.submit_batch(PROMPTS, 0.2, 100)
    results = await transport.retrieve_batch_text_results(batch_id)
    assert len(results) == 3
    assert all(isinstance(result, BatchItemError) for result in results)
    assert "duplicate" in str(results[0])
    assert "missing" in str(results[1])


@pytest.mark.asyncio
async def test_unknown_result_id_rejects_unalignable_provider_data():
    client = AzureClient([azure_row("other-batch", "wrong")])
    transport = azure_transport(client)
    batch_id = transport.submit_batch(PROMPTS, 0.2, 100)
    with pytest.raises(BatchProtocolError, match="unknown custom_id"):
        await transport.retrieve_batch_text_results(batch_id)


@pytest.mark.asyncio
async def test_fresh_transport_restores_input_order_from_azure_file():
    client = AzureClient([azure_row("third", "last"), azure_row("first", "start")])
    client.input_rows = [{"custom_id": "first"}, {"custom_id": "third"}]
    results = await azure_transport(client).retrieve_batch_text_results(
        "existing-batch"
    )
    assert results == ["start", "last"]
    assert ("content", "input-file") in client.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,finish_reason",
    [(None, "stop"), (17, "stop"), ("", "stop"), ("partial", "length"), ("valid", [])],
)
async def test_invalid_and_truncated_azure_outputs_are_item_errors(text, finish_reason):
    client = AzureClient([azure_row("0", text, finish_reason=finish_reason)])
    transport = azure_transport(client)
    batch_id = transport.submit_batch(PROMPTS[:1], 0.2, 100)
    results = await transport.retrieve_batch_text_results(batch_id)
    assert len(results) == 1
    assert isinstance(results[0], BatchItemError)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_per_item_http_errors_preserve_fail_fast_status(status):
    row = azure_row("0")
    row["response"]["status_code"] = status
    client = AzureClient([row])
    transport = azure_transport(client)
    batch_id = transport.submit_batch(PROMPTS[:1], 0.2, 100)
    result = (await transport.retrieve_batch_text_results(batch_id))[0]
    assert isinstance(result, BatchItemError)
    assert result.status_code == status


def http_failure(status):
    request = httpx.Request("GET", "https://provider.example.invalid/batches/batch")
    return httpx.HTTPStatusError(
        "provider failure",
        request=request,
        response=httpx.Response(status, request=request),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory,client_factory,complete,pending",
    [
        (azure_transport, AzureClient, "completed", "in_progress"),
        (
            cohere_transport,
            CohereClient,
            "BATCH_STATUS_COMPLETED",
            "BATCH_STATUS_QUEUED",
        ),
    ],
)
async def test_polling_retries_transient_failures_without_resubmitting(
    factory, client_factory, complete, pending
):
    client = client_factory()
    client.statuses = deque([http_failure(503), pending, complete])
    assert await factory(client).wait_for_completion("batch") is True
    assert [call[0] for call in client.calls] == ["retrieve", "retrieve", "retrieve"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_polling_authentication_and_invalid_input_errors_fail_immediately(status):
    client = AzureClient()
    client.statuses = deque([http_failure(status), "completed"])
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await azure_transport(client).wait_for_completion("batch")
    assert raised.value.response.status_code == status
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_poll_retry_budget_is_not_reset_by_intermediate_pending_status():
    client = AzureClient()
    client.statuses = deque(
        [http_failure(503), "in_progress", http_failure(429), "completed"]
    )
    with pytest.raises(httpx.HTTPStatusError):
        await azure_transport(client, max_poll_retries=1).wait_for_completion("batch")
    assert len(client.calls) == 3


@pytest.fixture
def fixed_polling_clock(monkeypatch):
    # Keep the polling clock before its deadline while the real asyncio timer
    # expires. Event-loop timer resolution can produce this ordering on Windows.
    api = SimpleNamespace(
        **{
            name: getattr(asyncio, name)
            for name in (
                "create_task",
                "wait_for",
                "to_thread",
                "sleep",
                "TimeoutError",
            )
        },
        get_running_loop=lambda: SimpleNamespace(time=lambda: 100.0),
    )
    monkeypatch.setattr(provider_batches, "asyncio", api)
    return api


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize("request_timeout", [1.0, 2.0])
async def test_owned_overall_expiry_does_not_require_clock_to_reach_deadline(
    provider, request_timeout, fixed_polling_clock
):
    timers = []

    async def expire_early(awaitable, *, timeout):
        timers.append(timeout)
        return await asyncio.wait_for(awaitable, timeout=0)

    fixed_polling_clock.wait_for = expire_early
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    transport = factory(
        client, timeout=1.0, request_timeout=request_timeout, max_poll_retries=0
    )
    assert await transport.wait_for_completion("batch") is False
    assert timers == [1.0]
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("timeout_type", [asyncio.TimeoutError, SDKTimeoutError])
async def test_sdk_timeout_keeps_its_retry_budget_and_exception_identity(
    provider, complete, timeout_type, fixed_polling_clock
):
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    completed = "completed" if provider == "azure" else "BATCH_STATUS_COMPLETED"
    pending = "in_progress" if provider == "azure" else "BATCH_STATUS_QUEUED"
    error = timeout_type("SDK request timeout")
    cause = asyncio.CancelledError("SDK internal cancellation")
    error.__cause__ = cause
    client.statuses = deque(
        [error, pending, completed] if complete else [error, pending, error, completed]
    )
    # The overall deadline limits every request here. An SDK TimeoutError must
    # still retry and then propagate; its type or cause does not prove ownership.
    transport = factory(client, timeout=10.0, request_timeout=20.0, max_poll_retries=1)
    if complete:
        assert await transport.wait_for_completion("batch") is True
    else:
        with pytest.raises(asyncio.TimeoutError) as raised:
            await transport.wait_for_completion("batch")
        assert type(raised.value) is timeout_type
        assert str(raised.value) == "SDK request timeout"
        # CPython converts exact built-in timeouts at the thread-future boundary.
        # SDK subclasses retain their identity and cause across that boundary.
        if timeout_type is SDKTimeoutError:
            assert raised.value is error
            assert raised.value.__cause__ is cause
    assert client.calls == [("retrieve", "batch")] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize("complete", [False, True])
async def test_owned_request_expiry_retries_without_becoming_overall_expiry(
    provider, complete, fixed_polling_clock
):
    timers = []

    async def expire_requests(awaitable, *, timeout):
        timers.append(timeout)
        if complete and len(timers) == 2:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        return await asyncio.wait_for(awaitable, timeout=0)

    fixed_polling_clock.wait_for = expire_requests
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    transport = factory(client, timeout=10.0, request_timeout=1.0, max_poll_retries=1)
    if complete:
        assert await transport.wait_for_completion("batch") is True
        assert client.calls == [("retrieve", "batch")]
    else:
        with pytest.raises(asyncio.TimeoutError):
            await transport.wait_for_completion("batch")
        assert client.calls == []
    assert timers == [1.0, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["azure", "cohere"])
async def test_poll_cancellation_propagates_while_status_request_is_awaited(
    provider, fixed_polling_clock
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def waiting_call(function, *args):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    fixed_polling_clock.to_thread = waiting_call
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    waiter = asyncio.create_task(factory(client).wait_for_completion("batch"))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert stopped.is_set()
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["azure", "cohere"])
@pytest.mark.parametrize("outcome", ["failed", "unknown", "pending_deadline"])
async def test_status_outcomes_remain_distinct_from_request_expiry(
    provider, outcome, fixed_polling_clock
):
    clock = [100.0]

    async def reach_deadline(delay):
        clock[0] += 1.0

    fixed_polling_clock.get_running_loop = lambda: SimpleNamespace(
        time=lambda: clock[0]
    )
    fixed_polling_clock.sleep = reach_deadline
    client = AzureClient() if provider == "azure" else CohereClient()
    factory = azure_transport if provider == "azure" else cohere_transport
    status = {
        "azure": {"failed": "failed", "pending_deadline": "in_progress"},
        "cohere": {
            "failed": "BATCH_STATUS_FAILED",
            "pending_deadline": "BATCH_STATUS_QUEUED",
        },
    }[provider].get(outcome, "undocumented_status")
    client.statuses = deque([status])
    transport = factory(client, timeout=1.0)
    if outcome == "unknown":
        with pytest.raises(BatchProtocolError, match="Unknown batch status"):
            await transport.wait_for_completion("batch")
    else:
        assert await transport.wait_for_completion("batch") is False
    assert client.calls == [("retrieve", "batch")]


@pytest.mark.asyncio
async def test_wait_timeout_is_bounded_and_cancellation_propagates():
    client = AzureClient()
    client.statuses = deque(["in_progress"])
    assert (
        await azure_transport(client, timeout=0.01).wait_for_completion("batch")
        is False
    )
    waiter = asyncio.create_task(azure_transport(client).wait_for_completion("batch"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert all(call[0] != "create" for call in client.calls)


@pytest.mark.parametrize(
    "factory,client_factory",
    [(azure_transport, AzureClient), (cohere_transport, CohereClient)],
)
def test_cancel_uses_own_provider_batch_api(factory, client_factory):
    client = client_factory()
    factory(client).cancel_batch("to-cancel")
    assert client.calls == [("cancel", "to-cancel")]


@pytest.mark.asyncio
async def test_cohere_output_download_respects_size_bound(avro_download):
    add, _ = avro_download
    add("output-dataset", [cohere_row("0")])
    transport = cohere_transport(CohereClient(), max_download_bytes=8)
    batch_id = transport.submit_batch(PROMPTS[:1], 0.2, 100)
    with pytest.raises(BatchProtocolError, match="max_download_bytes"):
        await transport.retrieve_batch_text_results(batch_id)


def test_azure_sdk_construction_uses_requested_endpoint_and_bounded_requests(
    monkeypatch,
):
    sdk = pytest.importorskip("openai")
    client = AzureClient()
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        return client

    monkeypatch.setattr(sdk, "AzureOpenAI", construct)
    transport = AzureBatchTransport(
        "fixture",
        "https://named-resource.openai.azure.com",
        "deployment",
        request_timeout=12,
    )
    assert transport.client is client
    assert calls == [
        {
            "api_key": "fixture",
            "azure_endpoint": "https://named-resource.openai.azure.com",
            "api_version": "2024-10-21",
            "timeout": 12,
            "max_retries": 0,
        }
    ]


def test_real_cohere_sdk_serializes_dataset_and_batch_requests_with_mock_http():
    sdk = pytest.importorskip("cohere")
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/datasets"):
            assert b"Caf" in request.content
            return httpx.Response(200, json={"id": "dataset-fixture"})
        if request.method == "GET" and request.url.path.endswith(
            "/datasets/dataset-fixture"
        ):
            return httpx.Response(
                200,
                json={
                    "dataset": {
                        "id": "dataset-fixture",
                        "validation_status": "validated",
                    }
                },
            )
        if request.method == "POST" and request.url.path.endswith("/batches"):
            assert json.loads(request.content) == {
                "name": "toponymy-chat-batch",
                "input_dataset_id": "dataset-fixture",
                "model": "command-r-08-2024",
            }
            return httpx.Response(200, json={"batch": {"id": "batch-fixture"}})
        raise AssertionError(f"Unexpected SDK request {request.method} {request.url}")

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = sdk.ClientV2(
            api_key="fixture", httpx_client=http_client, max_retries=0
        )
        transport = cohere_transport(client)
        assert transport.submit_batch(PROMPTS[:1], 0.2, 100) == "batch-fixture"
    assert len(requests) == 3
