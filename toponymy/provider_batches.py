"""Cohere dataset batches and Azure OpenAI JSONL batches.

Submission and explicit status/cancellation calls are synchronous SDK operations.
Async polling and retrieval offload those operations and preserve cancellation.
Provider SDKs are optional until a transport is constructed.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from contextvars import ContextVar
from threading import Event
from typing import TypeVar

import httpx

T = TypeVar("T")
Prompt = Mapping[str, object]
Record = Mapping[str, object]

_submission_cancel_event: ContextVar[Event | None] = ContextVar(
    "toponymy_batch_submission_cancel_event", default=None
)


class _BatchPreparationCancelled(Exception):
    """The caller cancelled before remote batch creation was dispatched."""


def _check_submission_cancelled(event):
    if event is not None and event.is_set():
        raise _BatchPreparationCancelled()


# Exact documented model identifiers only: a custom Azure deployment name does
# not identify its underlying model. Callers can declare a known deployment's
# capability explicitly without a model-discovery request.
_AZURE_SCHEMA_MODELS = frozenset(
    {
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
    }
)
_COHERE_SCHEMA_MODELS = frozenset(
    {
        "command-r",
        "command-r-08-2024",
        "command-r-plus",
        "command-r-plus-08-2024",
        "command-r7b-12-2024",
        "command-a-03-2025",
    }
)


class BatchProtocolError(ValueError):
    """A provider response cannot be associated with the submitted batch."""

    def __init__(self, message: str, *, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class BatchItemError(RuntimeError):
    """One submitted request failed or has a missing/invalid result."""

    def __init__(self, custom_id: str, message: str, *, status_code: int | None = None):
        self.custom_id = custom_id
        self.status_code = status_code
        super().__init__(f"Batch request {custom_id}: {message}")


def _field(value: object, name: str, default: object = None) -> object:
    return (
        value.get(name, default)
        if isinstance(value, Mapping)
        else getattr(value, name, default)
    )


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BatchProtocolError(f"{label} must be a nonempty string")
    return value


def _object(value: object, label: str) -> Record:
    # AVRO records may encode nested response bodies as JSON strings.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise BatchProtocolError(f"{label} contains invalid JSON") from error
    if not isinstance(value, Mapping):
        raise BatchProtocolError(f"{label} must be an object")
    return value


def _messages(prompt: Prompt) -> list[dict[str, str]]:
    if not isinstance(prompt, Mapping):
        raise ValueError("Batch transports require normalized system/user prompts")
    system, user = prompt.get("system", ""), prompt.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError("Normalized prompt system and user fields must be strings")
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": user}]


def _schema_policy(use_json_schema, use_json_object, capability, model, *, azure):
    for name, value in (
        ("use_json_schema", use_json_schema),
        ("use_json_object", use_json_object),
        ("supports_json_schema", capability),
    ):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean or None")
    if use_json_schema is True and use_json_object is True:
        raise ValueError("use_json_schema and use_json_object cannot both be True")
    if capability is not None:
        return capability
    known_models = _AZURE_SCHEMA_MODELS if azure else _COHERE_SCHEMA_MODELS
    return model in known_models


def _cohere_schema_problem(schema):
    """Check documented limitations without rewriting the requested schema."""
    if schema.get("type") != "object":
        return "the root type must be object"
    unsupported = {
        "allOf",
        "oneOf",
        "not",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "uniqueItems",
    }
    pending = [schema]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        rejected = unsupported.intersection(node)
        if rejected:
            return f"unsupported keyword {sorted(rejected)[0]}"
        if "format" in node and node["format"] not in {
            "date-time",
            "uuid",
            "date",
            "time",
        }:
            return f"unsupported format {node['format']}"
        if "pattern" in node and any(
            token in node["pattern"] for token in ("^", "$", "?=", "?!")
        ):
            return "unsupported pattern anchor or lookahead"
        # Traverse schemas, not arbitrary values or names in properties/enum.
        for key in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ):
            pending.extend(node.get(key, {}).values())
        for key in ("items", "additionalProperties", "contains", "propertyNames"):
            child = node.get(key)
            pending.extend(child if isinstance(child, list) else [child])
        pending.extend(node.get("anyOf", []))
    return None


def _response_format(
    prompt, *, azure, use_json_schema, use_json_object, supports_json_schema
):
    schema = prompt.get("json_schema")
    if use_json_schema is True:
        if schema is None:
            raise ValueError("use_json_schema=True requires a prompt json_schema")
        if not supports_json_schema:
            raise ValueError("The configured batch model does not support JSON Schema")
    if use_json_object is True:
        return {"type": "json_object"}
    if schema is None or use_json_schema is False or not supports_json_schema:
        return None
    if not isinstance(schema, dict):
        raise ValueError("prompt json_schema must be a JSON Schema object")
    from jsonschema import Draft202012Validator, SchemaError

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"Invalid prompt json_schema: {error.message}") from error
    if not azure and (problem := _cohere_schema_problem(schema)) is not None:
        if use_json_schema is True:
            raise ValueError(f"Cohere batch JSON Schema is unsupported: {problem}")
        return None
    schema = deepcopy(schema)
    if azure:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "toponymy_response",
                "strict": True,
                "schema": schema,
            },
        }
    # Current /v2/chat reference and cohere 7.0.8 ResponseFormatV2 use
    # json_schema; the older structured-output guide calls this field schema.
    return {"type": "json_object", "json_schema": schema}


def _requests(
    prompts: Sequence[Prompt],
    temperature: float,
    max_tokens: int,
    model: str,
    *,
    azure: bool,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
    supports_json_schema: bool = False,
) -> list[dict[str, object]]:
    _schema_policy(
        use_json_schema, use_json_object, supports_json_schema, model, azure=azure
    )
    if not prompts:
        raise ValueError("Cannot submit an empty batch")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
        or (azure and temperature > 2)
    ):
        raise ValueError(
            "temperature must be finite and nonnegative, and at most 2 for Azure"
        )
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    rows = []
    for index, prompt in enumerate(prompts):
        body = {
            "messages": _messages(prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response_format = _response_format(
            prompt,
            azure=azure,
            use_json_schema=use_json_schema,
            use_json_object=use_json_object,
            supports_json_schema=supports_json_schema,
        )
        if response_format is not None:
            body["response_format"] = response_format
        row = {"custom_id": str(index), "body": body}
        if azure:
            body["model"] = model
            row.update(method="POST", url="/chat/completions")
        rows.append(row)
    return rows


def _jsonl(rows: Iterable[Record]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    ).encode("utf-8")


def _jsonl_records(text: str) -> list[Record]:
    return [
        _object(line, "Batch JSONL record")
        for line in text.splitlines()
        if line.strip()
    ]


def _response(row: Record) -> Record:
    if row.get("error") is not None:
        error = row["error"]
        if isinstance(error, str):
            try:
                error = json.loads(error)
            except json.JSONDecodeError:
                pass
        status = None
        if isinstance(error, Mapping):
            code = error.get("code", error.get("type"))
            status = error.get("status_code")
            if not isinstance(status, int) or isinstance(status, bool):
                known_status = {
                    "invalid_request": 400,
                    "invalid_request_error": 400,
                    "bad_request": 400,
                    "authentication_error": 401,
                    "invalid_api_key": 401,
                    "unauthorized": 401,
                    "permission_error": 403,
                    "permission_denied": 403,
                    "not_found": 404,
                    "not_found_error": 404,
                    "unprocessable_entity": 422,
                    "rate_limit_error": 429,
                }
                status = known_status.get(code) if isinstance(code, str) else None
        raise BatchProtocolError(f"provider error: {error}", status_code=status)
    response = _object(row.get("response"), "Batch response")
    status = response.get("status_code", 200)
    if isinstance(status, bool) or not isinstance(status, int):
        raise BatchProtocolError("response status_code must be an integer")
    if not 200 <= status < 300:
        raise BatchProtocolError(
            f"provider returned HTTP {status}: {response.get('body')}",
            status_code=status,
        )
    return _object(response.get("body", response), "Batch response body")


def _cohere_text(row: Record) -> str:
    body = _response(row)
    finish_reason = body.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise BatchProtocolError("Cohere finish_reason must be a string")
    if finish_reason in {"MAX_TOKENS", "ERROR", "ERROR_TOXIC"}:
        raise BatchProtocolError(f"incomplete generation: {body['finish_reason']}")
    message = _object(body.get("message"), "Cohere response message")
    content = message.get("content")
    if not isinstance(content, list):
        raise BatchProtocolError("Cohere response content must be a list")
    parts = []
    for block in content:
        block = _object(block, "Cohere content block")
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise BatchProtocolError("Cohere text block must contain a string")
            parts.append(text)
    if not parts or not "".join(parts).strip():
        raise BatchProtocolError("Cohere response contains no text")
    return "".join(parts)


def _azure_text(row: Record) -> str:
    body = _response(row)
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BatchProtocolError("Azure response must contain exactly one choice")
    choice = _object(choices[0], "Azure choice")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise BatchProtocolError("Azure finish_reason must be a string")
    if finish_reason in {"length", "content_filter", "tool_calls"}:
        raise BatchProtocolError(f"incomplete generation: {choice['finish_reason']}")
    message = _object(choice.get("message"), "Azure response message")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise BatchProtocolError("Azure response content must be a string")
    return text


def _aligned_results(
    rows: Iterable[Record],
    expected: Sequence[str],
    text_from_row: Callable[[Record], str],
) -> list[str | Exception]:
    positions = {custom_id: index for index, custom_id in enumerate(expected)}
    if len(positions) != len(expected):
        raise BatchProtocolError("Submitted input contains duplicate custom_id values")
    results: list[str | Exception] = [
        BatchItemError(key, "missing result") for key in expected
    ]
    seen = set()
    for value in rows:
        row = _object(value, "Batch result record")
        key = _identifier(row.get("custom_id"), "Result custom_id")
        if key not in positions:
            raise BatchProtocolError(f"Result has unknown custom_id {key!r}")
        position = positions[key]
        if key in seen:
            results[position] = BatchItemError(key, "duplicate result custom_id")
            continue
        seen.add(key)
        try:
            results[position] = text_from_row(row)
        except BatchProtocolError as error:
            results[position] = BatchItemError(
                key, str(error), status_code=error.status_code
            )
    return results


def _settings(
    model: str,
    polling_interval: float,
    timeout: float,
    request_timeout: float,
    max_poll_retries: int,
) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    for name, value in (
        ("polling_interval", polling_interval),
        ("timeout", timeout),
        ("request_timeout", request_timeout),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(max_poll_retries, bool)
        or not isinstance(max_poll_retries, int)
        or max_poll_retries < 0
    ):
        raise ValueError("max_poll_retries must be a nonnegative integer")


def _transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
    if status is not None:
        return status in {408, 429, 500, 502, 503, 504}
    return isinstance(
        error,
        (TimeoutError, asyncio.TimeoutError, ConnectionError, httpx.TransportError),
    ) or isinstance(
        error.__cause__,
        (TimeoutError, asyncio.TimeoutError, ConnectionError, httpx.TransportError),
    )


class _ThreadCallTimeout(asyncio.TimeoutError):
    """The local await expired, rather than the SDK raising its own timeout."""


async def _thread_call(function: Callable[..., T], *args: object, timeout: float) -> T:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError as error:
        if task.cancelled():
            raise _ThreadCallTimeout() from error
        raise


async def _wait_for_status(
    get_status: Callable[[str], str],
    batch_id: str,
    *,
    timeout: float,
    request_timeout: float,
    interval: float,
    retries: int,
    errors: tuple[type[Exception], ...],
    completed: str,
    pending: set[str],
    failed: set[str],
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    remaining_retries = retries
    while (remaining := deadline - loop.time()) > 0:
        try:
            status = await _thread_call(
                get_status, batch_id, timeout=min(request_timeout, remaining)
            )
        except errors as error:
            # A timer may fire before a separate clock read reaches its deadline.
            # Only our own deadline-limited await proves overall expiry; an SDK
            # timeout still follows the transient-error policy below.
            if (
                isinstance(error, _ThreadCallTimeout) and remaining <= request_timeout
            ) or deadline <= loop.time():
                return False
            if not _transient(error) or remaining_retries == 0:
                raise
            remaining_retries -= 1
        else:
            if status == completed:
                return True
            if status in failed:
                return False
            if status not in pending:
                raise BatchProtocolError(f"Unknown batch status {status!r}")
        await asyncio.sleep(min(interval, max(0, deadline - loop.time())))
    return False


def _remember(sizes: OrderedDict[str, int], batch_id: str, count: int) -> None:
    sizes[batch_id] = count
    sizes.move_to_end(batch_id)
    if len(sizes) > 128:
        sizes.popitem(last=False)


class CohereBatchTransport:
    """Use Cohere's batch-chat-v2-input datasets and /v2/batches endpoints.

    Injected SDK clients must provide their own bounded HTTP timeouts. Submission
    waits for dataset validation; async callers should offload submission. The
    SDK's AVRO dependency is imported only when output records are downloaded.

    use_json_schema=True requires a schema on every prompt and a supported model
    before uploading anything. None uses a supplied schema for known models;
    otherwise requests stay unformatted unless use_json_object=True. False
    disables schema use. Both flags cannot be True. A capability override applies
    to the configured model; server-side schema limitations still apply. Schemas
    are preserved without deleting unsupported constraints. Known unsupported
    Cohere constraints fail in required mode and use plain requests in auto mode.
    JSON-object prompts
    should explicitly ask the model to generate JSON, as required by Cohere.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "command-r-08-2024",
        polling_interval: float = 60,
        timeout: float = 7200,
        *,
        client: object = None,
        request_timeout: float = 30,
        max_poll_retries: int = 2,
        max_download_bytes: int = 64 * 1024 * 1024,
        use_json_schema: bool | None = None,
        use_json_object: bool | None = None,
        supports_json_schema: bool | None = None,
    ):
        _settings(model, polling_interval, timeout, request_timeout, max_poll_retries)
        self.supports_json_schema = _schema_policy(
            use_json_schema, use_json_object, supports_json_schema, model, azure=False
        )
        self.use_json_schema = use_json_schema
        self.use_json_object = use_json_object
        if (
            isinstance(max_download_bytes, bool)
            or not isinstance(max_download_bytes, int)
            or max_download_bytes <= 0
        ):
            raise ValueError("max_download_bytes must be a positive integer")
        self._poll_errors = (
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
            httpx.HTTPError,
        )
        try:
            from cohere.core import ApiError
        except ImportError:
            if client is None:
                raise ImportError("Cohere batches require cohere>=7.0.8") from None
        else:
            self._poll_errors += (ApiError,)
        if client is None:
            from cohere import ClientV2

            client = ClientV2(api_key=api_key, timeout=request_timeout, max_retries=0)
        self.client = client
        self.model = model
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.max_poll_retries = max_poll_retries
        self.max_download_bytes = max_download_bytes
        self._batch_sizes: OrderedDict[str, int] = OrderedDict()

    def submit_batch(
        self, prompts: Sequence[Prompt], temperature: float, max_tokens: int
    ) -> str:
        cancel_event = _submission_cancel_event.get()
        _check_submission_cancelled(cancel_event)
        rows = _requests(
            prompts,
            temperature,
            max_tokens,
            self.model,
            azure=False,
            use_json_schema=self.use_json_schema,
            use_json_object=self.use_json_object,
            supports_json_schema=self.supports_json_schema,
        )
        _check_submission_cancelled(cancel_event)
        with io.BytesIO(_jsonl(rows)) as data:
            data.name = "toponymy-batch.jsonl"
            uploaded = self.client.datasets.create(
                name="toponymy-chat-input",
                data=data,
                type="batch-chat-v2-input",
                keep_fields=["custom_id"],
                skip_malformed_input=False,
            )
        dataset_id = _identifier(_field(uploaded, "id"), "Cohere input dataset ID")
        _check_submission_cancelled(cancel_event)
        deadline = time.monotonic() + self.timeout
        remaining_retries = self.max_poll_retries
        while time.monotonic() < deadline:
            _check_submission_cancelled(cancel_event)
            try:
                dataset = _field(self.client.datasets.get(id=dataset_id), "dataset")
            except self._poll_errors as error:
                if not _transient(error) or remaining_retries == 0:
                    raise
                remaining_retries -= 1
            else:
                _check_submission_cancelled(cancel_event)
                status = _field(dataset, "validation_status")
                if status == "validated":
                    break
                if status not in {"unknown", "pending", "validating"}:
                    raise BatchProtocolError(
                        f"Dataset {dataset_id} validation status={status}: {_field(dataset, 'validation_error')}"
                    )
            delay = min(self.polling_interval, max(0, deadline - time.monotonic()))
            if cancel_event is None:
                time.sleep(delay)
            elif cancel_event.wait(delay):
                raise _BatchPreparationCancelled()
        else:
            raise TimeoutError(f"Dataset {dataset_id} validation timed out")
        _check_submission_cancelled(cancel_event)
        response = self.client.batches.create(
            request={
                "name": "toponymy-chat-batch",
                "input_dataset_id": dataset_id,
                "model": self.model,
            }
        )
        batch_id = _identifier(
            _field(_field(response, "batch"), "id"), "Cohere batch ID"
        )
        _remember(self._batch_sizes, batch_id, len(rows))
        return batch_id

    def _get_batch(self, batch_id: str) -> object:
        return _field(
            self.client.batches.retrieve(_identifier(batch_id, "batch_id")), "batch"
        )

    def get_batch_status(self, batch_id: str) -> str:
        return _identifier(
            _field(self._get_batch(batch_id), "status"), "Cohere batch status"
        )

    async def wait_for_completion(self, batch_id: str) -> bool:
        return await _wait_for_status(
            self.get_batch_status,
            batch_id,
            timeout=self.timeout,
            request_timeout=self.request_timeout,
            interval=self.polling_interval,
            retries=self.max_poll_retries,
            errors=self._poll_errors,
            completed="BATCH_STATUS_COMPLETED",
            pending={
                "BATCH_STATUS_UNSPECIFIED",
                "BATCH_STATUS_QUEUED",
                "BATCH_STATUS_IN_PROGRESS",
            },
            failed={
                "BATCH_STATUS_FAILED",
                "BATCH_STATUS_CANCELED",
                "BATCH_STATUS_CANCELLED",
                "BATCH_STATUS_CANCELING",
            },
        )

    def _dataset_records(self, dataset_id: str) -> list[Record]:
        from fastavro import reader

        dataset = _field(self.client.datasets.get(id=dataset_id), "dataset")
        parts = _field(dataset, "dataset_parts")
        if not isinstance(parts, list) or not parts:
            raise BatchProtocolError(f"Dataset {dataset_id} has no downloadable parts")
        records = []
        downloaded = 0
        for part in parts:
            url = _identifier(_field(part, "url"), "Dataset download URL")
            if not url.startswith("https://"):
                raise BatchProtocolError("Dataset downloads require HTTPS")
            with httpx.stream("GET", url, timeout=self.request_timeout) as response:
                response.raise_for_status()
                with io.BytesIO() as content:
                    for chunk in response.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > self.max_download_bytes:
                            raise BatchProtocolError(
                                "Dataset exceeds max_download_bytes"
                            )
                        content.write(chunk)
                    content.seek(0)
                    records.extend(reader(content))
        return records

    def _retrieve(self, batch_id: str) -> list[str | Exception]:
        batch = self._get_batch(batch_id)
        count = self._batch_sizes.get(batch_id)
        if count is None:
            source = _identifier(
                _field(batch, "input_dataset_id"), "Cohere input dataset ID"
            )
            expected = [
                _identifier(row.get("custom_id"), "Input custom_id")
                for row in self._dataset_records(source)
            ]
        else:
            expected = [str(index) for index in range(count)]
        output = _field(batch, "output_dataset_id")
        if output is None:
            return [
                BatchItemError(
                    key, f"no output dataset; status={_field(batch, 'status')}"
                )
                for key in expected
            ]
        return _aligned_results(
            self._dataset_records(_identifier(output, "Cohere output dataset ID")),
            expected,
            _cohere_text,
        )

    async def retrieve_batch_text_results(self, batch_id: str) -> list[str | Exception]:
        return await _thread_call(self._retrieve, batch_id, timeout=self.timeout)

    def cancel_batch(self, batch_id: str) -> object:
        return self.client.batches.cancel(_identifier(batch_id, "batch_id"))


class AzureBatchTransport:
    """Use an Azure OpenAI deployment's files and chat-completions batch APIs.

    Schema policy matches CohereBatchTransport. Unknown deployment names default
    to no known schema support; set supports_json_schema=True for a deployment
    whose underlying model supports structured output. Required schema requests
    fail before upload if support is unknown or explicitly disabled. Auto mode
    preserves plain requests when no supported schema is supplied. The schema is
    forwarded unchanged under response_format.json_schema.schema with strict=True.
    """

    def __init__(
        self,
        api_key: str | None,
        endpoint: str,
        model: str,
        polling_interval: float = 60,
        timeout: float = 7200,
        *,
        client: object = None,
        api_version: str = "2024-10-21",
        request_timeout: float = 30,
        max_poll_retries: int = 2,
        use_json_schema: bool | None = None,
        use_json_object: bool | None = None,
        supports_json_schema: bool | None = None,
    ):
        _settings(model, polling_interval, timeout, request_timeout, max_poll_retries)
        self.supports_json_schema = _schema_policy(
            use_json_schema, use_json_object, supports_json_schema, model, azure=True
        )
        self.use_json_schema = use_json_schema
        self.use_json_object = use_json_object
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("endpoint must be an HTTPS Azure OpenAI resource URL")
        self._poll_errors = (
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
            httpx.HTTPError,
        )
        try:
            from openai import APIError
        except ImportError:
            if client is None:
                raise ImportError(
                    "Azure OpenAI batches require the openai package"
                ) from None
        else:
            self._poll_errors += (APIError,)
        if client is None:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                timeout=request_timeout,
                max_retries=0,
            )
        self.client = client
        self.model = model
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.max_poll_retries = max_poll_retries
        self._batch_sizes: OrderedDict[str, int] = OrderedDict()

    def submit_batch(
        self, prompts: Sequence[Prompt], temperature: float, max_tokens: int
    ) -> str:
        rows = _requests(
            prompts,
            temperature,
            max_tokens,
            self.model,
            azure=True,
            use_json_schema=self.use_json_schema,
            use_json_object=self.use_json_object,
            supports_json_schema=self.supports_json_schema,
        )
        with io.BytesIO(_jsonl(rows)) as data:
            data.name = "toponymy-batch.jsonl"
            uploaded = self.client.files.create(file=data, purpose="batch")
        file_id = _identifier(_field(uploaded, "id"), "Azure input file ID")
        batch = self.client.batches.create(
            input_file_id=file_id, endpoint="/chat/completions", completion_window="24h"
        )
        batch_id = _identifier(_field(batch, "id"), "Azure batch ID")
        _remember(self._batch_sizes, batch_id, len(rows))
        return batch_id

    def _get_batch(self, batch_id: str) -> object:
        return self.client.batches.retrieve(_identifier(batch_id, "batch_id"))

    def get_batch_status(self, batch_id: str) -> str:
        return _identifier(
            _field(self._get_batch(batch_id), "status"), "Azure batch status"
        )

    async def wait_for_completion(self, batch_id: str) -> bool:
        return await _wait_for_status(
            self.get_batch_status,
            batch_id,
            timeout=self.timeout,
            request_timeout=self.request_timeout,
            interval=self.polling_interval,
            retries=self.max_poll_retries,
            errors=self._poll_errors,
            completed="completed",
            pending={"validating", "in_progress", "finalizing"},
            failed={"failed", "expired", "cancelled", "cancelling"},
        )

    def _file_records(self, file_id: str) -> list[Record]:
        response = self.client.files.content(file_id)
        text = _field(response, "text")
        if not isinstance(text, str):
            raise BatchProtocolError("Azure file content must be text")
        return _jsonl_records(text)

    def _retrieve(self, batch_id: str) -> list[str | Exception]:
        batch = self._get_batch(batch_id)
        count = self._batch_sizes.get(batch_id)
        if count is None:
            source = _identifier(_field(batch, "input_file_id"), "Azure input file ID")
            expected = [
                _identifier(row.get("custom_id"), "Input custom_id")
                for row in self._file_records(source)
            ]
        else:
            expected = [str(index) for index in range(count)]
        rows = []
        for field in ("output_file_id", "error_file_id"):
            file_id = _field(batch, field)
            if file_id is not None:
                rows.extend(self._file_records(_identifier(file_id, f"Azure {field}")))
        return _aligned_results(rows, expected, _azure_text)

    async def retrieve_batch_text_results(self, batch_id: str) -> list[str | Exception]:
        return await _thread_call(self._retrieve, batch_id, timeout=self.timeout)

    def cancel_batch(self, batch_id: str) -> object:
        return self.client.batches.cancel(_identifier(batch_id, "batch_id"))
