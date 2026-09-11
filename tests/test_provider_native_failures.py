"""Native provider lifecycle contracts using bounded, local SDK doubles."""

import asyncio
from collections import deque
import sys
import threading
from types import SimpleNamespace

import pytest

from toponymy import llm_wrappers as wrappers
from toponymy.templates import Prompt


class NativeAPIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


class ResultStream:
    def __init__(self, error=None, close_error=None):
        self.error = error
        self.close_error = close_error
        self.closed = False

    def __iter__(self):
        yield SimpleNamespace(
            custom_id="0",
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(
                    content=[SimpleNamespace(type="text", text='{"topic_name":"Done"}')]
                ),
            ),
        )
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class NativeAnthropicClient:
    def __init__(self, statuses=("ended",)):
        self.statuses = deque(statuses)
        self.calls = []
        self.closed = False
        self.stream = ResultStream()
        self.poll_entered = threading.Event()
        self.poll_release = None
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(
                create=self.create,
                retrieve=self.retrieve,
                results=self.results,
                cancel=self.cancel,
            )
        )

    def create(self, *, requests):
        self.calls.append(("create", requests))
        return SimpleNamespace(id="anthropic-owned-job")

    def retrieve(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        self.poll_entered.set()
        if self.poll_release is not None:
            if not self.poll_release.wait(timeout=2):
                raise RuntimeError("Native poll fixture gate was not released")
        status = self.statuses.popleft() if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, BaseException):
            raise status
        return SimpleNamespace(processing_status=status)

    def results(self, batch_id):
        self.calls.append(("results", batch_id))
        return self.stream

    def cancel(self, batch_id):
        self.calls.append(("cancel", batch_id))
        return SimpleNamespace(processing_status="canceling")

    def close(self):
        self.closed = True


def anthropic_wrapper(monkeypatch, client, **options):
    construction = []

    def create_client(**kwargs):
        construction.append(kwargs)
        return client

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(
            Anthropic=create_client,
            APIError=NativeAPIError,
            APIStatusError=NativeAPIError,
            APIConnectionError=ConnectionError,
        ),
    )
    wrapper = wrappers.BatchAnthropicNamer(
        "fixture-key", polling_interval=0.001, **options
    )
    assert construction[0]["max_retries"] == 0
    return wrapper


def call_names(client):
    return [name for name, _ in client.calls]


@pytest.mark.asyncio
async def test_anthropic_close_releases_its_owned_native_client(monkeypatch):
    client = NativeAnthropicClient()
    wrapper = anthropic_wrapper(monkeypatch, client)
    await wrapper.close()
    assert client.closed, "The wrapper owns this SDK client and must release it"


@pytest.mark.asyncio
async def test_anthropic_result_stream_closes_when_iteration_fails(monkeypatch):
    client = NativeAnthropicClient()
    error = ValueError("Malformed native JSONL fixture")
    client.stream = ResultStream(error)
    wrapper = anthropic_wrapper(monkeypatch, client)
    with pytest.raises(ValueError) as raised:
        await wrapper.generate_topic_names([Prompt("system", "user")])
    assert raised.value is error
    assert client.stream.closed, "An incomplete native stream must be closed"
    assert call_names(client) == ["create", "retrieve", "results", "cancel"]


@pytest.mark.asyncio
async def test_anthropic_stream_close_failure_preserves_read_failure(
    monkeypatch, caplog
):
    client = NativeAnthropicClient()
    error = ValueError("Invalid stream item")
    client.stream = ResultStream(error, OSError("Failed to close stream"))
    wrapper = anthropic_wrapper(monkeypatch, client)
    with pytest.raises(ValueError) as raised:
        await wrapper.generate_topic_names([Prompt("system", "user")])
    assert raised.value is error
    assert client.stream.closed
    assert "Failed to close stream" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True])
async def test_anthropic_real_sdk_connection_error_uses_one_poll_retry_budget(
    monkeypatch, exhausted
):
    anthropic = pytest.importorskip("anthropic")
    error = anthropic.APIConnectionError(request=SimpleNamespace())
    # A non-httpx cause reproduces SDK families whose transport is not httpx 1.
    error.__cause__ = OSError("Local connection fixture")
    client = NativeAnthropicClient([error] if exhausted else [error, "ended"])
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    wrapper = wrappers.BatchAnthropicNamer(
        "fixture-key", polling_interval=0.001, timeout=1.0
    )
    if exhausted:
        with pytest.raises(anthropic.APIConnectionError) as raised:
            await wrapper.generate_topic_names([Prompt("system", "user")])
        assert raised.value is error
        assert call_names(client) == [
            "create",
            "retrieve",
            "retrieve",
            "retrieve",
            "cancel",
        ]
    else:
        assert await wrapper.generate_topic_names([Prompt("system", "user")]) == [
            "Done"
        ]
        assert call_names(client) == ["create", "retrieve", "retrieve", "results"]
        assert client.stream.closed


@pytest.mark.asyncio
async def test_anthropic_provider_canceling_can_finish_with_partial_results(
    monkeypatch,
):
    client = NativeAnthropicClient(["canceling", "ended"])
    wrapper = anthropic_wrapper(monkeypatch, client, timeout=1.0)
    assert await wrapper.generate_topic_names([Prompt("system", "user")]) == ["Done"]
    assert call_names(client) == ["create", "retrieve", "retrieve", "results"]
    assert client.stream.closed


@pytest.mark.asyncio
async def test_anthropic_cancellation_during_callback_prevents_submission(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def callback(event):
        entered.set()
        assert release.wait(timeout=2), "Callback fixture was not released"

    client = NativeAnthropicClient()
    wrapper = anthropic_wrapper(monkeypatch, client, callback=callback)
    task = asyncio.create_task(wrapper.generate_topic_names([Prompt("system", "user")]))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel("Cancelled during local preparation")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 503])
async def test_anthropic_transient_poll_retries_read_without_resubmission(
    monkeypatch, status_code
):
    client = NativeAnthropicClient(
        [
            NativeAPIError(status_code, "Temporary fixture failure"),
            "in_progress",
            "ended",
        ]
    )
    wrapper = anthropic_wrapper(monkeypatch, client, timeout=1.0)
    assert await wrapper.generate_topic_names([Prompt("system", "user")]) == ["Done"]
    assert call_names(client) == [
        "create",
        "retrieve",
        "retrieve",
        "retrieve",
        "results",
    ]


@pytest.mark.asyncio
async def test_anthropic_exhausted_poll_retry_budget_is_finite(monkeypatch):
    error = NativeAPIError(503, "Persistent fixture failure")
    client = NativeAnthropicClient([error])
    wrapper = anthropic_wrapper(monkeypatch, client, timeout=1.0)
    with pytest.raises(NativeAPIError) as raised:
        await wrapper.generate_topic_names([Prompt("system", "user")])
    assert raised.value is error
    # Existing Azure/Cohere policy allows two retries, and Anthropic should share
    # that same bounded default rather than silently retrying an entire job.
    assert call_names(client) == [
        "create",
        "retrieve",
        "retrieve",
        "retrieve",
        "cancel",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
async def test_anthropic_auth_and_invalid_poll_errors_are_not_retried(
    monkeypatch, status_code
):
    error = NativeAPIError(status_code, "Nonretryable native fixture failure")
    client = NativeAnthropicClient([error, "ended"])
    wrapper = anthropic_wrapper(monkeypatch, client, timeout=1.0)
    with pytest.raises(NativeAPIError) as raised:
        await wrapper.generate_topic_names([Prompt("system", "user")])
    assert raised.value is error
    assert call_names(client) == ["create", "retrieve", "cancel"]


@pytest.mark.asyncio
async def test_anthropic_local_deadline_does_not_wait_for_late_status_success(
    monkeypatch,
):
    client = NativeAnthropicClient(["ended"])
    client.poll_release = threading.Event()
    wrapper = anthropic_wrapper(monkeypatch, client, timeout=0.03)
    task = asyncio.create_task(wrapper.generate_topic_names([Prompt("system", "user")]))
    try:
        assert await asyncio.to_thread(client.poll_entered.wait, 1)
        done, _ = await asyncio.wait({task}, timeout=0.3)
        completed_before_release = task in done
    finally:
        client.poll_release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=2
        )
    assert completed_before_release, "Local deadline must bound the status await"
    assert isinstance(outcomes[0], TimeoutError)
    assert call_names(client) == ["create", "retrieve", "cancel"]


class NativeAzureClient:
    def __init__(self, status="completed"):
        self.status = status
        self.calls = []
        self.upload_entered = threading.Event()
        self.upload_release = None
        self.files = SimpleNamespace(create=self.upload)
        self.batches = SimpleNamespace(
            create=self.create, retrieve=self.retrieve, cancel=self.cancel
        )

    def upload(self, *, file, purpose):
        self.calls.append(("upload", purpose))
        self.upload_entered.set()
        if self.upload_release is not None:
            if not self.upload_release.wait(timeout=2):
                raise RuntimeError("Native upload fixture gate was not released")
        return SimpleNamespace(id="azure-input-file")

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return SimpleNamespace(id="azure-owned-job")

    def retrieve(self, batch_id):
        self.calls.append(("retrieve", batch_id))
        return SimpleNamespace(status=self.status)

    def cancel(self, batch_id):
        self.calls.append(("cancel", batch_id))
        return SimpleNamespace(status="cancelling")


def azure_wrapper(monkeypatch, client, *, timeout=1.0):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(APIError=NativeAPIError))
    return wrappers.BatchAzureAINamer(
        "fixture-key",
        "https://fixture.invalid",
        "fixture-deployment",
        polling_interval=0.001,
        timeout=timeout,
        client=client,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
async def test_azure_terminal_failure_is_not_a_local_timeout(monkeypatch, status):
    client = NativeAzureClient(status)
    wrapper = azure_wrapper(monkeypatch, client)
    with pytest.raises(Exception) as raised:
        await wrapper.generate_topic_names([Prompt("system", "user")])
    assert not isinstance(raised.value, TimeoutError)
    assert "azure-owned-job" in str(raised.value)
    assert status in str(raised.value)
    assert call_names(client) == ["upload", "create", "retrieve", "cancel"]


@pytest.mark.asyncio
async def test_azure_pending_deadline_remains_a_local_timeout(monkeypatch):
    client = NativeAzureClient("in_progress")
    wrapper = azure_wrapper(monkeypatch, client, timeout=0.02)
    with pytest.raises(TimeoutError):
        await wrapper.generate_topic_names([Prompt("system", "user")])
    names = call_names(client)
    assert names.count("upload") == names.count("create") == names.count("cancel") == 1
    assert names.count("retrieve") >= 1


@pytest.mark.asyncio
async def test_azure_cancellation_during_upload_never_creates_a_paid_job(monkeypatch):
    client = NativeAzureClient()
    client.upload_release = threading.Event()
    wrapper = azure_wrapper(monkeypatch, client)
    task = asyncio.create_task(wrapper.generate_topic_names([Prompt("system", "user")]))
    try:
        assert await asyncio.to_thread(client.upload_entered.wait, 1)
        task.cancel("Caller stopped before batch creation")
        # Let the cancellation handler set the per-submission event before
        # releasing the still-blocked upload; this is not an in-flight create.
        await asyncio.sleep(0)
        client.upload_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        client.upload_release.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
    assert call_names(client) == ["upload"]
