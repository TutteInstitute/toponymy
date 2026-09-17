"""Provider-managed job ownership tested entirely at fake SDK boundaries."""

import asyncio
import copy
import sys
import threading
from types import SimpleNamespace

import pytest
import httpx

from toponymy import llm_wrappers as wrappers
from toponymy.templates import Prompt

BATCH_CLASSES = (
    wrappers.CohereBatchNamer,
    wrappers.BatchAnthropicNamer,
    wrappers.BatchAzureAINamer,
)


def lifecycle_wrapper(cls, phase, cleanup_failure=False):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release_submission = threading.Event()
    calls = {"submit": 0, "cancel": [], "original_cancel": None}
    wrapper = object.__new__(cls)
    wrapper.model = "fixture"
    wrapper.transport = SimpleNamespace(
        supports_json_schema=False, use_json_schema=None, use_json_object=None
    )
    wrapper.use_json_schema = None
    wrapper.use_json_object = None
    wrapper._schema_capability = False

    def submit(prompts, temperature, max_tokens):
        calls["submit"] += 1
        if phase == "submission":
            loop.call_soon_threadsafe(entered.set)
            if not release_submission.wait(timeout=5):
                raise TimeoutError("test submission gate was not released")
        return "owned-job-17"

    async def gate():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            calls["original_cancel"] = error
            raise

    async def wait(batch_id):
        assert batch_id == "owned-job-17"
        if phase == "waiting":
            await gate()
        if phase == "timeout":
            return False
        return True

    async def retrieve(batch_id):
        assert batch_id == "owned-job-17"
        if phase == "retrieval":
            await gate()
        return [wrappers.CallResult(value='{"topic_name":"Done"}')]

    def cancel(batch_id):
        calls["cancel"].append(batch_id)
        if cleanup_failure:
            raise RuntimeError("cleanup fixture failure")

    wrapper.submit_batch = submit
    wrapper._wait_for_completion_async = wait
    wrapper._retrieve_batch_results = retrieve
    wrapper.cancel_batch = cancel
    return wrapper, calls, entered, release_submission


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("phase", ["before", "submission", "waiting", "retrieval"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_batch_cancellation_owns_submission_and_cancels_exactly_once(
    cls, phase, cleanup_failure, caplog
):
    wrapper, calls, entered, release = lifecycle_wrapper(cls, phase, cleanup_failure)
    api_entered = asyncio.Event()
    api_cancellations = []

    async def observe_api_boundary():
        api_entered.set()
        try:
            return await wrapper.generate_topic_names([Prompt("s", "u")])
        except asyncio.CancelledError as error:
            # Observe Toponymy's actual exception before it crosses the native
            # Task boundary. CPython 3.10 creates a fresh CancelledError there
            # and keeps this original object as its exception context.
            api_cancellations.append(error)
            raise

    task = asyncio.create_task(observe_api_boundary())
    try:
        if phase == "before":
            # This direct Event await resumes us at the helper's initial yield,
            # before it creates its owned submission task.
            await api_entered.wait()
        else:
            await asyncio.wait_for(entered.wait(), timeout=3)
        task.cancel("original caller cancellation")
        if phase == "submission":
            # The caller is cancelled while the SDK is still creating the job.
            # Releasing its result must lead to exactly one remote cancellation.
            await asyncio.sleep(0)
            assert not task.done()
            assert calls["cancel"] == []
            release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, timeout=3)
        assert len(api_cancellations) == 1
        original = api_cancellations[0]
        assert original.args == ("original caller cancellation",)
        assert raised.value is original or raised.value.__context__ is original
        if calls["original_cancel"] is not None:
            assert original is calls["original_cancel"]
        assert calls["submit"] == (0 if phase == "before" else 1)
        assert calls["cancel"] == ([] if phase == "before" else ["owned-job-17"])
        if cleanup_failure and phase != "before":
            assert "Failed to cancel abandoned batch owned-job-17" in caplog.text
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_wait_for", [False, True])
async def test_stdlib_task_cancellation_preserves_original_as_exception_or_context(
    use_wait_for,
):
    """Control for native Task semantics, independent of Toponymy and providers."""
    entered = asyncio.Event()
    observed = []

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    task = asyncio.create_task(operation())
    await entered.wait()
    task.cancel("stdlib cancellation sentinel")
    # Bound completion without retrieving the task's exception first: doing so
    # consumes CPython 3.10's saved cancellation context.
    done, pending = await asyncio.wait({task}, timeout=3)
    assert done == {task} and not pending
    with pytest.raises(asyncio.CancelledError) as raised:
        if use_wait_for:
            await asyncio.wait_for(task, timeout=3)
        else:
            await task
    assert len(observed) == 1
    assert observed[0].args == ("stdlib cancellation sentinel",)
    assert raised.value is observed[0] or raised.value.__context__ is observed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
async def test_batch_cancelled_before_task_start_never_submits(cls):
    wrapper, calls, _, _ = lifecycle_wrapper(cls, "before")
    task = asyncio.create_task(wrapper.generate_topic_names([Prompt("s", "u")]))
    task.cancel("cancelled before task start")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)
    assert calls["submit"] == 0
    assert calls["cancel"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_batch_timeout_cancels_once_and_retains_timeout(cls, cleanup_failure):
    wrapper, calls, _, _ = lifecycle_wrapper(cls, "timeout", cleanup_failure)
    with pytest.raises(TimeoutError, match="owned-job-17 did not complete"):
        await wrapper.generate_topic_names([Prompt("s", "u")])
    assert calls["submit"] == 1
    assert calls["cancel"] == ["owned-job-17"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
async def test_successful_batch_is_not_cancelled_or_resubmitted(cls):
    wrapper, calls, _, _ = lifecycle_wrapper(cls, "success")
    assert await wrapper.generate_topic_names([Prompt("s", "u")]) == ["Done"]
    assert calls["submit"] == 1
    assert calls["cancel"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("phase", ["waiting", "retrieval"])
@pytest.mark.parametrize(
    "error_type", [httpx.ReadTimeout, ConnectionError, RuntimeError]
)
@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_batch_sdk_errors_cancel_known_job_and_preserve_original(
    cls, phase, error_type, cleanup_failure
):
    wrapper, calls, _, _ = lifecycle_wrapper(cls, "success", cleanup_failure)
    error = error_type("original SDK failure")

    async def fail(batch_id):
        assert batch_id == "owned-job-17"
        raise error

    if phase == "waiting":
        wrapper._wait_for_completion_async = fail
    else:
        wrapper._retrieve_batch_results = fail
    with pytest.raises(error_type) as raised:
        await wrapper.generate_topic_names([Prompt("s", "u")])
    assert raised.value is error
    assert calls["submit"] == 1
    assert calls["cancel"] == ["owned-job-17"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
async def test_ambiguous_batch_submission_failure_is_never_retried(cls):
    wrapper, calls, _, _ = lifecycle_wrapper(cls, "success")
    error = httpx.ReadTimeout("creation response was lost")

    def submit(*args):
        calls["submit"] += 1
        raise error

    wrapper.submit_batch = submit
    with pytest.raises(httpx.ReadTimeout) as raised:
        await wrapper.generate_topic_names([Prompt("s", "u")])
    assert raised.value is error
    assert calls["submit"] == 1
    assert calls["cancel"] == []


def anthropic_fixture(monkeypatch, **options):
    calls = []

    def create(**request):
        calls.append(request)
        return SimpleNamespace(id="schema-batch")

    client = SimpleNamespace(
        messages=SimpleNamespace(batches=SimpleNamespace(create=create))
    )
    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **kwargs: client)
    )
    return wrappers.BatchAnthropicNamer("fixture-key", **options), calls


SUPPORTED_SCHEMA = {
    "type": "object",
    "properties": {"topic_name": {"type": "string"}},
    "required": ["topic_name"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("option", [None, True, False])
def test_anthropic_batch_native_schema_request_is_owned(monkeypatch, option):
    wrapper, calls = anthropic_fixture(monkeypatch, use_json_schema=option)
    schema = copy.deepcopy(SUPPORTED_SCHEMA)
    prompt = Prompt("system", "user", schema)
    assert wrapper.submit_batch([prompt], 0.2, 256) == "schema-batch"
    assert len(calls) == 1
    params = calls[0]["requests"][0]["params"]
    assert params["messages"] == [{"role": "user", "content": "user"}]
    assert params["max_tokens"] == 256
    if option is False:
        assert "output_config" not in params
    else:
        assert params["output_config"] == {
            "format": {"type": "json_schema", "schema": schema}
        }
        params["output_config"]["format"]["schema"]["properties"]["topic_name"][
            "type"
        ] = "integer"
        assert prompt.json_schema == schema


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {"type": "object", "properties": {"x": {"type": "string"}}},
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "number", "minimum": 0}},
        },
        {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        {"type": "array", "items": {"type": "string"}, "minItems": 2},
        {"$ref": "#"},
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "array",
            "items": [{"type": "number", "minimum": 0}],
        },
    ],
)
def test_anthropic_required_unsupported_schema_fails_before_create(monkeypatch, schema):
    wrapper, calls = anthropic_fixture(monkeypatch, use_json_schema=True)
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrapper.submit_batch([Prompt("s", "u", schema)], 0.2, 256)
    assert calls == []


def test_anthropic_auto_preserves_unsupported_schema_by_falling_back_to_text(
    monkeypatch,
):
    wrapper, calls = anthropic_fixture(monkeypatch)
    schema = copy.deepcopy(SUPPORTED_SCHEMA)
    schema["properties"]["topic_name"]["minLength"] = 1
    before = copy.deepcopy(schema)
    wrapper.submit_batch([Prompt("s", "u", schema)], 0.2, 256)
    assert "output_config" not in calls[0]["requests"][0]["params"]
    assert schema == before


def test_anthropic_supports_local_nonrecursive_schema_pointer(monkeypatch):
    wrapper, calls = anthropic_fixture(monkeypatch, use_json_schema=True)
    schema = copy.deepcopy(SUPPORTED_SCHEMA)
    schema["$defs"] = {"name": {"type": "string"}}
    schema["properties"]["topic_name"] = {"$ref": "#/$defs/name"}
    wrapper.submit_batch([Prompt("s", "u", schema)], 0.2, 256)
    assert (
        calls[0]["requests"][0]["params"]["output_config"]["format"]["schema"] == schema
    )


@pytest.mark.parametrize(
    "options",
    [
        {"use_json_schema": True, "supports_json_schema": False},
        {"use_json_schema": True, "model": "unverified-model"},
    ],
)
def test_anthropic_required_model_capability_fails_before_create(monkeypatch, options):
    wrapper, calls = anthropic_fixture(monkeypatch, **options)
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrapper.submit_batch([Prompt("s", "u", SUPPORTED_SCHEMA)], 0.2, 256)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", BATCH_CLASSES)
async def test_required_batch_schema_preflight_precedes_any_submission(cls):
    wrapper, calls, _, _ = lifecycle_wrapper(cls, "success")
    wrapper.use_json_schema = True
    with pytest.raises(wrappers.InvalidLLMInputError, match="does not support"):
        await wrapper.generate_topic_names([Prompt("s", "u", SUPPORTED_SCHEMA)])
    assert calls["submit"] == 0
    assert calls["cancel"] == []


def test_anthropic_explicit_output_mode_conflicts_fail_locally(monkeypatch):
    with pytest.raises(wrappers.InvalidLLMInputError):
        anthropic_fixture(monkeypatch, use_json_schema=True, use_json_object=True)
    with pytest.raises(wrappers.InvalidLLMInputError, match="JSON-object"):
        anthropic_fixture(monkeypatch, use_json_object=True)


@pytest.mark.parametrize("field", ["polling_interval", "timeout"])
@pytest.mark.parametrize(
    "value", [True, False, 0, -1, float("nan"), float("inf"), 10**400, "1"]
)
def test_anthropic_invalid_timers_fail_before_sdk_client_creation(
    monkeypatch, field, value
):
    def unexpected_client(**kwargs):
        raise AssertionError("Invalid timer reached SDK client construction")

    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=unexpected_client)
    )
    with pytest.raises(wrappers.InvalidLLMInputError, match="finite positive"):
        wrappers.BatchAnthropicNamer("fixture-key", **{field: value})
