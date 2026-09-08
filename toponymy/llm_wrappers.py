from warnings import warn
from copy import deepcopy

from toponymy.templates import (
    GET_TOPIC_CLUSTER_NAMES_REGEX,
    GET_TOPIC_NAME_REGEX,
    default_extract_topic_names,
    Prompt,
)
from toponymy.tools.notebook_test_helpers import (
    notebook_test_replacement,
    get_test_ollama_model,
)
from toponymy._utils import resolve_api_key
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Generic, TypeVar, Callable, Any
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception,
    AsyncRetrying,
    wait_random_exponential,
)

from dataclasses import dataclass
from .response_parsing import (
    ResponseParseError,
    extract_response,
    string_field,
    topic_fields,
    topic_name_mapping,
)

import os
import httpx
import json
import asyncio

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
DebugCallback = Callable[[dict[str, Any]], None]


@dataclass
class CallResult(Generic[T]):
    value: Optional[T] = None
    error: Optional[Exception] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class InvalidLLMInputError(ValueError):
    """A custom exception for invalid LLM input. In these cases we do not want to retry, as the input will not change."""

    pass


class FailFastLLMError(RuntimeError):
    """
    A non-retryable error that is not caused by invalid input, but by a configuration
    or provider issue (e.g. bad API key, insufficient permissions, model not found).
    Retrying will not resolve these errors.
    """

    def __init__(self, message: str = "", original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class LLMBatchItemError(RuntimeError):
    """A provider-managed batch reported a failed item."""


class _SystemPromptFallback(RuntimeError):
    """Retry the same prompt without a system role, within its attempt budget."""


def _should_retry(e: Exception) -> bool:
    if isinstance(e, (InvalidLLMInputError, FailFastLLMError)):
        return False
    status = getattr(e, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or 500 <= status < 600
    return isinstance(
        e,
        (
            ResponseParseError,
            _SystemPromptFallback,
            TimeoutError,
            ConnectionError,
            httpx.TransportError,
        ),
    )


class LLMErrorHandlingMixin:
    """
    A mixin class that provides standardized error handling for LLM wrappers.

    This mixin centralizes error detection and safe async call patterns,
    ensuring consistent behavior across synchronous and asynchronous LLM wrappers.
    Subclasses should declare FAIL_FAST_EXCEPTIONS to specify which exceptions
    should trigger an immediate failure without retrying.

    Attributes:
    -----------
    FAIL_FAST_EXCEPTIONS: tuple
        A tuple of exception types that should trigger an immediate failure without
        retrying. These are typically configuration or provider-level errors such as
        invalid API keys, insufficient permissions, or model not found errors.
        Default is an empty tuple, meaning no exceptions are treated as fail-fast.
    """

    FAIL_FAST_EXCEPTIONS: tuple = ()

    def _handle_exception(self, e: Exception) -> None:
        if isinstance(e, InvalidLLMInputError):
            raise e

        if isinstance(e, self.FAIL_FAST_EXCEPTIONS) or getattr(
            e, "status_code", None
        ) in (400, 401, 403, 404, 422):
            raise FailFastLLMError(
                message=(
                    f"Non-retryable error for model "
                    f"'{getattr(self, 'model', '<unknown>')}': {e}"
                ),
                original_exception=e,
            ) from None

        raise e

    async def _safe_call_with_retry_result(
        self,
        fn,
        *args,
        **kwargs,
    ) -> CallResult:
        prompt = kwargs.get("prompt", args[0] if args else None)
        routine = kwargs.pop("routine", None)
        response_parser = kwargs.pop("response_parser", None)
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_random_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception(
                    lambda error: not isinstance(error, self.FAIL_FAST_EXCEPTIONS)
                    and _should_retry(error)
                ),
                reraise=True,
            ):
                with attempt:
                    prompt_type = "system" if self.supports_system_prompts else "single"
                    self._emit_debug_callback(
                        {
                            "event": "llm_call_start",
                            "routine": routine,
                            "prompt": prompt,
                        }
                    )
                    try:
                        value = await fn(*args, **kwargs)
                    except Exception as error:
                        self._emit_debug_callback(
                            {
                                "event": "llm_call_error",
                                "prompt_type": prompt_type,
                                "routine": routine,
                                "prompt": prompt,
                                "error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                            }
                        )
                        raise
                    # SUCCESS emit
                    self._emit_debug_callback(
                        {
                            "event": "llm_call_success",
                            "prompt_type": prompt_type,
                            "routine": routine,
                            "prompt": prompt,
                            "raw_response": value,
                        }
                    )
                    return CallResult(
                        value=response_parser(value) if response_parser else value
                    )
        except Exception as e:
            if isinstance(e, self.FAIL_FAST_EXCEPTIONS) or not _should_retry(e):
                self._handle_exception(e)

            # For other exceptions, we log a warning and return the error in the CallResult for potential handling by the caller.
            logger.warning(
                "%s exhausted retries for LLM call (%s): %s",
                self.__class__.__name__,
                type(e).__name__,
                str(e)[:200],
            )
            return CallResult(error=e)

    def _raise_fail_fast_from_batch_error(self, error) -> None:
        """
        Handle a provider-level error surfaced from a batch response item.

        Some provider batch APIs return errors inline as response fields rather than raising.
        This method provides a hook for subclasses to inspect those inline errors
        and raise FailFastLLMError if appropriate.

        A subclass that uses a batch API should override this method to handle
        provider-specific error formats. Subclasses that do not use a batch API
        do not need to override this method.

        Parameters:
        -----------
        error:
            The provider-specific error object from a batch response item.
            If None, the method returns immediately.
        """
        if error is None:
            return
        warn(
            f"{self.__class__.__name__} received a batch item error but did not "
            f"override _raise_fail_fast_from_batch_error: {error}"
        )


class DebugCallbackMixin:
    """
    Mixin providing optional debug callback support for LLM wrappers.

    This mixin allows wrappers to emit structured debug events (e.g., prompts,
    raw LLM responses, errors, and metadata) to a user-supplied callback
    function. The callback is intended for debugging, logging, or observability
    purposes such as inspecting prompts/responses or recording them to a file.

    Wrappers opt into emitting events by setting `_supports_debug_callback = True`.

    The helper `_warn_if_debug_callback_unsupported` provides a check to warn if
    a debug callback is provided but not supported.
    """

    _supports_debug_callback: bool = False
    callback: DebugCallback | None = None

    def _emit_debug_callback(self, payload: dict[str, Any]) -> None:
        callback = getattr(self, "callback", None)
        if callback is None:
            return

        try:
            callback(
                {
                    "wrapper": self.__class__.__name__,
                    "model": getattr(self, "model", None),
                    **payload,
                }
            )
        except Exception:
            logger.exception("LLM debug callback failed")

    def _warn_if_debug_callback_unsupported(self) -> None:
        callback = getattr(self, "callback", None)

        if callback is not None and not self._supports_debug_callback:
            warn(
                (
                    f"{self.__class__.__name__} received a debug callback, but "
                    "this wrapper does not currently support debug callback events."
                ),
                UserWarning,
                stacklevel=2,
            )


def llm_output_to_result(llm_output: str, regex: str = GET_TOPIC_NAME_REGEX) -> dict:
    """Parse legacy output structurally; regex arguments only identify the contract."""

    def validate(value):
        if regex == GET_TOPIC_CLUSTER_NAMES_REGEX:
            topic_name_mapping(value)
        elif regex == GET_TOPIC_NAME_REGEX:
            topic_fields(value, "topic_name")
        return value

    return extract_response(llm_output, validate)


def validate_prompt(prompt: Any, supports_system_prompts: bool) -> Dict[str, Any]:
    """Normalize canonical prompts and explicit legacy renderings at one boundary."""
    if isinstance(prompt, Prompt):
        prompt = {
            "system": prompt.system,
            "user": prompt.user,
            "json_schema": prompt.json_schema,
        }
    elif isinstance(prompt, str):
        prompt = {"system": "", "user": prompt, "combined": prompt}
    elif isinstance(prompt, dict):
        prompt = dict(prompt)
    else:
        raise InvalidLLMInputError(
            "Prompt must be a Prompt, string, or rendering dictionary"
        )
    if "combined" not in prompt and "system" in prompt and "user" in prompt:
        if not isinstance(prompt["system"], str) or not isinstance(prompt["user"], str):
            raise InvalidLLMInputError("Prompt messages must be strings")
        prompt["combined"] = prompt["system"] + "\n\n" + prompt["user"]
    required = ("system", "user") if supports_system_prompts else ("combined",)
    if any(not isinstance(prompt.get(key), str) for key in required):
        raise InvalidLLMInputError(f"Prompt requires string renderings: {required}")
    schema = prompt.get("json_schema")
    if schema is not None and (not isinstance(schema, dict) or not schema):
        raise InvalidLLMInputError(
            "Prompt json_schema must be a nonempty JSON Schema object"
        )
    if schema is not None:
        _validate_json_schema(schema)
    return prompt


def _validate_json_schema(schema: dict) -> None:
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
    from referencing.exceptions import Unresolvable
    from referencing.jsonschema import UnknownDialect, specification_with

    try:
        json.dumps(schema, allow_nan=False)
        if "$schema" in schema and validator_for(schema, default=None) is None:
            raise InvalidLLMInputError("Unsupported JSON Schema dialect")
        validator_for(schema).check_schema(schema)
    except (SchemaError, TypeError, ValueError) as error:
        raise InvalidLLMInputError("Invalid response JSON Schema") from error
    specification = specification_with(
        schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    )
    root_resource = Resource.from_contents(schema, default_specification=specification)
    resolver = Registry().resolver_with_root(root_resource)
    pending = [(schema, resolver, specification)]
    while pending:
        value, resolver, specification = pending.pop()
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                reference = value.get(key)
                if reference is None:
                    continue
                if not isinstance(reference, str) or not reference.startswith("#"):
                    raise InvalidLLMInputError(
                        "Response schemas may use only local references"
                    )
                try:
                    resolved = resolver.lookup(reference)
                except Unresolvable as error:
                    raise InvalidLLMInputError(
                        f"Unresolvable local response schema reference: {reference}"
                    ) from error
                if not isinstance(resolved.contents, (dict, bool)):
                    raise InvalidLLMInputError(
                        "Response schema reference must target a schema"
                    )
            children = []
            for key in (
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
                "dependencies",
            ):
                child_map = value.get(key)
                if isinstance(child_map, dict):
                    children.extend(child_map.values())
            for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
                child_list = value.get(key)
                if isinstance(child_list, list):
                    children.extend(child_list)
            for key in (
                "items",
                "additionalItems",
                "additionalProperties",
                "unevaluatedItems",
                "unevaluatedProperties",
                "contains",
                "propertyNames",
                "not",
                "if",
                "then",
                "else",
                "contentSchema",
                "extends",
            ):
                child = value.get(key)
                children.extend(child if isinstance(child, list) else [child])
            for child in children:
                if isinstance(child, dict):
                    try:
                        child_specification = (
                            specification_with(child["$schema"])
                            if "$schema" in child
                            else specification
                        )
                        resource = Resource.from_contents(
                            child, default_specification=child_specification
                        )
                    except UnknownDialect as error:
                        raise InvalidLLMInputError(
                            "Unsupported nested JSON Schema dialect"
                        ) from error
                    pending.append(
                        (child, resolver.in_subresource(resource), child_specification)
                    )


def _validate_generation_options(temperature, max_tokens):
    import math

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise InvalidLLMInputError("temperature must be a finite nonnegative number")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise InvalidLLMInputError("max_tokens must be a positive integer")


class LLMWrapper(DebugCallbackMixin, LLMErrorHandlingMixin, ABC):
    FAIL_FAST_EXCEPTIONS: tuple = ()

    @property
    def supports_json_schema(self) -> bool:
        return False

    @abstractmethod
    def _call_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        """
        Call the LLM with the combined rendering of the given prompt.

        Implementations should send ``prompt["combined"]``, which carries the whole
        instruction in a single message. This is the path taken when the wrapper
        reports `supports_system_prompts` as False.

        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _call_llm_with_system_prompt(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        """
        Call the LLM with the system/user rendering of the given prompt.

        Implementations should send ``prompt["system"]`` and ``prompt["user"]`` as a
        system message and a user message. This is the path taken when the wrapper
        reports `supports_system_prompts` as True.

        This method should be implemented by subclasses.
        """
        pass

    def _safe_call_llm(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> str:
        try:
            self._emit_debug_callback(
                {"event": "llm_call_start", "routine": routine, "prompt": prompt}
            )
            raw_response = self._call_llm(prompt, temperature, max_tokens)

            self._emit_debug_callback(
                {
                    "event": "llm_call_success",
                    "prompt_type": "single",
                    "routine": routine,
                    "prompt": prompt,
                    "raw_response": raw_response,
                }
            )
            return raw_response

        except Exception as e:
            self._emit_debug_callback(
                {
                    "event": "llm_call_error",
                    "prompt_type": "single",
                    "routine": routine,
                    "prompt": prompt,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                }
            )
            self._handle_exception(e)

    def _safe_call_llm_with_system_prompt(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> str:
        try:
            self._emit_debug_callback(
                {"event": "llm_call_start", "routine": routine, "prompt": prompt}
            )
            raw_response = self._call_llm_with_system_prompt(
                prompt, temperature, max_tokens
            )

            self._emit_debug_callback(
                {
                    "event": "llm_call_success",
                    "prompt_type": "system",
                    "routine": routine,
                    "prompt": prompt,
                    "raw_response": raw_response,
                }
            )
            return raw_response

        except Exception as e:
            self._emit_debug_callback(
                {
                    "event": "llm_call_error",
                    "prompt_type": "system",
                    "routine": routine,
                    "prompt": prompt,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                }
            )
            self._handle_exception(e)

    def _call_llm_for_prompt(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> str:
        """
        Send a prompt using whichever rendering this wrapper's provider supports.

        This is the single point at which a prompt's renderings are resolved down to
        one provider call, so that everything upstream of the wrapper can stay
        provider agnostic.
        """
        prompt = validate_prompt(prompt, self.supports_system_prompts)

        if self.supports_system_prompts:
            return self._safe_call_llm_with_system_prompt(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                routine=routine,
            )

        return self._safe_call_llm(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            routine=routine,
        )

    @staticmethod
    def _topic_name_error_callback(retry_state):
        raise retry_state.outcome.exception()

    # @abstractmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=_topic_name_error_callback,
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_name(
        self,
        prompt: Prompt | str | dict,
        temperature: float = 0.4,
        topic_extraction_function=None,
        get_topic_name_regex=GET_TOPIC_NAME_REGEX,
        max_tokens: int | None = None,
        *,
        response_parser: Callable | None = None,
    ) -> str | tuple:
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_topic_name", 128)
        _validate_generation_options(temperature, max_tokens)
        raw = self._call_llm_for_prompt(
            prompt, temperature, max_tokens, routine="generate_topic_name"
        )
        if response_parser is not None:
            return response_parser(raw)
        info = llm_output_to_result(raw, get_topic_name_regex)
        return (
            topic_extraction_function(info)
            if topic_extraction_function
            else string_field(info, "topic_name")
        )

    @staticmethod
    def _topic_cluster_names_error_callback(retry_state):
        raise retry_state.outcome.exception()

    # @abstractmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=_topic_cluster_names_error_callback,
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_cluster_names(
        self,
        prompt: Prompt | str | dict,
        old_names: List[str],
        temperature: float = 0.4,
        extract_topic_names_function=default_extract_topic_names,
        get_topic_names_regex=GET_TOPIC_CLUSTER_NAMES_REGEX,
        max_tokens: int | None = None,
        *,
        response_parser: Callable | None = None,
    ) -> List[str]:
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_cluster_names", 1024)
        _validate_generation_options(temperature, max_tokens)
        raw = self._call_llm_for_prompt(
            prompt, temperature, max_tokens, routine="generate_topic_cluster_names"
        )
        names = (
            response_parser(raw)
            if response_parser
            else extract_topic_names_function(
                llm_output_to_result(raw, get_topic_names_regex), old_names, raw
            )
        )
        if len(names) != len(old_names):
            raise ResponseParseError("Response must contain one name per input topic")
        return names

    @property
    def supports_system_prompts(self) -> bool:
        """
        Check if the LLM wrapper supports system prompts.
        By default, it does. Override in subclasses if not supported.
        """
        return True

    def test_llm_connectivity(self) -> str:
        result = self.connectivity_status()

        if result["success"]:
            logger.info(
                " Connected to %s using %s",
                result["wrapper"],
                result["model"],
            )
            return result["response"]

        logger.warning(
            "  Failed to connect to %s using %s",
            result["wrapper"],
            result["model"],
        )
        logger.warning(
            "  Cause:  %s: %s",
            result["error_type"],
            result["error_message"],
        )
        return "<error>"

    def connectivity_status(
        self,
        prompt: str = (
            "Respond with exactly this JSON and nothing else.\n"
            "Do not use markdown or code blocks.\n\n"
            '{"status": "ok"}'
        ),
        system_prompt: str | None = None,
    ) -> dict:
        result = {
            "success": False,
            "model": self.model,
            "wrapper": self.__class__.__name__,
            "response": None,
            "error_type": None,
            "error_message": None,
            "original_exception": None,
        }

        try:
            if system_prompt is None:
                response = self._call_llm(
                    {"combined": prompt},
                    temperature=0.4,
                    max_tokens=128,
                )
            else:
                response = self._call_llm_with_system_prompt(
                    {"system": system_prompt, "user": prompt},
                    temperature=0.4,
                    max_tokens=128,
                )

            result["success"] = True
            result["response"] = response

        except Exception as e:
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)
            result["original_exception"] = e

        return result


def _result_values(results):
    for result in results:
        if result.error is not None:
            raise result.error
    return [result.value for result in results]


def _transport_batch_results(results):
    from .provider_batches import BatchItemError

    aligned = []
    for result in results:
        if isinstance(result, BatchItemError):
            if getattr(result, "status_code", None) in (400, 401, 403, 404, 422):
                raise FailFastLLMError(
                    str(result), original_exception=result
                ) from result
            aligned.append(CallResult(error=LLMBatchItemError(str(result))))
        elif isinstance(result, Exception):
            raise result
        else:
            aligned.append(CallResult(value=result))
    return aligned


def _ordered_anthropic_results(records):
    indexed = {}
    for record in records:
        key = record.custom_id
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise InvalidLLMInputError("Invalid batch result ID")
        index = int(key)
        if index in indexed:
            raise InvalidLLMInputError("Duplicate batch result ID")
        result = record.result
        if result.type == "succeeded":
            text = "".join(
                block.text
                for block in result.message.content
                if getattr(block, "type", None) == "text"
            )
            indexed[index] = CallResult(value=text)
        else:
            error = getattr(result, "error", None)
            error = getattr(error, "error", error)
            kind = getattr(error, "type", result.type)
            message = getattr(error, "message", str(error))
            if kind in (
                "authentication_error",
                "permission_error",
                "invalid_request_error",
                "not_found_error",
            ):
                raise FailFastLLMError(f"Batch item {key}: {kind}: {message}")
            indexed[index] = CallResult(
                error=LLMBatchItemError(f"Batch item {key}: {kind}: {message}")
            )
    if sorted(indexed) != list(range(len(indexed))):
        raise InvalidLLMInputError("Batch result IDs are missing or misaligned")
    return [indexed[index] for index in range(len(indexed))]


async def _await_owned_batch_operation(task):
    """Finish a bounded SDK operation even if its caller is cancelled again."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _run_managed_batch(wrapper, prompts, temperature, max_tokens):
    """Own one submission and cancel its job once when the operation is abandoned.

    Cancellation during synchronous submission waits for its eventual identifier
    before cancelling. SDK request timeouts bound those operations. Cleanup errors
    are logged and never replace the original cancellation or exception. Submission
    errors without an identifier are propagated; submission is never retried.
    """
    await asyncio.sleep(0)
    submission = asyncio.create_task(
        asyncio.to_thread(wrapper.submit_batch, prompts, temperature, max_tokens)
    )
    batch_id = None
    try:
        batch_id = await asyncio.shield(submission)
        if not await wrapper._wait_for_completion_async(batch_id):
            raise TimeoutError(f"Batch job {batch_id} did not complete")
        return await wrapper._retrieve_batch_results(batch_id)
    except (asyncio.CancelledError, Exception):
        if batch_id is None:
            # A failed submission exposes no job identifier. In particular, do
            # not retry an ambiguous SDK timeout which may already have created
            # a paid job. Preserve the caller's original cancellation if it won
            # a race with submission failure.
            if submission.done() and (
                submission.cancelled() or submission.exception() is not None
            ):
                raise
            try:
                batch_id = await _await_owned_batch_operation(submission)
            except (Exception, asyncio.CancelledError):
                logger.exception(
                    "Batch submission ended without an identifier during cleanup"
                )
        if batch_id is not None:
            cancellation = asyncio.create_task(
                asyncio.to_thread(wrapper.cancel_batch, batch_id)
            )
            try:
                await _await_owned_batch_operation(cancellation)
            except (Exception, asyncio.CancelledError):
                logger.exception("Failed to cancel abandoned batch %s", batch_id)
        raise


class AsyncLLMWrapper(DebugCallbackMixin, LLMErrorHandlingMixin, ABC):
    @property
    def supports_json_schema(self) -> bool:
        return False

    async def _call_single_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        """
        Execute a single provider request for the combined rendering of one prompt
        and return the raw text result from the model.

        Implementations should send ``prompt["combined"]``, which carries the whole
        instruction in a single message. This is the path taken when the wrapper
        reports `supports_system_prompts` as False.

        Subclasses should implement this method when their provider interaction
        follows the common pattern of issuing one request per prompt.

        This method should contain only provider-specific mechanics, such as:
            - constructing the provider request
            - calling the async SDK/API
            - extracting the returned text from the provider response
            - applying provider-specific concurrency controls (e.g., semaphores)

        This method should NOT implement:
            - retry logic
            - fail-fast handling
            - fallback behavior
            - batching or orchestration across prompts

        Those responsibilities are handled by the base class through
        `_safe_call_with_retry_result` and the batch orchestration methods.

        Override this method for most new async wrappers.

        To support true provider-managed batch jobs, subclasses may
        leave this unimplemented and instead override `_call_llm_batch` directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either _call_single_llm "
            f"or override _call_llm_batch"
        )

    async def _call_single_llm_with_system(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Execute a single provider request for the system/user rendering of one prompt
        and return the raw text result from the model.

        Implementations should send ``prompt["system"]`` and ``prompt["user"]`` as a
        system message and a user message. This is the path taken when the wrapper
        reports `supports_system_prompts` as True.

        Subclasses should implement this method when the provider supports system
        prompts and uses a one-request-per-prompt execution model.

        This method should contain only provider-specific mechanics, such as:
            - formatting the provider request with system and user prompts
            - calling the async SDK/API
            - extracting the returned text from the provider response
            - applying provider-specific concurrency controls (e.g., semaphores)

        This method should NOT implement:
            - retry logic
            - fail-fast handling
            - fallback behavior
            - batching or orchestration across prompts

        Those behaviors are handled by the base class through
        `_safe_call_with_retry_result` and `_call_llm_with_system_prompt_batch`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either "
            f"_call_single_llm_with_system or override "
            f"_call_llm_with_system_prompt_batch"
        )

    async def _call_llm_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> List[CallResult[str]]:
        """
        Process a batch of prompts using their combined rendering, and return one
        CallResult per prompt.

        The default implementation wraps `_call_single_llm` with retry and error
        handling via `_safe_call_with_retry_result` and runs all prompts concurrently
        using `asyncio.gather`.

        This produces the standard async behavior used by most wrappers:

            - retryable errors are retried per prompt
            - fail-fast errors abort the entire batch/layer
            - exhausted retryable errors return CallResult(error=...)
            - successful calls return CallResult(value=<text>)

        Subclasses normally should NOT override this method if their provider interaction model
        is "one async request per prompt". Instead, implement `_call_single_llm`
        and inherit this default batching behavior.

        Override this method when using a fundamentally different batch
        model than concurrent single-call execution, such as:

            - provider-managed batch job APIs
            - bulk endpoints accepting multiple prompts in one request
            - server-side batching that must be coordinated as a unit

        In the current architecture, such providers may still inherit from
        AsyncLLMWrapper and override this method directly.

        If a dedicated batch wrapper base class (for example LLMBatchWrapper) is
        introduced in the future, these implementations may move there instead.

        Note:
            Some legacy wrapper implementations override `_call_llm_batch`
            directly even with a "one async request per prompt". Those implementations
            remain supported and will take precedence over this default method.
        """
        tasks = [
            self._safe_call_with_retry_result(
                self._call_single_llm,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                routine=routine,
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def _call_llm_with_system_prompt_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> List[CallResult[str]]:
        """
        Process a batch of prompts using their system/user rendering, and return one
        CallResult per prompt.

        The default implementation wraps `_call_single_llm_with_system` with retry and
        error handling via `_safe_call_with_retry_result` and executes all prompts
        concurrently using `asyncio.gather`.

        This produces the standard async behavior used by most wrappers:

            - retryable errors are retried per prompt
            - fail-fast errors abort the entire batch/layer
            - exhausted retryable errors return CallResult(error=...)
            - successful calls return CallResult(value=<text>)

        Subclasses normally should NOT override this method if their provider model is
        "one async request per prompt". Instead, implement
        `_call_single_llm_with_system` and inherit this default batching behavior.

        Override this method when the provider uses a fundamentally different batch
        model than concurrent single-call execution, such as:

            - provider-managed batch job APIs
            - bulk endpoints accepting multiple prompts in one request
            - server-side batching that must be coordinated as a unit

        In the current architecture, such providers may still inherit from
        AsyncLLMWrapper and override this method directly.

        If a dedicated batch wrapper base class (for example LLMBatchWrapper) is
        introduced in the future, these implementations may move there instead.

        Note:
            Some legacy wrapper implementations override `_call_llm_with_system_prompt_batch`
            with a "one async request per prompt" model. Those implementations remain
            supported and will take precedence over this default method.
        """
        tasks = [
            self._safe_call_with_retry_result(
                self._call_single_llm_with_system,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                routine=routine,
            )
            for prompt in prompts
        ]

        return await asyncio.gather(*tasks)

    async def _call_llm_batch_for_prompts(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> List[CallResult[str]]:
        """
        Send a batch of prompts using whichever rendering this wrapper's provider
        supports.

        This is the single point at which a prompt's renderings are resolved down to
        one provider call, so that everything upstream of the wrapper can stay
        provider agnostic.
        """
        supports_system_prompts = self.supports_system_prompts
        prompts = [
            validate_prompt(prompt, supports_system_prompts) for prompt in prompts
        ]

        if supports_system_prompts:
            return await self._call_llm_with_system_prompt_batch(
                prompts, temperature, max_tokens=max_tokens
            )

        return await self._call_llm_batch(prompts, temperature, max_tokens=max_tokens)

    async def generate_topic_names(
        self,
        prompts: List[Prompt | str | dict],
        temperature: float = 0.4,
        extract_topic_name_function=None,
        get_topic_name_regex=GET_TOPIC_NAME_REGEX,
        null_result_value=None,
        max_tokens: int | None = None,
        *,
        response_parser: Callable | None = None,
        return_results: bool = False,
    ) -> List:
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_topic_name", 128)

        def parse(raw):
            if response_parser is not None:
                return response_parser(raw)
            info = llm_output_to_result(raw, get_topic_name_regex)
            return (
                extract_topic_name_function(info)
                if extract_topic_name_function
                else string_field(info, "topic_name")
            )

        results = await self._generate_results(
            prompts, [parse] * len(prompts), temperature, max_tokens
        )
        if return_results:
            return results
        values = []
        for result in results:
            if result.error is not None:
                if null_result_value is None:
                    raise result.error
                values.append(null_result_value)
            else:
                values.append(result.value)
        return values

    async def _generate_results(self, prompts, parsers, temperature, max_tokens):
        _validate_generation_options(temperature, max_tokens)
        normalized = [
            validate_prompt(prompt, self.supports_system_prompts) for prompt in prompts
        ]
        if not normalized:
            return []
        if getattr(self, "use_json_schema", None) is True:
            request_builder = getattr(self, "_provider_kwargs", None)
            if request_builder is not None:
                for prompt in normalized:
                    request_builder([], temperature, max_tokens, prompt=prompt)
            else:
                if not self.supports_json_schema:
                    raise InvalidLLMInputError(
                        "This batch provider/model does not support JSON Schema"
                    )
                if any(prompt.get("json_schema") is None for prompt in normalized):
                    raise InvalidLLMInputError(
                        "use_json_schema=True requires a schema for every prompt"
                    )
        batch_method = (
            self._call_llm_with_system_prompt_batch
            if self.supports_system_prompts
            else self._call_llm_batch
        )
        default_method = (
            AsyncLLMWrapper._call_llm_with_system_prompt_batch
            if self.supports_system_prompts
            else AsyncLLMWrapper._call_llm_batch
        )
        if getattr(batch_method, "__func__", None) is not default_method:
            # Provider-managed jobs are submitted once; never resubmit an entire
            # paid batch to repair one malformed item.
            responses = await batch_method(normalized, temperature, max_tokens)
            if len(responses) != len(normalized):
                raise InvalidLLMInputError(
                    "Batch result count does not match prompt count"
                )
            results = []
            for response, parse in zip(responses, parsers):
                if isinstance(response, CallResult):
                    if response.error is not None:
                        if not isinstance(
                            response.error, LLMBatchItemError
                        ) and not _should_retry(response.error):
                            self._handle_exception(response.error)
                        results.append(response)
                        continue
                    response = response.value
                try:
                    results.append(CallResult(value=parse(response)))
                except ResponseParseError as error:
                    results.append(CallResult(error=error))
            return results
        method = (
            self._call_single_llm_with_system
            if self.supports_system_prompts
            else self._call_single_llm
        )

        tasks = [
            asyncio.create_task(
                self._safe_call_with_retry_result(
                    method,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_parser=parse,
                )
            )
            for prompt, parse in zip(normalized, parsers)
        ]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def generate_topic_cluster_names(
        self,
        prompts: List[Prompt | str | dict],
        old_names_list: List[List[str]],
        temperature: float = 0.4,
        extract_topic_names_function=default_extract_topic_names,
        get_topic_names_regex=GET_TOPIC_CLUSTER_NAMES_REGEX,
        max_tokens: int | None = None,
        *,
        response_parser: Callable | None = None,
        return_results: bool = False,
    ) -> List:
        if len(prompts) != len(old_names_list):
            raise InvalidLLMInputError("Number of prompts must match old_names lists")
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_cluster_names", 1024)

        def parser(old_names):
            def parse(raw):
                names = (
                    response_parser(raw)
                    if response_parser
                    else extract_topic_names_function(
                        llm_output_to_result(raw, get_topic_names_regex), old_names, raw
                    )
                )
                if len(names) != len(old_names):
                    raise ResponseParseError(
                        "Response must contain one name per input topic"
                    )
                return names

            return parse

        results = await self._generate_results(
            prompts,
            [parser(names) for names in old_names_list],
            temperature,
            max_tokens,
        )
        if return_results:
            return results
        for result in results:
            if result.error is not None:
                raise result.error
        return [result.value for result in results]

    def _parse_cluster_response(
        self, response, old_names, extract_topic_names_function, get_topic_names_regex
    ):
        return extract_topic_names_function(
            llm_output_to_result(response, get_topic_names_regex), old_names, response
        )

    @property
    def supports_system_prompts(self) -> bool:
        """
        Check if the LLM wrapper supports system prompts.
        By default, it does. Override in subclasses if not supported.
        """
        return True

    async def test_llm_connectivity(self) -> str:
        result = await self.connectivity_status()

        if result["success"]:
            logger.info(
                " Connected to %s using %s",
                result["wrapper"],
                result["model"],
            )
            return result["response"]

        logger.warning(
            "  Failed to connect to %s using %s",
            result["wrapper"],
            result["model"],
        )
        logger.warning(
            "  Cause:  %s: %s",
            result["error_type"],
            result["error_message"],
        )

        return "<error>"

    async def connectivity_status(
        self,
        prompt: str = (
            "Identify yourself and explain that you will be providing "
            "topic names for clusters in JSON format"
        ),
        *,
        system_prompt: str | None = None,
    ) -> dict:
        result = {
            "success": False,
            "model": self.model,
            "wrapper": self.__class__.__name__,
            "response": None,
            "error_type": None,
            "error_message": None,
            "original_exception": None,
        }

        try:
            if system_prompt is None:
                probe_prompt = {"combined": prompt}
                try:
                    response = await self._call_single_llm(
                        probe_prompt, temperature=0.4, max_tokens=128
                    )
                except NotImplementedError:
                    responses = await self._call_llm_batch(
                        [probe_prompt], temperature=0.4, max_tokens=128
                    )
                    if not responses:
                        raise RuntimeError("Connectivity probe returned no responses")
                    response = responses[0]
            else:
                probe_prompt = {"system": system_prompt, "user": prompt}
                try:
                    response = await self._call_single_llm_with_system(
                        probe_prompt,
                        temperature=0.4,
                        max_tokens=128,
                    )
                except NotImplementedError:
                    responses = await self._call_llm_with_system_prompt_batch(
                        [probe_prompt],
                        temperature=0.4,
                        max_tokens=128,
                    )
                    if not responses:
                        raise RuntimeError("Connectivity probe returned no responses")
                    response = responses[0]

            if isinstance(response, CallResult):
                if not response.ok:
                    raise response.error
                response = response.value

            result["success"] = True
            result["response"] = response

        except Exception as e:
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)
            result["original_exception"] = e

        return result

    async def close(self) -> None:
        """
        Optional cleanup hook for LLM wrappers that manage network clients or connection pools.
        """
        pass


class LLMWrapperImportError(ImportError):
    """A custom exception for missing package dependencies required by LLM wrappers. In these cases we do not want to retry, as the error will not resolve until the required package is installed."""

    pass


class FailedImportLLMWrapper(LLMWrapper):

    @classmethod
    def _import_error_message(cls):
        return f"Failed to import LLMWrapper for {cls.__name__}. This is likely because the required package is not installed. Please install the required package and try again."

    def __init__(self, *args, **kwds):
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm_with_system_prompt(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        raise LLMWrapperImportError(self._import_error_message())

    def test_llm_connectivity(
        self,
        prompt="Identify yourself and explain that you will be providing topic names for clusters",
    ):
        return LLMWrapperImportError(self._import_error_message())


class FailedImportAsyncLLMWrapper(AsyncLLMWrapper):
    @classmethod
    def _import_error_message(cls):
        return f"Failed to import AsyncLLMWrapper for {cls.__name__}. This is likely because the required package is not installed. Please install the required package and try again."

    def __init__(self, *args, **kwds):
        raise LLMWrapperImportError(self._import_error_message())

    async def _call_llm_batch(
        self, prompts: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())

    async def _call_llm_with_system_prompt_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())

    async def test_llm_connectivity(
        self,
        prompt="Identify yourself and explain that you will be providing topic names for clusters",
    ):
        return LLMWrapperImportError(self._import_error_message())


####
# LLM Wrappers
####


# Model String Helpers to convert to LiteLLM-style
def _openai_model(model: str) -> str:
    return f"openai/{model}" if "/" not in model else model


def _anthropic_model(model: str) -> str:
    return f"anthropic/{model}" if "/" not in model else model


def _cohere_model(model: str) -> str:
    return f"cohere/{model}" if "/" not in model else model


def _together_model(model: str) -> str:
    return f"together_ai/{model}" if "together_ai/" not in model else model


def _azure_model(model: str) -> str:
    return f"azure_ai/{model}" if "azure_ai/" not in model else model


def _gemini_model(model: str) -> str:
    return f"gemini/{model}" if "gemini/" not in model else model


def _ollama_model(model: str) -> str:
    return f"ollama_chat/{model}" if "ollama_chat/" not in model else model


def _replicate_model(model: str) -> str:
    return f"replicate/{model}" if "replicate/" not in model else model


def _get_litellm():
    # Importing LiteLLM can initialize tokenizers and provider metadata. Keep it
    # outside package import and postpone it until an integration is requested.
    import importlib

    return importlib.import_module("litellm")


def _validate_json_options(use_json_schema, use_json_object, provider_kwargs):
    if provider_kwargs is not None and not isinstance(provider_kwargs, dict):
        raise InvalidLLMInputError("provider_kwargs must be a dictionary")
    for value in (use_json_schema, use_json_object):
        if value is not None and not isinstance(value, bool):
            raise InvalidLLMInputError("JSON output options must be bool or None")
    if use_json_schema is True and use_json_object is True:
        raise InvalidLLMInputError(
            "use_json_schema and use_json_object cannot both be True"
        )
    if (
        provider_kwargs
        and "response_format" in provider_kwargs
        and (use_json_schema is not None or use_json_object is not None)
    ):
        raise InvalidLLMInputError(
            "response_format conflicts with explicit JSON output options"
        )
    reserved = {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "api_key",
        "api_base",
        "num_retries",
        "max_retries",
    }
    conflicts = reserved.intersection(provider_kwargs or {})
    if conflicts:
        raise InvalidLLMInputError(
            f"provider_kwargs conflicts with named request options: {sorted(conflicts)}"
        )
    if provider_kwargs and "response_format" in provider_kwargs:
        response_format = provider_kwargs["response_format"]
        if not isinstance(response_format, dict) or response_format.get("type") not in (
            "text",
            "json_object",
            "json_schema",
        ):
            raise InvalidLLMInputError(
                "response_format must declare text, json_object, or json_schema"
            )
        if response_format["type"] == "json_schema":
            specification = response_format.get("json_schema")
            if (
                not isinstance(specification, dict)
                or not isinstance(specification.get("name"), str)
                or not specification["name"]
            ):
                raise InvalidLLMInputError(
                    "response_format json_schema requires a name and schema"
                )
            if not isinstance(specification.get("schema"), dict):
                raise InvalidLLMInputError(
                    "response_format json_schema requires a schema object"
                )
            _validate_json_schema(specification["schema"])


def _provider_request_kwargs(wrapper, prompt, messages, temperature, max_tokens):
    _validate_json_options(
        wrapper.use_json_schema, wrapper.use_json_object, wrapper.provider_kwargs
    )
    _validate_generation_options(temperature, max_tokens)
    kwargs = dict(wrapper.provider_kwargs)
    if "response_format" in kwargs:
        kwargs["response_format"] = deepcopy(kwargs["response_format"])
    explicit_format = kwargs.get("response_format", {}).get("type")
    if explicit_format == "json_schema" and not wrapper.supports_json_schema:
        raise InvalidLLMInputError(
            f"Model {wrapper.model} does not support JSON Schema"
        )
    if explicit_format == "json_object" and not wrapper._detect_json_object_support():
        raise InvalidLLMInputError(
            f"Model {wrapper.model} does not support JSON object output"
        )
    schema = prompt.get("json_schema") if prompt else None
    if schema is not None:
        if not isinstance(schema, dict) or not schema:
            raise InvalidLLMInputError(
                "Prompt json_schema must be a nonempty schema object"
            )
        _validate_json_schema(schema)
    if wrapper.use_json_schema is True and schema is None:
        raise InvalidLLMInputError("use_json_schema=True requires a prompt json_schema")
    schema_supported = False
    if (
        schema is not None
        and wrapper.use_json_schema is not False
        and wrapper.use_json_object is not True
        and "response_format" not in kwargs
    ):
        schema_supported = wrapper.supports_json_schema
    if wrapper.use_json_schema is True and not schema_supported:
        raise InvalidLLMInputError(
            f"Model {wrapper.model} does not support JSON Schema"
        )
    if "response_format" not in kwargs:
        if schema is not None and schema_supported:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "toponymy_response",
                    "strict": True,
                    "schema": deepcopy(schema),
                },
            }
        elif wrapper.use_json_object is not False:
            object_supported = wrapper._detect_json_object_support()
            if wrapper.use_json_object is True and not object_supported:
                raise InvalidLLMInputError(
                    f"Model {wrapper.model} does not support JSON object output"
                )
            if object_supported:
                kwargs["response_format"] = {"type": "json_object"}
    kwargs.update(
        model=wrapper.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=0,
    )
    if wrapper.api_key is not None:
        kwargs["api_key"] = wrapper.api_key
    if wrapper.api_base is not None:
        kwargs["api_base"] = wrapper.api_base
    return kwargs


class LiteLLMNamer(LLMWrapper):
    """
    Provides access to any LLM supported by LiteLLM using a unified interface.
    LiteLLM supports 100+ providers including OpenAI, Anthropic, Cohere, HuggingFace,
    Together, Replicate, and more. For more information, see https://docs.litellm.ai.

    Parameters
    ----------
    api_key: str, optional
        The API key for the provider. If not provided, LiteLLM will look for the
        appropriate environment variable for the provider (e.g. OPENAI_API_KEY,
        ANTHROPIC_API_KEY).

    model: str, optional
        The LiteLLM model string, e.g. "openai/gpt-4o-mini",
        "anthropic/claude-haiku-4-5-20251001", etc.

    api_base: str, optional
        Optional LiteLLM/OpenAI-compatible API base. Alias-style convenience.

    llm_specific_instructions: str, optional
        Additional instructions appended to the user prompt.

    use_json_object: bool, optional
        Whether to request JSON object output via response_format={"type": "json_object"}.
        If None (default), support is detected automatically by check if response_format is supported
        for the specified model. Set to True to force JSON object mode, or False to
        disable it.

    disable_system_prompts: bool, False
        Set to True to override to use plain calls instead of system prompts.
        If False (default), system prompt support is detected automatically and will flatten system prompts
        if unsupported for a given model.

    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().

    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().

    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.

    provider_kwargs : dict[str, Any], optional
        Additional keyword arguments passed directly to `_get_litellm().completion()` /
        `_get_litellm().acompletion()`. This allows callers to use LiteLLM-specific
        features such as provider routing, request timeouts, custom headers,
        user identifiers, or other provider parameters without modifying the
        wrapper.

        These values are merged into the completion call arguments but may be
        overridden by core wrapper parameters such as `model`, `messages`,
        `temperature`, and `max_tokens`.

    Attributes
    ----------
    model: str
        The LiteLLM model string being used.

    extra_prompting: str
        Additional instructions appended to the prompt.

    max_tokens_topic_name: int
        Default maximum tokens for topic name generation.

    max_tokens_cluster_names: int
        Default maximum tokens for cluster name generation.

    use_json_object: bool
        Whether response_format={"type": "json_object"} will be sent.
    """

    FAIL_FAST_EXCEPTIONS = ()
    _supports_debug_callback = True

    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/gpt-4o-mini",
        api_base: str = None,
        llm_specific_instructions=None,
        use_json_object: bool = None,
        disable_system_prompts: bool = False,
        max_tokens_topic_name: int = 128,
        max_tokens_cluster_names: int = 1024,
        temperature_override: float | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        callback: DebugCallback | None = None,
        use_json_schema: bool | None = None,
    ):

        _validate_json_options(use_json_schema, use_json_object, provider_kwargs)
        self.use_json_schema = use_json_schema
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.temperature_override = temperature_override
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.use_json_object = use_json_object  # set by user
        self._resolved_use_json_object: bool | None = None  # set internally
        self.disable_system_prompts = disable_system_prompts
        self._system_prompt_capability: bool | None = None
        self.max_tokens_topic_name = max_tokens_topic_name
        self.max_tokens_cluster_names = max_tokens_cluster_names
        self.provider_kwargs = dict(provider_kwargs) if provider_kwargs else {}
        if "response_format" in self.provider_kwargs:
            self.provider_kwargs["response_format"] = deepcopy(
                self.provider_kwargs["response_format"]
            )

    @property
    def supports_system_prompts(self) -> bool:
        if getattr(self, "_capability_model", None) != self.model:
            self._system_prompt_capability = None
            self._capability_model = self.model
        if self.disable_system_prompts:
            return False
        return True

    def _looks_like_unsupported_system_prompt_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            s in message
            for s in (
                "system role",
                "system message",
                "unsupported role",
                "invalid role",
                "does not support system",
            )
        )

    def _flatten_system_into_user(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": f"System: {system_prompt}\n\nUser: {user_prompt + self.extra_prompting}",
            }
        ]

    def _detect_json_object_support(self) -> bool:
        supported = _get_litellm().get_supported_openai_params(model=self.model)
        return "response_format" in (supported or [])

    def _should_use_json_object(self) -> bool:
        if self.use_json_object is not None:
            return self.use_json_object

        return self._detect_json_object_support()

    def _provider_kwargs(self, messages, temperature, max_tokens, prompt=None) -> dict:
        return _provider_request_kwargs(self, prompt, messages, temperature, max_tokens)

    @property
    def supports_json_schema(self) -> bool:
        return bool(_get_litellm().supports_response_schema(model=self.model))

    def _completion_with_messages(
        self,
        messages,
        temperature: float,
        max_tokens: int,
        prompt=None,
    ) -> str:
        response = _get_litellm().completion(
            **self._provider_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt,
            )
        )
        return response.choices[0].message.content

    def _call_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        effective_temperature = (
            self.temperature_override
            if self.temperature_override is not None
            else temperature
        )
        return self._completion_with_messages(
            messages=[
                {
                    "role": "user",
                    "content": prompt["combined"] + self.extra_prompting,
                },
            ],
            temperature=effective_temperature,
            max_tokens=max_tokens,
            prompt=prompt,
        )

    def _call_llm_with_system_prompt(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        system_prompt = prompt["system"]
        user_prompt = prompt["user"]
        effective_temperature = (
            self.temperature_override
            if self.temperature_override is not None
            else temperature
        )
        if self._system_prompt_capability is False:
            messages = self._flatten_system_into_user(system_prompt, user_prompt)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ]

        try:
            result = self._completion_with_messages(
                messages=messages,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                prompt=prompt,
            )
            if self._system_prompt_capability is None:
                self._system_prompt_capability = True
            return result

        except self.FAIL_FAST_EXCEPTIONS:
            raise

        except Exception as e:
            if (
                getattr(e, "status_code", None) not in (400, 422)
                or not any(message["role"] == "system" for message in messages)
                or not self._looks_like_unsupported_system_prompt_error(e)
            ):
                raise

            self._system_prompt_capability = False
            raise _SystemPromptFallback(
                "Provider requires system instructions in the user message"
            ) from e


class AsyncLiteLLMNamer(AsyncLLMWrapper):
    """
    Provides access to any LLM supported by LiteLLM with asynchronous support.
    This allows for concurrent processing of multiple prompts across 100+ providers
    including OpenAI, Anthropic, Cohere, HuggingFace, Together, Replicate, and more.
    For more information, see https://docs.litellm.ai.

    As an asynchronous wrapper this will potentially speed up topic naming, particularly
    when you have a large number of topics. If, however, there are quirks in your data,
    or bugs in Toponymy's prompt generation, you will potentially quickly spend money on
    API calls.

    Uses _get_litellm().acompletion() and an asyncio semaphore for bounded
    concurrency. Since this wrapper does not create a persistent SDK client,
    close() is a no-op.


    Parameters:
    -----------
    api_key: str, optional
        The API key for the provider. If not provided, LiteLLM will look for the
        appropriate environment variable for the provider (e.g. OPENAI_API_KEY,
        ANTHROPIC_API_KEY).

    model: str
        The model to use in LiteLLM format, e.g. "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku-20240307", "together_ai/mistralai/Mixtral-8x7B-v0.1".
        See https://docs.litellm.ai/docs/providers for the full list.

    api_base: str, optional
        The base URL for the provider API. Useful for self-hosted models or proxies.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt.

    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the provider API. Default is 10.
        This can be adjusted based on your application's needs and the rate limits of
        the provider. Higher values may improve throughput but could lead to rate limiting.

    disable_system_prompts: bool, False
        Set to True to override to use plain calls instead of system prompts.
        If False (default), system prompt support is detected automatically and will flatten system prompts
        if unsupported for a given model.

    use_json_object: bool, optional
        Whether to request JSON object output via response_format={"type": "json_object"}.
        If None (default), support is detected automatically by check if response_format is supported
        for the specified model. Set to True to force JSON object mode, or False to
        disable it.

    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_names().

    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().

    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.

    provider_kwargs : dict[str, Any], optional
        Additional keyword arguments passed directly to `_get_litellm().completion()` /
        `_get_litellm().acompletion()`. This allows callers to use LiteLLM-specific
        features such as provider routing, request timeouts, custom headers,
        user identifiers, or other provider parameters without modifying the
        wrapper.

        These values are merged into the completion call arguments but may be
        overridden by core wrapper parameters such as `model`, `messages`,
        `temperature`, and `max_tokens`.

    Attributes:
    -----------
    model: str
        The LiteLLM model string being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    max_tokens_topic_name: int
        Default maximum tokens for topic name generation.

    max_tokens_cluster_names: int
        Default maximum tokens for cluster name generation.
    """

    FAIL_FAST_EXCEPTIONS = ()
    _supports_debug_callback = True

    def __init__(
        self,
        api_key: str = None,
        model: str = "openai/gpt-4o-mini",
        api_base: str = None,
        llm_specific_instructions: str = None,
        max_concurrent_requests: int = 10,
        use_json_object: bool | None = None,
        disable_system_prompts: bool = False,
        max_tokens_topic_name: int = 128,
        max_tokens_cluster_names: int = 1024,
        temperature_override: float | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        callback: DebugCallback | None = None,
        use_json_schema: bool | None = None,
    ):

        _validate_json_options(use_json_schema, use_json_object, provider_kwargs)
        self.use_json_schema = use_json_schema
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.temperature_override = temperature_override
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        if (
            isinstance(max_concurrent_requests, bool)
            or not isinstance(max_concurrent_requests, int)
            or max_concurrent_requests < 1
        ):
            raise InvalidLLMInputError(
                "max_concurrent_requests must be a positive integer"
            )
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.use_json_object = use_json_object
        self._resolved_use_json_object: bool | None = None
        self.disable_system_prompts = disable_system_prompts
        self._system_prompt_capability: bool | None = None
        self.max_tokens_topic_name = max_tokens_topic_name
        self.max_tokens_cluster_names = max_tokens_cluster_names
        self.provider_kwargs = dict(provider_kwargs) if provider_kwargs else {}
        if "response_format" in self.provider_kwargs:
            self.provider_kwargs["response_format"] = deepcopy(
                self.provider_kwargs["response_format"]
            )

    @property
    def supports_system_prompts(self) -> bool:
        if getattr(self, "_capability_model", None) != self.model:
            self._system_prompt_capability = None
            self._capability_model = self.model
        if self.disable_system_prompts:
            return False
        return True

    def _looks_like_unsupported_system_prompt_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            s in message
            for s in (
                "system role",
                "system message",
                "unsupported role",
                "invalid role",
                "does not support system",
            )
        )

    def _flatten_system_into_user(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": f"System: {system_prompt}\n\nUser: {user_prompt + self.extra_prompting}",
            }
        ]

    def _detect_json_object_support(self) -> bool:
        supported = _get_litellm().get_supported_openai_params(model=self.model)
        return "response_format" in (supported or [])

    def _should_use_json_object(self) -> bool:
        if self.use_json_object is not None:
            return self.use_json_object

        return self._detect_json_object_support()

    def _provider_kwargs(self, messages, temperature, max_tokens, prompt=None) -> dict:
        return _provider_request_kwargs(self, prompt, messages, temperature, max_tokens)

    @property
    def supports_json_schema(self) -> bool:
        return bool(_get_litellm().supports_response_schema(model=self.model))

    async def _acompletion_with_messages(
        self,
        messages,
        temperature: float,
        max_tokens: int,
        prompt=None,
    ) -> str:
        async with self.semaphore:
            response = await _get_litellm().acompletion(
                **self._provider_kwargs(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prompt=prompt,
                )
            )
        return response.choices[0].message.content

    async def _call_single_llm(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        effective_temperature = (
            self.temperature_override
            if self.temperature_override is not None
            else temperature
        )
        return await self._acompletion_with_messages(
            messages=[
                {
                    "role": "user",
                    "content": prompt["combined"] + self.extra_prompting,
                }
            ],
            temperature=effective_temperature,
            max_tokens=max_tokens,
            prompt=prompt,
        )

    async def _call_single_llm_with_system(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        system_prompt = prompt["system"]
        user_prompt = prompt["user"]
        effective_temperature = (
            self.temperature_override
            if self.temperature_override is not None
            else temperature
        )
        if self._system_prompt_capability is False:
            messages = self._flatten_system_into_user(system_prompt, user_prompt)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ]
        try:
            # If the model doesn't support system prompts, this will raise an error which we
            # catch to disable system prompt usage for future calls. Everything else raises as normal.
            result = await self._acompletion_with_messages(
                messages=messages,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                prompt=prompt,
            )
            if self._system_prompt_capability is None:
                self._system_prompt_capability = True
            return result

        except self.FAIL_FAST_EXCEPTIONS:
            raise

        except Exception as e:
            if (
                getattr(e, "status_code", None) not in (400, 422)
                or not any(message["role"] == "system" for message in messages)
                or not self._looks_like_unsupported_system_prompt_error(e)
            ):
                raise

            self._system_prompt_capability = False
            raise _SystemPromptFallback(
                "Provider requires system instructions in the user message"
            ) from e

    async def close(self):
        """No-op for parity with other async wrappers."""
        return None


def AnthropicNamer(
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    Create a LiteLLMNamer configured for Anthropic with convenient defaults for
    topic naming. For more flexibility, use LiteLLMNamer directly with the model and parameters of your choice.

    All namers share the same interface once constructed — AnthropicNamer is a
    convenience entry point, not a special case.

    Parameters
    ----------
    model : str, optional
        Anthropic model to use. Default is "claude-haiku-4-5-20251001".
        May be in LiteLLM format ("anthropic/claude-haiku-4-5-20251001")
    api_key : str, optional
        Anthropic API key. Falls back to the ANTHROPIC_API_KEY environment variable.
    api_base : str, optional
        Override the Anthropic API endpoint. Useful for proxies or Anthropic-compatible
        local servers (e.g. vLLM, LM Studio). Can use the ANTHROPIC_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.

    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().

    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().

    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.

    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AnthropicNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AnthropicNamer(model="claude-3-5-20251001", api_key="my-api-key")

    Using an Anthropic-compatible local server::

        namer = AnthropicNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    return LiteLLMNamer(
        model=_anthropic_model(model),
        api_key=api_key,
        api_base=api_base,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncAnthropicNamer(
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    Create an AsyncLiteLLMNamer configured for Anthropic with convenient defaults.
    For more flexibility, use AsyncLiteLLMNamer directly with the model and parameters of your choice.

    All namers share the same interface once constructed — AnthropicNamer is a
    convenience entry point, not a special case.

    Parameters
    ----------
    model : str, optional
        Anthropic model to use. Default is "claude-haiku-4-5-20251001". Must be in LiteLLM format ("anthropic/claude-haiku-4-5-20251001")
        or bare Anthropic format ("claude-haiku-4-5-20251001") — both are accepted.
    api_key : str, optional
        Anthropic API key. Falls back to the ANTHROPIC_API_KEY environment variable.
    api_base : str, optional
        Override the Anthropic API endpoint. Useful for proxies or Anthropic-compatible
        local servers (e.g. vLLM, LM Studio). Can use the ANTHROPIC_API_BASE environment variable.
        Default is the standard Anthropic endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Anthropic API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Anthropic API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncAnthropicNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AsyncAnthropicNamer(model="claude-3-5-20251001", api_key="my-api-key")

    Using an Anthropic-compatible local server::

        namer = AsyncAnthropicNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    return AsyncLiteLLMNamer(
        model=_anthropic_model(model),
        api_key=api_key,
        api_base=api_base,
        disable_system_prompts=False,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def _resolve_cohere_api_base(
    api_base: str | None,
    base_url: str | None,
    env_new: str = "COHERE_API_BASE",
    env_legacy: str = "CO_API_URL",
) -> str | None:
    """Helper function for migrating previous base_url defaults"""

    if api_base is not None:
        return api_base

    if base_url is not None:
        warn(
            "base_url is deprecated. Use api_base instead.",
            FutureWarning,
            stacklevel=3,
        )
        return base_url

    new_val = os.getenv(env_new)
    if new_val:
        return new_val

    legacy_val = os.getenv(env_legacy)
    if legacy_val:
        warn(
            f"{env_legacy} is deprecated. Use {env_new} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return legacy_val

    return None


def CohereNamer(
    model: str = "command-r-08-2024",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str | None = None,  # deprecated, renamed to api_base
    httpx_client: Optional[httpx.Client] = None,  # deprecated
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    Create a LiteLLMNamer configured for Cohere with convenient defaults for
    topic naming. For more flexibility, use LiteLLMNamer directly with the model and parameters of your choice.

    All namers share the same interface once constructed — CohereNamer is a
    convenience entry point, not a special case.

    Parameters
    ----------
    model : str, optional
        Cohere model to use. Default is "command-r-08-2024".
        May be in LiteLLM format ("cohere/command-r-08-2024")
    api_key : str, optional
        Cohere API key. Falls back to the COHERE_API_KEY environment variable.
    api_base : str, optional
        Override the Cohere API endpoint. Useful for proxies or Cohere-compatible
        local servers (e.g. vLLM, LM Studio). Can use the COHERE_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    base_url : str, optional
        Deprecated. Use ``api_base`` instead.
    httpx_client : optional
        Deprecated. Pass via ``provider_kwargs={'httpx_client': <client>}`` instead.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = CohereNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = CohereNamer(model="command-r-08-2024", api_key="my-api-key")

    Using a Cohere-compatible local server::

        namer = CohereNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    if httpx_client is not None:
        warn(
            "httpx_client is deprecated. "
            "Pass via provider_kwargs={'httpx_client': httpx_client} instead.",
            FutureWarning,
            stacklevel=2,
        )
        provider_kwargs = provider_kwargs or {}
        provider_kwargs["httpx_client"] = httpx_client
    return LiteLLMNamer(
        model=_cohere_model(model),
        api_key=resolve_api_key(
            api_key=api_key, env_new="COHERE_API_KEY", env_legacy="CO_API_KEY"
        ),
        api_base=_resolve_cohere_api_base(api_base, base_url),
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncCohereNamer(
    model: str = "command-r-08-2024",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str = None,
    httpx_client: Optional[httpx.Client] = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    Create an AsyncLiteLLMNamer configured for Cohere with convenient defaults.
    For more flexibility, use AsyncLiteLLMNamer directly with the model and parameters of your choice.

    All namers share the same interface once constructed — CohereNamer is a
    convenience entry point, not a special case.

    Parameters
    ----------
    model : str, optional
        Cohere model to use. Default is "command-r-08-2024". Must be in LiteLLM format ("cohere/command-r-08-2024")
        or bare Cohere format ("command-r-08-2024") — both are accepted.
    api_key : str, optional
        Cohere API key. Falls back to the COHERE_API_KEY environment variable.
    api_base : str, optional
        Override the Cohere API endpoint. Useful for proxies or Cohere-compatible
        local servers (e.g. vLLM, LM Studio). Can use the COHERE_API_BASE environment variable.
        Default is the standard Cohere endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Cohere API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Cohere API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    base_url : str, optional
        Deprecated. Use ``api_base`` instead.
    httpx_client : optional
        Deprecated. Pass via ``provider_kwargs={'httpx_client': <client>}`` instead.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncCohereNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AsyncCohereNamer(model="command-r-08-2024", api_key="my-api-key")

    Using a Cohere-compatible local server::

        namer = AsyncCohereNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    if httpx_client is not None:
        warn(
            "httpx_client is deprecated. "
            "Pass via provider_kwargs={'httpx_client': httpx_client} instead.",
            FutureWarning,
            stacklevel=2,
        )
        provider_kwargs = provider_kwargs or {}
        provider_kwargs["httpx_client"] = httpx_client
    return AsyncLiteLLMNamer(
        model=_cohere_model(model),
        api_key=resolve_api_key(
            api_key=api_key, env_new="COHERE_API_KEY", env_legacy="CO_API_KEY"
        ),
        api_base=_resolve_cohere_api_base(api_base, base_url),
        disable_system_prompts=False,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def TogetherNamer(
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    Deprecated. Use LiteLLMNamer(model="together_ai/<model_name>") instead.

    Parameters
    ----------
    model : str, optional
        Together AI model to use. Default is "meta-llama/Llama-3.3-70B-Instruct-Turbo".
        May be in LiteLLM format ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
    api_key : str, optional
        Together AI API key. Falls back to the TOGETHERAI_API_KEY environment variable.
    api_base : str, optional
        Override the Together AI API endpoint. Can use the TOGETHERAI_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = TogetherNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = TogetherNamer(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key="my-api-key")

    Using a Together AI-compatible local server::

        namer = TogetherNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    warn(
        (
            "TogetherNamer is deprecated and will be removed in a future "
            "release. Use LiteLLMNamer(model='together_ai/<model_name>') directly instead."
        ),
        FutureWarning,
        stacklevel=2,
    )
    return LiteLLMNamer(
        model=_together_model(model),
        api_key=api_key,
        api_base=api_base,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
    )


def AsyncTogether(
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    Deprecated. Use AsyncLiteLLMNamer(model="together_ai/<model_name>") instead.

    Parameters
    ----------
    model : str, optional
        Together AI model to use. Default is "meta-llama/Llama-3.3-70B-Instruct-Turbo". Must be in LiteLLM format ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
        or bare Together AI format ("meta-llama/Llama-3.3-70B-Instruct-Turbo") — both are accepted.
    api_key : str, optional
        Together AI API key. Falls back to the TOGETHERAI_API_KEY environment variable.
    api_base : str, optional
        Override the Together AI API endpoint. Useful for proxies or Together AI-compatible
        local servers. Can use the TOGETHERAI_API_BASE environment variable.
        Default is the standard Together AI endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Together AI API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Together AI API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncTogether(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AsyncTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key="my-api-key")

    Using a Together AI-compatible local server::

        namer = AsyncTogether(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    warn(
        (
            "AsyncTogether is deprecated and will be removed in a future "
            "release. Use AsyncLiteLLMNamer(model='together_ai/<model_name>') directly instead."
        ),
        FutureWarning,
        stacklevel=2,
    )

    return AsyncLiteLLMNamer(
        model=_together_model(model),
        api_key=api_key,
        api_base=api_base,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
    )


class LlamaCppNamer(LLMWrapper):
    """
    Provides Access to LlamaCpp models with the Toponymy framework. For more information on LlamaCpp, see
    https://github.com/abetlen/llama-cpp-python. You will need llamma-cpp-python installed to make use of
    this wrapper, and you will need a local model file downloaded to use it. This Wrapper allows you
    to use local models, rather than requiring a service API key. However this does require you to have the model
    and suitable hardware to run it.

    Note: This wrapper does not support system prompts, as LlamaCpp does not support them.

    Parameters:
    -----------

    model_path: str
        The path to the local LlamaCpp model file.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt.

    **kwargs: dict, optional
        Additional keyword arguments passed to the LlamaCpp model initialization.

    Attributes:
    -----------
    model_path: str
        The path to the local LlamaCpp model file.

    llm: llama_cpp.Llama
        The LlamaCpp model instance.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For LlamaCpp, this is always False.
    """

    def __init__(
        self,
        model_path: str,
        llm_specific_instructions=None,
        callback: DebugCallback | None = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.model = model_path
        for arg, val in kwargs.items():
            if arg == "n_ctx":
                continue
            setattr(self, arg, val)
        import llama_cpp

        self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        response = self.llm(
            prompt["combined"] + self.extra_prompting,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = response["choices"][0]["text"]
        return result

    def _call_llm_with_system_prompt(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise InvalidLLMInputError(
            "System prompts are not supported for LlamaCpp wrapper"
        )

    @property
    def supports_system_prompts(self) -> bool:
        return False


class HuggingFaceNamer(LLMWrapper):
    """
    Provides access to Huggingface models from Huggingface Hub with the Toponymy framework.
    For more information on Huggingface, see https://huggingface.co/docs/transformers/index.
    You will need the transformers library installed to make use of this wrapper, and you will need a model
    available on Huggingface Hub. This wrapper allows you to use models hosted on Huggingface Hub,
    rather than requiring a service API key. However, this does require you to have access to the model
    and suitable hardware to run it.

    Parameters:
    -----------
    model: str
        The name of the Huggingface model to use, e.g. "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-3-1b-it", etc.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt.

    **kwargs: dict, optional
        Additional keyword arguments passed to the Huggingface model initialization.

    Attributes:
    -----------
    model: str
        The name of the Huggingface model to use.

    llm: transformers.pipeline
        The Huggingface model instance.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For Huggingface, this is always True.
    """

    def __init__(
        self,
        model: str,
        llm_specific_instructions=None,
        callback: DebugCallback | None = None,
        **kwargs,
    ):
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        from transformers import pipeline

        self.llm = pipeline("text-generation", model=model, **kwargs)
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        response = self.llm(
            [
                {
                    "role": "user",
                    "content": prompt["combined"] + self.extra_prompting,
                }
            ],
            return_full_text=False,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )
        result = response[0]["generated_text"]
        return result

    def _call_llm_with_system_prompt(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.llm(
            [
                {"role": "system", "content": prompt["system"]},
                {
                    "role": "user",
                    "content": prompt["user"] + self.extra_prompting,
                },
            ],
            return_full_text=False,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )
        result = response[0]["generated_text"]
        print(result)
        return result


class AsyncHuggingFaceNamer(AsyncLLMWrapper):
    """This class is essentially for testing purposes only, allowing testing of the Async API with local models."""

    def __init__(
        self,
        model: str,
        llm_specific_instructions: Optional[str] = None,
        max_concurrent_requests: int = 10,
        callback: DebugCallback | None = None,
        **kwargs,
    ):
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        from transformers import pipeline

        self.llm = pipeline("text-generation", model=model, **kwargs)
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.max_concurrent_requests = max_concurrent_requests

    async def _call_llm_batch(
        self, prompts: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> List[str]:
        responses = []
        for prompt in prompts:
            response = await asyncio.to_thread(
                self.llm,
                [
                    {
                        "role": "user",
                        "content": prompt["combined"] + self.extra_prompting,
                    }
                ],
                return_full_text=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            responses.append(response[0]["generated_text"])
        return responses

    async def _call_llm_with_system_prompt_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        responses = []
        for prompt in prompts:
            response = await asyncio.to_thread(
                self.llm,
                [
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": prompt["user"] + self.extra_prompting,
                    },
                ],
                return_full_text=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            responses.append(response[0]["generated_text"])
        return responses


class VLLMNamer(LLMWrapper):
    """
    Provides access to Huggingface models from Huggingface Hub ran via vLLM, with the Toponymy framework.
    For more information on vLLM, see https://docs.vllm.ai/en/latest/.
    You will need the vllm library installed to make use of this wrapper, and you will need a model
    available on Huggingface Hub. This wrapper allows you to use models hosted on Huggingface Hub,
    rather than requiring a service API key. However, this does require you to have access to the model
    and suitable hardware to run it.

    Parameters:
    -----------
    model: str
        The name of the Huggingface model to use, e.g. "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-3-1b-it", etc.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt.

    **kwargs: dict, optional
        Additional keyword arguments passed to the vLLM model initialization.

    Attributes:
    -----------
    model: str
        The name of the Huggingface model to use.

    llm: transformers.pipeline
        The vLLM model instance.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For Huggingface, this is always True.
    """

    def __init__(
        self,
        model: str,
        llm_specific_instructions=None,
        callback: DebugCallback | None = None,
        **kwargs,
    ):
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        import vllm
        from vllm.v1.engine.exceptions import EngineDeadError

        self._vllm = vllm
        self._engine_dead_error = EngineDeadError
        self.kwargs = kwargs
        self._start_engine()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _start_engine(self):
        """
        Start the VLLM engine. This is necessary to initialize the model.
        """
        self.llm = self._vllm.LLM(model=self.model, **self.kwargs)

    def _call_llm(
        self, prompt: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        sampling_params = self._vllm.SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )
        message = [
            {"role": "user", "content": prompt["combined"] + self.extra_prompting}
        ]
        try:
            outputs = self.llm.chat(message, sampling_params=sampling_params)
        except self._engine_dead_error:
            self._start_engine()
            # Retry after restarting the engine
            outputs = self.llm.chat(message, sampling_params=sampling_params)
        result = outputs[0].outputs[0].text
        return result

    def _call_llm_with_system_prompt(
        self,
        prompt: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        sampling_params = self._vllm.SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"] + self.extra_prompting},
        ]

        try:
            outputs = self.llm.chat(messages, sampling_params=sampling_params)
        except self._engine_dead_error:
            self._start_engine()
            outputs = self.llm.chat(messages, sampling_params=sampling_params)

        result = outputs[0].outputs[0].text
        return result


class AsyncVLLMNamer(AsyncLLMWrapper):
    """This class is essentially for testing purposes only, allowing testing of the Async API with local models."""

    def __init__(
        self,
        model: str,
        llm_specific_instructions: Optional[str] = None,
        max_concurrent_requests: int = 10,
        callback: DebugCallback | None = None,
        **kwargs,
    ):
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        import vllm
        from vllm.v1.engine.exceptions import EngineDeadError

        self._vllm = vllm
        self._engine_dead_error = EngineDeadError
        self.kwargs = kwargs
        self._start_engine()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.max_concurrent_requests = max_concurrent_requests

    def _start_engine(self):
        self.llm = self._vllm.LLM(model=self.model, **self.kwargs)

    async def _call_llm_batch(
        self, prompts: List[Dict[str, Any]], temperature: float, max_tokens: int
    ) -> List[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": prompt["combined"] + self.extra_prompting,
                }
            ]
            for prompt in prompts
        ]
        sampling_params = self._vllm.SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )

        try:
            outputs = await asyncio.to_thread(
                self.llm.chat, messages=messages, sampling_params=sampling_params
            )
        except self._engine_dead_error:
            self._start_engine()  # Restart the engine if it fails
            outputs = await asyncio.to_thread(
                self.llm.chat, messages=messages, sampling_params=sampling_params
            )

        return [output.outputs[0].text for output in outputs]

    async def _call_llm_with_system_prompt_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        messages = []
        for prompt in prompts:
            messages.append(
                [
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": prompt["user"] + self.extra_prompting,
                    },
                ]
            )
        sampling_params = self._vllm.SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )

        try:
            outputs = await asyncio.to_thread(
                self.llm.chat, messages=messages, sampling_params=sampling_params
            )
        except self._engine_dead_error:
            self._start_engine()  # Restart the engine if it fails
            outputs = await asyncio.to_thread(
                self.llm.chat, messages=messages, sampling_params=sampling_params
            )

        return [output.outputs[0].text for output in outputs]


class CohereBatchNamer(AsyncLLMWrapper):
    """Cohere batches with owned jobs and aligned per-item results.

    Cancellation or timeout cancels a submitted job exactly once, including a job
    whose identifier arrives after caller cancellation. Cleanup errors are logged
    while preserving the original exception. Cancellation can therefore take the
    bounded SDK submission/cancellation time to complete.

    ``use_json_schema=True`` requires a prompt schema supported by the model and
    provider grammar. Automatic mode uses supported schemas and otherwise sends
    ordinary text requests; constraints are never removed. ``use_json_object``
    selects JSON-object mode explicitly. ``supports_json_schema`` overrides model
    capability only when the caller has verified the configured model.
    """

    _supports_debug_callback = True

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-08-2024",
        llm_specific_instructions=None,
        polling_interval=60,
        timeout=7200,
        callback: DebugCallback | None = None,
        *,
        client=None,
        use_json_schema: bool | None = None,
        use_json_object: bool | None = None,
        supports_json_schema: bool | None = None,
    ):
        from .provider_batches import CohereBatchTransport

        self.transport = CohereBatchTransport(
            api_key=api_key,
            model=model,
            polling_interval=polling_interval,
            timeout=timeout,
            client=client,
            use_json_schema=use_json_schema,
            use_json_object=use_json_object,
            supports_json_schema=supports_json_schema,
        )
        self.use_json_schema = use_json_schema
        self.use_json_object = use_json_object
        self.client = self.transport.client
        self.model = model
        self.callback = callback
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.polling_interval = polling_interval
        self.timeout = timeout

    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        normalized = [validate_prompt(prompt, False) for prompt in prompts]
        return await self._call_llm_with_system_prompt_batch(
            [
                Prompt("", prompt["combined"], prompt.get("json_schema"))
                for prompt in normalized
            ],
            temperature,
            max_tokens,
        )

    async def _call_llm_with_system_prompt_batch(
        self, prompts, temperature, max_tokens
    ):
        return await _run_managed_batch(self, prompts, temperature, max_tokens)

    @property
    def supports_json_schema(self):
        return self.transport.supports_json_schema

    def submit_batch(self, prompts, temperature, max_tokens) -> str:
        _validate_generation_options(temperature, max_tokens)
        normalized = [validate_prompt(prompt, True) for prompt in prompts]
        for prompt in normalized:
            prompt["user"] += self.extra_prompting
            prompt["combined"] += self.extra_prompting
        self._emit_debug_callback(
            {
                "event": "llm_call_start",
                "routine": "submit_batch",
                "prompts": normalized,
            }
        )
        return self.transport.submit_batch(normalized, temperature, max_tokens)

    def get_batch_status(self, batch_id: str) -> str:
        return self.transport.get_batch_status(batch_id)

    async def _retrieve_batch_results(self, batch_id: str):
        results = _transport_batch_results(
            await self.transport.retrieve_batch_text_results(batch_id)
        )
        self._emit_debug_callback(
            {
                "event": "llm_call_success",
                "routine": "batch_results",
                "batch_id": batch_id,
                "results": results,
            }
        )
        return results

    async def retrieve_batch_text_results(
        self, batch_id: str, *, return_results: bool = False
    ):
        results = await self._retrieve_batch_results(batch_id)
        return results if return_results else _result_values(results)

    async def _wait_for_completion_async(self, batch_id: str) -> bool:
        return await self.transport.wait_for_completion(batch_id)

    def cancel_batch(self, batch_id: str):
        return self.transport.cancel_batch(batch_id)

    async def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


def _anthropic_schema_problem(schema):
    """Return an unsupported constraint without weakening the caller's schema.

    This is a conservative subset of Anthropic's documented structured-output
    grammar. Local nonrecursive JSON Pointer references are supported. Patterns,
    nested resource identifiers, anchors and dynamic references remain explicit
    unsupported cases until their complete grammar can be verified locally.
    """
    forbidden = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "maxItems",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "unevaluatedItems",
        "unevaluatedProperties",
        "patternProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "dependentRequired",
        "dependentSchemas",
        "dependencies",
        "not",
        "if",
        "then",
        "else",
        "$dynamicRef",
        "$recursiveRef",
        "$anchor",
        "$dynamicAnchor",
        "$id",
    }
    pending = [(schema, ())]
    while pending:
        value, ancestors = pending.pop()
        if not isinstance(value, dict):
            continue
        if id(value) in ancestors:
            return "recursive references"
        ancestors = (*ancestors, id(value))
        unsupported = forbidden.intersection(value)
        if unsupported:
            return f"unsupported constraint {sorted(unsupported)[0]}"
        if "minItems" in value and value["minItems"] not in (0, 1):
            return "minItems greater than one"
        if isinstance(value.get("items"), list):
            return "tuple-valued items are unsupported"
        types = value.get("type", [])
        types = [types] if isinstance(types, str) else types
        if ("object" in types or "properties" in value) and value.get(
            "additionalProperties"
        ) is not False:
            return "objects require additionalProperties=false"
        reference = value.get("$ref")
        if reference is not None:
            if "allOf" in value:
                return "allOf combined with $ref"
            if reference != "#" and not reference.startswith("#/"):
                return "only local JSON Pointer references are supported"
            target = schema
            from urllib.parse import unquote

            for part in unquote(reference[2:]).split("/") if reference != "#" else ():
                part = part.replace("~1", "/").replace("~0", "~")
                target = target[int(part)] if isinstance(target, list) else target[part]
            pending.append((target, ancestors))
        for key in ("properties", "$defs", "definitions"):
            pending.extend((child, ancestors) for child in value.get(key, {}).values())
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            pending.extend((child, ancestors) for child in value.get(key, []))
        if "items" in value:
            pending.append((value["items"], ancestors))
    return None


class BatchAnthropicNamer(AsyncLLMWrapper):
    """
    Provides access to Anthropic's Batch Processing API with asynchronous support.
    This allows for processing large batches of prompts over an extended period.
    For more information on Anthropic's Batch Processing, see https://docs.anthropic.com/docs/batch-processing.

    This wrapper conforms to the AsyncLLMWrapper interface, but note that it uses Anthropic's batch API,
    which processes jobs over hours rather than in real-time. The async methods will block until the batch job completes.

    This class provides a different tradeoff between speed and cost compared to the AsyncAnthropic wrapper.
    It is designed for scenarios where you have a large number of prompts to process and can afford to wait for the results.
    Anthropic's batch processing is more cost-effective (half the cost per token) for large volumes of data, but it does
    not provide immediate responses.

    Cancellation and timeout cancel the owned job exactly once. Cancellation
    during submission waits for the eventual job ID before cleanup; cleanup
    failures are logged without replacing the original cancellation or timeout.

    ``use_json_schema=True`` fails before submission when the model or the
    supported schema grammar cannot preserve the supplied constraints. Automatic
    mode uses supported schemas and otherwise sends a text request. In particular,
    numeric bounds in the default template schema cause text fallback; constraints
    are never silently removed. JSON-object mode is unsupported. A caller-verified
    model can opt into ``supports_json_schema=True``; the schema grammar is still
    checked.

    Parameters:
    -----------
    api_key: str
        Your Anthropic API key. You can set this as an environment variable ANTHROPIC_API_KEY or pass it directly.

    model: str, optional
        The name of the Anthropic model to use. Default is "claude-haiku-4-5-20251001". You can use any model available
        in the Anthropic API, but this is a good balance of performance and cost.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.

    polling_interval: int, optional
        The interval (in seconds) to poll the batch job status. Default is 60 seconds. This controls how often
        the wrapper checks the status of the batch job. A lower value will check more frequently, but may increase API usage.

    timeout: int, optional
        The maximum time (in seconds) to wait for the batch job to complete. Default is 7200 seconds (2 hours). If
        the job does not complete within this time, it will raise a RuntimeError. This is useful to prevent indefinite blocking
        if the batch job takes too long to process. You can adjust this based on your expected processing time.

    Attributes:
    -----------
    client: anthropic.Anthropic
        The Anthropic client instance for batch processing.

    model: str
        The name of the Anthropic model being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For Anthropic, this is always True.

    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        llm_specific_instructions=None,
        polling_interval: int = 60,
        timeout: int = 7200,
        callback: DebugCallback | None = None,
        *,
        use_json_schema: bool | None = None,
        use_json_object: bool | None = None,
        supports_json_schema: bool | None = None,
    ):
        _validate_json_options(use_json_schema, use_json_object, None)
        if supports_json_schema is not None and not isinstance(
            supports_json_schema, bool
        ):
            raise InvalidLLMInputError("supports_json_schema must be bool or None")
        if use_json_object is True:
            raise InvalidLLMInputError(
                "Anthropic batches support JSON Schema, not JSON-object mode"
            )
        import math

        try:
            valid_timers = all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and value > 0
                and math.isfinite(float(value))
                for value in (polling_interval, timeout)
            )
        except OverflowError:
            valid_timers = False
        if not valid_timers:
            raise InvalidLLMInputError(
                "Batch polling interval and timeout must be finite positive numbers"
            )
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=min(30, timeout)
        )
        self.model = model
        self.use_json_schema = use_json_schema
        self.use_json_object = use_json_object
        self._schema_capability = supports_json_schema
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.polling_interval = polling_interval
        self.timeout = timeout

    @property
    def supports_json_schema(self):
        if self._schema_capability is not None:
            return self._schema_capability
        return self.model in {
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
        }

    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        normalized = [validate_prompt(prompt, False) for prompt in prompts]
        return await self._call_llm_with_system_prompt_batch(
            [
                Prompt("", prompt["combined"], prompt.get("json_schema"))
                for prompt in normalized
            ],
            temperature,
            max_tokens,
        )

    async def _call_llm_with_system_prompt_batch(
        self, prompts, temperature, max_tokens
    ):
        return await _run_managed_batch(self, prompts, temperature, max_tokens)

    async def _wait_for_completion_async(self, batch_id: str) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while loop.time() < deadline:
            batch = await asyncio.to_thread(
                self.client.messages.batches.retrieve, batch_id
            )
            if batch.processing_status == "ended":
                return True
            if batch.processing_status in ("canceling", "canceled", "expired"):
                raise LLMBatchItemError(
                    f"Batch {batch_id} ended with {batch.processing_status}"
                )
            await asyncio.sleep(
                min(self.polling_interval, max(0, deadline - loop.time()))
            )
        return False

    async def _retrieve_batch_results(self, batch_id: str):
        def retrieve():
            return list(self.client.messages.batches.results(batch_id))

        return _ordered_anthropic_results(await asyncio.to_thread(retrieve))

    # Additional methods for non-blocking usage
    def submit_batch(self, prompts, temperature, max_tokens) -> str:
        _validate_generation_options(temperature, max_tokens)
        normalized = [validate_prompt(prompt, True) for prompt in prompts]
        if not normalized:
            raise InvalidLLMInputError("Cannot submit an empty provider batch")
        requests = []
        for index, prompt in enumerate(normalized):
            schema = prompt.get("json_schema")
            schema_problem = (
                _anthropic_schema_problem(schema)
                if schema is not None
                else "missing prompt schema"
            )
            if self.use_json_schema is True and (
                not self.supports_json_schema or schema_problem
            ):
                raise InvalidLLMInputError(
                    "Required Anthropic JSON Schema is unsupported: "
                    + (schema_problem or "model capability")
                )
            requests.append(
                {
                    "custom_id": str(index),
                    "params": {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "system": prompt["system"],
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt["user"] + self.extra_prompting,
                            }
                        ],
                        "temperature": temperature,
                    },
                }
            )
            if (
                self.use_json_schema is not False
                and self.supports_json_schema
                and schema is not None
                and schema_problem is None
            ):
                requests[-1]["params"]["output_config"] = {
                    "format": {"type": "json_schema", "schema": deepcopy(schema)}
                }
        self._emit_debug_callback(
            {
                "event": "llm_call_start",
                "routine": "submit_batch",
                "prompts": normalized,
            }
        )
        return self.client.messages.batches.create(requests=requests).id

    def get_batch_status(self, batch_id: str) -> str:
        return self.client.messages.batches.retrieve(batch_id).processing_status

    async def retrieve_batch_text_results(
        self, batch_id: str, *, return_results: bool = False
    ):
        results = await self._retrieve_batch_results(batch_id)
        return results if return_results else _result_values(results)

    def cancel_batch(self, batch_id: str):
        return self.client.messages.batches.cancel(batch_id)


# Ollama
def OllamaNamer(
    model: str = "llama3.2",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    host: str | None = None,  # deprecated, renamed to api_base
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    Convenience wrapper for a LiteLLMNamer configured for local Ollama use.

    For Ollama remote API use, use LiteLLMNamer(model="ollama_chat/<model_name>", api_key=<api_key>).

    Parameters
    ----------
    model : str, optional
        Ollama model to use. Default is "llama3.2",  Must be in LiteLLM format ("ollama_chat/llama3.2")
        or bare Ollama format ("llama3.2") — both are accepted.
    api_key : str, optional
        Used for authentication if your Ollama server requires it. Not needed for default local setup. Falls back to the OLLAMA_API_KEY environment variable if not provided.
    api_base : str, optional
        Override the Ollama host URL. Default is "http://localhost:11434".  Can use the OLLAMA_API_BASE environment variable.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    host : str, optional
        Deprecated. Use ``api_base`` instead.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = OllamaNamer()
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = OllamaNamer(model="llama3.2")

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    if host is not None:
        warn(
            "host is deprecated, use api_base instead.",
            FutureWarning,
            stacklevel=2,
        )
    api_base = api_base or host or "http://localhost:11434"

    return LiteLLMNamer(
        model=_ollama_model(model),
        api_key=api_key,
        api_base=api_base,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
    )


def AsyncOllamaNamer(
    model: str = "llama3.2",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 5,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    host: str | None = None,  # deprecated, renamed to api_base
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    Convenience wrapper for a AsyncLiteLLMNamer configured for local Ollama use.

    For Ollama remote API use, use AsyncLiteLLMNamer(model="ollama_chat/<model_name>", api_key=<api_key>).

    Parameters
    ----------
    model : str, optional
        Ollama model to use. Default is "llama3.2",  Must be in LiteLLM format ("ollama_chat/llama3.2")
        or bare Ollama format ("llama3.2") — both are accepted.
    api_key : str, optional
        Used for authentication if your Ollama server requires it. Not needed for default local setup. Falls back to the OLLAMA_API_KEY environment variable if not provided.
    api_base : str, optional
        Override the Ollama host URL. Default is "http://localhost:11434".  Can use the OLLAMA_API_BASE environment variable.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests. Default is 5. This can be adjusted based on your
        application's needs and the rate limits of the OpenAI API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    host : str, optional
        Deprecated. Use ``api_base`` instead.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncOllamaNamer()
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AsyncOllamaNamer(model="llama3.2")

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    if host is not None:
        warn(
            "host is deprecated, use api_base instead.",
            FutureWarning,
            stacklevel=2,
        )
    api_base = api_base or host or "http://localhost:11434"
    return AsyncLiteLLMNamer(
        model=_ollama_model(model),
        api_key=api_key,
        api_base=api_base,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
    )


## OpenAI Convenience Wrappers
def NotebookOpenAINamerMock(*args, **kwargs):
    """
    For mocking OpenAINamer calls with a local Ollama model.
    """
    logger.info("Using NotebookOpenAINamerMock instead of OpenAINamer")
    kwargs.pop("base_url", None)
    kwargs.pop("http_client", None)
    kwargs.pop("model", None)
    kwargs.pop("temperature_override", None)
    return OllamaNamer(
        model=get_test_ollama_model(), temperature_override=0.0, **kwargs
    )


@notebook_test_replacement(NotebookOpenAINamerMock)
def OpenAINamer(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str | None = None,  # deprecated, renamed to api_base
    http_client: "httpx.Client | None" = None,  # deprecated, pass via provider_kwargs instead
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    Create a LiteLLMNamer configured for OpenAI.

    All namers share the same interface once constructed — OpenAINamer is a
    convenience entry point, not a special case. For more information on OpenAI, see https://platform.openai.com/docs/models/overview.

    Parameters
    ----------
    model : str, optional
        OpenAI model to use. Default is "gpt-4o-mini", a good balance of
        quality and cost for topic naming. You can use more advanced models, but they have diminishing returns
        for this task, and are more expensive. Must be in LiteLLM format ("openai/gpt-4o-mini")
        or bare OpenAI format ("gpt-4o-mini") — both are accepted.
    api_key : str, optional
        OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
    api_base : str, optional
        Override the OpenAI API endpoint. Useful for proxies or OpenAI-compatible
        local servers (e.g. vLLM, LM Studio). Can use the OPENAI_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    use_json_object : bool, optional
        Request JSON object output via response_format. If None (default),
        support is detected automatically for the selected model. Set to False
        to disable if your model doesn't support it.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    base_url : str, optional
        Deprecated. Use ``api_base`` instead.
    http_client : optional
        Deprecated. Pass via ``provider_kwargs={'http_client': <client>}`` instead.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = OpenAINamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = OpenAINamer(model="gpt-4o", api_key="my-api-key")

    Using an OpenAI-compatible local server::

        namer = OpenAINamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    logger.info("Using OpenAINamer")
    if base_url is not None:
        warn(
            "base_url is deprecated, use api_base instead.",
            FutureWarning,
            stacklevel=2,
        )
    api_base = api_base or base_url
    if http_client is not None:
        warn(
            "http_client is deprecated. "
            "Pass via provider_kwargs={'http_client': http_client} instead.",
            FutureWarning,
            stacklevel=2,
        )
        provider_kwargs = provider_kwargs or {}
        provider_kwargs["http_client"] = http_client
    return LiteLLMNamer(
        model=_openai_model(model),
        api_key=api_key,
        api_base=api_base,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncOpenAINamer(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str | None = None,  # deprecated, renamed to api_base
    organization: str | None = None,  # deprecated, pass via provider_kwargs instead
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    Create an AsyncLiteLLMNamer configured for OpenAI.

    All namers share the same interface once constructed — OpenAINamer is a
    convenience entry point, not a special case. For more information on OpenAI, see https://platform.openai.com/docs/models/overview.

    Parameters
    ----------
    model : str, optional
        OpenAI model to use. Default is "gpt-4o-mini", a good balance of
        quality and cost for topic naming. You can use more advanced models, but they have diminishing returns
        for this task, and are more expensive. Must be in LiteLLM format ("openai/gpt-4o-mini")
        or bare OpenAI format ("gpt-4o-mini") — both are accepted.
    api_key : str, optional
        OpenAI API key. Falls back to the OPENAI_API_KEY environment variable.
    api_base : str, optional
        Override the OpenAI API endpoint. Useful for proxies or OpenAI-compatible
        local servers (e.g. vLLM, LM Studio). Can use the OPENAI_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    use_json_object : bool, optional
        Request JSON object output via response_format. If None (default),
        support is detected automatically for the selected model. Set to False
        to disable if your model doesn't support it.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the OpenAI API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the OpenAI API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    base_url : str, optional
        Deprecated. Use ``api_base`` instead.
    organization : str, optional
        Deprecated. Pass via ``provider_kwargs={'organization': organization}`` instead.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncOpenAINamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AsyncOpenAINamer(model="gpt-4o", api_key="my-api-key")

    Using an OpenAI-compatible local server::

        namer = AsyncOpenAINamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    if base_url is not None:
        warn(
            "base_url is deprecated, use api_base instead.",
            FutureWarning,
            stacklevel=2,
        )
    api_base = api_base or base_url
    if organization is not None:
        warn(
            "organization is deprecated. "
            "Pass via provider_kwargs={'organization': organization} instead.",
            FutureWarning,
            stacklevel=2,
        )
        provider_kwargs = provider_kwargs or {}
        provider_kwargs["organization"] = organization
    return AsyncLiteLLMNamer(
        model=_openai_model(model),
        api_key=api_key,
        api_base=api_base,
        disable_system_prompts=False,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AzureAINamer(
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    endpoint: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
        Create a LiteLLMNamer configured for Azure AI.

        All namers share the same interface once constructed — AzureAINamer is a convenience entry point, not a special case.

    Parameters
    ----------
    model : str,
        The deployed model name in Azure AI Foundry. Will be prefixed
        with "azure_ai/" automatically (e.g. "gpt-4o" → "azure_ai/gpt-4o").
    api_key : str, optional
        Azure API key. Falls back to the AZURE_AI_API_KEY environment variable.
    api_base : str, optional
        The Azure AI Foundry endpoint URL. Preferred over `endpoint` for
        consistency with other factory functions. Falls back to the AZURE_AI_API_BASE environment variable if not provided.
    endpoint : str, optional
        The Azure AI Foundry endpoint URL, e.g.
        "https://<your-resource-name>.openai.azure.com/".
        Alias for `api_base`; if both are provided, `api_base` takes precedence.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AzureAINamer(model="deployed-model-name", api_base="https://<your-resource-endpoint>")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)


    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    resolved_endpoint = api_base or endpoint
    return LiteLLMNamer(
        model=_azure_model(model),
        api_key=resolve_api_key(
            api_key=api_key, env_new="AZURE_AI_API_KEY", env_legacy="AZURE_API_KEY"
        ),
        api_base=resolved_endpoint,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncAzureAINamer(
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    endpoint: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    Create a LiteLLMNamer configured for Azure AI.

    All namers share the same interface once constructed — AsyncAzureAINamer is a convenience entry point, not a special case.


    Parameters
    ----------
    model : str
        The deployed model name in Azure AI Foundry. Will be prefixed
        with "azure_ai/" automatically (e.g. "gpt-4o" → "azure_ai/gpt-4o").
    api_key : str, optional
        Azure API key. Falls back to the AZURE_AI_API_KEY environment variable.
    api_base : str, optional
        The Azure AI Foundry endpoint URL. Preferred over `endpoint` for
        consistency with other factory functions. One of `api_base` or
        `endpoint` must be provided.
    endpoint : str, optional
        The Azure AI Foundry endpoint URL, e.g.
        "https://<your-resource-name>.openai.azure.com/".
        Alias for `api_base`; if both are provided, `api_base` takes precedence.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Anthropic API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Anthropic API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncAzureAINamer(model="deployed-model-name", api_base="https://<your-resource-endpoint>")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    resolved_endpoint = api_base or endpoint
    return AsyncLiteLLMNamer(
        model=_azure_model(model),
        api_key=resolve_api_key(
            api_key=api_key, env_new="AZURE_AI_API_KEY", env_legacy="AZURE_API_KEY"
        ),
        api_base=resolved_endpoint,
        disable_system_prompts=False,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


class BatchAzureAINamer(AsyncLLMWrapper):
    """Azure OpenAI batches with owned jobs and aligned per-item results.

    Cancellation and timeout cancel a submitted job exactly once, including a job
    whose ID arrives after caller cancellation. Cleanup errors are logged without
    replacing the original exception. SDK request timeouts bound cleanup work.

    ``use_json_schema=True`` requires a prompt schema and supported model before
    file upload. Automatic mode preserves supported schemas and otherwise sends
    plain text. Opaque Azure deployment names require a caller-verified
    ``supports_json_schema=True`` capability override. ``use_json_object=True``
    selects object mode explicitly and conflicts with required schema mode.
    """

    _supports_debug_callback = True

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        llm_specific_instructions=None,
        polling_interval=60,
        timeout=7200,
        callback: DebugCallback | None = None,
        *,
        client=None,
        use_json_schema: bool | None = None,
        use_json_object: bool | None = None,
        supports_json_schema: bool | None = None,
    ):
        from .provider_batches import AzureBatchTransport

        self.transport = AzureBatchTransport(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            polling_interval=polling_interval,
            timeout=timeout,
            client=client,
            use_json_schema=use_json_schema,
            use_json_object=use_json_object,
            supports_json_schema=supports_json_schema,
        )
        self.use_json_schema = use_json_schema
        self.use_json_object = use_json_object
        self.client = self.transport.client
        self.model = model
        self.callback = callback
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.polling_interval = polling_interval
        self.timeout = timeout

    async def _call_llm_batch(self, prompts, temperature, max_tokens):
        normalized = [validate_prompt(prompt, False) for prompt in prompts]
        return await self._call_llm_with_system_prompt_batch(
            [
                Prompt("", prompt["combined"], prompt.get("json_schema"))
                for prompt in normalized
            ],
            temperature,
            max_tokens,
        )

    async def _call_llm_with_system_prompt_batch(
        self, prompts, temperature, max_tokens
    ):
        return await _run_managed_batch(self, prompts, temperature, max_tokens)

    @property
    def supports_json_schema(self):
        return self.transport.supports_json_schema

    def submit_batch(self, prompts, temperature, max_tokens) -> str:
        _validate_generation_options(temperature, max_tokens)
        normalized = [validate_prompt(prompt, True) for prompt in prompts]
        for prompt in normalized:
            prompt["user"] += self.extra_prompting
            prompt["combined"] += self.extra_prompting
        self._emit_debug_callback(
            {
                "event": "llm_call_start",
                "routine": "submit_batch",
                "prompts": normalized,
            }
        )
        return self.transport.submit_batch(normalized, temperature, max_tokens)

    def get_batch_status(self, batch_id: str) -> str:
        return self.transport.get_batch_status(batch_id)

    async def _retrieve_batch_results(self, batch_id: str):
        results = _transport_batch_results(
            await self.transport.retrieve_batch_text_results(batch_id)
        )
        self._emit_debug_callback(
            {
                "event": "llm_call_success",
                "routine": "batch_results",
                "batch_id": batch_id,
                "results": results,
            }
        )
        return results

    async def retrieve_batch_text_results(
        self, batch_id: str, *, return_results: bool = False
    ):
        results = await self._retrieve_batch_results(batch_id)
        return results if return_results else _result_values(results)

    async def _wait_for_completion_async(self, batch_id: str) -> bool:
        return await self.transport.wait_for_completion(batch_id)

    def cancel_batch(self, batch_id: str):
        return self.transport.cancel_batch(batch_id)

    async def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)


def GoogleGeminiNamer(
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    GoogleGeminiNamer is deprecated and will be removed in a future release. Use LiteLLMNamer(model='gemini/<model_name>') directly instead.

    Parameters
    ----------
    model : str, optional
        Google Gemini model to use. Default is "gemini-2.5-flash-lite".
        May be in LiteLLM format ("google/gemini-2.5-flash-lite")
    api_key : str, optional
        Google Gemini API key. Falls back to the GEMINI_API_KEY environment variable.
    api_base : str, optional
        Override the Google Gemini API endpoint. Can use the GEMINI_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = GoogleGeminiNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = GoogleGeminiNamer(model="gemini-2.5-flash-lite",api_key="my-api-key")

    Using an Anthropic-compatible local server::

        namer = GoogleGeminiNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    warn(
        (
            "GoogleGeminiNamer is deprecated and will be removed in a future "
            "release. Use LiteLLMNamer(model='gemini/<model_name>') directly instead."
        ),
        FutureWarning,
        stacklevel=2,
    )
    return LiteLLMNamer(
        model=_gemini_model(model),
        api_key=resolve_api_key(
            api_key=api_key, env_new="GEMINI_API_KEY", env_legacy="GOOGLE_API_KEY"
        ),
        api_base=api_base,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncGoogleGeminiNamer(
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> AsyncLiteLLMNamer:
    """
    AsyncGoogleGeminiNamer is deprecated and will be removed in a future release. Use AsyncLiteLLMNamer(model='gemini/<model_name>') directly instead.

    Parameters
    ----------
    model : str, optional
        Google Gemini model to use. Default is "gemini-2.5-flash-lite", Must be in LiteLLM format ("google/gemini-2.5-flash-lite")
        or bare Google Gemini format ("gemini-2.5-flash-lite") — both are accepted.
    api_key : str, optional
        Google Gemini API key. Falls back to the GEMINI_API_KEY environment variable.
    api_base : str, optional
        Override the Google AI Studio API endpoint. Can use the GEMINI_API_BASE environment variable.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Gemini API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Gemini API. Higher values may improve throughput but could lead to rate limiting.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.

    Returns
    -------
    AsyncLiteLLMNamer
        A fully configured async namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = AsyncGoogleGeminiNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    Using a different model::

        namer = AsyncGoogleGeminiNamer(model="gemini-2.5-flash-lite",api_key="my-api-key")

    Using an Anthropic-compatible local server::

        namer = AsyncGoogleGeminiNamer(model="hosted-model", api_base="http://localhost:8000/v1", api_key="none")

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    warn(
        (
            "AsyncGoogleGeminiNamer is deprecated and will be removed in a future "
            "release. Use AsycLiteLLMNamer(model='gemini/<model_name>') directly instead."
        ),
        FutureWarning,
        stacklevel=2,
    )
    return AsyncLiteLLMNamer(
        model=_gemini_model(model),
        api_key=resolve_api_key(
            api_key=api_key, env_new="GEMINI_API_KEY", env_legacy="GOOGLE_API_KEY"
        ),
        api_base=api_base,
        disable_system_prompts=False,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def _resolve_replicate_api_key(
    api_key: str | None, api_token: str | None
) -> str | None:
    """Helper function to migrate from the old REPLICATE_API_TOKEN environment variable and api_token to the new REPLICATE_API_KEY, while still allowing explicit API keys to take precedence."""
    if api_key is not None:
        return api_key

    new_key = os.getenv("REPLICATE_API_KEY")
    legacy_key = os.getenv("REPLICATE_API_TOKEN")

    if new_key:
        return new_key

    if api_token is not None:
        warn(
            "api_token is deprecated and will be removed before 1.0. "
            "Please rename it to api_key.",
            FutureWarning,
            stacklevel=3,
        )
        return api_token

    if legacy_key:
        warn(
            "REPLICATE_API_TOKEN is deprecated and will be removed before 1.0. "
            "Please rename it to REPLICATE_API_KEY.",
            FutureWarning,
            stacklevel=3,
        )
        return legacy_key

    return None


def ReplicateNamer(
    model: str = "meta/llama-2-70b-chat",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    temperature_override: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    api_token: str = None,
    use_json_schema: bool | None = None,
    use_json_object: bool | None = None,
) -> LiteLLMNamer:
    """
    Deprecated. Use LiteLLMNamer(model="replicate/<model>") directly instead.

    Parameters
    ----------
    model : str, optional
        Replicate model to use. Default is "meta/llama-2-70b-chat".
        May be in LiteLLM format ("replicate/"meta/llama-2-70b-chat") or bare Replicate format ("meta/llama-2-70b-chat") — both are accepted.
    api_key : str, optional
        Replicate API key. Falls back to the REPLICATE_API_KEY environment variable.
    api_base : str, optional
        Override the Replicate API endpoint. Falls back to REPLICATE_API_BASE.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
    max_tokens_topic_name: int, optional
        Default maximum number of tokens for topic name generation. Default is 128.
        Can be overridden per-call in generate_topic_name().
    max_tokens_cluster_names: int, optional
        Default maximum number of tokens for cluster name generation. Default is 1024.
        Can be overridden per-call in generate_topic_cluster_names().
    temperature_override: float | None, optional
        If provided, this value overrides the temperature passed to the underlying
        LiteLLM completion calls, ensuring a fixed temperature regardless of per-call temperature
        arguments. Useful for test stability or reproducibility.
    provider_kwargs : dict, optional
        Additional keyword arguments passed directly to the LiteLLM completion
        call. Use for provider-specific features not covered by the parameters
        above, e.g. ``{"timeout": 30}``.
    callback : DebugCallback, optional
        Optional callback function for observability. Called on each LLM
        request and response with a structured payload. Useful for logging,
        debugging, or recording prompts and responses to a file.
    api_token : str, optional
        Deprecated alias for api_key. Use api_key instead.

    Returns
    -------
    LiteLLMNamer
        A fully configured namer ready for use with Toponymy.

    Examples
    --------
    Basic usage::

        namer = ReplicateNamer(api_key="my-api-key")
        toponymy = Toponymy(llm_wrapper=namer, text_embedding_model=...)

    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    warn(
        (
            "ReplicateNamer is deprecated and will be removed in a future "
            "release. Use LiteLLMNamer(model='replicate/<model_name>') directly instead."
        ),
        FutureWarning,
        stacklevel=2,
    )
    return LiteLLMNamer(
        model=_replicate_model(model),
        api_key=_resolve_replicate_api_key(api_key=api_key, api_token=api_token),
        api_base=api_base,
        use_json_object=use_json_object,
        use_json_schema=use_json_schema,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        temperature_override=temperature_override,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )
