import string
from unittest import result
from warnings import warn, filterwarnings
import tokenizers
import transformers

from toponymy.templates import (
    GET_TOPIC_CLUSTER_NAMES_REGEX,
    GET_TOPIC_NAME_REGEX,
    default_extract_topic_names,
)
from toponymy.tools.notebook_test_helpers import (
    notebook_test_replacement,
    get_test_ollama_model,
)
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

import re
import os
import httpx
import json
import asyncio

import logging

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
T = TypeVar("T")
DebugCallback = Callable[[dict[str, Any]], None]


# Ignore internal litellm warning
filterwarnings(
    "ignore",
    message="Support for class-based `config` is deprecated",
    category=DeprecationWarning,
    module="litellm",
)


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


def _should_retry(e: Exception) -> bool:
    if isinstance(e, InvalidLLMInputError):
        return False
    if isinstance(e, FailFastLLMError):
        return False
    return True


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

        if isinstance(e, self.FAIL_FAST_EXCEPTIONS):
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
        prompt = kwargs.get("prompt")
        routine = kwargs.pop("routine", None)
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_random_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception(_should_retry),
                reraise=True,
            ):
                with attempt:
                    value = await fn(*args, **kwargs)
                    # SUCCESS emit
                    self._emit_debug_callback(
                        {
                            "event": "llm_call_success",
                            "routine": routine,
                            "prompt": prompt,
                            "raw_response": value,
                        }
                    )
                    return CallResult(value=value)
        except Exception as e:
            if isinstance(e, (InvalidLLMInputError, *self.FAIL_FAST_EXCEPTIONS)):
                self._handle_exception(e)

            # For other exceptions, we log a warning and return the error in the CallResult for potential handling by the caller.
            logger.warning(
                "%s exhausted retries for LLM call (%s): %s",
                self.__class__.__name__,
                type(e).__name__,
                str(e)[:200],
            )
            # ERROR emit
            self._emit_debug_callback(
                {
                    "event": "llm_call_error",
                    "routine": routine,
                    "prompt": prompt,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                }
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
            pass

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


def repair_json_string_backslashes(s: str) -> str:
    """
    Attempts to repair a string that should be JSON by escaping unescaped backslashes.
    This focuses on the common issue of literal backslashes not being escaped.
    """
    # Define placeholders for known valid JSON escape sequences
    # This helps prevent double-escaping or breaking already correct sequences.
    placeholders = {
        "\\\\": "__DOUBLE_BACKSLASH_PLACEHOLDER__",
        '\\"': "__ESCAPED_QUOTE_PLACEHOLDER__",
        "\\n": "__NEWLINE_PLACEHOLDER__",
        "\\r": "__CARRIAGE_RETURN_PLACEHOLDER__",
        "\\t": "__TAB_PLACEHOLDER__",
        "\\b": "__BACKSPACE_PLACEHOLDER__",
        "\\f": "__FORMFEED_PLACEHOLDER__",
        "\\/": "__SOLIDUS_PLACEHOLDER__",  # Though '/' doesn't always need escaping
    }

    # Step 1: Protect existing valid escape sequences
    temp_s = s
    for original, placeholder in placeholders.items():
        temp_s = temp_s.replace(original, placeholder)

    # Step 2: Escape remaining single backslashes
    # These are likely the problematic ones intended to be literal backslashes.
    temp_s = temp_s.replace("\\", "\\\\")

    # Step 3: Restore the original valid escape sequences
    for original, placeholder in placeholders.items():
        temp_s = temp_s.replace(placeholder, original)

    return temp_s


def llm_output_to_result(llm_output: str, regex: str) -> dict:
    json_portion = re.findall(regex, llm_output, re.DOTALL)[0]
    try:
        result = json.loads(json_portion)
    except json.JSONDecodeError:
        # Attempt to repair the JSON string
        repaired_json = repair_json_string_backslashes(json_portion)
        result = json.loads(repaired_json)

    return result


class LLMWrapper(DebugCallbackMixin, LLMErrorHandlingMixin, ABC):
    FAIL_FAST_EXCEPTIONS: tuple = ()

    @abstractmethod
    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Call the LLM with the given prompt and temperature.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """
        Call the LLM with a system prompt and user prompt.
        This method should be implemented by subclasses.
        """
        pass

    def _safe_call_llm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> str:
        try:
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
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> str:
        prompt_payload = {
            "system": system_prompt,
            "user": user_prompt,
        }

        try:
            raw_response = self._call_llm_with_system_prompt(
                system_prompt, user_prompt, temperature, max_tokens
            )

            self._emit_debug_callback(
                {
                    "event": "llm_call_success",
                    "prompt_type": "system",
                    "routine": routine,
                    "prompt": prompt_payload,
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
                    "prompt": prompt_payload,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                }
            )
            self._handle_exception(e)

    @staticmethod
    def _topic_name_error_callback(retry_state):
        """Callback function for when all retries are exhausted in generate_topic_name. Logs the error and returns an empty string."""
        exc = retry_state.outcome.exception()
        if isinstance(exc, (FailFastLLMError, InvalidLLMInputError)):
            raise exc
        warn(
            f"All retries exhausted for generate_topic_name: {type(exc).__name__}: {exc}"
        )
        return ""

    # @abstractmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=_topic_name_error_callback,
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_name(
        self,
        prompt: Union[str, Dict[str, str]],
        temperature: float = 0.4,
        topic_extraction_function=lambda x: x["topic_name"],
        get_topic_name_regex=GET_TOPIC_NAME_REGEX,
        max_tokens: int | None = None,
    ) -> str | tuple:
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_topic_name", 128)

        if isinstance(prompt, str):
            topic_name_info_raw = self._safe_call_llm(
                prompt,
                temperature,
                max_tokens=max_tokens,
                routine="generate_topic_name",
            )
        elif isinstance(prompt, dict) and self.supports_system_prompts:
            topic_name_info_raw = self._safe_call_llm_with_system_prompt(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                temperature=temperature,
                max_tokens=max_tokens,
                routine="generate_topic_name",
            )
        else:
            warn(f"Prompt must be a string or a dictionary, got {type(prompt)}")
            raise InvalidLLMInputError(
                f"Prompt must be a string or a dictionary, got {type(prompt)}"
            )

        topic_name_info = llm_output_to_result(
            topic_name_info_raw, get_topic_name_regex
        )
        result = topic_extraction_function(topic_name_info)
        topic_name = result if isinstance(result, tuple) else str(result)
        return topic_name

    @staticmethod
    def _topic_cluster_names_error_callback(retry_state):
        exc = retry_state.outcome.exception()
        if isinstance(exc, (FailFastLLMError, InvalidLLMInputError)):
            raise exc
        old_names = (
            retry_state.args[2]  # args[0]=self, args[1]=prompt, args[2]=old_names
            if len(retry_state.args) > 2 and isinstance(retry_state.args[2], list)
            else []
        )
        warn(
            f"All retries exhausted for generate_topic_cluster_names: "
            f"{type(exc).__name__}: {exc}. Returning old names."
        )
        return old_names

    # @abstractmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=_topic_cluster_names_error_callback,
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_cluster_names(
        self,
        prompt: Union[str, Dict[str, str]],
        old_names: List[str],
        temperature: float = 0.4,
        extract_topic_names_function=default_extract_topic_names,
        get_topic_names_regex=GET_TOPIC_CLUSTER_NAMES_REGEX,
        max_tokens: int | None = None,
    ) -> List[str]:
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_cluster_names", 1024)

        if isinstance(prompt, str):
            topic_name_info_raw = self._safe_call_llm(
                prompt,
                temperature,
                max_tokens=max_tokens,
                routine="generate_topic_cluster_names",
            )
        elif isinstance(prompt, dict) and self.supports_system_prompts:
            topic_name_info_raw = self._safe_call_llm_with_system_prompt(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                temperature=temperature,
                max_tokens=max_tokens,
                routine="generate_topic_cluster_names",
            )
        else:
            raise InvalidLLMInputError(
                f"Prompt must be a string or a dictionary, got {type(prompt)}"
            )

        topic_name_info = llm_output_to_result(
            topic_name_info_raw, GET_TOPIC_CLUSTER_NAMES_REGEX
        )

        return extract_topic_names_function(
            topic_name_info, old_names, topic_name_info_raw
        )
        # mapping = topic_name_info["new_topic_name_mapping"]
        # if len(mapping) == len(old_names):
        #     result = []
        #     for i, old_name_val in enumerate(old_names, start=1):
        #         key_with_val = f"{i}. {old_name_val}"
        #         key_just_index = f"{i}."
        #         if key_with_val in mapping:
        #             result.append(mapping[key_with_val])
        #         elif (
        #             key_just_index in mapping
        #         ):  # This was `mapping.get(f"{n}.", name)` which is ambiguous
        #             result.append(mapping[key_just_index])
        #         else:
        #             result.append(
        #                 old_name_val
        #             )  # Fallback to old name to maintain length
        #     return result
        # else:
        #     # Fallback to just parsing the string as best we can
        #     mapping = re.findall(
        #         r'"new_topic_name_mapping":\s*\{(.*?)\}',
        #         topic_name_info_raw,
        #         re.DOTALL,
        #     )[0]
        #     new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
        #     if len(new_names) == len(old_names):
        #         return new_names
        #     else:
        #         raise ValueError(
        #             f"Failed to generate enough names when fixing {old_names}; got {mapping}"
        #         )

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
                    prompt,
                    temperature=0.4,
                    max_tokens=128,
                )
            else:
                response = self._call_llm_with_system_prompt(
                    system_prompt,
                    prompt,
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


class AsyncLLMWrapper(DebugCallbackMixin, LLMErrorHandlingMixin, ABC):

    async def _call_single_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """
        Execute a single provider request for one user prompt and return the raw
        text result from the model.

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
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Execute a single provider request for one system prompt + user prompt pair
        and return the raw text result from the model.

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
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> List[CallResult[str]]:
        """
        Process a batch of prompts and return one CallResult per prompt.

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
                prompt,
                temperature,
                max_tokens,
                routine=routine,
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
        routine: str | None = None,
    ) -> List[CallResult[str]]:
        """
        Process a batch of system prompt + user prompt pairs and return one CallResult
        per pair.

        The default implementation wraps `_call_single_llm_with_system` with retry and
        error handling via `_safe_call_with_retry_result` and executes all prompt pairs
        concurrently using `asyncio.gather`.

        This produces the standard async behavior used by most wrappers:

            - retryable errors are retried per prompt pair
            - fail-fast errors abort the entire batch/layer
            - exhausted retryable errors return CallResult(error=...)
            - successful calls return CallResult(value=<text>)

        Subclasses normally should NOT override this method if their provider model is
        "one async request per prompt pair". Instead, implement
        `_call_single_llm_with_system` and inherit this default batching behavior.

        Override this method when the provider uses a fundamentally different batch
        model than concurrent single-call execution, such as:

            - provider-managed batch job APIs
            - bulk endpoints accepting multiple prompt pairs in one request
            - server-side batching that must be coordinated as a unit

        In the current architecture, such providers may still inherit from
        AsyncLLMWrapper and override this method directly.

        If a dedicated batch wrapper base class (for example LLMBatchWrapper) is
        introduced in the future, these implementations may move there instead.

        Note:
            Some legacy wrapper implementations override `_call_llm_with_system_prompt_batch`
            with a "one async request per prompt pair" model. Those implementations remain
            supported and will take precedence over this default method.
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError(
                "Number of system prompts must match number of user prompts"
            )

        tasks = [
            self._safe_call_with_retry_result(
                self._call_single_llm_with_system,
                sys_prompt,
                user_prompt,
                temperature,
                max_tokens,
                routine=routine,
            )
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]

        return await asyncio.gather(*tasks)

    async def generate_topic_names(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float = 0.4,
        extract_topic_name_function=lambda x: x["topic_name"],
        get_topic_name_regex=GET_TOPIC_NAME_REGEX,
        null_result_value="",
        max_tokens: int | None = None,
    ) -> List[str]:
        """
        Generate topic names for a batch of prompts.
        Returns a list of topic names matching the input prompts.
        """
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_topic_name", 128)

        if not prompts:
            return []

        # Check the first prompt to determine type
        if isinstance(prompts[0], str):
            responses = await self._call_llm_batch(
                prompts, temperature, max_tokens=max_tokens
            )
        elif isinstance(prompts[0], dict) and self.supports_system_prompts:
            system_prompts = [p["system"] for p in prompts]
            user_prompts = [p["user"] for p in prompts]
            responses = await self._call_llm_with_system_prompt_batch(
                system_prompts, user_prompts, temperature, max_tokens=max_tokens
            )
        else:
            raise InvalidLLMInputError(
                f"Prompts must be strings or dictionaries, got {type(prompts[0])}"
            )

        # Parse responses
        results = []
        for response in responses:
            if isinstance(response, CallResult):
                if not response.ok:
                    warn(
                        f"Failed to generate topic name with "
                        f"{self.__class__.__name__}: {response.error}"
                    )
                    results.append(null_result_value)
                    continue
                response_text = response.value
            else:
                response_text = response

            if not response_text:
                results.append(null_result_value)
                continue

            # Attempt to parse the response
            try:
                topic_name_info = llm_output_to_result(
                    response_text, get_topic_name_regex
                )
                result = extract_topic_name_function(topic_name_info)
                topic_name = result if isinstance(result, tuple) else str(result)
                results.append(topic_name)
            except Exception as e:
                warn(
                    f"Failed to generate topic name with {self.__class__.__name__}: {e}"
                )
                results.append(
                    null_result_value
                )  # Fallback to null_result_value if parsing fails

        return results

    async def generate_topic_cluster_names(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        old_names_list: List[List[str]],
        temperature: float = 0.4,
        extract_topic_names_function=default_extract_topic_names,
        get_topic_names_regex=GET_TOPIC_CLUSTER_NAMES_REGEX,
        max_tokens: int | None = None,
    ) -> List[List[str]]:
        """
        Generate topic cluster names for a batch of prompts.
        Returns a list of lists of topic names matching the input prompts.
        """
        if max_tokens is None:
            max_tokens = getattr(self, "max_tokens_cluster_names", 1024)

        if len(prompts) != len(old_names_list):
            raise ValueError("Number of prompts must match number of old_names lists")

        if not prompts:
            return []

        # Check the first prompt to determine type
        if isinstance(prompts[0], str):
            responses = await self._call_llm_batch(
                prompts, temperature, max_tokens=max_tokens
            )
        elif isinstance(prompts[0], dict) and self.supports_system_prompts:
            system_prompts = [prompt["system"] for prompt in prompts]
            user_prompts = [prompt["user"] for prompt in prompts]
            responses = await self._call_llm_with_system_prompt_batch(
                system_prompts, user_prompts, temperature, max_tokens=max_tokens
            )
        else:
            raise InvalidLLMInputError(
                f"Prompts must be strings or dictionaries, got {type(prompts[0])}"
            )

        # Parse responses
        results = []
        for response, old_names in zip(responses, old_names_list):
            if isinstance(response, CallResult):
                if not response.ok:
                    warn(
                        f"Failed to generate cluster names with "
                        f"{self.__class__.__name__}: {response.error}"
                    )
                    results.append(old_names)
                    continue
                response_text = response.value
            else:
                response_text = response

            if not response_text:
                results.append(old_names)
                continue
            results.append(
                self._parse_cluster_response(
                    response_text,
                    old_names,
                    extract_topic_names_function,
                    get_topic_names_regex,
                )
            )

        return results

    def _parse_cluster_response(
        self,
        response: str,
        old_names: List[str],
        extract_topic_names_function,
        get_topic_names_regex,
    ) -> List[str]:
        """Parse a single cluster response."""
        try:
            topic_name_info = llm_output_to_result(response, get_topic_names_regex)
            return extract_topic_names_function(topic_name_info, old_names, response)
            # mapping = topic_name_info["new_topic_name_mapping"]

            # if len(mapping) == len(old_names):
            #     result = []
            #     for i, old_name_val in enumerate(old_names, start=1):
            #         key_with_val = f"{i}. {old_name_val}"
            #         key_just_index = f"{i}."
            #         if key_with_val in mapping:
            #             result.append(mapping[key_with_val])
            #         elif key_just_index in mapping:
            #             result.append(mapping[key_just_index])
            #         else:
            #             result.append(old_name_val)
            #     return result
            # else:
            #     # Fallback parsing
            #     mapping_str = re.findall(
            #         r'"new_topic_name_mapping":\s*\{(.*?)\}',
            #         response,
            #         re.DOTALL,
            #     )[0]
            #     new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping_str, re.DOTALL)
            #     if len(new_names) == len(old_names):
            #         return new_names
            #     else:
            #         raise ValueError(f"Failed to generate enough names; got {mapping}")
        except Exception as e:
            warn(f"Failed to parse cluster names: {e}")
            return old_names

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
                try:
                    response = await self._call_single_llm(
                        prompt, temperature=0.4, max_tokens=128
                    )
                except NotImplementedError:
                    responses = await self._call_llm_batch(
                        [prompt], temperature=0.4, max_tokens=128
                    )
                    if not responses:
                        raise RuntimeError("Connectivity probe returned no responses")
                    response = responses[0]
            else:
                try:
                    response = await self._call_single_llm_with_system(
                        system_prompt,
                        prompt,
                        temperature=0.4,
                        max_tokens=128,
                    )
                except NotImplementedError:
                    responses = await self._call_llm_with_system_prompt_batch(
                        [system_prompt],
                        [prompt],
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

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
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
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
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


def _resolve_api_key(
    api_key: str | None,
    env_new: str | None,
    env_legacy: str | None,
) -> str | None:
    """Helper function to migrate from the old environment variables to the new ones, while still allowing explicit API keys to take precedence."""
    if api_key is not None:
        return api_key

    new_key = os.getenv(env_new)
    legacy_key = os.getenv(env_legacy)

    if new_key:
        return new_key

    if legacy_key:
        warn(
            f"{env_legacy} is deprecated. Use {env_new} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return legacy_key

    return None


try:
    import litellm
    from litellm.exceptions import (
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError,
        NotFoundError,
        UnprocessableEntityError,
    )

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

        provider_kwargs : dict[str, Any], optional
            Additional keyword arguments passed directly to `litellm.completion()` /
            `litellm.acompletion()`. This allows callers to use LiteLLM-specific
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

        FAIL_FAST_EXCEPTIONS = (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
            UnprocessableEntityError,
        )
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
            provider_kwargs: dict[str, Any] | None = None,
            callback: DebugCallback | None = None,
        ):

            self.api_key = api_key
            self.model = model
            self.api_base = api_base
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

            filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
                module="pydantic",
            )
            filterwarnings(
                "ignore",
                message="Use 'content=<...>' to upload raw bytes/text content",
                category=DeprecationWarning,
                module="httpx",
            )

        @property
        def supports_system_prompts(self) -> bool:
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
            try:
                supported = litellm.get_supported_openai_params(model=self.model)
                return "response_format" in (supported or [])
            except Exception:
                logger.warning(
                    f"Failed to detect json_object support for model {self.model}, assuming not supported"
                )
                return False

        def _should_use_json_object(self) -> bool:
            if self.use_json_object is not None:
                return self.use_json_object

            if self._resolved_use_json_object is None:
                self._resolved_use_json_object = self._detect_json_object_support()
            return self._resolved_use_json_object

        def _provider_kwargs(
            self,
            messages,
            temperature: float,
            max_tokens: int,
        ) -> dict:
            kwargs = dict(self.provider_kwargs)
            kwargs.update(
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

            if self.api_base is not None:
                kwargs["api_base"] = self.api_base

            if self._should_use_json_object():
                kwargs["response_format"] = {"type": "json_object"}

            return kwargs

        def _completion_with_messages(
            self,
            messages,
            temperature: float,
            max_tokens: int,
        ) -> str:
            response = litellm.completion(
                **self._provider_kwargs(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return response.choices[0].message.content

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            return self._completion_with_messages(
                messages=[
                    {"role": "user", "content": prompt + self.extra_prompting},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
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
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if self._system_prompt_capability is None:
                    self._system_prompt_capability = True
                return result

            except self.FAIL_FAST_EXCEPTIONS:
                raise

            except Exception as e:
                if (
                    self._system_prompt_capability is not None
                    or not self._looks_like_unsupported_system_prompt_error(e)
                ):
                    raise

                self._system_prompt_capability = False
                return self._completion_with_messages(
                    messages=self._flatten_system_into_user(system_prompt, user_prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

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

        Uses litellm.acompletion() and an asyncio semaphore for bounded
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

        provider_kwargs : dict[str, Any], optional
            Additional keyword arguments passed directly to `litellm.completion()` /
            `litellm.acompletion()`. This allows callers to use LiteLLM-specific
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

        FAIL_FAST_EXCEPTIONS = (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
            UnprocessableEntityError,
        )
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
            provider_kwargs: dict[str, Any] | None = None,
            callback: DebugCallback | None = None,
        ):

            self.api_key = api_key
            self.model = model
            self.api_base = api_base
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

            self.use_json_object = use_json_object
            self._resolved_use_json_object: bool | None = None
            self.disable_system_prompts = disable_system_prompts
            self._system_prompt_capability: bool | None = None
            self.max_tokens_topic_name = max_tokens_topic_name
            self.max_tokens_cluster_names = max_tokens_cluster_names
            self.provider_kwargs = dict(provider_kwargs) if provider_kwargs else {}

        @property
        def supports_system_prompts(self) -> bool:
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
            try:
                supported = litellm.get_supported_openai_params(model=self.model)
                return "response_format" in (supported or [])
            except Exception:
                logger.warning(
                    f"Failed to detect json_object support for model {self.model}, assuming not supported"
                )
                return False

        def _should_use_json_object(self) -> bool:
            if self.use_json_object is not None:
                return self.use_json_object

            if self._resolved_use_json_object is None:
                self._resolved_use_json_object = self._detect_json_object_support()
            return self._resolved_use_json_object

        def _provider_kwargs(
            self,
            messages,
            temperature: float,
            max_tokens: int,
        ) -> dict:
            kwargs = dict(self.provider_kwargs)
            kwargs.update(
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

            if self.api_base is not None:
                kwargs["api_base"] = self.api_base

            if self._should_use_json_object():
                kwargs["response_format"] = {"type": "json_object"}

            return kwargs

        async def _acompletion_with_messages(
            self,
            messages,
            temperature: float,
            max_tokens: int,
        ) -> str:
            async with self.semaphore:
                response = await litellm.acompletion(
                    **self._provider_kwargs(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
            return response.choices[0].message.content

        async def _call_single_llm(
            self,
            prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return await self._acompletion_with_messages(
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        async def _call_single_llm_with_system(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
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
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if self._system_prompt_capability is None:
                    self._system_prompt_capability = True
                return result

            except self.FAIL_FAST_EXCEPTIONS:
                raise

            except Exception as e:
                if (
                    self._system_prompt_capability is not None
                    or not self._looks_like_unsupported_system_prompt_error(e)
                ):
                    raise

                self._system_prompt_capability = False
                return await self._acompletion_with_messages(
                    messages=self._flatten_system_into_user(system_prompt, user_prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        async def close(self):
            """No-op for parity with other async wrappers."""
            return None

except Exception as e:

    class LiteLLMNamer(FailedImportLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncLiteLLMNamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


def AnthropicNamer(
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        use_json_object=True,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        use_json_object=True,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str | None = None,  # deprecated, renamed to api_base
    httpx_client: Optional[httpx.Client] = None,  # deprecated
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        api_key=_resolve_api_key(
            api_key=api_key, env_new="COHERE_API_KEY", env_legacy="CO_API_KEY"
        ),
        api_base=_resolve_cohere_api_base(api_base, base_url),
        use_json_object=False,  # Cohere accepts this but the default prompts aren't strict enough to reliably produce non-empty JSON objects. Change when this is fixed.
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str = None,
    httpx_client: Optional[httpx.Client] = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        api_key=_resolve_api_key(
            api_key=api_key, env_new="COHERE_API_KEY", env_legacy="CO_API_KEY"
        ),
        api_base=_resolve_cohere_api_base(api_base, base_url),
        disable_system_prompts=False,
        use_json_object=False,  # Cohere accepts this but the default prompts aren't strict enough to reliably produce non-empty JSON objects. Change when this is fixed.
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def TogetherNamer(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
) -> LiteLLMNamer:
    """
    Deprecated. Use LiteLLMNamer(model="together_ai/<model_name>") instead.

    Parameters
    ----------
    model : str, optional
        Together AI model to use. Default is "meta-llama/Meta-Llama-3-8B-Instruct-Lite".
        May be in LiteLLM format ("together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite")
    api_key : str, optional
        Together AI API key. Falls back to the TOGETHERAI_API_KEY environment variable.
    api_base : str, optional
        Override the Together AI API endpoint. Can use the TOGETHERAI_API_BASE environment variable.
        Default is the standard OpenAI endpoint.
    llm_specific_instructions : str, optional
        Additional instructions appended to every prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

    Using a different model::

        namer = TogetherNamer(model="meta-llama/Meta-Llama-3-8B-Instruct-Lite", api_key="my-api-key")

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
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncTogether(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
) -> AsyncLiteLLMNamer:
    """
    Deprecated. Use AsyncLiteLLMNamer(model="together_ai/<model_name>") instead.

    Parameters
    ----------
    model : str, optional
        Together AI model to use. Default is "meta-llama/Meta-Llama-3-8B-Instruct-Lite". Must be in LiteLLM format ("together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite")
        or bare Together AI format ("meta-llama/Meta-Llama-3-8B-Instruct-Lite") — both are accepted.
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

    Using a different model::

        namer = AsyncTogether(model="meta-llama/Meta-Llama-3-8B-Instruct-Lite", api_key="my-api-key")

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
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


try:
    import llama_cpp

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
            for arg, val in kwargs.items():
                if arg == "n_ctx":
                    continue
                setattr(self, arg, val)
            self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm(
                prompt + self.extra_prompting,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            result = response["choices"][0]["text"]
            return result

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            raise InvalidLLMInputError(
                "System prompts are not supported for LlamaCpp wrapper"
            )

        @property
        def supports_system_prompts(self) -> bool:
            return False

except ImportError:

    class LlamaCppNamer(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    import huggingface_hub
    import transformers

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
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm(
                [{"role": "user", "content": prompt + self.extra_prompting}],
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
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            response = self.llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
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
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.max_concurrent_requests = max_concurrent_requests

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            responses = []
            for prompt in prompts:
                response = self.llm(
                    [{"role": "user", "content": prompt + self.extra_prompting}],
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
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            responses = []
            for system_prompt, user_prompt in zip(system_prompts, user_prompts):
                response = self.llm(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + self.extra_prompting},
                    ],
                    return_full_text=False,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.eos_token_id,
                )
                responses.append(response[0]["generated_text"])
            return responses

except:

    class HuggingFaceNamer(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncHuggingFaceNamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    import vllm
    import vllm.v1.engine.exceptions

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
            self.kwargs = kwargs
            self._start_engine()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _start_engine(self):
            """
            Start the VLLM engine. This is necessary to initialize the model.
            """
            self.llm = vllm.LLM(model=self.model, **self.kwargs)

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            sampling_params = vllm.SamplingParams(
                temperature=temperature, max_tokens=max_tokens
            )
            message = [{"role": "user", "content": prompt + self.extra_prompting}]
            try:
                outputs = self.llm.chat(message, sampling_params=sampling_params)
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()
                # Retry after restarting the engine
                outputs = self.llm.chat(message, sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            return result

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            sampling_params = vllm.SamplingParams(
                temperature=temperature, max_tokens=max_tokens
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ]

            try:
                outputs = self.llm.chat(messages, sampling_params=sampling_params)
            except vllm.v1.engine.exceptions.EngineDeadError:
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
            self.kwargs = kwargs
            self._start_engine()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.max_concurrent_requests = max_concurrent_requests

        def _start_engine(self):
            self.llm = vllm.LLM(model=self.model, **self.kwargs)

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            messages = [
                [{"role": "user", "content": prompt + self.extra_prompting}]
                for prompt in prompts
            ]
            sampling_params = vllm.SamplingParams(
                temperature=temperature, max_tokens=max_tokens
            )

            try:
                outputs = self.llm.chat(
                    messages=messages, sampling_params=sampling_params
                )
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()  # Restart the engine if it fails
                outputs = self.llm.chat(
                    messages=messages, sampling_params=sampling_params
                )

            return [output.outputs[0].text for output in outputs]

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            messages = []
            for system_prompt, user_prompt in zip(system_prompts, user_prompts):
                messages.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + self.extra_prompting},
                    ]
                )
            sampling_params = vllm.SamplingParams(
                temperature=temperature, max_tokens=max_tokens
            )

            try:
                outputs = self.llm.chat(
                    messages=messages, sampling_params=sampling_params
                )
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()  # Restart the engine if it fails
                outputs = self.llm.chat(
                    messages=messages, sampling_params=sampling_params
                )

            return [output.outputs[0].text for output in outputs]

except ImportError:

    class VLLMNamer(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncVLLMNamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    import cohere

    class CohereBatchNamer(FailedImportAsyncLLMWrapper):
        """
        Provides access to Cohere's Batch Processing API with asynchronous support.
        This allows for processing large batches of prompts over an extended period.
        For more information on Cohere's Batch Processing, see https://docs.cohere.com/docs/batch-processing.

        This wrapper conforms to the AsyncLLMWrapper interface, but note that it uses Cohere's batch API,
        which processes jobs over hours rather than in real-time. The async methods will block until the batch job completes.

        This class provides a different tradeoff between speed and cost compared to the AsyncCohere wrapper.
        It is designed for scenarios where you have a large number of prompts to process and can afford to wait for the results.
        Cohere's batch processing is more cost-effective (half the cost per token) for large volumes of data, but it does
        not provide immediate responses.

        Parameters:
        -----------
        api_key: str
            Your Cohere API key. You can set this as an environment variable CO_API_KEY or pass it directly

        model: str, optional
            The name of the Cohere model to use. Default is "command-r-08-2024". You can use any model available
            in the Cohere API, but this is a good balance of performance and cost.

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
        client: cohere.ClientV2
            The Cohere client instance for batch processing.

        model: str
            The name of the Cohere model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Cohere, this is always True.

        """

        def __init__(
            self,
            api_key: str,
            model: str = "command-r-08-2024",
            llm_specific_instructions=None,
            polling_interval: int = 60,
            timeout: int = 7200,
            callback: DebugCallback | None = None,
        ):
            self.client = cohere.ClientV2(api_key=api_key)
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.polling_interval = polling_interval
            self.timeout = timeout

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """
            Submit a batch job and wait for completion.
            This is a blocking operation that could take hours.
            """
            # Create batch requests
            requests = []
            for i, prompt in enumerate(prompts):
                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt + self.extra_prompting,
                                }
                            ],
                            "temperature": temperature,
                        },
                    }
                )

            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id

            # Wait for completion (with async sleep)
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """
            Submit a batch job with system prompts and wait for completion.
            """
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            # Create batch requests
            requests = []
            for i, (sys_prompt, user_prompt) in enumerate(
                zip(system_prompts, user_prompts)
            ):
                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {
                                    "role": "user",
                                    "content": user_prompt + self.extra_prompting,
                                },
                            ],
                            "temperature": temperature,
                        },
                    }
                )

            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id

            # Wait for completion
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _wait_for_completion_async(self, batch_id: str) -> bool:
            """
            Wait for a batch job to complete, using async sleep.
            Returns True if completed successfully, False if failed or timed out.
            """
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                batch = self.client.beta.messages.batches.retrieve(batch_id)

                if batch.processing_status == "ended":
                    return True
                elif batch.processing_status in ["canceling", "canceled", "expired"]:
                    warn(
                        f"Batch job {batch_id} ended with status: {batch.processing_status}"
                    )
                    return False

                # Use async sleep to not block the event loop
                await asyncio.sleep(self.polling_interval)

            warn(f"Batch job {batch_id} timed out after {self.timeout} seconds")
            return False

        async def _retrieve_batch_results(self, batch_id: str) -> List[str]:
            """
            Retrieve raw text results from a completed batch job.
            """
            # Run the synchronous API call in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            results_page = await loop.run_in_executor(
                None, self.client.beta.messages.batches.results, batch_id
            )

            # Sort by custom_id to maintain order
            sorted_results = sorted(
                results_page.results, key=lambda x: int(x.custom_id)
            )

            responses = []
            for result in sorted_results:
                if result.result.type == "succeeded":
                    responses.append(result.result.message.content[0].text)
                else:
                    warn(f"Request {result.custom_id} failed: {result.result.error}")
                    responses.append("")  # Empty string for failed requests

            return responses

        # Additional methods for non-blocking usage
        def submit_batch(
            self,
            prompts: List[Union[str, Dict[str, str]]],
            temperature: float,
            max_tokens: int,
        ) -> str:
            """
            Submit a batch job without waiting. Returns batch ID.
            This is for users who want to manage batch jobs manually.
            """
            requests = []

            for i, prompt in enumerate(prompts):
                if isinstance(prompt, str):
                    messages = [
                        {"role": "user", "content": prompt + self.extra_prompting}
                    ]
                elif isinstance(prompt, dict):
                    messages = [
                        {"role": "system", "content": prompt["system"]},
                        {
                            "role": "user",
                            "content": prompt["user"] + self.extra_prompting,
                        },
                    ]
                else:
                    raise InvalidLLMInputError(f"Prompt must be string or dict")

                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": messages,
                            "temperature": temperature,
                        },
                    }
                )

            batch = self.client.beta.messages.batches.create(requests=requests)
            return batch.id

        def get_batch_status(self, batch_id: str) -> str:
            """Check the status of a batch job."""
            batch = self.client.beta.messages.batches.retrieve(batch_id)
            return batch.processing_status

        async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
            """Retrieve raw text results from a completed batch."""
            return await self._retrieve_batch_results(batch_id)

        def cancel_batch(self, batch_id: str):
            """Cancel a running batch job."""
            self.client.beta.messages.batches.cancel(batch_id)

except:

    class CohereNamer(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncCohereNamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    import anthropic
    import time

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
        ):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.polling_interval = polling_interval
            self.timeout = timeout

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """
            Submit a batch job and wait for completion.
            This is a blocking operation that could take hours.
            """
            # Create batch requests
            requests = []
            for i, prompt in enumerate(prompts):
                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt + self.extra_prompting,
                                }
                            ],
                            "temperature": temperature,
                        },
                    }
                )

            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id

            # Wait for completion (with async sleep)
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """
            Submit a batch job with system prompts and wait for completion.
            """
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            # Create batch requests
            requests = []
            for i, (sys_prompt, user_prompt) in enumerate(
                zip(system_prompts, user_prompts)
            ):
                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "system": sys_prompt,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_prompt + self.extra_prompting,
                                },
                            ],
                            "temperature": temperature,
                        },
                    }
                )

            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id

            # Wait for completion
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _wait_for_completion_async(self, batch_id: str) -> bool:
            """
            Wait for a batch job to complete, using async sleep.
            Returns True if completed successfully, False if failed or timed out.
            """
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                batch = self.client.beta.messages.batches.retrieve(batch_id)

                if batch.processing_status == "ended":
                    return True
                elif batch.processing_status in ["canceling", "canceled", "expired"]:
                    warn(
                        f"Batch job {batch_id} ended with status: {batch.processing_status}"
                    )
                    return False

                # Use async sleep to not block the event loop
                await asyncio.sleep(self.polling_interval)

            warn(f"Batch job {batch_id} timed out after {self.timeout} seconds")
            return False

        async def _retrieve_batch_results(self, batch_id: str) -> List[str]:
            """
            Retrieve raw text results from a completed batch job.
            """
            # Run the synchronous API call in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            results_page = await loop.run_in_executor(
                None, self.client.beta.messages.batches.results, batch_id
            )

            # Sort by custom_id to maintain order
            sorted_results = sorted(
                results_page.results, key=lambda x: int(x.custom_id)
            )

            responses = []
            for result in sorted_results:
                if result.result.type == "succeeded":
                    responses.append(result.result.message.content[0].text)
                else:
                    warn(f"Request {result.custom_id} failed: {result.result.error}")
                    responses.append("")  # Empty string for failed requests

            return responses

        # Additional methods for non-blocking usage
        def submit_batch(
            self,
            prompts: List[Union[str, Dict[str, str]]],
            temperature: float,
            max_tokens: int,
        ) -> str:
            """
            Submit a batch job without waiting. Returns batch ID.
            This is for users who want to manage batch jobs manually.
            """
            requests = []

            for i, prompt in enumerate(prompts):
                if isinstance(prompt, str):
                    messages = [
                        {"role": "user", "content": prompt + self.extra_prompting}
                    ]
                elif isinstance(prompt, dict):
                    messages = [
                        {"role": "system", "content": prompt["system"]},
                        {
                            "role": "user",
                            "content": prompt["user"] + self.extra_prompting,
                        },
                    ]
                else:
                    raise InvalidLLMInputError(f"Prompt must be string or dict")

                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": messages,
                            "temperature": temperature,
                        },
                    }
                )

            batch = self.client.beta.messages.batches.create(requests=requests)
            return batch.id

        def get_batch_status(self, batch_id: str) -> str:
            """Check the status of a batch job."""
            batch = self.client.beta.messages.batches.retrieve(batch_id)
            return batch.processing_status

        async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
            """Retrieve raw text results from a completed batch."""
            return await self._retrieve_batch_results(batch_id)

        def cancel_batch(self, batch_id: str):
            """Cancel a running batch job."""
            self.client.beta.messages.batches.cancel(batch_id)

except:

    class BatchAnthropicNamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


# Ollama
def OllamaNamer(
    model: str = "llama3.2",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    host: str | None = None,  # deprecated, renamed to api_base
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncOllamaNamer(
    model: str = "llama3.2",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 5,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    host: str | None = None,  # deprecated, renamed to api_base
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


## OpenAI Convenience Wrappers
def NotebookOpenAINamerMock(*args, **kwargs):
    """
    For mocking OpenAINamer calls with a local Ollama model.
    """
    logger.info("Using NotebookOpenAINamerMock instead of OpenAINamer")
    return OllamaNamer(model=get_test_ollama_model())


@notebook_test_replacement(NotebookOpenAINamerMock)
def OpenAINamer(
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_tokens_topic_name: int = 128,
    max_tokens_cluster_names: int = 1024,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str | None = None,  # deprecated, renamed to api_base
    http_client: "httpx.Client | None" = None,  # deprecated, pass via provider_kwargs instead
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        use_json_object=True,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    base_url: str | None = None,  # deprecated, renamed to api_base
    organization: str | None = None,  # deprecated, pass via provider_kwargs instead
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        use_json_object=True,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)


    See Also
    --------
    LiteLLMNamer : The underlying namer, supports 100+ providers directly.
    """
    resolved_endpoint = api_base or endpoint
    return LiteLLMNamer(
        model=_azure_model(model),
        api_key=_resolve_api_key(
            api_key=api_key, env_new="AZURE_AI_API_KEY", env_legacy="AZURE_API_KEY"
        ),
        api_base=resolved_endpoint,
        use_json_object=True,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

    See Also
    --------
    AsyncLiteLLMNamer : The underlying async namer, supports 100+ providers directly.
    """
    resolved_endpoint = api_base or endpoint
    return AsyncLiteLLMNamer(
        model=_azure_model(model),
        api_key=_resolve_api_key(
            api_key=api_key, env_new="AZURE_AI_API_KEY", env_legacy="AZURE_API_KEY"
        ),
        api_base=resolved_endpoint,
        disable_system_prompts=False,
        use_json_object=True,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens_topic_name=max_tokens_topic_name,
        max_tokens_cluster_names=max_tokens_cluster_names,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.aio import (
        ChatCompletionsClient as AsyncChatCompletionsClient,
    )
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    class BatchAzureAINamer(AsyncLLMWrapper):
        """
        Provides access to Azure AI Foundry's Batch Processing API with asynchronous support.
        This allows for processing large batches of prompts over an extended period.
        For more information on Azure AI Foundry's Batch Processing, see https://learn.microsoft.com/en-us/azure/ai-services/ai-foundry/batch-processing.

        This wrapper conforms to the AsyncLLMWrapper interface, but note that it uses Azure's batch API,
        which processes jobs over hours rather than in real-time. The async methods will block until the batch job completes.

        This class provides a different tradeoff between speed and cost compared to the AsyncAzureAI wrapper.
        It is designed for scenarios where you have a large number of prompts to process and can afford to wait for the results.
        Azure's batch processing is more cost-effective (half the cost per token) for large volumes of data, but it does
        not provide immediate responses.

        Parameters:
        -----------
        api_key: str
            Your Azure API key. You can set this as an environment variable AZURE_API_KEY or pass it directly.

        endpoint: str
            The endpoint URL for your Azure AI Foundry model. This is typically in the format "https://<your-resource-name>.openai.azure.com/".

        model: str
            The name of the Azure AI Foundry model to use. This should match the model name you created in Azure AI Foundry.

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
        client: azure.ai.inference.ChatCompletionsClient
            The Azure AI Foundry LLM client instance.

        model: str
            The name of the Azure AI Foundry model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Azure AI Foundry, this is always True.

        """

        def __init__(
            self,
            api_key: str,
            endpoint: str,
            model: str,
            llm_specific_instructions=None,
            polling_interval: int = 60,
            timeout: int = 7200,
            callback: DebugCallback | None = None,
        ):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.polling_interval = polling_interval
            self.timeout = timeout
            self.callback = callback
            self._warn_if_debug_callback_unsupported()

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """
            Submit a batch job and wait for completion.
            This is a blocking operation that could take hours.
            """
            # Create batch requests
            requests = []
            for i, prompt in enumerate(prompts):
                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt + self.extra_prompting,
                                }
                            ],
                            "temperature": temperature,
                        },
                    }
                )

            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id

            # Wait for completion (with async sleep)
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """
            Submit a batch job with system prompts and wait for completion.
            """
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            # Create batch requests
            requests = []
            for i, (sys_prompt, user_prompt) in enumerate(
                zip(system_prompts, user_prompts)
            ):
                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {
                                    "role": "user",
                                    "content": user_prompt + self.extra_prompting,
                                },
                            ],
                            "temperature": temperature,
                        },
                    }
                )

            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id

            # Wait for completion
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _wait_for_completion_async(self, batch_id: str) -> bool:
            """
            Wait for a batch job to complete, using async sleep.
            Returns True if completed successfully, False if failed or timed out.
            """
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                batch = self.client.beta.messages.batches.retrieve(batch_id)

                if batch.processing_status == "ended":
                    return True
                elif batch.processing_status in ["canceling", "canceled", "expired"]:
                    warn(
                        f"Batch job {batch_id} ended with status: {batch.processing_status}"
                    )
                    return False

                # Use async sleep to not block the event loop
                await asyncio.sleep(self.polling_interval)

            warn(f"Batch job {batch_id} timed out after {self.timeout} seconds")
            return False

        async def _retrieve_batch_results(self, batch_id: str) -> List[str]:
            """
            Retrieve raw text results from a completed batch job.
            """
            # Run the synchronous API call in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            results_page = await loop.run_in_executor(
                None, self.client.beta.messages.batches.results, batch_id
            )

            # Sort by custom_id to maintain order
            sorted_results = sorted(
                results_page.results, key=lambda x: int(x.custom_id)
            )

            responses = []
            for result in sorted_results:
                if result.result.type == "succeeded":
                    responses.append(result.result.message.content[0].text)
                else:
                    warn(f"Request {result.custom_id} failed: {result.result.error}")
                    responses.append("")  # Empty string for failed requests

            return responses

        # Additional methods for non-blocking usage
        def submit_batch(
            self,
            prompts: List[Union[str, Dict[str, str]]],
            temperature: float,
            max_tokens: int,
        ) -> str:
            """
            Submit a batch job without waiting. Returns batch ID.
            This is for users who want to manage batch jobs manually.
            """
            requests = []

            for i, prompt in enumerate(prompts):
                if isinstance(prompt, str):
                    messages = [
                        {"role": "user", "content": prompt + self.extra_prompting}
                    ]
                elif isinstance(prompt, dict):
                    messages = [
                        {"role": "system", "content": prompt["system"]},
                        {
                            "role": "user",
                            "content": prompt["user"] + self.extra_prompting,
                        },
                    ]
                else:
                    raise InvalidLLMInputError(f"Prompt must be string or dict")

                requests.append(
                    {
                        "custom_id": str(i),
                        "params": {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": messages,
                            "temperature": temperature,
                        },
                    }
                )

            batch = self.client.beta.messages.batches.create(requests=requests)
            return batch.id

        def get_batch_status(self, batch_id: str) -> str:
            """Check the status of a batch job."""
            batch = self.client.beta.messages.batches.retrieve(batch_id)
            return batch.processing_status

        async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
            """Retrieve raw text results from a completed batch."""
            return await self._retrieve_batch_results(batch_id)

        def cancel_batch(self, batch_id: str):
            """Cancel a running batch job."""
            self.client.beta.messages.batches.cancel(batch_id)

except ImportError:

    class BatchAzureAINamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


def GoogleGeminiNamer(
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        api_key=_resolve_api_key(
            api_key=api_key, env_new="GEMINI_API_KEY", env_legacy="GOOGLE_API_KEY"
        ),
        api_base=api_base,
        use_json_object=True,
        disable_system_prompts=False,
        llm_specific_instructions=llm_specific_instructions,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )


def AsyncGoogleGeminiNamer(
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
    api_base: str | None = None,
    llm_specific_instructions: str | None = None,
    max_concurrent_requests: int = 10,
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        api_key=_resolve_api_key(
            api_key=api_key, env_new="GEMINI_API_KEY", env_legacy="GOOGLE_API_KEY"
        ),
        api_base=api_base,
        disable_system_prompts=False,
        use_json_object=True,
        llm_specific_instructions=llm_specific_instructions,
        max_concurrent_requests=max_concurrent_requests,
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
    provider_kwargs: dict[str, Any] | None = None,
    callback: DebugCallback | None = None,
    api_token: str = None,
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
        toponymy = Toponymy(embedding_model=..., llm_namer=namer)

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
        use_json_object=False,  # Replicate's API does not support this
        llm_specific_instructions=llm_specific_instructions,
        provider_kwargs=provider_kwargs,
        callback=callback,
    )
