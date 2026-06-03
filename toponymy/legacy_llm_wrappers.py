import openai

from .llm_wrappers import AsyncLLMWrapper, LLMWrapper, DebugCallback
from typing import List, Optional, Union, Dict, Generic, TypeVar, Callable, Any
import httpx

from openai import (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)

import anthropic
import time
from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
    PermissionDeniedError as AnthropicPermissionDeniedError,
    BadRequestError as AnthropicBadRequestError,
    NotFoundError as AnthropicNotFoundError,
)


class OpenAINamerLegacy(LLMWrapper):
    """
    Provides access to OpenAI's LLMs with the Toponymy framework. For more information on OpenAI, see
    https://platform.openai.com/docs/models/overview. You will need an OpenAI API key to use this wrapper.
    The default model is "gpt-4o-mini", which is a sufficiently powerful model for generating topic names and clusters,
    but inexpensive in terms of dollars per token. You can use more advanced models, but they have diminishing returns
    for this task, and are more expensive.

    Parameters:
    -----------

    api_key: str
        Your OpenAI API key. You can set this as an environment variable OPENAI_API_KEY or pass it directly

    model: str, optional
        The name of the OpenAI model to use. Default is "gpt-4o-mini". You can use any model available
        in the OpenAI API, but this is a good balance of performance and cost.

    base_url: str, optional
        The base URL for the OpenAI API. Default is None, which uses the default OpenAI endpoint.
        You can set this as an environment variable OPENAI_API_BASE to use a different endpoint, such as
        a hosted model supporting the openAI API.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.

    Attributes:
    -----------
    llm: openai.OpenAI
        The OpenAI LLM client instance.

    model: str
        The name of the OpenAI model being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For OpenAI, this is always True.

    Note:
    -----
    This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
    consider using the AsyncOpenAI wrapper instead.
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
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = None,
        http_client: "httpx.Client | None" = None,
        llm_specific_instructions=None,
        callback: DebugCallback | None = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set it as an environment variable OPENAI_API_KEY or pass it directly to the constructor."
            )

        self.llm = openai.OpenAI(
            api_key=api_key, base_url=base_url, http_client=http_client
        )
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.llm.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt + self.extra_prompting}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        return result

    def _call_llm_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.llm.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        return result


class AsyncOpenAINamerLegacy(AsyncLLMWrapper):
    """
    Provides access to OpenAI's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
    For more information on OpenAI, see https://platform.openai.com/docs/models/overview. You will need an OpenAI API key to use this wrapper.
    The default model is "gpt-4o-mini", which is a sufficiently powerful model for generating topic names and clusters,
    but inexpensive in terms of dollars per token. You can use more advanced models, but they have diminishing returns for this task,
    and are more expensive.

    As an asynchronous wrapper this will potentially speed up topic naming, particularly when you have a large number of topics. If,
    however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls.

    Parameters:
    -----------

    api_key: str
        Your OpenAI API key. You can set this as an environment variable OPENAI_API_KEY or pass it directly

    model: str, optional
        The name of the OpenAI model to use. Default is "gpt-4o-mini". You can use any model available
        in the OpenAI API, but this is a good balance of performance and cost.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
        model-specific instructions or context that may help improve the quality of the generated text.


    organization: str, optional
        The OpenAI organization ID to use for the API requests. If not provided, the default organization will be used.

    base_url: str, optional
        The base URL for the OpenAI API. Default is None, which uses the default OpenAI endpoint.
        You can set this as an environment variable OPENAI_API_BASE to use a different endpoint, such as
        a hosted model supporting the openAI API.

    Attributes:
    -----------

    client: openai.AsyncOpenAI
        The OpenAI asynchronous LLM client instance.

    model: str
        The name of the OpenAI model being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For OpenAI, this is always True.
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
        api_key: str,
        model: str = "gpt-4o-mini",
        llm_specific_instructions=None,
        max_concurrent_requests: int = 10,
        organization: str = None,
        base_url: str = None,
        callback: DebugCallback | None = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set it as an environment variable OPENAI_API_KEY or pass it directly to the constructor."
            )

        self.client = openai.AsyncOpenAI(api_key=api_key, organization=organization)
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _call_single_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call the LLM for a single prompt."""
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt + self.extra_prompting},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        return response.choices[0].message.content

    async def _call_single_llm_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the LLM for a single prompt with system prompt."""
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt + self.extra_prompting,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

    async def close(self):
        """Close the client connection."""
        await self.client.close()


## Anthropic Namers


class AnthropicNamerLegacy(LLMWrapper):
    """
    Provides access to Anthropic's LLMs with the Toponymy framework. For more information on Anthropic, see
    https://docs.anthropic.com/docs/overview. You will need an Anthropic API key to use this wrapper.
    The default model is "claude-haiku-4-5-20251001", which is the smallest model available, but is generally
    more than sufficient for generating topic names and clusters. You can use more advanced
    models, but they have diminishing returns for this task, and are more expensive.

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

    Attributes:
    -----------

    llm: anthropic.Anthropic
        The Anthropic LLM client instance.

    model: str
        The name of the Anthropic model being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For Anthropic, this is always True.

    Note:
    -----
    This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
    consider using the AsyncAnthropic wrapper instead.
    """

    FAIL_FAST_EXCEPTIONS = (
        AnthropicAuthenticationError,
        AnthropicPermissionDeniedError,
        AnthropicBadRequestError,
        AnthropicNotFoundError,
    )

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        llm_specific_instructions=None,
        callback: DebugCallback | None = None,
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set it as an environment variable ANTHROPIC_API_KEY or pass it directly to the constructor."
            )

        self.llm = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.llm.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt + self.extra_prompting}],
            temperature=temperature,
        )
        result = response.content[0].text
        return result

    def _call_llm_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.llm.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ],
            temperature=temperature,
        )
        result = response.content[0].text
        return result


class AsyncAnthropicNamerLegacy(AsyncLLMWrapper):
    """
    Provides access to Anthropic's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
    For more information on Anthropic, see https://docs.anthropic.com/docs/overview. You will need an Anthropic API key to use this wrapper.
    The default model is "claude-haiku-4-5-20251001", which is the smallest model available, but is generally
    more than sufficient for generating topic names and clusters. You can use more advanced models, but they have diminishing returns for this task,
    and are more expensive.

    As an asynchronous wrapper this will potentially speed up topic naming, particularly when you have a large number of topics. If,
    however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls.

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

    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Anthropic API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Anthropic API. Higher values may improve throughput but could lead to rate limiting.

    Attributes:
    -----------
    llm: anthropic.AsyncAnthropic
        The Anthropic asynchronous LLM client instance.

    model: str
        The name of the Anthropic model being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For Anthropic, this is always True.
    """

    FAIL_FAST_EXCEPTIONS = (
        AnthropicAuthenticationError,
        AnthropicPermissionDeniedError,
        AnthropicBadRequestError,
        AnthropicNotFoundError,
    )

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        llm_specific_instructions=None,
        max_concurrent_requests: int = 10,
        callback: DebugCallback | None = None,
    ):

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set it as an environment variable ANTHROPIC_API_KEY or pass it directly to the constructor."
            )

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _call_single_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call the LLM for a single prompt."""
        async with self.semaphore:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
            )
            return response.content[0].text

    async def _call_single_llm_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the LLM for a single prompt with system prompt."""
        async with self.semaphore:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                temperature=temperature,
            )
            return response.content[0].text


## Cohere wrappers

import cohere


class CohereNamerLegacy(LLMWrapper):
    """
    Provides access to Cohere's LLMs with the Toponymy framework. For more information on Cohere, see
    https://docs.cohere.com/docs/llm-overview. You will need a Cohere API key to use this wrapper. The
    default model is "command-r-08-2024", which is a sufficiently powerful to do a good job of generating
    topic names and clusters, but inexpensive in terms of dollars per token. You can use more advanced
    models, but they have diminishing returns for this task, and are more expensive.

    Parameters:
    -----------

    api_key: str
        Your Cohere API key. You can set this as an environment variable CO_API_KEY or pass it directly

    model: str, optional
        The name of the Cohere model to use. Default is "command-r-08-2024". You can use any model available
        in the Cohere API, but this is a good balance of performance and cost.

    base_url: str, optional
        The base URL for the Cohere API. Default is "https://api.cohere.com". You can set this as an environment
        variable CO_API_URL to use a different endpoint, such as for Cohere's private cloud.

    httpx_client: httpx.Client, optional
        An optional httpx client to use for making requests. If not provided, a default client will be created.
        This can be useful when using Cohere's private cloud or when you need to customize the HTTP client settings.

    llm_specific_instructions: str, optional
        Additional instructions specific to the LLM, appended to the prompt.

    Attributes:
    -----------

    llm: cohere.ClientV2
        The Cohere LLM client instance.

    model: str
        The name of the Cohere model being used.

    extra_prompting: str
        Additional instructions specific to the LLM, appended to the prompt.

    supports_system_prompts: bool
        Indicates whether the wrapper supports system prompts. For Cohere, this is always True.

    Note:
    -----
    This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
    consider using the AsyncCohere wrapper instead.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-08-2024",
        base_url: str = None,
        httpx_client: Optional[httpx.Client] = None,
        llm_specific_instructions=None,
        callback: DebugCallback | None = None,
    ):
        if base_url is None:
            base_url = os.getenv("CO_API_URL", "https://api.cohere.com")

        api_key = api_key or os.getenv("CO_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key is required. Set it as an environment variable CO_API_KEY or pass it directly to the constructor."
            )

        self.llm = cohere.ClientV2(
            api_key=api_key, base_url=base_url, httpx_client=httpx_client
        )

        try:
            self.llm.models.get(model)
        except cohere.errors.not_found_error.NotFoundError:
            models = [x.name for x in self.llm.models.list().models]
            msg = f"Model '{model}' not found, try one of {models}"
            raise ValueError(msg)
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.llm.chat(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt + self.extra_prompting}],
            temperature=temperature,
            # This results in failures more often than useful output
            # response_format={"type": "json_object"},
        )
        result = response.message.content[0].text
        return result

    def _call_llm_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.llm.chat(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ],
            temperature=temperature,
            # This results in failures more often than useful output
            # response_format={"type": "json_object"},
        )
        result = response.message.content[0].text
        return result


class AsyncCohereNamerLegacy(AsyncLLMWrapper):
    """
    Provides access to Cohere's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
    For more information on Cohere, see https://docs.cohere.com/docs/llm-overview. You will need a Cohere API key to use this wrapper.
    The default model is "command-r-08-2024", which is a sufficiently powerful model for generating topic names and clusters,
    but inexpensive in terms of dollars per token. You can use more advanced models, but they have diminishing returns for this task,
    and are more expensive.

    As an asynchronous wrapper this will potentially speed up topic naming, particlarly when you have a large number of topics. If,
    however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls."

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

    max_concurrent_requests: int, optional
        The maximum number of concurrent requests to the Cohere API. Default is 10. This can be adjusted based on your
        application's needs and the rate limits of the Cohere API. Higher values may improve throughput but could lead to rate limiting.

    base_url: str, optional
        The base URL for the Cohere API. Default is "https://api.cohere.com". You can set this as an environment
        variable CO_API_URL to use a different endpoint, such as for Cohere's private cloud.

    httpx_client: httpx.Client, optional
        An optional httpx client to use for making requests. If not provided, a default client will be created.

    Attributes:
    -----------
    llm: cohere.AsyncClientV2
        The Cohere asynchronous LLM client instance.

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
        max_concurrent_requests: int = 10,
        base_url: str = None,
        httpx_client: Optional[httpx.Client] = None,
        callback: DebugCallback | None = None,
    ):
        if base_url is None:
            base_url = os.getenv("CO_API_URL", "https://api.cohere.com")

        api_key = api_key or os.getenv("CO_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key is required. Set it as an environment variable CO_API_KEY or pass it directly to the constructor."
            )

        self.llm = cohere.AsyncClientV2(
            api_key=api_key, base_url=base_url, httpx_client=httpx_client
        )
        self.model = model
        self.callback = callback
        self._warn_if_debug_callback_unsupported()
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _call_single_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Call the LLM for a single prompt."""
        try:
            async with self.semaphore:
                response = await self.llm.chat(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt + self.extra_prompting}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.message.content[0].text
        except Exception as e:
            warn(f"Cohere API call failed: {str(e)[:100]}...")
            return ""

    async def _call_single_llm_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the LLM for a single prompt with system prompt."""
        try:
            async with self.semaphore:
                response = await self.llm.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt + self.extra_prompting,
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.message.content[0].text
        except Exception as e:
            warn(f"Cohere API call failed: {str(e)[:100]}...")
            return ""

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        """Process a batch of prompts concurrently."""
        tasks = [
            self._call_single_llm(prompt, temperature, max_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        """Process a batch of prompts with system prompts concurrently."""
        if len(system_prompts) != len(user_prompts):
            raise ValueError(
                "Number of system prompts must match number of user prompts"
            )

        tasks = [
            self._call_single_llm_with_system(
                sys_prompt, user_prompt, temperature, max_tokens
            )
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        return await asyncio.gather(*tasks)


try:
    import together

    class TogetherNamerLegacy(LLMWrapper):
        """
        Provides access to Together AI's LLMs with the Toponymy framework. Together AI provides access to various open-source models.
        For more information on Together AI, see https://together.ai/. You will need a Together API key to use this wrapper.

        Parameters:
        -----------
        api_key: str
            Your Together API key. You can set this as an environment variable TOGETHER_API_KEY or pass it directly.

        model: str, optional
            The name of the Together model to use. Default is "meta-llama/Llama-3-8b-chat-hf".
            Available models include various Llama, Mixtral, and other open-source models.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        Attributes:
        -----------
        client: together.Together
            The Together client instance.

        model: str
            The name of the Together model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Together, this is always True.
        """

        def __init__(
            self,
            api_key: str,
            model: str = "meta-llama/Llama-3-8b-chat-hf",
            llm_specific_instructions=None,
            callback: DebugCallback | None = None,
        ):
            api_key = api_key or os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError(
                    "Together API key is required. Set it as an environment variable TOGETHER_API_KEY or pass it directly to the constructor."
                )

            self.client = together.Together(api_key=api_key)
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    class AsyncTogetherLegacy(AsyncLLMWrapper):
        """
        Provides access to Together AI's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        Together AI provides access to various open-source models. For more information on Together AI, see https://together.ai/.
        You will need a Together API key to use this wrapper.

        Parameters:
        -----------
        api_key: str
            Your Together API key. You can set this as an environment variable TOGETHER_API_KEY or pass it directly.

        model: str, optional
            The name of the Together model to use. Default is "meta-llama/Llama-3-8b-chat-hf".
            Available models include various Llama, Mixtral, and other open-source models.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Together API. Default is 10.

        Attributes:
        -----------
        client: together.AsyncTogether
            The Together asynchronous client instance.

        model: str
            The name of the Together model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Together, this is always True.
        """

        def __init__(
            self,
            api_key: str,
            model: str = "meta-llama/Llama-3-8b-chat-hf",
            llm_specific_instructions=None,
            max_concurrent_requests: int = 10,
            callback: DebugCallback | None = None,
        ):
            api_key = api_key or os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError(
                    "Together API key is required. Set it as an environment variable TOGETHER_API_KEY or pass it directly to the constructor."
                )

            self.client = together.AsyncTogether(api_key=api_key)
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(
            self, prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt."""
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt + self.extra_prompting}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                warn(f"Together API call failed: {str(e)[:100]}...")
                return ""

        async def _call_single_llm_with_system(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": user_prompt + self.extra_prompting,
                            },
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                warn(f"Together API call failed: {str(e)[:100]}...")
                return ""

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            tasks = [
                self._call_single_llm_with_system(
                    sys_prompt, user_prompt, temperature, max_tokens
                )
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

        async def close(self):
            """Close the client connection."""
            await self.client.close()

except ImportError:

    class TogetherNamer(FailedImportLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncTogether(FailedImportAsyncLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.aio import (
        ChatCompletionsClient as AsyncChatCompletionsClient,
    )
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    class AzureAINamerLegacy(LLMWrapper):
        """
        Provides access to the Azure AI Foundry LLMs with the Toponymy framework. For more information on Azure AI, see
        https://learn.microsoft.com/en-us/azure/ai-services/overview. You will need an Azure API key for your Foundry model
        to use this wrapper. You will need to provide both the endpoint, and the model name per the instiated model on
        AI Foundry. For more information on creating models with Azure AI Foundry, see https://learn.microsoft.com/en-us/azure/ai-services/ai-foundry/create-models.

        Parameters:
        -----------
        api_key: str
            Your Azure API key. You can set this as an environment variable AZURE_API_KEY or pass it directly

        endpoint: str
            The endpoint URL for your Azure AI Foundry model. This is typically in the format "https://<your-resource-name>.openai.azure.com/".

        model: str
            The name of the Azure AI Foundry model to use. This should match the model name you created in Azure AI Foundry.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        Attributes:
        -----------

        llm: azure.ai.inference.ChatCompletionsClient
            The Azure AI Foundry LLM client instance.

        model: str
            The name of the Azure AI Foundry model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Azure AI Foundry, this is always True.

        Note:
        -----
        This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
        consider using the AsyncAzureAI wrapper instead.

        """

        def __init__(
            self,
            api_key: str,
            endpoint: str,
            model: str,
            llm_specific_instructions=None,
            callback: DebugCallback | None = None,
        ):
            self.endpoint = endpoint
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            api_key = api_key or os.getenv("AZURE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure API key is required. Set it as an environment variable AZURE_API_KEY or pass it directly to the constructor."
                )

            if not endpoint:
                raise ValueError(
                    "Azure endpoint is required. Provide the endpoint URL for your Azure AI Foundry model."
                )

            if not model:
                raise ValueError(
                    "Azure model name is required. Provide the name of the Azure AI Foundry model to use."
                )

            self.llm = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.complete(
                model=self.model,
                max_tokens=max_tokens,
                messages=[UserMessage(prompt + self.extra_prompting)],
                temperature=temperature,
            )
            result = response.choices[0].message.content
            return result

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            response = self.llm.complete(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    SystemMessage(system_prompt),
                    UserMessage(user_prompt + self.extra_prompting),
                ],
                temperature=temperature,
            )
            result = response.choices[0].message.content
            return result

    class AsyncAzureAINamerLegacy(AsyncLLMWrapper):
        """
        Provides access to the Azure AI Foundry LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        For more information on Azure AI, see https://learn.microsoft.com/en-us/azure/ai-services/overview. You will need an Azure API key for your Foundry model
        to use this wrapper. You will need to provide both the endpoint, and the model name per the instiated model on
        AI Foundry. For more information on creating models with Azure AI Foundry, see https://learn.microsoft.com/en-us/azure/ai-services/ai-foundry/create-models.

        This wrapper conforms to the AsyncLLMWrapper interface, and is designed for scenarios where you need to process multiple prompts concurrently.
        This is particularly useful for applications that require high throughput or need to process large volumes of data quickly.
        As an asynchronous wrapper this will potentially speed up topic naming, particularly when you have a large number of topics. If,
        however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls.

        Parameters:
        -----------
        api_key: str
            Your Azure API key. You can set this as an environment variable AZURE_API_KEY or pass it directly

        endpoint: str
            The endpoint URL for your Azure AI Foundry model. This is typically in the format "https://<your-resource-name>.openai.azure.com/".

        model: str
            The name of the Azure AI Foundry model to use. This should match the model name you created in Azure AI Foundry.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Azure AI Foundry API. Default is 10. This can be adjusted based on your
            application's needs and the rate limits of the Azure AI Foundry API. Higher values may improve throughput but could lead to rate limiting.

        Attributes:
        -----------
        client: azure.ai.inference.aio.ChatCompletionsClient
            The Azure AI Foundry asynchronous LLM client instance.

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
            max_concurrent_requests: int = 10,
            callback: DebugCallback | None = None,
        ):
            api_key = api_key or os.getenv("AZURE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure API key is required. Set it as an environment variable AZURE_API_KEY or pass it directly to the constructor."
                )

            if not endpoint:
                raise ValueError(
                    "Azure endpoint is required. Provide the endpoint URL for your Azure AI Foundry model."
                )

            if not model:
                raise ValueError(
                    "Azure model name is required. Provide the name of the Azure AI Foundry model to use."
                )
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.endpoint = endpoint
            self.model = model
            self.client = AsyncChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(
            self, prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt."""
            async with self.semaphore:
                try:
                    response = await self.client.complete(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[UserMessage(prompt + self.extra_prompting)],
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    warn(f"Azure API call failed: {str(e)[:100]}...")
                    return ""

        async def _call_single_llm_with_system(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            async with self.semaphore:
                try:
                    response = await self.client.complete(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[
                            SystemMessage(system_prompt),
                            UserMessage(user_prompt + self.extra_prompting),
                        ],
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    warn(f"Azure API call failed: {str(e)[:100]}...")
                    return ""

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            tasks = [
                self._call_single_llm_with_system(
                    sys_prompt, user_prompt, temperature, max_tokens
                )
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

except ImportError:

    class AzureAINamer(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncAzureAINamer(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    import google.generativeai as genai

    class GoogleGeminiNamerLegacy(LLMWrapper):
        """
        Provides access to Google's Gemini LLMs with the Toponymy framework. For more information on Google Gemini, see
        https://developers.google.com/generative-ai. You will need a Google API key to use this wrapper.
        The default model is "gemini-1.5-flash", which provides a good balance of performance and cost.

        Parameters:
        -----------
        api_key: str
            Your Google API key. You can set this as an environment variable GOOGLE_API_KEY or pass it directly.

        model: str, optional
            The name of the Gemini model to use. Default is "gemini-1.5-flash". Available models include
            "gemini-1.5-pro", "gemini-1.5-flash", etc.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        Attributes:
        -----------
        model: genai.GenerativeModel
            The Gemini model instance.

        model_name: str
            The name of the Gemini model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Gemini, this is always True.
        """

        def __init__(
            self,
            api_key: str,
            model: str = "gemini-1.5-flash",
            llm_specific_instructions=None,
            callback: DebugCallback | None = None,
        ):
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API key is required. Set it as an environment variable GOOGLE_API_KEY or pass it directly to the constructor."
                )

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.model_name = model
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = self.model.generate_content(
                prompt + self.extra_prompting, generation_config=generation_config
            )
            return response.text

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Gemini doesn't have explicit system prompts, so we combine them
            combined_prompt = (
                f"System: {system_prompt}\n\nUser: {user_prompt + self.extra_prompting}"
            )

            response = self.model.generate_content(
                combined_prompt, generation_config=generation_config
            )
            return response.text

    class AsyncGoogleGeminiNamerLegacy(AsyncLLMWrapper):
        """
        Provides access to Google's Gemini LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        For more information on Google Gemini, see https://developers.google.com/generative-ai. You will need a Google API key to use this wrapper.
        The default model is "gemini-1.5-flash", which provides a good balance of performance and cost.

        Parameters:
        -----------
        api_key: str
            Your Google API key. You can set this as an environment variable GOOGLE_API_KEY or pass it directly.

        model: str, optional
            The name of the Gemini model to use. Default is "gemini-1.5-flash". Available models include
            "gemini-1.5-pro", "gemini-1.5-flash", etc.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Gemini API. Default is 10.

        Attributes:
        -----------
        model: genai.GenerativeModel
            The Gemini model instance.

        model_name: str
            The name of the Gemini model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Gemini, this is always True.
        """

        def __init__(
            self,
            api_key: str,
            model: str = "gemini-1.5-flash",
            llm_specific_instructions=None,
            max_concurrent_requests: int = 10,
            callback: DebugCallback | None = None,
        ):
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google API key is required. Set it as an environment variable GOOGLE_API_KEY or pass it directly to the constructor."
                )

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.model_name = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(
            self, prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt."""
            try:
                async with self.semaphore:
                    generation_config = genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )

                    response = await self.model.generate_content_async(
                        prompt + self.extra_prompting,
                        generation_config=generation_config,
                    )
                    return response.text
            except Exception as e:
                warn(f"Google Gemini API call failed: {str(e)[:100]}...")
                return ""

        async def _call_single_llm_with_system(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            try:
                async with self.semaphore:
                    generation_config = genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )

                    # Gemini doesn't have explicit system prompts, so we combine them
                    combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt + self.extra_prompting}"

                    response = await self.model.generate_content_async(
                        combined_prompt, generation_config=generation_config
                    )
                    return response.text
            except Exception as e:
                warn(f"Google Gemini API call failed: {str(e)[:100]}...")
                return ""

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            tasks = [
                self._call_single_llm_with_system(
                    sys_prompt, user_prompt, temperature, max_tokens
                )
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

except ImportError:

    class GoogleGeminiNamer(FailedImportLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncGoogleGeminiNamer(FailedImportAsyncLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    import ollama

    class OllamaNamerLegacy(LLMWrapper):
        """
        Provides access to Olloma's local LLMs with the Toponymy framework. Ollama allows you to run large language models locally.
        For more information on Olloma, see https://ollama.ai/. You'll need to have Olloma installed and running locally.

        Parameters:
        -----------
        model: str
            The name of the Olloma model to use. Default is "llama3.2". You can use any model available
            in Olloma. Popular options include "llama3.2", "mistral", "codellama", etc.

        host: str, optional
            The host URL for the Olloma API. Default is "http://localhost:11434".

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        Attributes:
        -----------
        client: ollama.Client
            The Olloma client instance.

        model: str
            The name of the Olloma model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Olloma, this is always True.
        """

        def __init__(
            self,
            model: str = "llama3.2",
            host: str = "http://localhost:11434",
            llm_specific_instructions=None,
            callback: DebugCallback | None = None,
        ):
            self.client = ollama.Client(host=host)
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.client.generate(
                model=self.model,
                prompt=prompt + self.extra_prompting,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return response["response"]

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return response["message"]["content"]

    class AsyncOllamaNamerLegacy(AsyncLLMWrapper):
        """
        Provides access to Olloma's local LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        Olloma allows you to run large language models locally. For more information on Olloma, see https://ollama.ai/.
        You'll need to have Olloma installed and running locally.

        Parameters:
        -----------
        model: str
            The name of the Olloma model to use. Default is "llama3.2". You can use any model available
            in Olloma. Popular options include "llama3.2", "mistral", "codellama", etc.

        host: str, optional
            The host URL for the Olloma API. Default is "http://localhost:11434".

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Olloma API. Default is 5. Since Olloma runs locally,
            too many concurrent requests might overwhelm the local resources.

        Attributes:
        -----------
        client: ollama.AsyncClient
            The Olloma asynchronous client instance.

        model: str
            The name of the Olloma model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Olloma, this is always True.
        """

        def __init__(
            self,
            model: str = "llama3.2",
            host: str = "http://localhost:11434",
            llm_specific_instructions=None,
            max_concurrent_requests: int = 5,
            callback: DebugCallback | None = None,
        ):
            self.client = ollama.AsyncClient(host=host)
            self.model = model
            self.callback = callback
            self._warn_if_debug_callback_unsupported()
            self.extra_prompting = (
                "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            )
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(
            self, prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt."""
            try:
                async with self.semaphore:
                    response = await self.client.generate(
                        model=self.model,
                        prompt=prompt + self.extra_prompting,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    )
                    return response["response"]
            except Exception as e:
                warn(f"Olloma API call failed: {str(e)[:100]}...")
                return ""

        async def _call_single_llm_with_system(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            try:
                async with self.semaphore:
                    response = await self.client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": user_prompt + self.extra_prompting,
                            },
                        ],
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    )
                    return response["message"]["content"]
            except Exception as e:
                warn(f"Olloma API call failed: {str(e)[:100]}...")
                return ""

        async def _call_llm_batch(
            self, prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self,
            system_prompts: List[str],
            user_prompts: List[str],
            temperature: float,
            max_tokens: int,
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError(
                    "Number of system prompts must match number of user prompts"
                )

            tasks = [
                self._call_single_llm_with_system(
                    sys_prompt, user_prompt, temperature, max_tokens
                )
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

except ImportError:

    class OllamaNamer(FailedImportLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncOllamaNamer(FailedImportAsyncLLMWrapper):
        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)
