import openai

from .llm_wrappers import AsyncLLMWrapper, LLMWrapper, DebugCallback
from openai import (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
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
