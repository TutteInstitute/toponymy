import httpx
from openai import (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
    PermissionDeniedError as AnthropicPermissionDeniedError,
    BadRequestError as AnthropicBadRequestError,
    NotFoundError as AnthropicNotFoundError,
    RateLimitError as AnthropicRateLimitError,
    APITimeoutError as AnthropicAPITimeoutError,
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError as AnthropicAPIStatusError,
)

from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    PermissionDeniedError as LiteLLMPermissionDeniedError,
    BadRequestError as LiteLLMBadRequestError,
    NotFoundError as LiteLLMNotFoundError,
    UnprocessableEntityError as LiteLLMUnprocessableEntityError,
    RateLimitError as LiteLLMRateLimitError,
    APIConnectionError as LiteLLMAPIConnectionError,
    Timeout as LiteLLMTimeout,
    InternalServerError as LiteLLMInternalServerError,
    ServiceUnavailableError as LiteLLMServiceUnavailableError,
)

TEST_ERROR_MESSAGE = "test error"

## Open AI errors
OPENAI_FAIL_FAST = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
)

OPENAI_RETRYABLE = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

OPENAI_STATUS_CODES = {
    BadRequestError: 400,
    AuthenticationError: 401,
    PermissionDeniedError: 403,
    NotFoundError: 404,
    RateLimitError: 429,
}


def make_httpx_request(url: str) -> httpx.Request:
    return httpx.Request("POST", url)


def make_httpx_response(
    status_code: int,
    request: httpx.Request,
) -> httpx.Response:
    return httpx.Response(status_code, request=request)


def make_openai_error(error_class):
    message = TEST_ERROR_MESSAGE
    request = make_httpx_request("https://api.openai.com/v1/chat/completions")
    status = OPENAI_STATUS_CODES.get(error_class, 500)
    response = make_httpx_response(status, request)

    body = {
        "error": {
            "message": message,
            "type": "test_error",
            "code": "test",
        }
    }

    if error_class in (
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError,
        NotFoundError,
        RateLimitError,
    ):
        return error_class(message=message, response=response, body=body)

    elif error_class is APITimeoutError:
        return error_class(request=request)

    elif error_class is APIConnectionError:
        return error_class(message=message, request=request)

    elif error_class is APIError:
        return error_class(message=message, request=request, body=body)

    else:
        raise ValueError(f"Unknown error class: {error_class}")


## Anthropic errors
ANTHROPIC_FAIL_FAST = (
    AnthropicAuthenticationError,
    AnthropicPermissionDeniedError,
    AnthropicBadRequestError,
    AnthropicNotFoundError,
)

ANTHROPIC_RETRYABLE = (
    AnthropicRateLimitError,
    AnthropicAPITimeoutError,
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
)

ANTHROPIC_STATUS_CODES = {
    AnthropicBadRequestError: 400,
    AnthropicAuthenticationError: 401,
    AnthropicPermissionDeniedError: 403,
    AnthropicNotFoundError: 404,
    AnthropicRateLimitError: 429,
}


def make_anthropic_error(error_class):
    message = TEST_ERROR_MESSAGE
    request = make_httpx_request("https://api.anthropic.com/v1/messages")
    status = ANTHROPIC_STATUS_CODES.get(error_class, 500)
    response = make_httpx_response(status, request)

    body = {
        "type": "error",
        "error": {
            "type": "test_error",
            "message": message,
        },
    }

    if error_class in (
        AnthropicAuthenticationError,
        AnthropicPermissionDeniedError,
        AnthropicBadRequestError,
        AnthropicNotFoundError,
        AnthropicRateLimitError,
    ):
        return error_class(message=message, response=response, body=body)

    if error_class is AnthropicAPITimeoutError:
        return error_class(request=request)

    if error_class is AnthropicAPIConnectionError:
        return error_class(message=message, request=request)

    if error_class is AnthropicAPIStatusError:
        return error_class(message=message, response=response, body=body)

    raise ValueError(f"Unknown error class: {error_class}")


# LiteLLM Errors

LITELLM_FAIL_FAST = [
    LiteLLMAuthenticationError,
    LiteLLMPermissionDeniedError,
    LiteLLMBadRequestError,
    LiteLLMNotFoundError,
    LiteLLMUnprocessableEntityError,
]

LITELLM_RETRYABLE = [
    LiteLLMRateLimitError,
    LiteLLMAPIConnectionError,
    LiteLLMTimeout,
    LiteLLMInternalServerError,
    LiteLLMServiceUnavailableError,
]

LITELLM_STATUS_CODES = {
    LiteLLMBadRequestError: 400,
    LiteLLMAuthenticationError: 401,
    LiteLLMPermissionDeniedError: 403,
    LiteLLMNotFoundError: 404,
    LiteLLMUnprocessableEntityError: 422,
    LiteLLMRateLimitError: 429,
    LiteLLMInternalServerError: 500,
    LiteLLMServiceUnavailableError: 503,
}


def make_litellm_error(error_class, provider_name="openai", model="gpt-4"):
    message = TEST_ERROR_MESSAGE
    status = LITELLM_STATUS_CODES.get(error_class, 500)
    request = make_httpx_request("https://api.openai.com/v1/chat/completions")
    status = LITELLM_STATUS_CODES.get(error_class, 500)
    response = make_httpx_response(status, request)

    if error_class in (
        LiteLLMAuthenticationError,
        LiteLLMBadRequestError,
        LiteLLMNotFoundError,
        LiteLLMRateLimitError,
        LiteLLMAPIConnectionError,
        LiteLLMTimeout,
        LiteLLMInternalServerError,
        LiteLLMServiceUnavailableError,
    ):
        return error_class(
            message=message,
            llm_provider=provider_name,
            model=model,
        )
    if error_class in (
        LiteLLMPermissionDeniedError,
        LiteLLMUnprocessableEntityError,
    ):
        return error_class(
            message=message,
            llm_provider=provider_name,
            model=model,
            response=response,
        )
    if error_class is LiteLLMAPIConnectionError:
        return error_class(
            message=message,
            llm_provider=provider_name,
            model=model,
            request=request,
        )

    raise ValueError(f"Unknown error class: {error_class}")
