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

STATUS_CODES = {
    BadRequestError: 400,
    AuthenticationError: 401,
    PermissionDeniedError: 403,
    NotFoundError: 404,
    RateLimitError: 429,
}


def make_openai_error(error_class):
    message = "test error"
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    status = STATUS_CODES.get(error_class, 500)

    response = httpx.Response(status, request=request)

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