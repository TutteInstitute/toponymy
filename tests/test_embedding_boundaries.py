"""Real SDK transport and response contracts, with no external requests."""

import json
import socket
from types import SimpleNamespace
import numpy as np
import pytest


def test_cohere_retains_sdk_environment_endpoint_default():
    import os
    import subprocess
    import sys

    pytest.importorskip("cohere")
    # The SDK captures CO_API_URL at import, so exercise a fresh interpreter.
    code = """
import httpx
from toponymy.embedding_wrappers import CohereEmbedder
calls = []
def transport(request):
    assert request.url.host == "cohere-env.local.test"
    calls.append(request)
    return httpx.Response(200, json={"id": "fixture", "response_type": "embeddings_by_type",
        "embeddings": {"float": [[1.0, 0.0]]}, "texts": ["one"]})
with httpx.Client(transport=httpx.MockTransport(transport)) as client:
    embedder = CohereEmbedder(api_key="local-placeholder", httpx_client=client)
    with embedder.co:
        assert embedder.encode(["one"]).tolist() == [[1.0, 0.0]]
assert len(calls) == 1
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", code],
        env={**os.environ, "CO_API_URL": "https://cohere-env.local.test"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "CohereEmbedder",
        "OpenAIEmbedder",
        "AzureAIEmbedder",
        "MistralEmbedder",
        "VoyageAIEmbedder",
    ],
)
def test_empty_http_embedding_input_makes_no_request(name):
    from toponymy import embedding_wrappers

    cls = getattr(embedding_wrappers, name, None)
    if cls is None:
        pytest.skip(f"{name} optional dependency is unavailable")
    # An empty call must not even need a client or a configured model.
    result = cls.__new__(cls).encode([])
    assert result.shape == (0, 0)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "name", ["OpenAIEmbedder", "AzureAIEmbedder", "MistralEmbedder"]
)
@pytest.mark.parametrize("count", [2, 97])
def test_sdk_adapters_align_indices_within_each_request_batch(name, count):
    from toponymy import embedding_wrappers

    cls = getattr(embedding_wrappers, name, None)
    if cls is None:
        pytest.skip(f"{name} optional dependency is unavailable")
    sizes = []

    def create(**kwargs):
        assert kwargs["model"] == "local-fixture"
        texts = kwargs.get("input", kwargs.get("inputs"))
        sizes.append(len(texts))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(text), 1.0])
                for index, text in reversed(list(enumerate(texts)))
            ]
        )

    embedder = cls.__new__(cls)
    embedder.model = "local-fixture"
    embedder.client = SimpleNamespace(
        embed=create, embeddings=SimpleNamespace(create=create)
    )
    result = embedder.encode([str(index) for index in range(count)])
    np.testing.assert_array_equal(
        result, np.column_stack((np.arange(count), np.ones(count)))
    )
    assert sizes == ([2] if count == 2 else [96, 1])


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("This regression must use a local mock transport")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    httpx = pytest.importorskip("httpx")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)


def test_cohere_embedder_uses_supplied_endpoint_and_real_sdk_mock_transport():
    pytest.importorskip("cohere")
    import httpx
    from toponymy.embedding_wrappers import CohereEmbedder

    requests = []

    def transport(request):
        requests.append(request)
        assert request.url.host == "cohere.local.test"
        assert json.loads(request.content)["texts"] == ["one", "two"]
        return httpx.Response(
            200,
            json={
                "id": "local-response",
                "response_type": "embeddings_by_type",
                "embeddings": {"float": [[1.0, 0.0], [0.0, 1.0]]},
                "texts": ["one", "two"],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        embedder = CohereEmbedder(
            api_key="local-placeholder",
            base_url="https://cohere.local.test",
            httpx_client=client,
        )
        with embedder.co:
            actual = embedder.encode(["one", "two"])
            np.testing.assert_array_equal(actual, [[1.0, 0.0], [0.0, 1.0]])
            assert len(requests) == 1


@pytest.mark.parametrize(
    "response_items, expected",
    [
        (
            [
                {"index": 1, "embedding": [0.0, 1.0], "object": "embedding"},
                {"index": 0, "embedding": [1.0, 0.0], "object": "embedding"},
            ],
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            [
                {"index": 0, "embedding": [1.0, 0.0], "object": "embedding"},
                {"index": 0, "embedding": [0.0, 1.0], "object": "embedding"},
            ],
            None,
        ),
        ([{"index": 0, "embedding": [1.0, 0.0], "object": "embedding"}], None),
        (
            [
                {"index": 0, "embedding": [1.0, 0.0], "object": "embedding"},
                {"index": 1, "embedding": [0.0], "object": "embedding"},
            ],
            None,
        ),
    ],
    ids=["reordered", "duplicate", "missing", "ragged"],
)
def test_openai_embedding_rows_follow_request_indices_at_sdk_boundary(
    response_items, expected
):
    pytest.importorskip("openai")
    import httpx
    from toponymy.embedding_wrappers import OpenAIEmbedder

    requests = []

    def transport(request):
        requests.append(request)
        assert request.url.host == "openai.local.test"
        assert json.loads(request.content)["input"] == ["one", "two"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": response_items,
                "model": "local-embedding",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        embedder = OpenAIEmbedder(
            api_key="local-placeholder",
            model="local-embedding",
            base_url="https://openai.local.test/v1",
            http_client=client,
        )
        if expected is None:
            with pytest.raises(ValueError):
                embedder.encode(["one", "two"])
        else:
            np.testing.assert_array_equal(embedder.encode(["one", "two"]), expected)
        assert len(requests) == 1


def test_unavailable_anthropic_embedding_api_fails_with_an_explicit_contract():
    """Root should adapt/remove only after choosing the documented API migration."""
    pytest.importorskip("anthropic")
    from toponymy.embedding_wrappers import AnthropicEmbedder

    embedder = None
    try:
        with pytest.raises((ValueError, NotImplementedError), match="(?i)embedding"):
            embedder = AnthropicEmbedder(api_key="local-placeholder")
            embedder.encode(["one"])
    finally:
        if embedder is not None:
            embedder.client.close()


def test_azure_authentication_error_is_not_retried_by_embedding_wrapper(monkeypatch):
    pytest.importorskip("azure.ai.inference")
    from azure.core.exceptions import ClientAuthenticationError
    from toponymy.embedding_wrappers import AzureAIEmbedder

    calls = []

    class Client:
        def embed(self, *, model, input):
            calls.append((model, input))
            raise ClientAuthenticationError("local unauthorized response")

    # Exercise actual wrapper retry policy using the installed SDK exception,
    # with a narrow client method boundary. No MagicMock fabricates endpoints.
    embedder = AzureAIEmbedder.__new__(AzureAIEmbedder)
    embedder.client, embedder.model = Client(), "local-embedding"
    retry = getattr(AzureAIEmbedder._encode_batch, "retry", None)
    if retry is not None:
        monkeypatch.setattr(retry, "sleep", lambda seconds: None)
    with pytest.raises(ClientAuthenticationError):
        embedder.encode(["one"])
    assert len(calls) == 1


def test_azure_malformed_embedding_response_does_not_print_inputs_or_retry(
    monkeypatch, capsys
):
    pytest.importorskip("azure.ai.inference")
    from azure.ai.inference.models import EmbeddingsResult, EmbeddingsUsage
    from toponymy.embedding_wrappers import AzureAIEmbedder

    calls = []

    class Client:
        def embed(self, *, model, input):
            calls.append((model, input))
            return EmbeddingsResult(
                id="local-response",
                data=[],
                model=model,
                usage=EmbeddingsUsage(prompt_tokens=1, total_tokens=1),
            )

    embedder = AzureAIEmbedder.__new__(AzureAIEmbedder)
    embedder.client, embedder.model = Client(), "local-embedding"
    retry = getattr(AzureAIEmbedder._encode_batch, "retry", None)
    if retry is not None:
        monkeypatch.setattr(retry, "sleep", lambda seconds: None)
    with pytest.raises(ValueError):
        embedder.encode(["PRIVATE_SOURCE_MARKER"])
    assert len(calls) == 1
    assert "PRIVATE_SOURCE_MARKER" not in capsys.readouterr().out


def test_voyage_embedding_transport_has_a_finite_timeout(monkeypatch):
    requests = pytest.importorskip("requests")
    from toponymy.embedding_wrappers import VoyageAIEmbedder

    calls = []

    def post(url, *, headers, json, timeout=None):
        calls.append(url)
        values = timeout if isinstance(timeout, tuple) else (timeout,)
        assert all(
            isinstance(value, (int, float)) and 0 < value < np.inf for value in values
        )
        assert json["input"] == ["one"]
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"data": [{"index": 0, "embedding": [1.0, 0.0]}]}'
        response.headers["Content-Type"] = "application/json"
        return response

    monkeypatch.setattr(requests, "post", post)
    embedder = VoyageAIEmbedder(api_key="local-placeholder")
    np.testing.assert_array_equal(embedder.encode(["one"]), [[1.0, 0.0]])
    assert len(calls) == 1
