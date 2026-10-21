import asyncio
import os
import re

import pytest
from unittest.mock import AsyncMock, MagicMock

from toponymy.llm_wrappers import (
    AsyncOllamaNamer,
    OllamaNamer,
    _SyncBridge,
    AsyncLLMWrapper,
)

from toponymy.tests.helpers.llm_test_config import (
    EXTERNAL_SUPPORTED_NAMERS,
)
from toponymy.tests.helpers.dummy_wrappers import (
    DummySingleWrapper,
    DummyAsyncSingleWrapper,
)

ACCEPTED_PATTERNS = {
    "hockey_masks": [
        r"\bgoalie\b",
        r"\bgoaltend",
        r"\bmask",
    ]  # Note: do not match on hockey since it's too broad
}


def matches_expected_patterns(result, patterns):
    """Check if the result matches any of the expected patterns"""
    text = " ".join(result) if isinstance(result, tuple) else str(result)
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return any(re.search(p, normalized) for p in patterns)


def assert_valid_topic_name_result(result, example_id):
    assert result["success"] is True, result
    assert isinstance(result["topic_result"], str), result
    assert len(result["topic_result"]) > 0, result
    assert matches_expected_patterns(
        result["topic_result"], ACCEPTED_PATTERNS[example_id]
    ), result
    assert result["prompt_kind"] == "topic_name", result


class TestGoldenPrompts:
    """
    Test against real providers
    """

    @pytest.mark.parametrize(
        "ollama_namer_class",
        [OllamaNamer, AsyncOllamaNamer],
        ids=lambda cls: cls.__name__,
    )
    def test_ollama_smoke_test(self, ollama_namer_class, ollama_test_model):
        namer = ollama_namer_class(model=ollama_test_model, temperature_override=0.0)
        status = namer.connectivity_status()
        assert status["success"] is True, status

        example_id = "hockey_masks"
        result = namer.run_prompt_example(
            example_id=example_id,
            verbose=True,
        )

        assert_valid_topic_name_result(result, example_id)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_ollama_async_namer_mixed_call_styles(self, ollama_test_model):
        """
        Test separate clients for bridge loop and user loop
        """
        namer = AsyncOllamaNamer(model=ollama_test_model, temperature_override=0.0)
        example_id = "hockey_masks"
        try:
            sync_result = namer.run_prompt_example(example_id=example_id, verbose=True)
            async_result = await namer.arun_prompt_example(
                example_id=example_id, verbose=True
            )
        finally:
            await namer.close()

        assert_valid_topic_name_result(sync_result, example_id)
        assert_valid_topic_name_result(async_result, example_id)

    @pytest.mark.external
    @pytest.mark.parametrize(
        "namer_class, api_key_name, namer_kwargs",
        EXTERNAL_SUPPORTED_NAMERS,
        ids=lambda p: p.__name__ if isinstance(p, type) else None,
    )
    def test_supported_models_smoke_test(self, namer_class, api_key_name, namer_kwargs):
        if os.getenv(api_key_name) is None:
            pytest.skip(f"{api_key_name} is not set in the environment")

        namer = namer_class(**namer_kwargs)
        status = namer.connectivity_status()
        assert status["success"] is True

        example_id = "hockey_masks"

        result = namer.run_prompt_example(
            example_id=example_id,
            verbose=True,
        )

        assert_valid_topic_name_result(result, example_id)


ASYNC_METHOD_NAMES = {
    "run_prompt_example": "arun_prompt_example",
    "connectivity_status": "aconnectivity_status",
}


class TestRunPromptExampleAPI:
    """
    Test suite for the run_prompt_example API of various namer wrappers.
    Ensures consistent behavior across sync and async implementations.

    The parametrized dummy namers and call styles cover the four possible combinations of
    sync/async wrappers and sync/async call styles.
    """

    @pytest.fixture(params=["sync", "async"], ids=["sync_call", "async_call"])
    def call_style(self, request):
        def call(namer, method, **kwargs):
            if request.param == "sync":
                return getattr(namer, method)(**kwargs)
            async_method = getattr(namer, ASYNC_METHOD_NAMES[method])
            return asyncio.run(async_method(**kwargs))

        return call

    def test_run_prompt_example_invalid_example_id_raises(
        self, dummy_namer, call_style
    ):
        with pytest.raises(ValueError, match="example_id must be in"):
            call_style(
                dummy_namer, "run_prompt_example", example_id="not_a_real_example"
            )

    def test_run_prompt_example_invalid_prompt_kind_raises(
        self, dummy_namer, call_style
    ):
        with pytest.raises(ValueError, match="prompt_kind must be"):
            call_style(
                dummy_namer,
                "run_prompt_example",
                example_id="hockey_masks",
                prompt_kind="bad_kind",
            )

    @pytest.mark.filterwarnings("ignore:Connectivity check failed.")
    def test_run_prompt_example_connectivity_failure_returns_error(
        self, dummy_namer, call_style, monkeypatch
    ):
        failure = {
            "success": False,
            "error_type": "RuntimeError",
            "error_message": "boom",
            "original_exception": RuntimeError("boom"),
        }
        if isinstance(dummy_namer, AsyncLLMWrapper):
            attr, mock_cls = "aconnectivity_status", AsyncMock
        else:
            attr, mock_cls = "connectivity_status", MagicMock
        mock = mock_cls(return_value=failure)
        monkeypatch.setattr(dummy_namer, attr, mock)

        result = call_style(
            dummy_namer, "run_prompt_example", example_id="hockey_masks"
        )

        assert result["success"] is False, result
        assert result["error_type"] == "RuntimeError", result
        assert result["error_message"] == "boom", result
        assert isinstance(result["original_exception"], RuntimeError), result
        mock.assert_called_once()

    @pytest.mark.parametrize("verbose", [True, False], ids=["verbose", "non_verbose"])
    def test_sync_and_async_run_prompt_example_return_same_result_verbose(
        self, dummy_namer, verbose
    ):
        sync_result = dummy_namer.run_prompt_example(
            example_id="hockey_masks", verbose=verbose
        )
        async_result = asyncio.run(
            dummy_namer.arun_prompt_example(example_id="hockey_masks", verbose=verbose)
        )
        assert sync_result == async_result

    def test_sync_run_prompt_example_on_async_wrapper_propagates_exceptions(self):
        namer = DummyAsyncSingleWrapper()
        with pytest.raises(ValueError, match="prompt_kind must be"):
            namer.run_prompt_example(example_id="hockey_masks", prompt_kind="bad_kind")

    def test_async_run_prompt_example_on_sync_wrapper_propagates_exceptions(self):
        namer = DummySingleWrapper()
        with pytest.raises(ValueError, match="prompt_kind must be"):
            asyncio.run(
                namer.arun_prompt_example(
                    example_id="hockey_masks", prompt_kind="bad_kind"
                )
            )

    def test_run_prompt_example_non_verbose_omits_prompt_and_response(
        self, dummy_namer, call_style
    ):
        result = call_style(
            dummy_namer, "run_prompt_example", example_id="hockey_masks", verbose=False
        )

        assert result["success"] is True, result
        assert result["prompt"] is None, result
        assert result["response"] is None, result

    def test_run_prompt_example_verbose_includes_prompt_and_response(
        self, dummy_namer, call_style
    ):
        result = call_style(
            dummy_namer, "run_prompt_example", example_id="hockey_masks", verbose=True
        )
        assert result["success"] is True, result
        assert isinstance(result["prompt"], dict), result
        assert {"system", "user", "combined"} <= result["prompt"].keys(), result
        assert isinstance(result["response"], str) and result["response"], result


class TestSyncBridge:
    @pytest.mark.asyncio
    async def test_sync_run_prompt_example_works_inside_running_loop(self):
        """
        Tests that the synchronous run_prompt_example method works correctly
        when called inside an already running asyncio event loop.
        This simulates the behavior in environments like Jupyter notebooks.
        """
        asyncio.get_running_loop()  # Ensures we are inside a running event loop
        namer = DummyAsyncSingleWrapper()
        result = namer.run_prompt_example(example_id="hockey_masks")
        assert result["success"]

    def test_sync_method_called_from_bridge_loop_raises(self):
        namer = DummyAsyncSingleWrapper()

        async def call_sync_from_inside_bridge():
            return namer.connectivity_status()

        with pytest.raises(RuntimeError, match="sync bridge loop"):
            _SyncBridge.run(call_sync_from_inside_bridge())

    def test_sync_bridge_reuses_one_loop(self):
        loops = []

        async def record_loop():
            loops.append(asyncio.get_running_loop())

        _SyncBridge.run(record_loop())
        _SyncBridge.run(record_loop())
        assert loops[0] is loops[1]
