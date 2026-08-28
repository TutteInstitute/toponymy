import os
from pathlib import Path, PurePosixPath

import httpx
import nbformat
import pytest
import socket
from huggingface_hub import hf_hub_url

from nbformat.v4 import new_notebook, new_code_cell

from toponymy.tools.notebook_test_helpers import (
    get_test_ollama_model,
    notebook_test_replacement,
)
from toponymy.tools.notebook_data_load import (
    load_small_newsgroups,
    load_newsgroups,
    load_small_bundled_arxiv,
    load_bundled_arxiv,
    load_arxiv_ct,
    load_arxiv_ml,
    notebook_output_dir,
    HF_URL_NEWSGROUPS,
    HF_URL_ARXIV_CT,
    HF_URL_ARXIV_ML,
)
from toponymy.tools.notebook_runner import (
    _inject_logging_capture_cell,
    run_notebook,
    collect_log_lines,
)
from toponymy.llm_wrappers import OpenAINamer, OllamaNamer, LiteLLMNamer


def _deny_network(*args, **kwargs):
    raise AssertionError("Unexpected network call in HF fallback test")


def test_notebook_test_replacement_decorator(notebook_testing_env, monkeypatch):
    """Decorator should replace the wrapped function when NOTEBOOK_TESTING=true and pass through otherwise."""

    def replacement(a, b=2):
        return f"replaced:{a}:{b}"

    @notebook_test_replacement(replacement)
    def original(a, b=2):
        return f"original:{a}:{b}"

    assert original(1, b=3) == "replaced:1:3"

    monkeypatch.setenv("NOTEBOOK_TESTING", "false")
    assert original(1, b=3) == "original:1:3"


# Test local data loading functions to ensure they return expected sizes and shapes, and that the decorator replacement works as intended.
# Avoid testing the datasets that require network access
def test_load_small_newsgroups_size(monkeypatch):
    """Ensure the bundled small newsgroups dataset loads and has expected size."""

    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)

    df = load_small_newsgroups()
    assert hasattr(df, "shape")
    assert df.shape[0] == 150


def test_load_newsgroups_decorator_behaviour(notebook_testing_env, monkeypatch):
    """When NOTEBOOK_TESTING is set, load_newsgroups should use the small local parquet, not the hf:// URL."""

    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)

    df = load_newsgroups()
    assert len(df) == 150


def test_load_bundled_arxiv_small_size_and_shapes(monkeypatch):
    """Verify small bundled arXiv loader returns expected lengths and array shapes."""
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    docs, doc_vectors, cluster_vectors = load_small_bundled_arxiv()
    assert len(docs) == 350
    assert getattr(doc_vectors, "shape", (0,))[0] == 350
    assert getattr(cluster_vectors, "shape", (0,))[0] == 350


def test_load_bundled_arxiv_decorator_behaviour(notebook_testing_env, monkeypatch):
    """When NOTEBOOK_TESTING is set, load_bundled_arxiv should be replaced with the small loader unless override is specified."""
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)

    docs, _, _ = load_bundled_arxiv()
    assert len(docs) == 350
    docs, _, _ = load_bundled_arxiv(use_small=False)
    assert len(docs) == 10000


class _FakeDF:
    def __init__(self):
        self.calls = []
        self.sample_calls = []

    def sample(self, n, random_state):
        self.sample_calls.append((n, random_state))
        return self

    def reset_index(self, drop=True):
        return self


def _fake_read_parquet(path, *args, **kwargs):
    df = _FakeDF()
    df.calls.append(path)
    return df


@pytest.mark.parametrize(
    "loader, expected_url, expected_n, expected_random_state",
    [
        (load_arxiv_ct, HF_URL_ARXIV_CT, 3000, 99),
        (load_arxiv_ml, HF_URL_ARXIV_ML, 5000, 2),
    ],
)
def test_hf_loaders_fallback_small_path_without_network(
    notebook_testing_env,
    monkeypatch,
    loader,
    expected_url,
    expected_n,
    expected_random_state,
):
    """Under NOTEBOOK_TESTING, loader() should take the small-path sampling branch without network calls."""

    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(
        "toponymy.tools.notebook_data_load.pd.read_parquet", _fake_read_parquet
    )

    df = loader()

    assert df.calls == [expected_url]
    assert df.sample_calls == [(expected_n, expected_random_state)]


@pytest.mark.parametrize(
    "loader, expected_url",
    [
        (load_arxiv_ct, HF_URL_ARXIV_CT),
        (load_arxiv_ml, HF_URL_ARXIV_ML),
        (load_newsgroups, HF_URL_NEWSGROUPS),
    ],
)
def test_hf_loaders_use_small_false_uses_full_branch_without_network(
    notebook_testing_env, monkeypatch, loader, expected_url
):
    """Under NOTEBOOK_TESTING, loader(use_small=False) should bypass sampling and take the full branch without network calls."""

    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(
        "toponymy.tools.notebook_data_load.pd.read_parquet", _fake_read_parquet
    )

    df = loader(use_small=False)

    assert df.calls == [expected_url]
    assert df.sample_calls == []


@pytest.mark.parametrize(
    "loader, expected_url",
    [
        (load_arxiv_ct, HF_URL_ARXIV_CT),
        (load_arxiv_ml, HF_URL_ARXIV_ML),
        (load_newsgroups, HF_URL_NEWSGROUPS),
    ],
)
def test_hf_loaders_no_fallback_without_network(monkeypatch, loader, expected_url):
    """When NOTEBOOK_TESTING is not set, loader() should take the full dataset with no sampling."""
    monkeypatch.delenv("NOTEBOOK_TESTING", raising=False)
    monkeypatch.setattr(socket, "create_connection", _deny_network)
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    monkeypatch.setattr(
        "toponymy.tools.notebook_data_load.pd.read_parquet", _fake_read_parquet
    )

    df = loader()

    assert df.calls == [expected_url]
    assert df.sample_calls == []


@pytest.mark.external
@pytest.mark.parametrize(
    "hf_url",
    [HF_URL_NEWSGROUPS, HF_URL_ARXIV_CT, HF_URL_ARXIV_ML],
)
def test_hf_dataset_urls_are_reachable(hf_url):
    """Check that HF dataset URLs used in notebook_data_load are still reachable (HEAD request, no download)."""

    # hf://datasets/<owner>/<repo>/<path/to/file> -> strip scheme, parse with PurePosixPath
    _, owner, repo, *file_parts = PurePosixPath(hf_url.removeprefix("hf://")).parts
    url = hf_hub_url(
        repo_id=f"{owner}/{repo}",
        filename=str(PurePosixPath(*file_parts)),
        repo_type="dataset",
    )
    response = httpx.head(url, follow_redirects=True, timeout=10.0)
    assert response.status_code < 400, f"HF URL {url} returned {response.status_code}"


def test_inject_logging_cell_and_run(tmp_path):
    """Inject logging capture cell, write a simple notebook that logs, and run it to collect logs."""
    nb = new_notebook(
        cells=[new_code_cell("import logging\nlogging.warning('from test')")]
    )
    _inject_logging_capture_cell(nb)
    assert nb.cells[0].source.strip().startswith("import sys")

    path = tmp_path / "logging_capture.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    assert any("from test" in text for level, text in lines)


def test_collect_log_lines_from_stream_output():
    nb = new_notebook(cells=[new_code_cell("print('WARNING: hello from notebook')")])
    nb.cells[0].outputs = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": "WARNING: hello from notebook\n",
        }
    ]

    assert collect_log_lines(nb) == [("warning", "WARNING: hello from notebook")]


def test_run_notebook_captures_logger_output_on_success(tmp_path):
    path = tmp_path / "logging_capture.ipynb"
    nb = new_notebook(
        cells=[
            new_code_cell(
                "import logging\nlogging.warning('hello from logger')\nlogging.info('info from logger')"
            )
        ]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    assert any(
        level == "warning" and "hello from logger" in line for level, line in lines
    )
    assert any(level == "info" and "info from logger" in line for level, line in lines)


def test_run_notebook_ignores_litellm_output(tmp_path):
    path = tmp_path / "litellm_ignore.ipynb"
    nb = new_notebook(
        cells=[
            new_code_cell(
                "import logging\nlogging.warning('LiteLLM:WARNING: silly warning from provider')"
            )
        ]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(
        str(path),
        timeout=30,
        return_log_lines=True,
        ignore_litellm=True,
    )
    assert not lines


def test_run_notebook_captures_logs_on_failure(tmp_path):
    """Regression test: logs should be captured even when notebook fails to complete."""
    path = tmp_path / "failing_notebook.ipynb"
    nb = new_notebook(
        cells=[
            new_code_cell(
                "import logging\nlogging.warning('warning before failure')\nlogging.info('info before failure')"
            ),
            new_code_cell("raise ValueError('notebook execution failed')"),
        ]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    # Verify that logs from before the failure are still captured
    assert any(
        level == "warning" and "warning before failure" in line for level, line in lines
    )
    assert any(
        level == "info" and "info before failure" in line for level, line in lines
    )


def test_openainamer_fallback_to_notebook_mock(notebook_testing_env, monkeypatch):
    """When NOTEBOOK_TESTING=true, OpenAINamer should fallback to NotebookOpenAINamerMock which returns OllamaNamer."""
    # test fallback case
    namer = OpenAINamer()
    assert isinstance(namer, LiteLLMNamer)
    # Verify it uses the Ollama model (from the mock)
    assert get_test_ollama_model() in namer.model.lower()

    # Test the non-fallback case
    monkeypatch.setenv("NOTEBOOK_TESTING", "false")
    namer = OpenAINamer()
    # Without the env var, should use openai model
    assert isinstance(namer, LiteLLMNamer)
    assert "openai" in namer.model.lower()


def test_notebook_testing_env_fixture_sets_vars(notebook_testing_env):
    """Verify notebook_testing_env fixture sets the required environment variables."""
    assert os.environ.get("NOTEBOOK_TESTING") == "true"
    assert os.environ.get("OPENAI_API_KEY") == "notarealkey"
    assert "NB_TEST_OUTPUT_DIR" in os.environ


def test_notebook_output_dir_uses_pytest_fixture_env_var(notebook_testing_env):
    """When NB_TEST_OUTPUT_DIR is set by pytest fixture, notebook_output_dir should return that directory."""
    nb_output_dir = notebook_output_dir()

    assert isinstance(nb_output_dir, Path)
    assert nb_output_dir.exists()
    assert str(nb_output_dir) == os.environ.get("NB_TEST_OUTPUT_DIR")
    assert nb_output_dir != Path().resolve()


def test_notebook_output_dir_manual_fallback(monkeypatch):
    """When NOTEBOOK_TESTING=true but NB_TEST_OUTPUT_DIR is unset, fallback creates a temp directory."""
    monkeypatch.setenv("NOTEBOOK_TESTING", "true")
    monkeypatch.delenv("NB_TEST_OUTPUT_DIR", raising=False)

    nb_output_dir = notebook_output_dir()

    assert isinstance(nb_output_dir, Path)
    assert nb_output_dir.exists()
    assert nb_output_dir != Path().resolve()


def test_notebook_output_dir_normal_mode(monkeypatch):
    """When NOTEBOOK_TESTING is false, notebook_output_dir returns the current working directory."""
    monkeypatch.setenv("NOTEBOOK_TESTING", "false")

    nb_output_dir = notebook_output_dir()

    assert isinstance(nb_output_dir, Path)
    assert nb_output_dir == Path().resolve()
