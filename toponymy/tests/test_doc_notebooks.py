import pytest
from pathlib import Path
import logging
import os

from nbformat.v4 import new_notebook, new_code_cell

from conftest import ollama_has_model, ollama_running
from toponymy.tools.notebook_runner import collect_log_lines, run_notebook
from toponymy.tools.notebook_test_helpers import (
    doc_dir,
    get_notebooks,
    get_test_ollama_model,
)

logger = logging.getLogger(__name__)

NOTEBOOK_CONFIG = {
    "basic_usage.ipynb": {
        "has_openainamer": True,
        "run_in_pr": True,
        "timeout": 1000,
    },
    "clusterers.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 3600,
    },
    "clustering_options.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 300,
    },
    "exemplar_texts.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 300,
    },
    "how_toponymy_works.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 3600,
    },
    "keyphrases.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 1000,
    },
    "saving_loading.ipynb": {
        "has_openainamer": True,
        "run_in_pr": True,
        "timeout": 1000,
    },
    "test_audit_functionality.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 600,
    },
    "test_max_layers_newsgroups.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 300,
    },
    "topic_summaries.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 600,
    },
}


TEST_NOTEBOOKS = get_notebooks(doc_dir())


def is_pr_build() -> bool:
    return os.getenv("BUILD_REASON", "").lower() == "pullrequest"


def get_notebook_cfg(path: str, config=None):
    if config is None:
        config = NOTEBOOK_CONFIG
    name = Path(path).name
    return config.get(
        name, {"has_openainamer": True, "run_in_pr": True, "timeout": 6000}
    )


def should_run_in_pr(notebook: str, config=None) -> bool:
    if config is None:
        config = NOTEBOOK_CONFIG
    if is_pr_build():
        raw = os.getenv("CHANGED_NOTEBOOKS", "")
        changed_notebooks = {nb.strip() for nb in raw.split(",") if nb.strip()}
        if Path(notebook).name in changed_notebooks:
            return True
        return get_notebook_cfg(notebook, config)["run_in_pr"]
    return True


def active_notebook_list(
    notebook_list, has_openainamer: bool = False, config=None
) -> list[str]:
    """
    Get the list of active notebooks for testing.

    Parameters
    ----------
    notebook_list : list[str]
        List of notebook names to filter.
    has_openainamer : bool, optional
        If True, return notebooks that require OpenAINamer or have no config; if False, return those that do not.
        Default is False.
    config : dict, optional
        Configuration dict. If None, uses the global NOTEBOOK_CONFIG.
    """
    if config is None:
        config = NOTEBOOK_CONFIG
    if is_pr_build():
        active_list = [nb for nb in notebook_list if should_run_in_pr(nb, config)]
    else:
        active_list = notebook_list
    if has_openainamer:
        active_list = [
            nb
            for nb in active_list
            if get_notebook_cfg(nb, config).get("has_openainamer", True)
        ]
        logger.info(
            f"Active notebooks with OpenAINamer: {[Path(nb).name for nb in active_list]}"
        )
        return active_list
    else:
        active_list = [
            nb
            for nb in active_list
            if not get_notebook_cfg(nb, config).get("has_openainamer", True)
        ]
        logger.info(
            f"Active notebooks without OpenAINamer: {[Path(nb).name for nb in active_list]}"
        )
        return active_list


@pytest.mark.parametrize(
    "notebook", active_notebook_list(TEST_NOTEBOOKS, has_openainamer=False)
)
def test_doc_notebook_no_openainamer(notebook, notebook_testing_env):
    cfg = get_notebook_cfg(notebook)

    run_notebook(
        notebook,
        timeout=cfg["timeout"],
    )


@pytest.mark.parametrize(
    "notebook", active_notebook_list(TEST_NOTEBOOKS, has_openainamer=True)
)
def test_doc_notebook_has_openainamer(notebook, notebook_testing_env, ollama_running):
    cfg = get_notebook_cfg(notebook)

    model = get_test_ollama_model()
    logger.info(f"ollama running:{ollama_running}")
    logger.info(f"ollama_has_model {model}:{ollama_has_model(model)}")
    if not ollama_has_model(model):
        pytest.skip(f"{model} not available in local Ollama for OpenAI mocking")
    run_notebook(
        notebook,
        timeout=cfg["timeout"],
    )


# Tests for active_notebook_list() and its dependencies
@pytest.fixture
def test_config():
    """Fixture providing a test-specific notebook configuration."""
    return {
        "oan_true_pr_true.ipynb": {
            "has_openainamer": True,
            "run_in_pr": True,
        },
        "oan_false_pr_false.ipynb": {
            "has_openainamer": False,
            "run_in_pr": False,
            "timeout": 3600,
        },
        "oan_true_pr_false.ipynb": {
            "has_openainamer": True,
            "run_in_pr": False,
        },
        "oan_false_pr_true.ipynb": {
            "has_openainamer": False,
            "run_in_pr": True,
        },
    }


@pytest.fixture
def notebooks(tmp_path):
    """Provide a reusable, fully-qualified set of fake notebook paths for testing.

    Uses tmp_path so paths look like real filesystem locations, without
    needing actual notebook files to exist on disk for these unit tests.
    """
    names = [
        "oan_true_pr_true.ipynb",
        "oan_true_pr_false.ipynb",
        "oan_false_pr_false.ipynb",
        "oan_false_pr_true.ipynb",
        "unknown_notebook.ipynb",
    ]
    base = tmp_path / "notebooks"
    return [base / name for name in names]


class TestActiveNotebookListLogic:

    def test_get_notebook_cfg_known_notebook(self, test_config):
        cfg = get_notebook_cfg("oan_false_pr_false.ipynb", test_config)
        assert cfg["has_openainamer"] is False
        assert cfg["run_in_pr"] is False
        assert cfg["timeout"] == 3600

    def test_get_notebook_cfg_default_config(self, test_config):
        cfg = get_notebook_cfg("unknown_notebook.ipynb", test_config)
        assert cfg["has_openainamer"] is True
        assert cfg["timeout"] == 6000
        assert cfg["run_in_pr"] is True

    def test_get_notebook_cfg_extracts_filename(self, test_config):
        """Test that get_notebook_cfg extracts filename from full path."""
        cfg = get_notebook_cfg("/path/to/oan_false_pr_false.ipynb", test_config)
        assert cfg["has_openainamer"] is False
        assert cfg["run_in_pr"] is False
        assert cfg["timeout"] == 3600
        cfg2 = get_notebook_cfg("/different/path/oan_false_pr_false.ipynb", test_config)
        assert cfg == cfg2

    def test_should_run_in_pr_for_non_pr_build(self, monkeypatch, test_config):
        monkeypatch.setenv("BUILD_REASON", "Manual")

        assert should_run_in_pr("oan_false_pr_false.ipynb", test_config) is True

    def test_should_run_in_pr_config_based_filtering(self, monkeypatch, test_config):
        """Test should_run_in_pr logic respects run_in_pr config flag in PR builds."""
        monkeypatch.setenv("BUILD_REASON", "PullRequest")
        monkeypatch.delenv("CHANGED_NOTEBOOKS", raising=False)

        assert (
            get_notebook_cfg("oan_false_pr_true.ipynb", test_config)["run_in_pr"]
            is True
        )
        assert (
            get_notebook_cfg("oan_true_pr_false.ipynb", test_config)["run_in_pr"]
            is False
        )

    def test_should_run_in_pr_with_changed_notebooks_override(self, monkeypatch):
        monkeypatch.setenv("BUILD_REASON", "PullRequest")
        monkeypatch.setenv("CHANGED_NOTEBOOKS", "oan_false_pr_false.ipynb")

        assert should_run_in_pr("oan_false_pr_false.ipynb")

    def test_should_run_in_pr_handles_whitespace_in_changed_list(self, monkeypatch):
        monkeypatch.setenv("BUILD_REASON", "PullRequest")
        monkeypatch.setenv(
            "CHANGED_NOTEBOOKS",
            "  oan_true_pr_true.ipynb  ,  oan_false_pr_false.ipynb  ",
        )
        assert should_run_in_pr("oan_true_pr_true.ipynb")
        assert should_run_in_pr("oan_false_pr_false.ipynb")

    def test_active_notebook_list_non_pr_has_openainamer_false(
        self, monkeypatch, test_config, notebooks
    ):
        monkeypatch.setenv("BUILD_REASON", "Manual")

        result = [
            nb.name
            for nb in active_notebook_list(
                notebooks, has_openainamer=False, config=test_config
            )
        ]

        assert "oan_false_pr_false.ipynb" in result
        assert "oan_false_pr_true.ipynb" in result
        assert "oan_true_pr_true.ipynb" not in result
        assert "oan_true_pr_false.ipynb" not in result
        assert "unknown_notebook.ipynb" not in result

    def test_active_notebook_list_non_pr_has_openainamer_true(
        self, monkeypatch, test_config, notebooks
    ):
        monkeypatch.setenv("BUILD_REASON", "Manual")

        result = [
            nb.name
            for nb in active_notebook_list(
                notebooks, has_openainamer=True, config=test_config
            )
        ]

        assert "oan_false_pr_false.ipynb" not in result
        assert "oan_false_pr_true.ipynb" not in result
        assert "oan_true_pr_true.ipynb" in result
        assert "oan_true_pr_false.ipynb" in result
        assert "unknown_notebook.ipynb" in result

    def test_active_notebook_list_pr_build_default_filtering(
        self, monkeypatch, test_config, notebooks
    ):
        """Test that PR builds filter by run_in_pr flag first."""
        monkeypatch.setenv("BUILD_REASON", "PullRequest")
        monkeypatch.delenv("CHANGED_NOTEBOOKS", raising=False)

        result = [
            nb.name
            for nb in active_notebook_list(
                notebooks, has_openainamer=True, config=test_config
            )
        ]

        assert "oan_false_pr_false.ipynb" not in result
        assert "oan_false_pr_true.ipynb" not in result
        assert "oan_true_pr_true.ipynb" in result
        assert "oan_true_pr_false.ipynb" not in result
        assert "unknown_notebook.ipynb" in result

    def test_active_notebook_list_includes_pr_changed_notebooks(
        self, monkeypatch, test_config, notebooks
    ):
        monkeypatch.setenv("BUILD_REASON", "PullRequest")
        monkeypatch.setenv("CHANGED_NOTEBOOKS", "oan_false_pr_false.ipynb")

        result = [
            nb.name
            for nb in active_notebook_list(
                notebooks, has_openainamer=False, config=test_config
            )
        ]

        assert "oan_false_pr_false.ipynb" in result
        assert "oan_false_pr_true.ipynb" in result
        assert "oan_true_pr_true.ipynb" not in result
        assert "oan_true_pr_false.ipynb" not in result
        assert "unknown_notebook.ipynb" not in result

    def test_active_notebook_list_ignores_changed_notebook_not_in_input(
        self, monkeypatch, test_config, notebooks
    ):
        monkeypatch.setenv("BUILD_REASON", "PullRequest")
        monkeypatch.setenv("CHANGED_NOTEBOOKS", "deleted.ipynb")

        result = [
            nb.name
            for nb in active_notebook_list(
                notebooks, has_openainamer=True, config=test_config
            )
        ]

        assert "deleted.ipynb" not in result
        assert result == ["oan_true_pr_true.ipynb", "unknown_notebook.ipynb"]

    def test_active_notebook_list_empty_list(self, monkeypatch, test_config):
        monkeypatch.setenv("BUILD_REASON", "Manual")

        result = active_notebook_list([], has_openainamer=True, config=test_config)
        assert result == []
