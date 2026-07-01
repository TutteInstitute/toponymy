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
        "timeout": 800,
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
        "timeout": 1800,
    },
    "saving_loading.ipynb": {
        "has_openainamer": True,
        "run_in_pr": True,
        "timeout": 600,
    },
    "test_audit_functionality.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 600,
    },
    "test_max_layers_newsgroups.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 600,
    },
    "topic_summaries.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 300,
    },
}


def get_notebook_cfg(path: str):
    name = Path(path).name
    return NOTEBOOK_CONFIG.get(name, {"has_openainamer": True, "timeout": 6000})


# XXX for making this run pick a single notebook
# TEST_NOTEBOOKS = [get_notebooks(doc_dir())[4]]
TEST_NOTEBOOKS = get_notebooks(doc_dir())

CI = os.getenv("CI", "").lower() == "true"


def test_show_env():
    import os

    print("PYTEST process NOTEBOOK_TESTING:", os.getenv("NOTEBOOK_TESTING"))
    assert os.getenv("NOTEBOOK_TESTING") is None


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


def test_run_notebook_captures_logger_output(tmp_path):
    path = tmp_path / "logging_capture.ipynb"
    nb = new_notebook(
        cells=[new_code_cell("import logging\nlogging.warning('hello from logger')")]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    assert any(
        level == "warning" and "hello from logger" in line for level, line in lines
    )


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


# @pytest.mark.skipif(CI, reason="Skipping in CI environment")
@pytest.mark.parametrize("notebook", TEST_NOTEBOOKS)
def test_doc_notebook(notebook, notebook_testing_env):
    cfg = get_notebook_cfg(notebook)

    if not cfg.get("run_in_pr") and (os.getenv("BUILD_REASON") == "PullRequest"):
        pytest.skip(f"Skipped in PR CI")
    if cfg.get("has_openainamer", False):
        model = get_test_ollama_model()
        logger.info(f"get_test_ollama_model:{model}")
        logger.info(f"ollama running:{ollama_running()}")
        logger.info(f"ollama_has_model:{ollama_has_model(model)}")
        if not ollama_has_model(model):
            pytest.skip(f"{model} not available in local Ollama for OpenAI mocking")
    logger.info(f'OPENI_API_KEY reset:{os.environ["OPENAI_API_KEY"] == "notarealkey"}')
    logger.info(f'NOTEBOOK_TESTING set:{os.environ["NOTEBOOK_TESTING"] == "true"}')
    run_notebook(
        notebook,
        timeout=3600,  # cfg["timeout"],
    )
