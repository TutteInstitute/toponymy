import pytest
from pathlib import Path
import logging
import os

from conftest import ollama_has_model
from toponymy.tools.notebook_runner import run_notebook
from toponymy.tools.notebook_test_helpers import (
    doc_dir,
    get_notebooks,
    get_test_ollama_model,
)

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


# @pytest.mark.skipif(CI, reason="Skipping in CI environment")
@pytest.mark.parametrize("notebook", TEST_NOTEBOOKS)
def test_doc_notebook(notebook, notebook_testing_env):
    cfg = get_notebook_cfg(notebook)
    logging.info(notebook)
    if not cfg.get("run_in_pr") and (os.getenv("BUILD_REASON") == "PullRequest"):
        pytest.skip(f"Skipped in PR CI")
    if cfg.get("has_openainamer", False):
        model = get_test_ollama_model()
        if not ollama_has_model(model):
            pytest.skip(f"{model} not available in local Ollama for OpenAI mocking")
    run_notebook(
        notebook,
        timeout=3600,  # cfg["timeout"],
    )
