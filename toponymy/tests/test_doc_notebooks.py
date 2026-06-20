import pytest
from toponymy.tools.notebook_runner import (
    run_notebook,
    doc_dir,
    get_notebooks,
)

from pathlib import Path

import logging

print("LOGGERS:")
for name in logging.root.manager.loggerDict:
    print(name)


NOTEBOOK_CONFIG = {
    "basic_usage.ipynb": {
        "has_openainamer": True,
        "timeout": 800,
    },
    "clusterers.ipynb": {
        "has_openainamer": False,
        "timeout": 3600,
    },
    "clustering_options.ipynb": {
        "has_openainamer": False,
        "timeout": 300,
    },
    "exemplar_texts.ipynb": {
        "has_openainamer": False,
        "timeout": 300,
    },
    "how_toponymy_works.ipynb": {
        "has_openainamer": True,
        "timeout": 3600,
    },
    "keyphrases.ipynb": {
        "has_openainamer": False,
        "timeout": 1800,
    },
    "saving_loading.ipynb": {
        "has_openainamer": True,
        "timeout": 600,
    },
    "test_audit_functionality.ipynb": {
        "has_openainamer": True,
        "timeout": 600,
    },
    "test_max_layers_newsgroups.ipynb": {
        "has_openainamer": True,
        "timeout": 600,
    },
    "topic_summaries.ipynb": {
        "has_openainamer": True,
        "timeout": 6000,
    },
}


def get_notebook_cfg(path: str):
    name = Path(path).name
    return NOTEBOOK_CONFIG.get(name, {"has_openainamer": True, "timeout": 6000})


# XXX for making this run pick a single notebook
TEST_NOTEBOOKS = [get_notebooks(doc_dir)[10]]
# TEST_NOTEBOOKS = get_notebooks(doc_dir)


@pytest.mark.parametrize("notebook", TEST_NOTEBOOKS)
def test_doc_notebook(notebook, notebook_testing_env):
    cfg = get_notebook_cfg(notebook)

    run_notebook(
        notebook,
        timeout=cfg["timeout"],
    )
