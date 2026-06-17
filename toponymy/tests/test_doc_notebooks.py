import pytest
from toponymy.tools.notebook_runner import (
    run_notebook,
    doc_dir,
    get_doc_notebooks_via_sphinx,
)

from pathlib import Path

import logging

print("LOGGERS:")
for name in logging.root.manager.loggerDict:
    print(name)


DOC_NOTEBOOKS = [
    # doc_dir() / "basic_usage.ipynb",
    doc_dir() / "clustering_options.ipynb",
    doc_dir() / "exemplar_texts.ipynb",
]


def safe_load_notebooks(doc_dir: Path) -> list[Path]:
    try:
        return get_doc_notebooks_via_sphinx(doc_dir)
    except Exception as e:
        logging.error(f"Error loading notebooks via Sphinx: {e}")
        return []


# XXX for making this run pick a single notebook
SPHINX_NOTEBOOKS = [safe_load_notebooks(doc_dir())[4]]


def test_sphinx_notebook_loading():
    """
    This tests fails if safe_load_notebooks failed to return any notebooks and
    acts as a flag that test_doc_notebook tests will be skipped because it has no notebooks to test.
    """
    assert (
        len(SPHINX_NOTEBOOKS) > 0
    ), "No doc notebooks found via SPHINX, test_doc_notebook will be skipped"


@pytest.mark.parametrize("notebook", SPHINX_NOTEBOOKS)
def test_doc_notebook(notebook):
    run_notebook(notebook)
