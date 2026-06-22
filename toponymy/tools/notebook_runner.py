from nbclient import NotebookClient
import nbformat
import time
import logging
import warnings
import os
import functools

logger = logging.getLogger(__name__)

from pathlib import Path

import os
import warnings


def notebook_test_replacement(replacement):
    """
    Decorator for mocking function replacements for example notebook testing.
    """

    def decorator(func):
        if os.getenv("NOTEBOOK_TESTING", "").lower() == "true":

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return replacement(*args, **kwargs)

            return wrapper
        return func

    return decorator


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def doc_dir() -> Path:
    return PACKAGE_ROOT / "doc"


def examples_dir() -> Path:
    return PACKAGE_ROOT / "examples"


def get_notebooks(doc_dir: Path) -> list[Path]:
    """
    Get a list of all ipynb notebooks in the specified directory.
    """
    return sorted(Path(doc_dir).glob("*.ipynb"))


class InstrumentedNotebookClient(NotebookClient):
    """
    A NotebookClient that logs the start and end of each cell execution, as well as the total execution time of the notebook.
    """

    def on_cell_start(self, cell, cell_index, **kwargs):
        if not hasattr(self, "_notebook_start_time"):
            self._notebook_start_time = time.time()

        self._cell_start_time = time.time()
        logger.info("START cell %s", cell_index)

    def on_cell_executed(self, cell, cell_index, **kwargs):
        duration = time.time() - self._cell_start_time
        logger.info("END cell %s (%.2fs)", cell_index, duration)

    def on_notebook_complete(self, **kwargs):
        total = time.time() - self._notebook_start_time
        logger.info("NOTEBOOK DONE total time: %.2fs", total)


def run_notebook(
    path: str,
    timeout: int = 3000,
    kernel_name: str = "toponymy-uv",
    instrumented: bool = False,
):
    """
    Helper function to run a Jupyter notebook. Optionally uses an instrumented client to log execution times.
    """
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    client_cls = InstrumentedNotebookClient if instrumented else NotebookClient

    client = client_cls(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(doc_dir())}},
    )

    logger.info("Running %s", path)
    result = client.execute()

    return result


NOTEBOOKS = [
    "doc/basic_usage.ipynb",
]


def run_all(notebooks=NOTEBOOKS, instrumented: bool = False):
    """
    Helper function to run all notebooks in the NOTEBOOKS list, or a custom list of notebooks if provided. Optionally uses an instrumented client to log execution times.
    """
    notebooks = notebooks

    for nb in notebooks:
        run_notebook(nb, instrumented=instrumented)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Notebook(s) to run. If omitted, runs all preset NOTEBOOKS.",
    )
    parser.add_argument("--instrument", action="store_true")
    args = parser.parse_args()

    run_all(
        notebooks=args.notebooks or None,
        instrumented=args.instrument,
    )
