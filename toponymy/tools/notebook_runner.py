try:
    from nbclient import NotebookClient
    import nbformat
    from nbformat.v4 import new_code_cell
except ImportError as e:
    raise ImportError(
        "Notebook runner dependencies are not installed.\n\n"
        "Install with:\n"
        "  pip install 'toponymy[example-notebooks]'\n"
    ) from e

import re
import time
import logging
import os
from .notebook_test_helpers import doc_dir

logger = logging.getLogger(__name__)

LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")


def _inject_logging_capture_cell(nb):
    """Ensure logger output is routed to stdout in the notebook kernel."""
    setup_code = """
import sys
import logging
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger = logging.getLogger()
if not any(
    isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) is sys.stdout
    for h in root_logger.handlers
):
    root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
logging.captureWarnings(True)
"""
    nb.cells.insert(0, new_code_cell(setup_code))


def collect_log_lines(
    executed_nb, ignore_litellm: bool = False
) -> list[tuple[str, str]]:
    """Collect only log-like lines from notebook code-cell outputs.

    Returns a list of (level, text) tuples.
    """
    collected: list[tuple[str, str]] = []

    for i, cell in enumerate(executed_nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = out.get("text") or "".join(out.get("data", {}).get("text/plain", []))
            if not text:
                continue
            for line in str(text).splitlines():
                if ignore_litellm and "LiteLLM" in line:
                    continue
                match = LEVEL_RE.search(line)
                if match:
                    collected.append((match.group(1).lower(), line.rstrip()))

    return collected


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
    return_log_lines: bool = False,
    ignore_litellm: bool = True,
):
    """
    Helper function to run a Jupyter doc notebook. Optionally uses an instrumented client to log execution times.

    If ``return_log_lines`` is True, this returns only the collected log-like lines
    emitted by the notebook during execution.

    If ``ignore_litellm`` is True, notebook output lines containing "LiteLLM"
    are excluded from the returned log lines.
    """
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    _inject_logging_capture_cell(nb)

    client_cls = InstrumentedNotebookClient if instrumented else NotebookClient

    client = client_cls(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(doc_dir())}},
    )

    logger.info("Running %s", path)
    client.execute()
    try:
        executed_nb = client.nb
    except AttributeError:
        executed_nb = nb

    collected = collect_log_lines(executed_nb, ignore_litellm=ignore_litellm)

    logger.info("Collected %s logging lines from notebook %s", len(collected), path)
    for level, line in collected:
        normalized = level.lower()
        log_fn = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical,
        }.get(normalized, logger.info)
        log_fn("Notebook log line: %s", line)

    if return_log_lines:
        return collected

    return executed_nb


NOTEBOOKS = [
    "doc/basic_usage.ipynb",
]


def run_all(
    notebooks=NOTEBOOKS, instrumented: bool = False, ignore_litellm: bool = False
):
    """
    Helper function to run all notebooks in the NOTEBOOKS list, or a custom list of notebooks if provided. Optionally uses an instrumented client to log execution times.
    """
    notebooks = notebooks

    for nb in notebooks:
        run_notebook(nb, instrumented=instrumented, ignore_litellm=ignore_litellm)


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
