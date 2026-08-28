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
from pathlib import Path
from toponymy.tools.notebook_test_helpers import doc_dir

logger = logging.getLogger(__name__)

LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")


def _inject_logging_capture_cell(nb: nbformat.NotebookNode) -> None:
    """
    Inject a logging capture cell into the notebook to ensure logger output is routed to stdout.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        The notebook object to inject the logging setup cell into.

    Returns
    -------
    None
    """
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
    executed_nb: nbformat.NotebookNode, ignore_litellm: bool = False
) -> list[tuple[str, str]]:
    """
    Collect only log-like lines from notebook code-cell outputs.

    Scans all code cell outputs for lines matching logging level patterns (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    and returns them as (level, text) tuples.

    Parameters
    ----------
    executed_nb : nbformat.NotebookNode
        The executed notebook object with cell outputs populated.
    ignore_litellm : bool, optional
        If True, exclude lines containing the string "LiteLLM" from the collected output. Default is False.

    Returns
    -------
    list[tuple[str, str]]
        A list of (log_level, line_text) tuples where log_level is the lowercase level name.
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
) -> nbformat.NotebookNode | list[tuple[str, str]]:
    """
    Execute a Jupyter notebook with optional logging instrumentation and log-line collection.

    Runs the notebook in a kernel, injects a logging capture cell to ensure stdlib logging
    is routed to stdout, and optionally collects log-like lines from cell outputs. On exception,
    partial notebook outputs and collected logs are re-emitted before re-raising.

    Parameters
    ----------
    path : str
        Path to the .ipynb notebook file to execute.
    timeout : int, optional
        Timeout in seconds for notebook execution. Default is 3000.
    kernel_name : str, optional
        Name of the Jupyter kernel to use. Default is "toponymy-uv".
    instrumented : bool, optional
        If True, use InstrumentedNotebookClient to log cell start/end times and total runtime. Default is False.
    return_log_lines : bool, optional
        If True, return only the collected log-like lines instead of the executed notebook object. Default is False.
    ignore_litellm : bool, optional
        If True, exclude log lines containing "LiteLLM" from collection. Default is True.

    Returns
    -------
    nbformat.NotebookNode | list[tuple[str, str]]
        If return_log_lines is False, returns the executed notebook object.
        If return_log_lines is True, returns a list of (log_level, line_text) tuples.

    Raises
    ------
    Exception
        Any exception raised by the notebook kernel during execution is re-raised after logging.
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
    try:
        client.execute()
        try:
            executed_nb = client.nb
        except AttributeError:
            executed_nb = nb
    except Exception as exc:  # capture partial outputs on failure
        try:
            executed_nb = getattr(client, "nb", nb)
        except Exception:
            executed_nb = nb

        # collect and re-emit any log-like lines we captured so far
        collected = collect_log_lines(executed_nb, ignore_litellm=ignore_litellm)
        logger.info(
            "Collected %s logging lines from failed notebook %s", len(collected), path
        )
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

        # re-raise the original exception after logging
        raise

    # success path: collect and log normally
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
    doc_dir() / "basic_usage.ipynb",
]


def run_all(
    notebooks: list[str] | None = None,
    instrumented: bool = False,
    ignore_litellm: bool = True,
    kernel_name: str = "toponymy-uv",
) -> None:
    """
    Execute a sequence of notebooks with optional logging instrumentation.

    Iterates through the provided list of notebook paths and runs each one using run_notebook().

    Parameters
    ----------
    notebooks : list[str], optional
        List of notebook file paths to execute. If None, uses the default NOTEBOOKS list. Default is None.
    instrumented : bool, optional
        If True, use instrumented logging for each notebook. Default is False.
    ignore_litellm : bool, optional
        If True, exclude LiteLLM output lines from logging. Default is True.
    kernel_name : str, optional
        Jupyter kernel to use for executing the notebooks. Default is "toponymy-uv".

    Returns
    -------
    None
    """
    if notebooks is None:
        notebooks = NOTEBOOKS

    for nb in notebooks:
        run_notebook(
            nb,
            instrumented=instrumented,
            ignore_litellm=ignore_litellm,
            kernel_name=kernel_name,
        )


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
    parser.add_argument(
        "--allow-litellm-logs",
        action="store_true",
        help="Include LiteLLM log lines in output (default: ignored)",
    )
    parser.add_argument(
        "--kernel-name",
        default="toponymy-uv",
        help="Jupyter kernel to run notebooks with (default: toponymy-uv)",
    )

    args = parser.parse_args()

    notebooks = [Path.cwd() / nb for nb in args.notebooks]

    run_all(
        notebooks=notebooks or None,
        instrumented=args.instrument,
        ignore_litellm=not args.allow_litellm_logs,
        kernel_name=args.kernel_name,
    )
