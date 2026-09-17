"""Shared fixtures for local tests; notebook kernels never register with the user."""

import json
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture
def local_notebook_kernel(tmp_path, monkeypatch):
    """Expose both runner kernel names using the current test interpreter.

    Jupyter requires local sockets. Keep this fixture out of tests run under an
    all-sockets-denied guard; the examples use no external services or models.
    """
    name = "toponymy-local-doc-tests"
    data_directory = tmp_path / "jupyter"
    specification = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Toponymy local test interpreter",
        "language": "python",
    }
    for kernel_name in (name, "toponymy-uv"):
        kernel_directory = data_directory / "kernels" / kernel_name
        kernel_directory.mkdir(parents=True)
        (kernel_directory / "kernel.json").write_text(
            json.dumps(specification), encoding="utf-8"
        )
    existing = os.environ.get("JUPYTER_PATH")
    monkeypatch.setenv(
        "JUPYTER_PATH",
        str(data_directory) + (os.pathsep + existing if existing else ""),
    )
    existing_python_path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(Path(__file__).resolve().parent.parent)
        + (os.pathsep + existing_python_path if existing_python_path else ""),
    )
    runtime_directory = data_directory / "runtime"
    runtime_directory.mkdir()
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(runtime_directory))
    monkeypatch.delenv("NOTEBOOK_TESTING", raising=False)
    for key in tuple(os.environ):
        if any(
            marker in key.upper()
            for marker in ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET")
        ):
            monkeypatch.delenv(key)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        monkeypatch.setenv(key, "1")
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return name
