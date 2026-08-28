import functools
from pathlib import Path
import os
import httpx

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# Ollama test models
OLLAMA_CI_MODELS = ["qwen2.5:0.5b", "qwen2:0.5b", "tinydolphin:1.1b"]
OLLAMA_LOCAL_FAMILIES = ["llama3.2", "qwen2.5", "tinydolphin"]


def doc_dir() -> Path:
    return PACKAGE_ROOT / "doc"


def examples_dir() -> Path:
    return PACKAGE_ROOT / "examples"


def get_notebooks(doc_dir: Path) -> list[Path]:
    """
    Retrieve all non-test notebook files from a directory.

    Searches for .ipynb files and excludes those with "xxx" in the filename.

    Parameters
    ----------
    doc_dir : Path
        Path to the directory containing notebook files.

    Returns
    -------
    list[Path]
        Sorted list of Path objects for .ipynb notebooks excluding test files (those with "xxx" in the name).
    """
    return sorted(p for p in Path(doc_dir).glob("*.ipynb") if "xxx" not in p.name)


def get_test_ollama_model() -> str | None:
    """
    Returns the first available ollama model for testing, or None if none available.
    In CI, looks for exact matches from OLLAMA_CI_MODELS.
    Locally, looks for family matches from OLLAMA_LOCAL_FAMILIES.
    """

    response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
    response.raise_for_status()
    available = [m["name"] for m in response.json().get("models", [])]

    if os.getenv("CI", "").lower() in {"true", "1"}:
        for model in OLLAMA_CI_MODELS:
            if model in available:
                return model
    else:
        for family in OLLAMA_LOCAL_FAMILIES:
            match = next((m for m in available if m.startswith(family + ":")), None)
            if match:
                return match

    return None


def notebook_test_replacement(replacement):
    """
    Decorator for replacing function implementations during example notebook testing.

    When the NOTEBOOK_TESTING environment variable is set to "true", the decorated function
    is replaced with the provided replacement function. This allows lightweight mocking
    during automated notebook runs.

    Parameters
    ----------
    replacement : callable
        The replacement function to call when NOTEBOOK_TESTING is enabled.

    Returns
    -------
    callable
        A decorator that wraps the target function and applies the replacement conditionally.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv("NOTEBOOK_TESTING", "").lower() == "true":
                return replacement(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
