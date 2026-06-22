import functools
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def doc_dir() -> Path:
    return PACKAGE_ROOT / "doc"


def examples_dir() -> Path:
    return PACKAGE_ROOT / "examples"


def get_notebooks(doc_dir: Path) -> list[Path]:
    """
    Get a list of all ipynb notebooks in the specified directory.
    """
    return sorted(p for p in Path(doc_dir).glob("*.ipynb") if "xxx" not in p.name)


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
