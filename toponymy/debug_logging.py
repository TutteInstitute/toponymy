import json
from pathlib import Path
from typing import Any


class BasicDebugLogger:
    """
    Synchronous JSONL logger for LLM wrapper callbacks. Appends one JSON record
    per event to *path*, optionally truncating long ``prompt`` and
    ``raw_response`` fields so logs stay readable during development.

    Not suitable for high-volume runs — each event blocks on a file open/write.
    For production use, swap in a buffered or queue-backed callback instead.

    Args:
        path: Destination file. Appends to an existing file.
        truncate: If True, long string values in ``prompt`` and
                  ``raw_response`` are trimmed to *max_len* characters.
        max_len: Maximum characters kept per field when truncating
                 (half from the start, half from the end).
    """

    def __init__(
        self,
        path: str | Path = "llm_debug.jsonl",
        *,
        truncate: bool = True,
        max_len: int = 2000,
    ):
        self.path = Path(path)
        self.truncate = truncate
        self.max_len = max_len

    def _truncate(self, value: Any) -> Any:
        if not self.truncate or not isinstance(value, str):
            return value

        if len(value) <= self.max_len:
            return value

        half = self.max_len // 2
        return value[:half] + "\n...[truncated]...\n" + value[-half:]

    def _process(self, event: dict[str, Any]) -> dict[str, Any]:
        processed = {}

        for key, value in event.items():
            if key in {"prompt", "raw_response"}:
                if isinstance(value, dict):
                    processed[key] = {
                        subkey: self._truncate(subvalue)
                        for subkey, subvalue in value.items()
                    }
                else:
                    processed[key] = self._truncate(value)
            else:
                processed[key] = value

        return processed

    def __call__(self, event: dict[str, Any]) -> None:
        try:
            processed = self._process(event)
            line = json.dumps(processed, ensure_ascii=False)
        except Exception as e:
            line = json.dumps(
                {
                    "logger_error": "failed to process/serialize event",
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                },
                ensure_ascii=False,
            )

        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            warnings.warn(
                f"{self.__class__.__name__} could not write to {self.path}: {e}",
                UserWarning,
                stacklevel=2,
            )
