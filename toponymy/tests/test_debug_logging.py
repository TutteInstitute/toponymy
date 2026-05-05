import json
from toponymy.debug_logging import BasicDebugLogger


def test_basic_debug_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "debug.jsonl"
    logger = BasicDebugLogger(log_path, truncate=False)

    logger(
        {
            "event": "llm_call_success",
            "prompt": "test prompt",
            "raw_response": "test response",
        }
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "llm_call_success"
    assert record["prompt"] == "test prompt"
    assert record["raw_response"] == "test response"


def test_basic_debug_logger_truncates_strings(tmp_path):
    log_path = tmp_path / "debug.jsonl"
    logger = BasicDebugLogger(log_path, truncate=True, max_len=10)

    logger(
        {
            "prompt": "abcdefghijklmnopqrstuvwxyz",
            "raw_response": "0123456789abcdefghijklmnopqrstuvwxyz",
        }
    )

    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert "[truncated]" in record["prompt"]
    assert "[truncated]" in record["raw_response"]


def test_basic_debug_logger_truncates_dict_prompt(tmp_path):
    log_path = tmp_path / "debug.jsonl"
    logger = BasicDebugLogger(log_path, truncate=True, max_len=10)

    logger(
        {
            "prompt": {
                "system": "system-prompt-is-long",
                "user": "user-prompt-is-long",
            },
            "raw_response": "ok",
        }
    )

    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert "[truncated]" in record["prompt"]["system"]
    assert "[truncated]" in record["prompt"]["user"]
    assert record["raw_response"] == "ok"
