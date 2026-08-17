"""Structural extraction of complete JSON objects from model responses."""

import json
import math
from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")


class ResponseParseError(ValueError):
    """The response does not contain a complete object matching the contract."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseParseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ResponseParseError(f"Invalid JSON constant: {value}")


def json_objects(response: str) -> Iterator[dict]:
    """Yield complete outer objects, respecting nested braces and escaped quotes.

    Prose and code fences are allowed around objects. Incomplete objects are never
    repaired or mined for nested fragments that happen to look like valid output.
    """
    if not isinstance(response, str):
        raise ResponseParseError("An LLM response must be a string")
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
    position = 0
    while position < len(response):
        starts = [
            index
            for index in (response.find("{", position), response.find("[", position))
            if index >= 0
        ]
        if not starts:
            return
        start = min(starts)
        depth, quoted, escaped = 0, False, False
        end = start
        for end in range(start, len(response)):
            char = response[end]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    break
        if depth != 0 or quoted:
            return
        position = end + 1
        try:
            value, consumed = decoder.raw_decode(response[start:position])
        except (json.JSONDecodeError, ResponseParseError):
            continue
        if consumed == position - start and isinstance(value, dict):
            yield value


def extract_response(response: str, validate: Callable[[dict], T]) -> T:
    """Select the first complete candidate satisfying a field contract."""
    for candidate in json_objects(response):
        try:
            return validate(candidate)
        except ResponseParseError:
            continue
    raise ResponseParseError("No complete JSON object matches the response contract")


def string_field(value: dict, key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ResponseParseError(f"{key} must be a nonempty string")
    return result


def topic_fields(value: dict, *keys: str) -> tuple[str, ...]:
    if "topic_specificity" in value:
        _score(value["topic_specificity"])
    return tuple(string_field(value, key) for key in keys)


def _score(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        raise ResponseParseError(
            "Topic specificity must be a finite number between zero and one"
        )


def topic_name_mapping(value: dict) -> list[str]:
    mapping = value.get("new_topic_name_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ResponseParseError("new_topic_name_mapping must be a nonempty object")
    indexed = {}
    for key, name in mapping.items():
        # Legacy prompts used keys such as '1. Previous name'.
        prefix = key.strip().split(".", 1)[0]
        if not prefix.isdecimal() or int(prefix) < 1:
            raise ResponseParseError("Topic mapping keys must be positive indices")
        index = int(prefix)
        if index in indexed or not isinstance(name, str) or not name.strip():
            raise ResponseParseError("Topic mapping indices and names must be valid")
        indexed[index] = name
    if set(indexed) != set(range(1, len(indexed) + 1)):
        raise ResponseParseError("Topic mapping indices must be contiguous from one")
    if "topic_specificities" in value:
        scores = value["topic_specificities"]
        if not isinstance(scores, list) or len(scores) != len(indexed):
            raise ResponseParseError("Topic specificities must align with topic names")
        for score in scores:
            _score(score)
    return [indexed[index] for index in sorted(indexed)]
