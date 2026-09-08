"""Uncovered policy boundaries exercised without SDKs or external services."""

import copy
from types import SimpleNamespace

import pytest

from toponymy import llm_wrappers as wrappers
from toponymy import provider_batches as batches

NAME_SCHEMA = {
    "type": "object",
    "properties": {"topic_name": {"type": "string"}},
    "required": ["topic_name"],
    "additionalProperties": False,
}


def request_configuration(**overrides):
    """A capability boundary with no completion method or SDK dependency."""
    configuration = {
        "model": "fixture/model",
        "api_key": None,
        "api_base": None,
        "provider_kwargs": {},
        "use_json_schema": None,
        "use_json_object": None,
        "supports_json_schema": True,
        "_detect_json_object_support": lambda: True,
    }
    configuration.update(overrides)
    return SimpleNamespace(**configuration)


@pytest.mark.parametrize(
    "schema",
    [
        {"$schema": "https://example.invalid/unknown-dialect"},
        {
            "properties": {
                "topic_name": {"$schema": "https://example.invalid/nested-dialect"}
            }
        },
        {"description": "a scalar, not a schema", "$ref": "#/description"},
        {"enum": [float("nan")]},
    ],
)
def test_response_schema_rejects_unknown_dialects_and_non_schema_references(schema):
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrappers._validate_json_schema(schema)


def test_response_schema_accepts_boolean_and_object_combinator_children():
    schema = {
        "allOf": [True, {"type": "object"}],
        "anyOf": [{"properties": {"topic_name": {"type": "string"}}}, False],
        "oneOf": [{"required": ["topic_name"]}],
        "prefixItems": [True, {"type": "string"}],
    }
    before = copy.deepcopy(schema)
    assert wrappers._validate_json_schema(schema) is None
    assert schema == before


@pytest.mark.parametrize(
    "temperature", [-0.01, True, "warm", float("nan"), float("inf")]
)
def test_invalid_generation_temperature_is_rejected(temperature):
    with pytest.raises(wrappers.InvalidLLMInputError, match="temperature"):
        wrappers._validate_generation_options(temperature, 128)


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5, "128"])
def test_invalid_generation_token_budget_is_rejected(max_tokens):
    with pytest.raises(wrappers.InvalidLLMInputError, match="max_tokens"):
        wrappers._validate_generation_options(0.4, max_tokens)


def test_zero_temperature_and_single_token_are_valid_boundaries():
    assert wrappers._validate_generation_options(0, 1) is None


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"response_format": "json_schema"},
        {"response_format": {"type": "unknown"}},
        {"response_format": {"type": "json_schema"}},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "", "schema": NAME_SCHEMA},
            }
        },
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": 17, "schema": NAME_SCHEMA},
            }
        },
        {"response_format": {"type": "json_schema", "json_schema": {"name": "named"}}},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "named", "schema": []},
            }
        },
    ],
)
def test_explicit_output_format_requires_a_valid_named_schema_envelope(options):
    with pytest.raises(wrappers.InvalidLLMInputError):
        wrappers._validate_json_options(None, None, options)


def test_explicit_named_schema_and_connection_settings_survive_request_building():
    specification = {
        "name": "caller_chosen_name",
        "strict": True,
        "schema": copy.deepcopy(NAME_SCHEMA),
    }
    options = {
        "response_format": {"type": "json_schema", "json_schema": specification},
        "timeout": 12,
    }
    before = copy.deepcopy(options)
    configuration = request_configuration(
        provider_kwargs=options,
        api_key="fixture-key",
        api_base="https://example.invalid/v1",
    )
    messages = [{"role": "user", "content": "Name this topic"}]
    result = wrappers._provider_request_kwargs(configuration, {}, messages, 0, 1)
    assert result == {
        "response_format": before["response_format"],
        "timeout": 12,
        "model": "fixture/model",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1,
        "num_retries": 0,
        "api_key": "fixture-key",
        "api_base": "https://example.invalid/v1",
    }
    result["response_format"]["json_schema"]["schema"]["properties"]["topic_name"][
        "type"
    ] = "integer"
    assert options == before


def test_explicit_text_output_does_not_require_structured_output_capabilities():
    configuration = request_configuration(
        provider_kwargs={"response_format": {"type": "text"}},
        supports_json_schema=False,
        _detect_json_object_support=lambda: False,
    )
    result = wrappers._provider_request_kwargs(configuration, {}, [], 0.4, 128)
    assert result["response_format"] == {"type": "text"}


@pytest.mark.parametrize("mode", ["json_schema", "json_object"])
def test_explicit_output_format_requires_provider_capability(mode):
    response_format = {"type": mode}
    if mode == "json_schema":
        response_format["json_schema"] = {"name": "topic", "schema": NAME_SCHEMA}
    configuration = request_configuration(
        provider_kwargs={"response_format": response_format},
        supports_json_schema=False,
        _detect_json_object_support=lambda: False,
    )
    with pytest.raises(wrappers.InvalidLLMInputError, match="does not support"):
        wrappers._provider_request_kwargs(configuration, {}, [], 0.4, 128)


def test_explicit_json_object_request_does_not_silently_downgrade():
    configuration = request_configuration(
        use_json_object=True, _detect_json_object_support=lambda: False
    )
    with pytest.raises(wrappers.InvalidLLMInputError, match="JSON object"):
        wrappers._provider_request_kwargs(configuration, {}, [], 0.4, 128)


@pytest.mark.parametrize("schema", [{}, [], True])
def test_request_builder_rechecks_programmatically_supplied_prompt_schema(schema):
    with pytest.raises(wrappers.InvalidLLMInputError, match="nonempty schema object"):
        wrappers._provider_request_kwargs(
            request_configuration(), {"json_schema": schema}, [], 0.4, 128
        )


def test_post_construction_option_conflict_is_rechecked_at_request_boundary():
    configuration = request_configuration()
    configuration.use_json_schema = True
    configuration.use_json_object = True
    with pytest.raises(wrappers.InvalidLLMInputError, match="cannot both"):
        wrappers._provider_request_kwargs(
            configuration, {"json_schema": NAME_SCHEMA}, [], 0.4, 128
        )


@pytest.mark.parametrize("minimum", [0, 1])
def test_anthropic_array_items_and_boolean_schema_children_are_inspected(minimum):
    schema = {
        "type": "array",
        "minItems": minimum,
        "items": {"anyOf": [True, {"type": "string"}]},
    }
    before = copy.deepcopy(schema)
    assert wrappers._anthropic_schema_problem(schema) is None
    assert schema == before


def test_anthropic_rejects_combining_reference_and_allof():
    schema = {
        "$defs": {"name": {"type": "string"}},
        "$ref": "#/$defs/name",
        "allOf": [{"type": "string"}],
    }
    assert wrappers._anthropic_schema_problem(schema) == "allOf combined with $ref"


def test_anthropic_reports_anchor_reference_outside_its_supported_pointer_subset():
    schema = {
        "$ref": "#named-anchor",
        "$defs": {"name": {"$anchor": "named-anchor", "type": "string"}},
    }
    assert (
        wrappers._anthropic_schema_problem(schema)
        == "only local JSON Pointer references are supported"
    )


def test_custom_legacy_parser_selector_still_requires_structural_json():
    result = wrappers.llm_output_to_result(
        'prefix {"custom_field":"value"} suffix', regex="custom parser selector"
    )
    assert result == {"custom_field": "value"}


def test_raw_batch_result_values_preserve_order_and_original_failure():
    assert wrappers._result_values([]) == []
    assert wrappers._result_values(
        [wrappers.CallResult(value="A"), wrappers.CallResult(value="B")]
    ) == ["A", "B"]
    failure = RuntimeError("item failed")
    with pytest.raises(RuntimeError) as raised:
        wrappers._result_values(
            [wrappers.CallResult(value="A"), wrappers.CallResult(error=failure)]
        )
    assert raised.value is failure


def test_transport_batch_partial_results_keep_successes_in_original_positions():
    failure = batches.BatchItemError("1", "rate limit", status_code=429)
    converted = wrappers._transport_batch_results(["A", failure, "C"])
    assert [item.value for item in converted] == ["A", None, "C"]
    assert converted[0].error is converted[2].error is None
    assert isinstance(converted[1].error, wrappers.LLMBatchItemError)
    assert str(converted[1].error) == "Batch request 1: rate limit"
    assert wrappers._transport_batch_results([]) == []


def test_transport_batch_auth_failure_retains_original_error():
    failure = batches.BatchItemError("0", "permission denied", status_code=403)
    with pytest.raises(wrappers.FailFastLLMError) as raised:
        wrappers._transport_batch_results([failure])
    assert raised.value.original_exception is failure
    assert raised.value.__cause__ is failure


def test_transport_batch_unexpected_errors_are_not_converted_to_item_failures():
    failure = TypeError("provider integration bug")
    with pytest.raises(TypeError) as raised:
        wrappers._transport_batch_results([failure])
    assert raised.value is failure


def test_anthropic_inline_auth_error_fails_fast():
    record = SimpleNamespace(
        custom_id="0",
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(
                error=SimpleNamespace(type="authentication_error", message="bad key")
            ),
        ),
    )
    with pytest.raises(
        wrappers.FailFastLLMError, match="Batch item 0: authentication_error: bad key"
    ):
        wrappers._ordered_anthropic_results([record])


def test_normalized_batch_messages_reject_non_mapping_input():
    with pytest.raises(ValueError, match="normalized system/user"):
        batches._messages("un-normalized prompt")


def test_cohere_rejects_unsupported_format_without_mutating_schema():
    schema = copy.deepcopy(NAME_SCHEMA)
    schema["properties"]["topic_name"]["format"] = "email"
    before = copy.deepcopy(schema)
    assert batches._cohere_schema_problem(schema) == "unsupported format email"
    assert schema == before


@pytest.mark.parametrize(
    "error,expected_status",
    [
        ("plain provider error", None),
        ('{"code":"rate_limit_error"}', 429),
        ({"status_code": 418, "message": "explicit HTTP status"}, 418),
        ({"status_code": True, "code": "permission_error"}, 403),
        ({"code": 17}, None),
    ],
)
def test_batch_error_records_preserve_explicit_or_recognized_status(
    error, expected_status
):
    with pytest.raises(batches.BatchProtocolError) as raised:
        batches._response({"custom_id": "0", "error": error})
    assert raised.value.status_code == expected_status


@pytest.mark.parametrize("status", [True, "200"])
def test_batch_response_rejects_non_integer_http_status(status):
    with pytest.raises(
        batches.BatchProtocolError, match="status_code must be an integer"
    ):
        batches._response({"response": {"status_code": status, "body": {}}})


@pytest.mark.parametrize(
    "body,problem",
    [
        ({"finish_reason": 12}, "finish_reason must be a string"),
        ({"finish_reason": "MAX_TOKENS"}, "incomplete generation"),
        ({"message": {"content": "not a list"}}, "content must be a list"),
        (
            {"message": {"content": [{"type": "text", "text": 42}]}},
            "text block must contain a string",
        ),
        ({"message": {"content": []}}, "contains no text"),
        (
            {"message": {"content": [{"type": "text", "text": "  "}]}},
            "contains no text",
        ),
    ],
)
def test_cohere_incomplete_or_malformed_content_is_an_explicit_protocol_error(
    body, problem
):
    with pytest.raises(batches.BatchProtocolError, match=problem):
        batches._cohere_text({"response": {"status_code": 200, "body": body}})


def test_cohere_text_blocks_are_joined_in_order_and_non_text_blocks_are_ignored():
    record = {
        "response": {
            "status_code": 200,
            "body": {
                "finish_reason": "COMPLETE",
                "message": {
                    "content": [
                        {"type": "tool_call", "id": "tool-17"},
                        {"type": "text", "text": "Café "},
                        {"type": "text", "text": "trams"},
                    ]
                },
            },
        },
    }
    assert batches._cohere_text(record) == "Café trams"


@pytest.mark.parametrize("choices", [None, [], [{}, {}]])
def test_azure_requires_exactly_one_completion_choice(choices):
    with pytest.raises(batches.BatchProtocolError, match="exactly one choice"):
        batches._azure_text({"response": {"body": {"choices": choices}}})


def test_duplicate_submitted_ids_are_rejected_before_reading_results():
    def unexpected_parser(row):
        raise AssertionError("A duplicate input index reached response extraction")

    with pytest.raises(batches.BatchProtocolError, match="input contains duplicate"):
        batches._aligned_results(
            [{"custom_id": "same"}], ["same", "same"], unexpected_parser
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "  "},
        {"model": 12},
        {"polling_interval": True},
        {"timeout": float("nan")},
        {"request_timeout": 0},
        {"max_poll_retries": -1},
        {"max_poll_retries": True},
        {"max_poll_retries": 1.5},
    ],
)
def test_batch_configuration_rejects_invalid_settings_before_sdk_construction(changed):
    configuration = dict(
        model="fixture",
        polling_interval=1,
        timeout=10,
        request_timeout=2,
        max_poll_retries=0,
    )
    configuration.update(changed)
    with pytest.raises(ValueError):
        batches._settings(**configuration)


def test_zero_poll_retries_is_a_valid_explicit_budget():
    assert batches._settings("fixture", 1, 10, 2, 0) is None
