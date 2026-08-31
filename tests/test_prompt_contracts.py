import copy
import json
import os

import pytest
from hypothesis import given, settings, strategies as st

from toponymy.response_parsing import ResponseParseError, extract_response, string_field


def _template_module():
    import toponymy.templates as templates

    return templates


def test_text_name_and_prompt_characterization():
    template = _template_module().TextTemplate("article", "urban reports")
    prompt = template.cluster_prompt({"cluster_sentences": ["Bus lanes."]}, "specific")
    assert "urban reports" in prompt.system
    assert "Bus lanes." in prompt.user
    assert (
        template.extract_name('{"topic_name":"Transit","topic_specificity":0.8}')
        == "Transit"
    )


def test_summary_field_order_characterization():
    template = _template_module().SummaryTemplate("article", "urban reports")
    assert template.extract_name(
        '{"topic_analysis":"analysis","topic_summary":"summary","topic_name":"Transit","topic_specificity":0.8}'
    ) == ("Transit", "summary", "analysis")


def test_numeric_disambiguation_characterization():
    template = _template_module().TextTemplate("article", "urban reports")
    names = [
        "Cars",
        "Bus",
        "Metro",
        "Tram",
        "Cycling",
        "Walking",
        "Ferries",
        "Flights",
        "Taxis",
        "Rail",
    ]
    response = json.dumps(
        {
            "new_topic_name_mapping": {
                str(index): names[index - 1] for index in range(10, 0, -1)
            },
            "topic_specificities": [0.8] * 10,
        }
    )
    assert template.extract_disambiguated_names(response) == names


@pytest.mark.parametrize("wrapper", ["{}", "prefix {} suffix", "```json\n{}\n```"])
def test_structural_candidates_unicode_nesting_and_order(wrapper):
    record = {
        "extra": {"quotes": 'a } brace and \\" quote'},
        "topic_specificity": 0.8,
        "topic_name": 'Café "bus" \\ routes\nrail',
    }
    text = '{"wrong": "object"} ' + wrapper.format(
        json.dumps(record, ensure_ascii=True)
    )
    assert _template_module().TextTemplate.extract_name(text) == record["topic_name"]


@pytest.mark.parametrize(
    "text",
    [
        '{"topic_name": 42}',
        '{"topic_name": null}',
        '{"topic_name": []}',
        '{"topic_name": {"topic_name":"nested"}}',
        '{"topic_name": ""}',
        '{"topic_name":"truncated"',
        '{"outer":{"topic_name":"fragment"}',
        '{"topic_name":"first","topic_name":"second"}',
        '{"topic_name":"name","score":NaN}',
        "no JSON here",
    ],
)
def test_invalid_responses_are_not_repaired_into_names(text):
    with pytest.raises(ResponseParseError):
        _template_module().TextTemplate.extract_name(text)


def test_invalid_candidate_can_be_followed_by_valid_candidate():
    assert (
        _template_module().TextTemplate.extract_name(
            '{"topic_name": 42}\n{"topic_name":"valid"}'
        )
        == "valid"
    )


@pytest.mark.parametrize("score", [10**400, -(10**400), True, False, -0.1, 1.1])
def test_invalid_specificity_is_a_parse_failure_and_later_candidate_is_selected(score):
    response = json.dumps({"topic_name": "invalid", "topic_specificity": score})
    with pytest.raises(ResponseParseError):
        _template_module().TextTemplate.extract_name(response)
    assert (
        _template_module().TextTemplate.extract_name(
            response + '{"topic_name":"valid"}'
        )
        == "valid"
    )


@pytest.mark.parametrize("mapping", [{"1": "A", "3": "B"}, {"2": "A", "4": "B"}])
def test_disambiguation_rejects_missing_indices(mapping):
    with pytest.raises(ResponseParseError):
        _template_module().TextTemplate.extract_disambiguated_names(
            json.dumps({"new_topic_name_mapping": mapping})
        )


def test_templates_do_not_mutate_features_and_schema_is_owned():
    features = {
        "cluster_subtopics": {"major": ["Transit"], "minor": None},
        "cluster_sentences": ["Buses"],
    }
    before = copy.deepcopy(features)
    template = _template_module().TextTemplate("article", "urban reports")
    first = template.cluster_prompt(features, "specific")
    template.disambiguate_prompt(["Transit"], [features], "specific")
    assert features == before
    first.json_schema["properties"]["topic_name"]["type"] = "integer"
    assert (
        template.cluster_prompt(features, "specific").json_schema["properties"][
            "topic_name"
        ]["type"]
        == "string"
    )


def test_prompt_schema_is_owned_and_inspection_is_defensive():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    prompt = _template_module().Prompt("s", "u", schema)
    schema["properties"]["name"]["type"] = "integer"
    exposed = prompt.json_schema
    exposed["properties"]["name"]["type"] = "array"
    assert prompt.json_schema["properties"]["name"]["type"] == "string"
    prompt._asdict()["json_schema"].clear()
    assert prompt.json_schema["type"] == "object"
    with pytest.raises(AttributeError):
        prompt.user = "changed"


def test_summary_kind_is_template_configuration():
    template = _template_module().SummaryTemplate(
        "article", "urban reports", summary_kind="two sentences"
    )
    assert "two sentences" in template.cluster_prompt({}, "specific").system


def test_disambiguation_validates_alignment_and_types():
    template = _template_module().TextTemplate("article", "urban reports")
    with pytest.raises(ValueError):
        template.disambiguate_prompt(["one"], [{}, {}], "specific")
    for mapping in ({"x": "name"}, {"0": "name"}, {"1": 42}, {"1": "A", "1. Old": "B"}):
        with pytest.raises(ResponseParseError):
            template.extract_disambiguated_names(
                json.dumps({"new_topic_name_mapping": mapping})
            )


def test_extractor_does_not_swallow_programming_errors():
    with pytest.raises(TypeError, match="bug"):
        extract_response(
            '{"valid": true}', lambda value: (_ for _ in ()).throw(TypeError("bug"))
        )


@settings(
    max_examples=2000 if os.getenv("TOPONYMY_EXTENDED_TESTS") else 200, deadline=None
)
@given(
    name=st.text(min_size=1).filter(lambda text: bool(text.strip())),
    ensure_ascii=st.booleans(),
    field_order=st.permutations(("topic_name", "topic_specificity", "metadata")),
)
def test_generated_unicode_objects_round_trip(name, ensure_ascii, field_order):
    fields = {
        "topic_name": name,
        "topic_specificity": 0.5,
        "metadata": {"nested": [{}, '} quote \\"', ["{"]]},
    }
    response = json.dumps(
        {key: fields[key] for key in field_order}, ensure_ascii=ensure_ascii
    )
    assert (
        _template_module().TextTemplate.extract_name(
            "An unrelated object: {}\n```json\n" + response + "\n```"
        )
        == name
    )
    with pytest.raises(ResponseParseError):
        _template_module().TextTemplate.extract_name(response[:-1])
