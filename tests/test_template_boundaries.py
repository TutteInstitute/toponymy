import pytest

from toponymy.response_parsing import ResponseParseError
from toponymy.templates import (
    MultilingualENFRTemplate,
    TextTemplate,
    default_extract_topic_names,
)


def test_bilingual_prompt_preserves_both_required_response_fields():
    prompt = MultilingualENFRTemplate("article", "city").cluster_prompt(
        {"cluster_sentences": ["Le tramway relie les quartiers."]}, "specific"
    )
    schema = prompt.json_schema
    assert set(schema["required"]) == {
        "english_topic_name",
        "nom_du_sujet_en_français",
        "topic_specificity",
    }
    assert schema["properties"]["nom_du_sujet_en_français"] == {"type": "string"}
    assert "Le tramway relie les quartiers." in prompt.user


def test_list_subtopics_render_without_changing_the_callers_features():
    children = ["Rail transit", "Bus routes"]
    features = {"cluster_subtopics": children}
    prompt = TextTemplate("article", "city").cluster_prompt(features, "general")
    assert all(name in prompt.user for name in children)
    assert features == {"cluster_subtopics": ["Rail transit", "Bus routes"]}
    assert features["cluster_subtopics"] is children


@pytest.mark.parametrize("old_names, valid", [(["A", "B"], True), (["A"], False)])
def test_legacy_mapping_callback_requires_one_result_per_input(old_names, valid):
    response = {"new_topic_name_mapping": {"2. B": "Beta", "1. A": "Alpha"}}
    if valid:
        assert default_extract_topic_names(response, old_names) == ["Alpha", "Beta"]
    else:
        with pytest.raises(ResponseParseError, match="one name per input"):
            default_extract_topic_names(response, old_names)
