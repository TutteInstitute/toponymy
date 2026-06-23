from toponymy.new_templates import (
    MultilingualENFRTemplate,
    SummaryTemplate,
    TextTemplate,
)


def test_text_extract():
    template = TextTemplate("document", "corpus")

    result = template.extract_name(
        '{"topic_name": "Transit Policy", "topic_specificity": 0.8}'
    )

    assert result == "Transit Policy"


def test_text_prompt():
    template = TextTemplate("article", "urban planning reports")
    features = {
        "cluster_keywords": ["transit", "zoning"],
        "cluster_subtopics": {
            "major": ["bus priority lanes"],
            "minor": ["parking reform"],
            "misc": ["bike parking"],
        },
        "cluster_sentences": ["Cities expanded bus lanes downtown."],
    }

    prompt = template.cluster_prompt(features, "very specific topic")

    assert "urban planning reports" in prompt.system
    assert "article" in prompt.system
    assert "very specific topic" in prompt.system
    assert '"topic_name":<NAME>' in prompt.system
    assert "transit, zoning" in prompt.user
    assert "<SUBTOPIC>\nbus priority lanes\n</SUBTOPIC>" in prompt.user
    assert "<SUBTOPIC>\nparking reform\n</SUBTOPIC>" in prompt.user
    assert "<SUBTOPIC>\nbike parking\n</SUBTOPIC>" in prompt.user
    assert '    * "Cities expanded bus lanes downtown."\n' in prompt.user


def test_multi_extract():
    template = MultilingualENFRTemplate("document", "corpus")

    result = template.extract_name(
        '{"english_topic_name": "Transit Policy", '
        '"nom_du_sujet_en_fran\\u00e7ais": "Politique des transports", '
        '"topic_specificity": 0.8}'
    )

    assert result == "Transit Policy / Politique des transports"


def test_summary_extract():
    template = SummaryTemplate("document", "corpus")

    result = template.extract_name(
        '{"topic_analysis": "Analysis text", '
        '"topic_summary": "Summary text", '
        '"topic_name": "Transit Policy", '
        '"topic_specificity": 0.8}'
    )

    assert result == ("Transit Policy", "Summary text", "Analysis text")


def test_disambig_extract():
    template = TextTemplate("document", "corpus")

    result = template.extract_disambiguated_names(
        '{"new_topic_name_mapping": {'
        '"2": "Regional Transit", '
        '"10": "Freight Rail", '
        '"1": "Urban Buses"}, '
        '"topic_specificities": [0.8, 0.7, 0.9]}'
    )

    assert result == ["Urban Buses", "Regional Transit", "Freight Rail"]
