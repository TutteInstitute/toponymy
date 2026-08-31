"""Retained upstream refactor template tests, using the canonical module.

The disambiguation case now supplies all ten request positions because the
canonical alignment contract deliberately rejects gaps; its numeric ordering
assertion and the original three named positions are retained.
"""

from toponymy.templates import (
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
        '"1": "Urban Buses", '
        '"9": "Air Routes", "8": "Taxis", "7": "Trams", '
        '"6": "Ferries", "5": "Cycling", "4": "Walking", "3": "Metro"}, '
        '"topic_specificities": [0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8]}'
    )
    assert result == [
        "Urban Buses",
        "Regional Transit",
        "Metro",
        "Walking",
        "Cycling",
        "Ferries",
        "Trams",
        "Taxis",
        "Air Routes",
        "Freight Rail",
    ]
