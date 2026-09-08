"""Fixed prompt programs must not cache configuration or rendered topic data."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import fields
from threading import Barrier

import jinja2

from toponymy.templates import (
    MultilingualENFRTemplate,
    SummaryTemplate,
    TextTemplate,
)


def test_four_compiled_programs_are_shared_across_instances_and_subclasses():
    programs = {
        name: value
        for name, value in vars(TextTemplate).items()
        if isinstance(value, jinja2.Template)
    }
    assert len(programs) == 4
    assert len({id(program) for program in programs.values()}) == 4

    for template in (
        TextTemplate("articles", "city planning"),
        TextTemplate("letters", "community feedback"),
        SummaryTemplate("reports", "marine ecology"),
        MultilingualENFRTemplate("notices", "public services"),
    ):
        assert not programs.keys() & {field.name for field in fields(template)}
        assert not programs.keys() & vars(template).keys()
        for name, program in programs.items():
            assert getattr(template, name) is program


def test_repeated_rendering_never_compiles_after_import(monkeypatch):
    def unexpected_compilation(*args, **kwargs):
        raise AssertionError("A fixed prompt program was compiled during rendering")

    monkeypatch.setattr(jinja2.Environment, "compile", unexpected_compilation)
    for template in (
        TextTemplate("articles", "city planning"),
        SummaryTemplate("articles", "city planning"),
        MultilingualENFRTemplate("articles", "city planning"),
    ):
        for index in range(8):
            feature = {"cluster_keywords": [f"tram-route-{index}"]}
            prompt = template.cluster_prompt(feature, "very specific")
            renamed = template.disambiguate_prompt(
                [f"Old route {index}"], [feature], "general"
            )
            assert "classifying articles from city planning" in prompt.system
            assert f"Keywords for this group include: tram-route-{index}" in prompt.user
            assert f"Current auto-generated name: Old route {index}" in renamed.user
            assert "Please provide new general names for each topic" in renamed.user


def test_changed_instance_configuration_and_features_are_rendered_fresh():
    template = TextTemplate("articles", "historic transport")
    feature = {
        "cluster_keywords": ["steam locomotives"],
        "cluster_subtopics": {"major": ["coal engines"]},
        "cluster_sentences": ["Steam powered the first trains."],
    }
    original_features = deepcopy(feature)
    first = template.cluster_prompt(feature, "very specific")
    first_renamed = template.disambiguate_prompt(["Steam rail"], [feature], "general")
    assert feature == original_features

    template.document_type = "field notes"
    template.corpus_description = "urban pollinators"
    template.cluster_task = "identify the pollinator habitat"
    template.cluster_response_description = "a JSON object describing the habitat"
    template.user_request = "name the habitat"
    template.subtopic_start = "[habitat:"
    template.subtopic_end = "]"
    feature["cluster_keywords"] = ["mason bees", "orchards"]
    feature["cluster_subtopics"] = {
        "major": ["apple trees"],
        "minor": ["nesting boxes"],
        "misc": ["flower strips"],
    }
    feature["cluster_sentences"] = ["Mason bees visit apple blossoms."]
    feature["exemplar_start_delimiter"] = "[observation:"
    feature["exemplar_end_delimiter"] = "]"
    updated_features = deepcopy(feature)

    second = template.cluster_prompt(feature, "general")
    second_renamed = template.disambiguate_prompt(
        ["Orchard bees"], [feature], "very specific"
    )

    assert feature == updated_features
    assert "classifying field notes from urban pollinators" in second.system
    assert "Your task is to identify the pollinator habitat" in second.system
    assert "response must be a JSON object describing the habitat" in second.system
    assert "Make every requested output broad enough" in second.system
    assert "Make every requested output precise" not in second.system
    assert "name the habitat for this group of field notes" in second.user
    for prompt in (second, second_renamed):
        assert "Keywords for this group include: mason bees, orchards" in prompt.user
        for subtopic in ("apple trees", "nesting boxes", "flower strips"):
            assert f"[habitat:{subtopic}]" in prompt.user
        assert "[observation:Mason bees visit apple blossoms.]" in prompt.user
        assert "steam locomotives" not in prompt.user
        assert "historic transport" not in prompt.system
    assert "Current auto-generated name: Orchard bees" in second_renamed.user
    assert "Each new topic name should be specific" in second_renamed.system
    assert "<SUBTOPIC>\ncoal engines\n</SUBTOPIC>" in first.user
    assert "historic transport" in first.system
    assert "Current auto-generated name: Steam rail" in first_renamed.user
    assert "orchards" not in first.user


def test_class_configuration_updates_respect_subclasses_and_instance_overrides(
    monkeypatch,
):
    class MuseumTemplate(TextTemplate):
        user_request = "name the collection"

    first = MuseumTemplate("catalogues", "ceramics")
    second = MuseumTemplate("catalogues", "textiles")
    second.user_request = "name the textile collection"
    ordinary = TextTemplate("articles", "archaeology")

    assert (
        "name the collection for this group" in first.cluster_prompt({}, "short").user
    )
    monkeypatch.setattr(MuseumTemplate, "user_request", "name the exhibit")
    assert "name the exhibit for this group" in first.cluster_prompt({}, "short").user
    assert (
        "name the textile collection for this group"
        in second.cluster_prompt({}, "short").user
    )
    assert "provide a name for this group" in ordinary.cluster_prompt({}, "short").user
    assert "textiles" not in first.cluster_prompt({}, "short").system
    assert "ceramics" not in second.cluster_prompt({}, "short").system


def test_summary_and_bilingual_instances_keep_distinct_current_configuration():
    summary = SummaryTemplate("reports", "wetlands", summary_kind="two sentences")
    other_summary = SummaryTemplate("reports", "forests", summary_kind="a bullet list")
    bilingual = MultilingualENFRTemplate("notices", "municipal services")
    ordinary = TextTemplate("reports", "wetlands")

    initial = summary.cluster_prompt({}, "general")
    summary.summary_kind = "one sentence"
    current = summary.cluster_prompt({}, "very specific")
    other = other_summary.cluster_prompt({}, "general")
    translated = bilingual.cluster_prompt({}, "general")
    plain = ordinary.cluster_prompt({}, "general")

    assert "two sentences summary" in initial.system
    assert "one sentence summary" in current.system
    assert "its form should be one sentence" in current.system
    assert "two sentences" not in current.system
    assert "a bullet list summary" in other.system
    assert "one sentence" not in other.system
    assert '"english_topic_name":<NAME>' in translated.system
    assert '"nom_du_sujet_en_français":<NOM>' in translated.user
    assert '"topic_summary":<SUMMARY>' not in translated.system
    assert '"english_topic_name"' not in plain.system
    assert "If a summary is requested" not in plain.system
    assert set(current.json_schema["required"]) == {
        "topic_analysis",
        "topic_summary",
        "topic_name",
        "topic_specificity",
    }
    assert set(translated.json_schema["required"]) == {
        "english_topic_name",
        "nom_du_sujet_en_français",
        "topic_specificity",
    }
    assert plain.json_schema["required"] == ["topic_name", "topic_specificity"]


def test_concurrent_rendering_isolates_features_names_and_schemas():
    templates = (
        TextTemplate("articles", "urban transport"),
        SummaryTemplate("reports", "marine ecology", summary_kind="two sentences"),
        MultilingualENFRTemplate("notices", "public services"),
    )
    markers = [f"record-{index:02d}-é-{{{{literal}}}}" for index in range(12)]
    barrier = Barrier(len(markers), timeout=10)

    def render(index):
        marker = markers[index]
        feature = {
            "cluster_keywords": [marker],
            "cluster_subtopics": {"major": [f"Habitat {marker}"]},
            "cluster_sentences": [f"Observed {marker}."],
        }
        template = templates[index % 3]
        barrier.wait()
        cluster = template.cluster_prompt(feature, "very specific")
        renamed = template.disambiguate_prompt(
            [f"First {marker}", f"Second {marker}"],
            [feature, {"cluster_keywords": [f"Alternative {marker}"]}],
            "general",
        )
        inspected_schema = cluster.json_schema
        inspected_schema["required"].append(marker)
        return cluster, renamed

    with ThreadPoolExecutor(max_workers=len(markers)) as executor:
        prompts = list(executor.map(render, range(len(markers))))

    for index, (cluster, renamed) in enumerate(prompts):
        marker = markers[index]
        assert f"Keywords for this group include: {marker}" in cluster.user
        assert f"<SUBTOPIC>\nHabitat {marker}\n</SUBTOPIC>" in cluster.user
        assert f'    * "Observed {marker}."\n' in cluster.user
        assert f'"1":\n- Current auto-generated name: First {marker}' in renamed.user
        assert f'"2":\n- Current auto-generated name: Second {marker}' in renamed.user
        assert f"Keywords for this group include: Alternative {marker}" in renamed.user
        assert "Please provide new general names for each topic" in renamed.user
        for foreign_marker in markers[:index] + markers[index + 1 :]:
            assert foreign_marker not in cluster.user
            assert foreign_marker not in renamed.user
        assert marker not in cluster.json_schema["required"]
        mapping = renamed.json_schema["properties"]["new_topic_name_mapping"]
        assert mapping["required"] == ["1", "2"]
        assert mapping["properties"] == {
            "1": {"type": "string"},
            "2": {"type": "string"},
        }
        if index % 3 == 0:
            assert "classifying articles from urban transport" in cluster.system
            assert "If a summary is requested" not in cluster.system
        elif index % 3 == 1:
            assert "classifying reports from marine ecology" in cluster.system
            assert "its form should be two sentences" in cluster.system
        else:
            assert "classifying notices from public services" in cluster.system
            assert '"nom_du_sujet_en_français":<NOM>' in cluster.system
