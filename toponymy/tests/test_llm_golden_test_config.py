import itertools
import json
import re
import warnings

import pytest

from toponymy.templates import MULTILINGUAL_EN_FR_PROMPT_TEMPLATES, SUMMARY_KINDS
from toponymy.tools.load_prompt_examples import example_ids, load_prompt_params

from toponymy.tests.helpers.llm_golden_test_config import (
    EXAMPLE_VOCABULARY,
    TEMPLATE_CASES,
    PartShape,
    assert_valid_topic_name_result,
    check,
    check_part,
    count_pattern_hits,
    normalize,
    paragraph_count,
    sentence_count,
    split_topic_result,
    summary_kind_word_range,
    word_count,
)

HOCKEY_EN = EXAMPLE_VOCABULARY["hockey_masks"]["en"]
MULTILINGUAL_LAYER = MULTILINGUAL_EN_FR_PROMPT_TEMPLATES["layer"]


def test_check_passes_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(True, "unused")
        check(True, "unused", soft=True)


def test_check_hard_failure_raises():
    with pytest.raises(AssertionError, match="it broke"):
        check(False, "it broke")


def test_check_soft_failure_warns_instead_of_raising():
    with pytest.warns(UserWarning, match="it broke"):
        check(False, "it broke", soft=True)


@pytest.mark.parametrize(
    "summary_kind, expected",
    [
        ("domain expert level (8 to 15 word)", (8, 15)),
        ("clear and concise (3 to 6 word)", (3, 6)),
        ("simple (1 or 2 word)", (1, 2)),
    ],
)
def test_summary_kind_word_range_values(summary_kind, expected):
    assert summary_kind_word_range(summary_kind) == expected


def test_summary_kind_word_range_rejects_unparseable():
    with pytest.raises(ValueError, match="Could not parse a word range"):
        summary_kind_word_range("simple and short")


@pytest.mark.parametrize("summary_kind", SUMMARY_KINDS)
def test_every_summary_kind_parses(summary_kind):
    low, high = summary_kind_word_range(summary_kind)
    assert 1 <= low <= high


# Prefixes include digits, commas, apostrophes and hyphens, to check that stray
# numbers or punctuation outside the parentheses aren't picked up.
ROUND_TRIP_PREFIXES = ["", "simple", "top 10 style", "domain-expert, level's"]
ROUND_TRIP_BOUNDS = [
    (1, 2),
    (3, 6),
    (8, 15),
    (12, 100),
    (6, 4),
]  # (6, 4): not reordered
ROUND_TRIP_CONNECTORS = ["to", "or"]


@pytest.mark.parametrize(
    "prefix, bounds, connector",
    itertools.product(ROUND_TRIP_PREFIXES, ROUND_TRIP_BOUNDS, ROUND_TRIP_CONNECTORS),
)
def test_summary_kind_word_range_round_trips(prefix, bounds, connector):
    low, high = bounds
    summary_kind = f"{prefix} ({low} {connector} {high} word)"
    assert summary_kind_word_range(summary_kind) == (low, high)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "simple",
        "top 10 style",
        "(1 to 2 words)",
        "1 to 2 word",
        "(one to two word)",
    ],
)
def test_summary_kind_word_range_rejects_text_without_range(text):
    with pytest.raises(ValueError, match="Could not parse a word range"):
        summary_kind_word_range(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Goalie mask designs", 3),
        ("Masques de gardien de but", 5),
        ("défense", 1),  # accents don't split words
        ("l'équipe", 1),
        ("l’équipe", 1),  # curly apostrophe
        ("face-off", 1),
        ("Art & Design", 2),  # standalone punctuation isn't a word
        ("", 0),
    ],
)
def test_word_count(text, expected):
    assert word_count(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One. Two! Three?", 3),
        ("No terminal punctuation", 1),
        ("Ends with a period.", 1),
        ("", 0),
    ],
)
def test_sentence_count(text, expected):
    assert sentence_count(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("one\n\ntwo", 2),
        ("one\ntwo", 1),  # a single newline is not a paragraph break
        ("one\n   \ntwo", 2),  # a whitespace-only line is a blank line
        ("", 0),
    ],
)
def test_paragraph_count(text, expected):
    assert paragraph_count(text) == expected


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Goalie-Mask") == "goalie mask"


def test_count_pattern_hits_counts_distinct_patterns_once():
    assert count_pattern_hits("mask mask mask", (r"\bmask", r"\bgoalie\b")) == 1


def test_count_pattern_hits_is_case_insensitive():
    assert count_pattern_hits("GOALIE MASK", HOCKEY_EN) == 2


def test_count_pattern_hits_respects_hyphen_boundaries():
    assert count_pattern_hits("goalie-mask", (r"\bgoalie\b",)) == 1


def test_count_pattern_hits_does_not_split_accented_words():
    # "é" is a word character, so there is no boundary before "fense".
    assert count_pattern_hits("défense", (r"\bfense",)) == 0


def test_count_pattern_hits_does_not_split_underscores():
    assert count_pattern_hits("goalie_mask", (r"\bgoalie\b",)) == 0


def test_count_pattern_hits_matches_accented_patterns():
    assert count_pattern_hits("Il protège le visage", (r"\bprotège",)) == 1


def test_word_range_uses_fixed_bounds_by_default():
    shape = PartShape("summary", "en", min_words=15, max_words=150)
    assert shape.word_range() == (15, 150)
    assert shape.word_range("ignored (1 to 2 word)") == (15, 150)


def test_word_range_from_summary_kind_ignores_fixed_bounds():
    shape = PartShape("name", "en", min_words=50, words_from_summary_kind=True)
    assert shape.word_range("clear and concise (3 to 6 word)") == (3, 6)


def test_word_range_slack_widens_both_sides():
    shape = PartShape("nom", "fr", words_from_summary_kind=True, word_slack=3)
    assert shape.word_range("domain expert level (8 to 15 word)") == (5, 18)


def test_word_range_slack_never_goes_below_one():
    shape = PartShape("nom", "fr", words_from_summary_kind=True, word_slack=3)
    assert shape.word_range("simple (1 or 2 word)") == (1, 5)


def test_word_range_from_summary_kind_requires_summary_kind():
    shape = PartShape("name", "en", words_from_summary_kind=True)
    with pytest.raises(ValueError, match="no summary_kind was given"):
        shape.word_range()


def multilingual_result(en, fr, topic_result=None):
    """A run_prompt_example-style result for the multilingual template."""
    response = json.dumps(
        {
            "english_topic_name": en,
            "nom_du_sujet_en_français": fr,
            "topic_specificity": 0.9,
        },
        ensure_ascii=False,  # the template regex matches the literal "ç"
    )
    return {
        "topic_result": topic_result if topic_result is not None else f"{en} / {fr}",
        "response": response,
    }


def test_split_passes_tuples_through():
    result = {"topic_result": ("a", "b", "c"), "response": None}
    assert split_topic_result(result, 3, MULTILINGUAL_LAYER) == ("a", "b", "c")


def test_split_single_part_needs_no_response():
    result = {"topic_result": "Goalie Masks", "response": None}
    assert split_topic_result(result, 1, MULTILINGUAL_LAYER) == ("Goalie Masks",)


def test_split_joined_result_from_raw_response():
    result = multilingual_result("Goalie Masks", "Masques de gardien")
    assert split_topic_result(result, 2, MULTILINGUAL_LAYER) == (
        "Goalie Masks",
        "Masques de gardien",
    )


def test_split_tolerates_separator_inside_a_part():
    result = multilingual_result("Masks / Helmets", "Masques de gardien")
    assert split_topic_result(result, 2, MULTILINGUAL_LAYER) == (
        "Masks / Helmets",
        "Masques de gardien",
    )


def test_split_orders_parts_as_the_extractor_does():
    reversed_layer = {
        "get_topic_name_regex": MULTILINGUAL_LAYER["get_topic_name_regex"],
        "extract_topic_name": lambda j: (
            f'{j["nom_du_sujet_en_français"]} / {j["english_topic_name"]}'
        ),
    }
    result = multilingual_result(
        "Goalie Masks", "Masques de gardien", "Masques de gardien / Goalie Masks"
    )
    assert split_topic_result(result, 2, reversed_layer) == (
        "Masques de gardien",
        "Goalie Masks",
    )


def test_split_rejects_result_not_matching_extraction():
    result = multilingual_result("Goalie Masks", "Masques de gardien", "Goalie Masks")
    with pytest.raises(AssertionError, match="does not match"):
        split_topic_result(result, 2, MULTILINGUAL_LAYER)


def test_split_rejects_field_missing_from_result():
    english_only = {
        "get_topic_name_regex": MULTILINGUAL_LAYER["get_topic_name_regex"],
        "extract_topic_name": lambda j: j["english_topic_name"],
    }
    result = multilingual_result("Goalie Masks", "Masques de gardien", "Goalie Masks")
    with pytest.raises(AssertionError, match="does not appear"):
        split_topic_result(result, 2, english_only)


# check part

NAME_SHAPE = PartShape("name", "en", max_words=3)


def test_check_part_passes_valid_part():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_part("Goalie masks", NAME_SHAPE, HOCKEY_EN, (1, 3), {})


def test_check_part_rejects_empty_part():
    with pytest.raises(AssertionError):
        check_part("   ", NAME_SHAPE, HOCKEY_EN, (1, 3), {})


def test_check_part_word_range_is_hard_by_default():
    with pytest.raises(AssertionError, match="words > 3"):
        check_part("Goalie masks of the past", NAME_SHAPE, HOCKEY_EN, (1, 3), {})


def test_check_part_word_range_can_be_soft():
    shape = PartShape("name", "en", soft_word_range=True)
    with pytest.warns(UserWarning, match="words > 3"):
        check_part("Goalie masks of the past", shape, HOCKEY_EN, (1, 3), {})


def test_check_part_reports_range_source():
    shape = PartShape("name", "en", soft_word_range=True)
    with pytest.warns(UserWarning, match="for summary_kind 'simple"):
        check_part(
            "Goalie masks of the past",
            shape,
            HOCKEY_EN,
            (1, 2),
            {},
            range_source="summary_kind 'simple (1 or 2 word)'",
        )


def test_check_part_enforces_max_sentences():
    shape = PartShape("summary", "en", max_sentences=1)
    with pytest.raises(AssertionError, match="sentences"):
        check_part("Goalie masks. More masks.", shape, HOCKEY_EN, (1, None), {})


def test_check_part_enforces_max_paragraphs():
    shape = PartShape("summary", "en", max_paragraphs=1)
    with pytest.raises(AssertionError, match="paragraphs"):
        check_part("Goalie masks.\n\nMore masks.", shape, HOCKEY_EN, (1, None), {})


def test_check_part_enforces_pattern_hits():
    with pytest.raises(AssertionError, match="matched 0"):
        check_part("Ice skating", NAME_SHAPE, HOCKEY_EN, (1, 3), {})


# assert_valid_topic_name_result

EN_WORDS = ("Goalie mask art and design in the National Hockey League").split()
FR_WORDS = (
    "Masques de gardien de but et leur conception artistique "
    "dans le Ligue nationale de hockey"
).split()

SUMMARY_TEXT = (
    "This group covers goalie masks, their design history, and the protective "
    "role they play for goaltenders in professional and amateur leagues."
)
ANALYSIS_TEXT = " ".join(
    [
        "Goalie masks evolved from fiberglass shells into painted helmet and cage designs."
    ]
    * 6
)


def hockey_name_words():
    """The minimum name length the hockey_masks summary kind asks for."""
    low, _ = summary_kind_word_range(load_prompt_params("hockey_masks")["summary_kind"])
    return low


def topic_name_result(topic_result, response=None, **overrides):
    result = {
        "success": True,
        "prompt_kind": "topic_name",
        "topic_result": topic_result,
        "response": response,
    }
    result.update(overrides)
    return result


@pytest.mark.filterwarnings("error")
def test_valid_default_result_passes():
    name = " ".join(EN_WORDS[: hockey_name_words()])
    assert_valid_topic_name_result(topic_name_result(name), "hockey_masks", "default")


@pytest.mark.filterwarnings("error")
def test_valid_summary_result_passes():
    name = " ".join(EN_WORDS[: hockey_name_words()])
    result = topic_name_result((name, SUMMARY_TEXT, ANALYSIS_TEXT))
    assert_valid_topic_name_result(result, "hockey_masks", "summary")


@pytest.mark.filterwarnings("error")
def test_valid_multilingual_result_passes():
    n = hockey_name_words()
    joined = multilingual_result(" ".join(EN_WORDS[:n]), " ".join(FR_WORDS[:n]))
    result = topic_name_result(joined["topic_result"], joined["response"])
    assert_valid_topic_name_result(result, "hockey_masks", "multilingual_en_fr")


def test_failed_call_is_rejected():
    result = topic_name_result("", success=False)
    with pytest.raises(AssertionError):
        assert_valid_topic_name_result(result, "hockey_masks", "default")


def test_wrong_prompt_kind_is_rejected():
    result = topic_name_result("Goalie Masks", prompt_kind="topic_cluster_names")
    with pytest.raises(AssertionError):
        assert_valid_topic_name_result(result, "hockey_masks", "default")


def test_wrong_number_of_parts_is_rejected():
    result = topic_name_result(("Goalie Masks", SUMMARY_TEXT))
    with pytest.raises(AssertionError, match="Expected"):
        assert_valid_topic_name_result(result, "hockey_masks", "summary")


def test_missing_vocabulary_is_reported(monkeypatch):
    monkeypatch.delitem(EXAMPLE_VOCABULARY["hockey_masks"], "fr")
    n = hockey_name_words()
    joined = multilingual_result(" ".join(EN_WORDS[:n]), " ".join(FR_WORDS[:n]))
    result = topic_name_result(joined["topic_result"], joined["response"])
    with pytest.raises(AssertionError, match="has no 'fr' vocabulary"):
        assert_valid_topic_name_result(result, "hockey_masks", "multilingual_en_fr")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analysis_repeating_summary_is_rejected():
    name = " ".join(EN_WORDS[: hockey_name_words()])
    result = topic_name_result((name, SUMMARY_TEXT, SUMMARY_TEXT))
    with pytest.raises(AssertionError, match="should not repeat"):
        assert_valid_topic_name_result(result, "hockey_masks", "summary")


@pytest.mark.parametrize("example_id", EXAMPLE_VOCABULARY)
def test_vocabulary_examples_exist(example_id):
    assert example_id in example_ids()


@pytest.mark.parametrize("example_id", EXAMPLE_VOCABULARY)
def test_example_summary_kind_parses(example_id):
    summary_kind_word_range(load_prompt_params(example_id)["summary_kind"])


@pytest.mark.parametrize("template_id", TEMPLATE_CASES)
@pytest.mark.parametrize("example_id", EXAMPLE_VOCABULARY)
def test_examples_provide_every_vocabulary_templates_use(example_id, template_id):
    vocabulary = EXAMPLE_VOCABULARY[example_id]
    for shape in TEMPLATE_CASES[template_id].parts:
        assert shape.vocabulary in vocabulary, (example_id, template_id, shape.label)


@pytest.mark.parametrize("example_id", EXAMPLE_VOCABULARY)
def test_vocabulary_patterns_compile(example_id):
    for patterns in EXAMPLE_VOCABULARY[example_id].values():
        for pattern in patterns:
            re.compile(pattern)


@pytest.mark.parametrize("template_id", TEMPLATE_CASES)
def test_template_layer_declares_its_parsing(template_id):
    layer = TEMPLATE_CASES[template_id].templates["layer"]
    assert "extract_topic_name" in layer
    assert "get_topic_name_regex" in layer
