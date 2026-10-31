import re
from dataclasses import dataclass
import warnings

from toponymy.tools.load_prompt_examples import load_prompt_params
from toponymy.templates import (
    PROMPT_TEMPLATES,
    SUMMARY_PROMPT_TEMPLATES,
    MULTILINGUAL_EN_FR_PROMPT_TEMPLATES,
)

# ---------------------------------------------------------------------------
# example_id vocabulary to be checked against the patterns
# ---------------------------------------------------------------------------

EXAMPLE_VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    "hockey_masks": {
        # Note: do not match on hockey since it's too broad
        "en": (r"\bgoalie\b", r"\bgoaltend", r"\bmask"),
        "fr": (r"\bgardien", r"\bmasque", r"\bportier"),
        "themes": (
            r"\bgoalie|\bgoaltend",
            r"\bmask",
            r"\bhelmet|\bcage",
            r"\bprotect|\bsafety|\binjur",
            r"\bpaint|\bartwork|\bart\b|\bdesign|\bdecorat",
            r"\bfib(?:er|re)glass|\bmaterial|\bplastic|\bcarbon",
        ),
    },
}

# ---------------------------------------------------------------------------
# Output shape by template
# ---------------------------------------------------------------------------
SUMMARY_KIND_WORD_RANGE_REGEX = re.compile(r"\((\d+)\s+(?:to|or)\s+(\d+)\s+word\)")


def summary_kind_word_range(summary_kind: str) -> tuple[int, int]:
    """
    Parse the topic-name word range out of a summary kind, e.g.
    "clear and concise (3 to 6 word)" -> (3, 6), "simple (1 or 2 word)" -> (1, 2).
    """
    match = SUMMARY_KIND_WORD_RANGE_REGEX.search(summary_kind)
    if match is None:
        raise ValueError(
            f"Could not parse a word range from summary_kind {summary_kind!r}"
        )
    return int(match.group(1)), int(match.group(2))


def word_count(text):
    # \w is Unicode-aware, so accented words count once; apostrophes and hyphens
    # join rather than split words ("l'équipe", "face-off").
    return len(re.findall(r"\w+(?:['’-]\w+)*", text))


def check(condition, message, *, soft=False):
    """Assert, or when soft, warn so the run continues and the miss is still reported."""
    if condition:
        return
    if soft:
        warnings.warn(message, stacklevel=3)
    else:
        raise AssertionError(message)


@dataclass(frozen=True)
class PartShape:
    """
    Expectations for one part of a topic result, independent of the example.

    A template's extraction function may return several parts (for instance a
    name, a summary and an analysis). Each part is described by a PartShape,
    which sets its expected length and structure and names the example
    vocabulary its content is matched against. Because a PartShape refers to
    a vocabulary by key rather than holding patterns, the same shape applies
    to every example.

    Attributes
    ----------
    label : str
        Name for this part, used in failure messages and to pair parts up in
        the cross-part checks (e.g. "name", "summary", "analysis").

    vocabulary : str
        Key into ``EXAMPLE_VOCABULARY[example_id]`` selecting the regex patterns
        this part's text is matched against.

    min_pattern_hits : int, default=1
        Minimum number of distinct vocabulary patterns that must match the
        part's text. Raise this to require broader coverage of a topic.

    min_words : int, default=1
        Minimum word count. Ignored when ``words_from_summary_kind`` is True.

    max_words : int or None, default=None
        Maximum word count, or None for no upper limit. Ignored when
        ``words_from_summary_kind`` is True.

    words_from_summary_kind : bool, default=False
        If True, take the word range from the example's ``summary_kind``
        (e.g. "clear and concise (3 to 6 word)" gives 3 to 6) instead of from
        ``min_words`` and ``max_words``.

    word_slack : int, default=0
        Number of words by which to widen a ``summary_kind`` range on either
        side, for parts such as translations that need extra function words.
        Only applies when ``words_from_summary_kind`` is True.

    soft_word_range : bool, default=False
        If True, a word count outside the range emits a GoldenSoftCheckWarning
        instead of failing the test, so the miss is reported without being
        enforced.

    min_sentences : int, default=1
        Minimum number of sentences.

    max_sentences : int or None, default=None
        Maximum number of sentences, or None for no upper limit.

    max_paragraphs : int or None, default=None
        Maximum number of paragraphs, or None for no upper limit.
    """

    label: str
    vocabulary: str
    min_pattern_hits: int = 1
    min_words: int = 1
    max_words: int | None = None
    words_from_summary_kind: bool = False
    word_slack: int = 0
    soft_word_range: bool = False
    min_sentences: int = 1
    max_sentences: int | None = None
    max_paragraphs: int | None = None

    def word_range(self, summary_kind: str | None = None) -> tuple[int, int | None]:
        """
        Resolve the word-count range this part must fall within.

        Parameters
        ----------
        summary_kind : str or None, default=None
            The example's summary kind, e.g. "clear and concise (3 to 6 word)".
            Required when ``words_from_summary_kind`` is True, ignored otherwise.

        Returns
        -------
        min_words : int
            The minimum word count, never less than 1.

        max_words : int or None
            The maximum word count, or None for no upper limit.

        Raises
        ------
        ValueError
            If ``words_from_summary_kind`` is True and ``summary_kind`` is None or
            has no parseable word range.
        """
        if not self.words_from_summary_kind:
            return self.min_words, self.max_words
        if summary_kind is None:
            raise ValueError(
                f"PartShape {self.label!r} takes its word range from summary_kind, "
                f"but no summary_kind was given"
            )
        low, high = summary_kind_word_range(summary_kind)
        return max(1, low - self.word_slack), high + self.word_slack


NAME_EN = PartShape("name", "en", words_from_summary_kind=True, soft_word_range=True)
NAME_FR = PartShape(
    "nom", "fr", words_from_summary_kind=True, word_slack=3, soft_word_range=True
)
SUMMARY = PartShape(
    "summary", "en", min_words=15, max_words=150, max_sentences=6, max_paragraphs=1
)
ANALYSIS = PartShape(
    "analysis",
    "themes",
    min_pattern_hits=2,
    min_words=60,
    min_sentences=1,
    soft_word_range=True,
)


@dataclass(frozen=True)
class TemplateCase:
    templates: dict
    parts: tuple[PartShape, ...]
    max_tokens: int | None = None


TEMPLATE_CASES: dict[str, TemplateCase] = {
    "default": TemplateCase(PROMPT_TEMPLATES, (NAME_EN,)),
    "summary": TemplateCase(
        SUMMARY_PROMPT_TEMPLATES, (NAME_EN, SUMMARY, ANALYSIS), max_tokens=1024
    ),
    "multilingual_en_fr": TemplateCase(
        MULTILINGUAL_EN_FR_PROMPT_TEMPLATES, (NAME_EN, NAME_FR)
    ),
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

MULTILINGUAL_SEPARATOR = " / "


def normalize(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())


def count_pattern_hits(text, patterns):
    normalized = normalize(text)
    return sum(1 for p in patterns if re.search(p, normalized))


def word_count(text):
    return len(text.split())


def sentence_count(text):
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def paragraph_count(text):
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def split_topic_result(topic_result, n_parts):
    if isinstance(topic_result, tuple):
        return topic_result
    if n_parts == 1:
        return (topic_result,)
    return tuple(part.strip() for part in topic_result.split(MULTILINGUAL_SEPARATOR))


def assert_valid_part(part, shape, patterns, word_range, result, range_source=None):
    label = shape.label
    context = (label, part, result)

    assert isinstance(part, str) and part.strip(), context

    min_words, max_words = word_range
    words = word_count(part)
    source = f" for {range_source}" if range_source else ""
    check(
        words >= min_words,
        f"{label} {part!r}: {words} words < {min_words}{source}",
        soft=shape.soft_word_range,
    )
    if max_words is not None:
        check(
            words <= max_words,
            f"{label} {part!r}: {words} words > {max_words}{source}",
            soft=shape.soft_word_range,
        )

    sentences = sentence_count(part)
    assert sentences >= shape.min_sentences, (
        f"{label}: {sentences} sentences < {shape.min_sentences}",
        *context,
    )
    if shape.max_sentences is not None:
        assert sentences <= shape.max_sentences, (
            f"{label}: {sentences} sentences > {shape.max_sentences}",
            *context,
        )

    if shape.max_paragraphs is not None:
        paragraphs = paragraph_count(part)
        assert paragraphs <= shape.max_paragraphs, (
            f"{label}: {paragraphs} paragraphs > {shape.max_paragraphs}",
            *context,
        )

    hits = count_pattern_hits(part, patterns)
    assert hits >= shape.min_pattern_hits, (
        f"{label}: matched {hits} of {patterns}, need {shape.min_pattern_hits}",
        *context,
    )


def assert_valid_topic_name_result(result, example_id, template_id):
    assert result["success"] is True, result
    assert result["prompt_kind"] == "topic_name", result

    topic_result = result["topic_result"]
    assert isinstance(topic_result, (str, tuple)), result

    shapes = TEMPLATE_CASES[template_id].parts
    vocabulary = EXAMPLE_VOCABULARY[example_id]
    summary_kind = (
        load_prompt_params(example_id)["summary_kind"]
        if any(shape.words_from_summary_kind for shape in shapes)
        else None
    )
    parts = split_topic_result(topic_result, len(shapes))
    assert len(parts) == len(shapes), (
        f"Expected {[s.label for s in shapes]} for {template_id!r}, got {parts!r}",
        result,
    )

    for part, shape in zip(parts, shapes):
        assert shape.vocabulary in vocabulary, (
            f"Example {example_id!r} has no {shape.vocabulary!r} vocabulary, "
            f"needed by template {template_id!r}"
        )
        assert_valid_part(
            part,
            shape,
            vocabulary[shape.vocabulary],
            shape.word_range(summary_kind),
            result,
            range_source=(
                f"summary_kind {summary_kind!r}"
                if shape.words_from_summary_kind
                else None
            ),
        )

    by_label = dict(zip((s.label for s in shapes), parts))
    if "summary" in by_label and "analysis" in by_label:
        summary, analysis = by_label["summary"], by_label["analysis"]
        check(
            word_count(analysis) > word_count(summary),
            f"analysis should be longer than the summary: analysis={analysis}, summary={summary}",
            soft=True,
        )
        assert normalize(analysis) != normalize(summary), (
            "analysis should not repeat the summary",
            result,
        )
    if "name" in by_label and "summary" in by_label:
        assert word_count(by_label["summary"]) > word_count(by_label["name"]), (
            "summary should be longer than the name",
            result,
        )
