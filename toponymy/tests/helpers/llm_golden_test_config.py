"""
Configuration, expected patterns, and helper functions for test_golden_prompts.py.
"""

import re
import warnings
from dataclasses import dataclass


from toponymy.llm_wrappers import llm_output_to_result
from toponymy.tools.load_prompt_examples import load_prompt_params
from toponymy.templates import (
    PROMPT_TEMPLATES,
    SUMMARY_PROMPT_TEMPLATES,
    MULTILINGUAL_EN_FR_PROMPT_TEMPLATES,
)

EXAMPLE_VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    "hockey_masks": {
        # Note: do not match on hockey since it's too broad
        "en": (r"\bgoalie\b", r"\bgoaltend", r"\bmask"),
        "fr": (r"\bgardien", r"\bmasque", r"\bcagette", r"\bportier"),
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
"""
Regex patterns a correct answer should match, by example id and vocabulary name.

Each example maps vocabulary names to tuples of patterns. A ``PartShape`` names the
vocabulary its part is matched against, so every example must provide each
vocabulary used by the templates it is tested with. Patterns are matched
case-insensitively against the original text, so word boundaries are the real
ones and accented patterns work.
"""

SUMMARY_KIND_WORD_RANGE_REGEX = re.compile(r"\((\d+)\s+(?:to|or)\s+(\d+)\s+word\)")


def check(condition, message, *, soft=False):
    """
    Assert a condition, or when soft, warn so the run continues and the miss is
    still reported. Useful since some basic models used for testing rarely meet word
    count or length expectations.

    Parameters
    ----------
    condition : bool
        The condition that should hold.

    message : str
        Description of the failure, used for the AssertionError or the warning.

    soft : bool, default=False
        If True, a failed condition emits a UserWarning instead of raising.

    Raises
    ------
    AssertionError
        If ``condition`` is false and ``soft`` is False.
    """
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
        If True, a word count outside the range emits a warning
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
    """
    A template set under test and the shape of what it returns.

    Attributes
    ----------
    templates : dict
        The template set, as passed to ``run_prompt_example(prompt_templates=...)``.

    parts : tuple of PartShape
        One shape per part of the topic result, in the order the template's
        ``extract_topic_name`` returns them.

    max_tokens : int or None, default=None
        Token budget for the call, or None for the namer's default. Set this for
        templates whose output won't fit in the default topic-name budget.
    """

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


def normalize(text):
    """
    Lowercase text and replace every character other than ASCII letters, digits
    and whitespace with a space.

    Used only to compare two texts for near-identity (e.g. summary vs analysis). Not for pattern
    matching, since it turns accented letters into spaces and so creates word
    boundaries that aren't in the original text.

    Parameters
    ----------
    text : str or object
        The text to normalize. Non-string values are converted with ``str``.

    Returns
    -------
    normalized : str
        The lowercased text with every other character replaced by a space.
    """
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower())


def count_pattern_hits(text, patterns):
    """
    Count the distinct patterns that match somewhere in the text.

    Each pattern counts at most once, however many times it matches. Matching is
    case-insensitive and runs on the original text, so word boundaries and
    accented characters are respected.

    Parameters
    ----------
    text : str
        The text to search.

    patterns : iterable of str
        Regex patterns to look for.

    Returns
    -------
    hits : int
        The number of patterns with at least one match.
    """
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def summary_kind_word_range(summary_kind: str) -> tuple[int, int]:
    """
    Parse the topic-name word range out of a summary kind, e.g.
    "clear and concise (3 to 6 word)" -> (3, 6), "simple (1 or 2 word)" -> (1, 2).

    Parameters
    ----------
    summary_kind : str
        A summary kind such as "clear and concise (3 to 6 word)" or
        "simple (1 or 2 word)".

    Returns
    -------
    low, high : int
        The bounds as written, e.g. (3, 6) or (1, 2). They are not reordered.

    Raises
    ------
    ValueError
        If no "(N to M word)" or "(N or M word)" range is found.
    """
    match = SUMMARY_KIND_WORD_RANGE_REGEX.search(summary_kind)
    if match is None:
        raise ValueError(
            f"Could not parse a word range from summary_kind {summary_kind!r}"
        )
    return int(match.group(1)), int(match.group(2))


def word_count(text):
    """
    Count the words in a text.

    Words are runs of ``\w`` characters, which are Unicode-aware, so accented words
    count once. Apostrophes (straight or curly) and hyphens join rather than split
    words, so "l'équipe" and "face-off" are one word each. Standalone punctuation
    such as "&" is not a word.

    Parameters
    ----------
    text : str
        The text to count words in.

    Returns
    -------
    count : int
        The number of words in the text.
    """
    return len(re.findall(r"\w+(?:['’-]\w+)*", text))


def sentence_count(text):
    """
    Count the sentences in a text.

    A sentence boundary is ".", "!" or "?" followed by whitespace. Text with no
    terminal punctuation counts as one sentence, and empty text as none. This is
    a heuristic: abbreviations such as "e.g. " also count as boundaries.

    Parameters
    ----------
    text : str
        The text to count sentences in.

    Returns
    -------
    count : int
        The number of sentences in the text.
    """
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def paragraph_count(text):
    """
    Count the paragraphs in a text.

    Paragraphs are separated by one or more blank lines.

    Parameters
    ----------
    text : str
        The text to count paragraphs in.

    Returns
    -------
    count : int
        The number of paragraphs in the text.
    """
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def split_topic_result(result, n_parts, template):
    """
    Return the parts of a topic result, one per expected part.

    Tuples are already split, and a single expected part is the string itself.
    A string that joins several parts (e.g. "English / Français") is not split on
    its separator, which may also appear inside a part. Instead the parts are read
    from the string fields of the raw JSON response, checked against the
    template's own extract_topic_name applied to that response, and returned in
    the order the extractor placed them in the joined string.

    Parameters
    ----------
    result : dict
        The result of run_prompt_example, run with verbose=True so the raw
        response is available.

    n_parts : int
        The number of parts expected, one per PartShape declared for the template.

    template : dict
        The template's "layer" entry, providing "get_topic_name_regex" and
        "extract_topic_name".

    Returns
    -------
    parts : tuple of str
        The parts of the topic result, in the order they appear in it.
    """
    topic_result = result["topic_result"]
    if isinstance(topic_result, tuple):
        return topic_result
    if n_parts == 1:
        return (topic_result,)

    parsed = llm_output_to_result(result["response"], template["get_topic_name_regex"])

    expected = template["extract_topic_name"](parsed)
    expected = expected if isinstance(expected, tuple) else str(expected)
    assert topic_result == expected, (
        f"topic_result {topic_result!r} does not match the template's "
        f"extraction of the raw response {expected!r}",
        result,
    )

    parts = [value.strip() for value in parsed.values() if isinstance(value, str)]
    for part in parts:
        assert part in topic_result, (
            f"part {part!r} does not appear in topic_result {topic_result!r}",
            result,
        )
    return tuple(sorted(parts, key=topic_result.index))


def check_part(part, shape, patterns, word_range, result, range_source=None):
    """
    Check one part of a topic result against its shape.

    A missing or empty part always fails. The word-count check fails, or warns
    if ``shape.soft_word_range`` is True. The sentence, paragraph and pattern
    checks always fail.

    Parameters
    ----------
    part : str
        The part's text.

    shape : PartShape
        The expectations for this part.

    patterns : tuple of str
        The vocabulary patterns this part is matched against.

    word_range : tuple of (int, int or None)
        The resolved minimum and maximum word counts, from ``shape.word_range``.

    result : dict
        The full result, included in hard-failure messages for context.

    range_source : str or None, default=None
        Where the word range came from (e.g. "summary_kind '...'"), appended to
        word-count messages when given.

    Raises
    ------
    AssertionError
        If a hard check fails.
    """
    label = shape.label
    context = (label, part, result)

    assert isinstance(part, str) and part.strip(), context

    min_words, max_words = word_range
    n_words = word_count(part)
    source = f" for {range_source}" if range_source else ""
    check(
        n_words >= min_words,
        f"{label} {part!r}: {n_words} words < {min_words}{source}",
        soft=shape.soft_word_range,
    )
    if max_words is not None:
        check(
            n_words <= max_words,
            f"{label} {part!r}: {n_words} words > {max_words}{source}",
            soft=shape.soft_word_range,
        )

    n_sentences = sentence_count(part)
    assert n_sentences >= shape.min_sentences, (
        f"{label}: {n_sentences} sentences < {shape.min_sentences}",
        *context,
    )
    if shape.max_sentences is not None:
        assert n_sentences <= shape.max_sentences, (
            f"{label}: {n_sentences} sentences > {shape.max_sentences}",
            *context,
        )

    if shape.max_paragraphs is not None:
        n_paragraphs = paragraph_count(part)
        assert n_paragraphs <= shape.max_paragraphs, (
            f"{label}: {n_paragraphs} paragraphs > {shape.max_paragraphs}",
            *context,
        )

    n_hits = count_pattern_hits(part, patterns)
    assert n_hits >= shape.min_pattern_hits, (
        f"{label}: matched {n_hits} of {patterns}, need {shape.min_pattern_hits}",
        *context,
    )


def assert_valid_topic_name_result(result, example_id, template_id):
    """
    Check a run_prompt_example topic-name result for an example and template.

    Checks that the call succeeded, splits the topic result into the parts the
    template declares, checks each part against its shape and the example's
    vocabulary, and finally checks the parts against each other:

    - the analysis should be longer than the summary (soft),
    - the analysis must not repeat the summary,
    - the summary must be longer than the name.

    Parameters
    ----------
    result : dict
        The result of run_prompt_example with prompt_kind="topic_name" and
        verbose=True.

    example_id : str
        The example the prompt was rendered from; a key of EXAMPLE_VOCABULARY.

    template_id : str
        The template set the prompt was rendered with; a key of TEMPLATE_CASES.

    Raises
    ------
    AssertionError
        If any hard check fails. Soft checks emit a UserWarning instead.
    """
    assert result["success"] is True, result
    assert result["prompt_kind"] == "topic_name", result

    topic_result = result["topic_result"]
    assert isinstance(topic_result, (str, tuple)), result

    case = TEMPLATE_CASES[template_id]
    shapes = case.parts
    prompt_template = case.templates["layer"]
    vocabulary = EXAMPLE_VOCABULARY[example_id]
    summary_kind = (
        load_prompt_params(example_id)["summary_kind"]
        if any(shape.words_from_summary_kind for shape in shapes)
        else None
    )
    parts = split_topic_result(result, len(shapes), prompt_template)
    assert len(parts) == len(shapes), (
        f"Expected {[s.label for s in shapes]} for {template_id!r}, got {parts!r}",
        result,
    )

    for part, shape in zip(parts, shapes):
        assert shape.vocabulary in vocabulary, (
            f"Example {example_id!r} has no {shape.vocabulary!r} vocabulary, "
            f"needed by template {template_id!r}"
        )
        check_part(
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
