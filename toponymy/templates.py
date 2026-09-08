from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jinja2

from .response_parsing import (
    ResponseParseError,
    extract_response,
    topic_fields,
    topic_name_mapping,
)

GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*[\w.]+\s*\}'
GET_TOPIC_CLUSTER_NAMES_REGEX = (
    r'\{\s*"new_topic_name_mapping":\s*.*?,\s*"topic_specificities": .*?\}'
)
TopicNameResult = str | tuple[str, str, str]


@dataclass(frozen=True, init=False)
class Prompt:
    """Inspectable provider-independent messages and optional JSON Schema."""

    system: str
    user: str
    _json_schema: dict[str, Any] | None = field(repr=False)

    def __init__(
        self, system: str, user: str, json_schema: dict[str, Any] | None = None
    ):
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "user", user)
        object.__setattr__(self, "_json_schema", deepcopy(json_schema))

    @property
    def json_schema(self) -> dict[str, Any] | None:
        """Return an owned schema copy; inspection cannot alter future requests."""
        return deepcopy(self._json_schema)

    def _asdict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "user": self.user,
            "json_schema": self.json_schema,
        }


class Template(ABC):
    """Interface for prompt templates and response parsers."""

    @abstractmethod
    def cluster_prompt(self, features: dict[str, Any], name_kind: str) -> Prompt:
        """Build the prompt used to name one topic cluster."""
        pass

    @staticmethod
    @abstractmethod
    def extract_name(response: str) -> TopicNameResult:
        """Extract the generated name from a model response."""
        pass

    @abstractmethod
    def disambiguate_prompt(
        self,
        names: list[str],
        features: list[dict[str, Any]],
        name_kind: str,
    ) -> Prompt:
        """Build the prompt used to rename similar topic clusters."""
        pass

    @staticmethod
    @abstractmethod
    def extract_disambiguated_names(response: str) -> list[str]:
        """Extract renamed topics from a disambiguation response."""
        pass


@dataclass
class TextTemplate(Template):
    document_type: str
    corpus_description: str

    cluster_task: ClassVar[str] = (
        "analyze the provided group information and assign a name"
    )
    cluster_response_description: ClassVar[str] = (
        'in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>}\n'
        "where NAME is the topic name you generate and SCORE is a float value between "
        "0.0 and 1.0,\nrepresenting how specific and well-defined the topic name is "
        "given the input information."
    )
    user_request: ClassVar[str] = "provide a name"
    subtopic_start: ClassVar[str] = "<SUBTOPIC>\n"
    subtopic_end: ClassVar[str] = "\n</SUBTOPIC>"

    # Fixed programs are shared; every render receives its own fresh context.
    _disambiguation_system_program: ClassVar[jinja2.Template] = jinja2.Template("""
You are an expert in {{corpus_description}}. You have been asked to provide more specific and distinguishing names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Your task is to generate a new {{name_kind}} name for each topic group presented.
You should make use of the relative relationships between these topics, their keywords, subtopic information, and sample {{document_type}} to generate new, distinct topic names.
The new names must be in the same order as the original topics are presented.
There should be no duplicate topic names in your final list of new names.

{% if "very specific" in name_kind %}
Each new topic name should be specific to the information of that topic and sufficiently detailed to ensure it can be distinguished from all the other similar topics listed.
{% elif "general" in name_kind %}
Each topic name should be broad and simple enough to capture the overall sense of the large and diverse range of {{document_type}} contained in it at a glance, while still separating it from the other topics listed.
{% endif %}
{% if features_list | selectattr("cluster_subtopics.major") | list %}
When major subtopics are present, primarily make use of the major and minor subtopics, and ensure each generated topic name reflects the core essence of *all* major subtopics.
{% endif %}

The response must be formatted as a single JSON object in the format:
{"new_topic_name_mapping": {"1": "NEW_TOPIC_NAME1", "2": "NEW_TOPIC_NAME2", ... }, "topic_specificities": [<NEW_TOPIC_SCORE1>, <NEW_TOPIC_SCORE2>, ...]}
where each NEW_TOPIC_NAME value is a JSON string containing the complete new topic name, and SCORE is a float value between 0.0 and 1.0 representing the quality and specificity of the new name.
If the current auto-generated names are bilingual, keep the same "English / French" style in a single string value.
Ensure your entire response is only the JSON object, with no other text before or after it.
""")

    _disambiguation_user_program: ClassVar[jinja2.Template] = jinja2.Template("""
Below are the auto-generated topic names, along with keywords, subtopics, and sample {{document_type}} for each topic area.

Corpus description: {{corpus_description}}

{% for features, name in feature_names %}
"{{loop.index}}":
- Current auto-generated name: {{name}}
{% if features["cluster_keywords"] %}
- Keywords for this group include: {{", ".join(features["cluster_keywords"])}}
{% endif %}
{%- if features["cluster_subtopics"]["major"] %}
- Major subtopics of this group are:
{%- for subtopic in features["cluster_subtopics"]["major"] %}
{{features["subtopic_start"]}}{{subtopic}}{{features["subtopic_end"]}}
{%- endfor %}
{%- endif %}
{%- if features["cluster_subtopics"]["minor"] %}
- Minor subtopics of this group are:
{%- for subtopic in features["cluster_subtopics"]["minor"] %}
{{features["subtopic_start"]}}{{subtopic}}{{features["subtopic_end"]}}
{%- endfor %}
{%- endif %}
{%- if features["cluster_subtopics"]["misc"] %}
- Other miscellaneous specific subtopics of this group in order of relevance (from most to least) include:
{%- for subtopic in features["cluster_subtopics"]["misc"] %}
{{features["subtopic_start"]}}{{subtopic}}{{features["subtopic_end"]}}
{%- endfor %}
{%- endif %}
{%- if features["cluster_sentences"] %}
- Sample {{document_type}} from this group include:
{%- for sentence in features["cluster_sentences"] %}
{{features["exemplar_start_delimiter"]}}{{sentence}}{{features["exemplar_end_delimiter"]}}
{%- endfor %}
{%- endif %}
{% endfor %}

Please provide new {{name_kind}} names for each topic, following the JSON output format specified.
""")

    _cluster_system_program: ClassVar[jinja2.Template] = jinja2.Template("""
You are an expert at classifying {{document_type}} from {{corpus_description}} into topics.
Your task is to {{cluster_task}} for this group of {{document_type}} from {{corpus_description}}. The name should be {{name_kind}}.
The response must be {{cluster_response_description}}
A score of 1.0 means a perfectly descriptive and specific name, while 0.0 would be a completely generic or unrelated name.
{% if "very specific" in name_kind %}
Make every requested output precise, detailed, and distinguishing. The topic name should be specific to the information given and sufficiently detailed to distinguish it from similarly detailed topics. If analysis or summary fields are requested, apply the same specificity level to those fields.
{% elif "general" in name_kind %}
Make every requested output broad enough to capture the overall range at a glance. The topic name should be broad and simple enough to capture the overall sense of the large and diverse range of {{document_type}} contained in it. If analysis or summary fields are requested, apply the same generality level to those fields.
{% endif %}
{% if cluster_subtopics["major"] %}
When major subtopics are present, primarily make use of the major and minor subtopics, and ensure each generated topic name reflects the core essence of *all* major subtopics.
{% endif %}
Ensure your entire response is only the JSON object, with no other text before or after it.
{% if summary_kind %}
If a summary is requested, its form should be {{summary_kind}}.
{% endif %}
Keep all JSON string values on a single line (escape any newlines as \\n).
""")

    _cluster_user_program: ClassVar[jinja2.Template] = jinja2.Template("""
Here is the information about the group of {{document_type}}:
{% if cluster_keywords %}
- Keywords for this group include: {{", ".join(cluster_keywords)}}
{% endif %}
{%- if cluster_subtopics["major"] %}
- Major subtopics of this group are:
{%- for subtopic in cluster_subtopics["major"] %}
{{subtopic_start}}{{subtopic}}{{subtopic_end}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["minor"] %}
- Minor subtopics of this group are:
{%- for subtopic in cluster_subtopics["minor"] %}
{{subtopic_start}}{{subtopic}}{{subtopic_end}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["misc"] %}
- Other miscellaneous detailed subtopics of this group in order of relevance (from most to least) include:
{%- for subtopic in cluster_subtopics["misc"] %}
{{subtopic_start}}{{subtopic}}{{subtopic_end}}
{%- endfor %}
{%- endif %}
{%- if cluster_sentences %}
- Sample {{document_type}} from this group include:
{%- for sentence in cluster_sentences %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}

Based on this information, {{user_request}} for this group of {{document_type}}. The name should be {{name_kind}}.
Recall that the response must be {{cluster_response_description}}
""")

    def _add_template_features(
        self, features: dict[str, Any], name_kind: str
    ) -> dict[str, Any]:
        context = dict(features)
        subtopics = features.get("cluster_subtopics") or {}
        if isinstance(subtopics, list):
            subtopics = {"major": subtopics}
        context.update(
            document_type=self.document_type,
            corpus_description=self.corpus_description,
            name_kind=name_kind,
            subtopic_start=self.subtopic_start,
            subtopic_end=self.subtopic_end,
            cluster_keywords=features.get("cluster_keywords") or [],
            cluster_subtopics={
                key: subtopics.get(key) or [] for key in ("major", "minor", "misc")
            },
            cluster_sentences=features.get("cluster_sentences") or [],
            cluster_task=self.cluster_task,
            cluster_response_description=self.cluster_response_description,
            user_request=self.user_request,
            summary_kind=getattr(self, "summary_kind", None),
        )
        if context["summary_kind"]:
            context["cluster_task"] = context["cluster_task"].replace(
                "a short paragraph", context["summary_kind"]
            )
            context["cluster_response_description"] = context[
                "cluster_response_description"
            ].replace("a short paragraph", context["summary_kind"])
        context.setdefault("exemplar_start_delimiter", '    * "')
        context.setdefault("exemplar_end_delimiter", '"\n')
        return context

    def _disambiguation_context(
        self, names: list[str], features: list[dict[str, Any]], name_kind: str
    ) -> dict[str, Any]:
        if len(names) != len(features):
            raise ValueError("Names and features must have the same length")
        contexts = [
            self._add_template_features(feature, name_kind) for feature in features
        ]
        return {
            "corpus_description": self.corpus_description,
            "document_type": self.document_type,
            "name_kind": name_kind,
            "features_list": contexts,
            "feature_names": list(zip(contexts, names)),
        }

    def disambiguate_prompt(
        self,
        names: list[str],
        features: list[dict[str, Any]],
        name_kind: str,
    ) -> Prompt:
        context = self._disambiguation_context(names, features, name_kind)
        return Prompt(
            self._disambiguation_system_program.render(**context),
            self._disambiguation_user_program.render(**context),
            _disambiguation_schema(len(names)),
        )

    @staticmethod
    def extract_disambiguated_names(response: str) -> list[str]:
        return extract_response(response, topic_name_mapping)

    def cluster_prompt(self, features: dict[str, Any], name_kind: str) -> Prompt:
        context = self._add_template_features(features, name_kind)
        return Prompt(
            self._cluster_system_program.render(**context),
            self._cluster_user_program.render(**context),
            _name_schema(self),
        )

    @staticmethod
    def extract_name(response: str) -> TopicNameResult:
        return extract_response(
            response, lambda value: topic_fields(value, "topic_name")[0]
        )


@dataclass
class MultilingualENFRTemplate(TextTemplate):
    cluster_response_description: ClassVar[str] = (
        'in JSON formatted as {"english_topic_name":<NAME>, '
        '"nom_du_sujet_en_français":<NOM>, "topic_specificity":<SCORE>}\n'
        "where NAME is the English topic name, NOM is the French topic name, and "
        "SCORE is a float value between 0.0 and 1.0,\nrepresenting how specific and "
        "well-defined the topic name is given the input information."
    )

    @staticmethod
    def extract_name(response: str) -> str:
        return extract_response(
            response,
            lambda value: " / ".join(
                topic_fields(value, "english_topic_name", "nom_du_sujet_en_français")
            ),
        )


@dataclass
class SummaryTemplate(TextTemplate):
    summary_kind: str = "a short paragraph"

    cluster_task: ClassVar[str] = (
        "analyze the provided group information and provide a thorough analysis of "
        "the topic,\na short paragraph summary, and a name"
    )
    cluster_response_description: ClassVar[str] = (
        'in JSON formatted as {"topic_analysis":<ANALYSIS>, '
        '"topic_summary":<SUMMARY>, "topic_name":<NAME>, '
        '"topic_specificity":<SCORE>}\n'
        "where ANALYSIS is a thorough analytical discussion of the topic's content, "
        "key themes, sub-areas, and relationships\n(written to inform the summary "
        "and name that follow), SUMMARY is a short paragraph summary of the topic,\n"
        "NAME is the topic name you generate, and SCORE is a float value between "
        "0.0 and 1.0,\nrepresenting how specific and well-defined the topic name is "
        "given the input information."
    )
    user_request: ClassVar[str] = (
        "provide a detailed topic analysis, a summary, and a name"
    )

    @staticmethod
    def extract_name(response: str) -> tuple[str, str, str]:
        name, summary, analysis = extract_response(
            response,
            lambda value: topic_fields(
                value, "topic_name", "topic_summary", "topic_analysis"
            ),
        )
        return name, summary, analysis


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _name_schema(template: TextTemplate) -> dict[str, Any]:
    fields = ["topic_name"]
    if isinstance(template, SummaryTemplate):
        fields = ["topic_analysis", "topic_summary", "topic_name"]
    elif isinstance(template, MultilingualENFRTemplate):
        fields = ["english_topic_name", "nom_du_sujet_en_français"]
    properties: dict[str, dict[str, str | int]] = {
        field: {"type": "string"} for field in fields
    }
    properties["topic_specificity"] = {"type": "number", "minimum": 0, "maximum": 1}
    return _object_schema(properties)


def _disambiguation_schema(count: int) -> dict[str, Any]:
    return _object_schema(
        {
            "new_topic_name_mapping": _object_schema(
                {str(index): {"type": "string"} for index in range(1, count + 1)}
            ),
            "topic_specificities": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": count,
                "maxItems": count,
            },
        }
    )


def default_extract_topic_names(json_response, old_names, topic_name_info_raw=None):
    """Compatibility parser for the legacy provider callback signature."""
    names = topic_name_mapping(json_response)
    if len(names) != len(old_names):
        raise ResponseParseError("Response must contain one name per input topic")
    return names
