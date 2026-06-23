import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import jinja2

GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*[\w.]+\s*\}'
GET_TOPIC_NAME_AND_SUMMARY_REGEX = (
    r'\{\s*"topic_analysis":\s*.*?,\s*"topic_summary":\s*.*?,'
    r'\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*[\w.]+\s*\}'
)
GET_MULTILINGUAL_EN_FR_TOPIC_NAME_REGEX = (
    r'\{\s*"english_topic_name":\s*.*?,'
    r'\s*"nom_du_sujet_en_fran(?:\u00e7|\\u00e7)ais":\s*.*?,'
    r'\s*"topic_specificity":\s*[\w.]+\s*\}'
)
GET_TOPIC_CLUSTER_NAMES_REGEX = (
    r'\{\s*"new_topic_name_mapping":\s*.*?,\s*"topic_specificities": .*?\}'
)


class Prompt(NamedTuple):
    system: str
    user: str


class Template(ABC):
    @abstractmethod
    def cluster_prompt(self, features: dict[str, Any], name_kind: str) -> Prompt:
        pass

    @abstractmethod
    def extract_name(self, response: str) -> Any:
        pass

    @abstractmethod
    def disambiguate_prompt(
        self,
        names: list[str],
        features: list[dict[str, Any]],
        name_kind: str,
    ) -> Prompt:
        pass

    @abstractmethod
    def extract_disambiguated_names(self, response: str) -> list[str]:
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

    def _add_template_features(
        self,
        features: dict[str, Any],
        name_kind: str,
    ) -> dict[str, Any]:
        features["document_type"] = self.document_type
        features["corpus_description"] = self.corpus_description
        features["name_kind"] = name_kind
        features["subtopic_start"] = self.subtopic_start
        features["subtopic_end"] = self.subtopic_end
        features["cluster_keywords"] = features.get("cluster_keywords") or []
        features["cluster_subtopics"] = features.get("cluster_subtopics") or {
            "major": [],
            "minor": [],
            "misc": [],
        }
        features["cluster_subtopics"]["major"] = (
            features["cluster_subtopics"].get("major") or []
        )
        features["cluster_subtopics"]["minor"] = (
            features["cluster_subtopics"].get("minor") or []
        )
        features["cluster_subtopics"]["misc"] = (
            features["cluster_subtopics"].get("misc") or []
        )
        features["cluster_sentences"] = features.get("cluster_sentences") or []
        features.setdefault("exemplar_start_delimiter", '    * "')
        features.setdefault("exemplar_end_delimiter", '"\n')
        features["cluster_task"] = self.cluster_task
        features["cluster_response_description"] = self.cluster_response_description
        features["user_request"] = self.user_request
        return features

    def _disambiguation_context(
        self,
        names: list[str],
        features: list[dict[str, Any]],
        name_kind: str,
    ) -> dict[str, Any]:
        for feature in features:
            self._add_template_features(feature, name_kind)
        return {
            "corpus_description": self.corpus_description,
            "document_type": self.document_type,
            "name_kind": name_kind,
            "features_list": features,
            "feature_names": list(zip(features, names)),
        }

    def disambiguate_prompt(
        self,
        names: list[str],
        features: list[dict[str, Any]],
        name_kind: str,
    ) -> Prompt:
        system_prompt = jinja2.Template("""
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
        user_prompt = jinja2.Template("""
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
        context = self._disambiguation_context(names, features, name_kind)
        return Prompt(system_prompt.render(**context), user_prompt.render(**context))

    def extract_disambiguated_names(self, response: str) -> list[str]:
        try:
            response_json = _json_from_response(response, GET_TOPIC_CLUSTER_NAMES_REGEX)
            mapping = response_json["new_topic_name_mapping"]
            return [mapping[key] for key in sorted(mapping, key=_numeric_mapping_key)]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            mapping_match = re.findall(
                r'"new_topic_name_mapping":\s*\{(.*?)\}',
                response,
                re.DOTALL,
            )
            if mapping_match:
                new_names = [
                    name
                    for _, name in sorted(
                        (
                            (_numeric_mapping_key(index), name)
                            for index, name in re.findall(
                                r'"\s*(\d+)(?:\.[^"]*)?\s*":\s*"(.*?)",?',
                                mapping_match[0],
                                re.DOTALL,
                            )
                        )
                    )
                ]
                if new_names:
                    return new_names
            raise ValueError(
                f"Failed to extract disambiguated topic names from response: {response}"
            )

    def cluster_prompt(self, features: dict[str, Any], name_kind: str) -> Prompt:
        system_prompt = jinja2.Template("""
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
Keep all JSON string values on a single line (escape any newlines as \\n).
""")
        user_prompt = jinja2.Template("""
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
        context = self._add_template_features(features, name_kind)
        return Prompt(system_prompt.render(**context), user_prompt.render(**context))

    def extract_name(self, response: str) -> str:
        try:
            response_json = _json_from_response(response, GET_TOPIC_NAME_REGEX)
            return str(response_json["topic_name"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            match = re.search(r'"topic_name"\s*:\s*"(.*?)"', response, re.DOTALL)
            if match:
                return match.group(1)
            raise ValueError(f"Failed to extract topic name from response: {response}")


@dataclass
class MultilingualENFRTemplate(TextTemplate):
    cluster_response_description: ClassVar[str] = (
        'in JSON formatted as {"english_topic_name":<NAME>, '
        '"nom_du_sujet_en_fran\u00e7ais":<NOM>, "topic_specificity":<SCORE>}\n'
        "where NAME is the English topic name, NOM is the French topic name, and "
        "SCORE is a float value between 0.0 and 1.0,\nrepresenting how specific and "
        "well-defined the topic name is given the input information."
    )

    def extract_name(self, response: str) -> str:
        try:
            response_json = _json_from_response(
                response,
                GET_MULTILINGUAL_EN_FR_TOPIC_NAME_REGEX,
            )
            french_name = response_json["nom_du_sujet_en_fran\u00e7ais"]
            return f'{response_json["english_topic_name"]} / {french_name}'
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            english_match = re.search(
                r'"english_topic_name"\s*:\s*"(.*?)"',
                response,
                re.DOTALL,
            )
            french_match = re.search(
                r'"nom_du_sujet_en_fran(?:\u00e7|\\u00e7)ais"\s*:\s*"(.*?)"',
                response,
                re.DOTALL,
            )
            if english_match and french_match:
                return f"{english_match.group(1)} / {french_match.group(1)}"
            raise ValueError(f"Failed to extract topic name from response: {response}")


@dataclass
class SummaryTemplate(TextTemplate):
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

    def extract_name(self, response: str) -> tuple[str, str, str]:
        try:
            response_json = _json_from_response(
                response,
                GET_TOPIC_NAME_AND_SUMMARY_REGEX,
            )
            return (
                response_json["topic_name"],
                response_json["topic_summary"],
                response_json["topic_analysis"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            name_match = re.search(r'"topic_name"\s*:\s*"(.*?)"', response, re.DOTALL)
            summary_match = re.search(
                r'"topic_summary"\s*:\s*"(.*?)"',
                response,
                re.DOTALL,
            )
            analysis_match = re.search(
                r'"topic_analysis"\s*:\s*"(.*?)"',
                response,
                re.DOTALL,
            )
            if name_match and summary_match and analysis_match:
                return (
                    name_match.group(1),
                    summary_match.group(1),
                    analysis_match.group(1),
                )
            raise ValueError(
                f"Failed to extract topic summary from response: {response}"
            )


def _json_from_response(response: str, regex: str) -> dict[str, Any]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        matches = re.findall(regex, response, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        raise


def _numeric_mapping_key(key: Any) -> int:
    match = re.match(r"\s*(\d+)", str(key))
    if not match:
        raise ValueError(f"Mapping key does not start with a numeric index: {key}")
    return int(match.group(1))
