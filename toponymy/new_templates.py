import jinja2
import json
import re
from templates import GET_TOPIC_NAME_REGEX, GET_TOPIC_CLUSTER_NAMES_REGEX
from typing import Any, NamedTuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Prompt(NamedTuple):
    system: str
    user: str


class Template(ABC):
    @abstractmethod
    def cluster_prompt(self, features: dict[str, Any]) -> Prompt:
        pass

    @abstractmethod
    def extract_name(self, response: str) -> str:
        pass

    @abstractmethod
    def disambiguate_prompt(
        self,
        name_a: str,
        features_a: dict[str, Any],
        name_b: str,
        features_b: dict[str, Any],
    ) -> Prompt:
        pass

    @abstractmethod
    def extract_disambiguated_names(self, response: str) -> tuple[str, str]:
        pass

@dataclass
class TextTemplate(Template):
    document_type: str
    corpus_description: str

    def cluster_prompt(self, features: dict[str, Any], name_kind: str) -> Prompt:
        SYSTEM_PROMPT = jinja2.Template("""
You are an expert at classifying {{document_type}} from {{corpus_description}} into topics.
Your task is to analyze information about a group of {{document_type}} and assign a {{summary_kind}} name to this group.
The response must be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>}
where NAME is the topic name you generate and SCORE is a float value between 0.0 and 1.0,
representing how specific and well-defined the topic name is given the input information.
A score of 1.0 means a perfectly descriptive and specific name, while 0.0 would be a completely generic or unrelated name.
{% if "very specific" in summary_kind %}
The topic name should be specific to the information given and sufficiently detailed to ensure
it can be distinguished from other similarly detailed topics.
{% elif "general" in summary_kind %}
The topic name should be broad and simple enough to capture the overall sense of the
large and diverse range of {{document_type}} contained in it at a glance.
{% endif %}
{% if cluster_subtopics["major"] %}
You should primarily make use of the major and minor subtopics of this group to generate a name,
and ensure the topic name reflects the core essence of *all* major subtopics.
{% endif %}
Ensure your entire response is only the JSON object, with no other text before or after it.
""")
        USER_PROMPT = jinja2.Template("""
Here is the information about the group of {{document_type}}:
{% if cluster_keywords %}
- Keywords for this group include: {{", ".join(cluster_keywords)}}
{% endif %}
{%- if cluster_subtopics["major"] %}
- Major subtopics of this group are:
{%- for subtopic in cluster_subtopics["major"] %}
  * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["minor"] %}
- Minor subtopics of this group are:
{%- for subtopic in cluster_subtopics["minor"] %}
  * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["misc"] %}
- Other miscellaneous detailed subtopics of this group in order of relevance (from most to least) include:
{%- for subtopic in cluster_subtopics["misc"] %}
  * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_sentences %}
- Sample {{document_type}} from this group include:
{%- for sentence in cluster_sentences %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}

Based on this information, provide a {{summary_kind}} name for this group.
Recall the output format: {"topic_name":<NAME>, "topic_specificity":<SCORE>}.
""")

        features["document_type"] = self.document_type
        features["corpus_description"] = self.corpus_description
        features["summary_kind"] = name_kind
        return Prompt(SYSTEM_PROMPT.render(**features), USER_PROMPT.render(**features))

    def disambiguate_prompt(
        self,
        names: list[str],
        features: list[dict[str, Any]],
        name_kind: str
    ) -> Prompt:
        SYSTEM_PROMPT = jinja2.Template("""
You are an expert in {{corpus_description}}. You have been asked to provide more specific and distinguishing names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Your task is to generate a new {{summary_kind}} name for each topic group presented.
You should make use of the relative relationships between these topics, their keywords, subtopic information, and sample {{document_type}} to generate new, distinct topic names.
The new names must be in the same order as the original topics are presented.
There should be no duplicate topic names in your final list of new names.

{% if "very specific" in summary_kind %}
Each new topic name should be specific to the information of that topic and sufficiently detailed to ensure it can be distinguished from all the other similar topics listed.
{% elif "general" in summary_kind %}
Each topic name should be broad and simple enough to capture the overall sense of the large and diverse range of {{document_type}} contained in it at a glance, while still separating it from the other topics listed.
{% endif %}
{% if cluster_subtopics["major"] | select | list %}
For each topic, you should primarily make use of its major and minor subtopics to generate a name, and ensure the new topic name reflects the core essence of *all* of its major subtopics.
{% endif %}

The response must be formatted as a single JSON object in the format:
{"new_topic_name_mapping": {"1": <NEW_NAME1>, "2": <NEW_NAME2>, ... }, "topic_specificities": [<NEW_TOPIC_SCORE1>, <NEW_TOPIC_SCORE2>, ...]}
where SCORE is a float value between 0.0 and 1.0 representing the quality and specificity of the new name.
Ensure your entire response is only the JSON object, with no other text before or after it.
""")
        USER_PROMPT = jinja2.Template("""
Below are the auto-generated topic names, along with keywords, subtopics, and sample {{document_type}} for each topic area.

Corpus description: {{corpus_description}}

{% for features, name in features_list | zip(names) %}
"{{loop.index}}. name":
{% if features["cluster_keywords"] %}
- Keywords for this group include: {{", ".join(features["cluster_keywords"])}}
{% endif %}
{%- if features["cluster_subtopics"]["major"] %}
- Major subtopics of this group are:
{%- for subtopic in features["cluster_subtopics"]["major"] %}
    * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if features["cluster_subtopics"]["minor"] %}
- Minor subtopics of this group are:
{%- for subtopic in features["cluster_subtopics"]["minor"] %}
    * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if features["cluster_subtopics"]["misc"] %}
- Other miscellaneous specific subtopics of this group in order of relevance (from most to least) include:
{%- for subtopic in features["cluster_subtopics"]["misc"] %}
    * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if features["cluster_sentences"] %}
- Sample {{document_type}} from this group include:
{%- for sentence in features["cluster_sentences"] %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}
{% endfor %}

Please provide new {{summary_kind}} names for each topic, following the JSON output format specified.
""")
        content = {
            "corpus_description": self.corpus_description,
            "document_type": self.document_type,
            "summary_kind": name_kind,
            "features_list": features,
            "names": names,
        }
        return Prompt(SYSTEM_PROMPT.render(**content), USER_PROMPT.render(**content))
    
    def extract_name(response: str) -> str:
        try:
            response_json = json.loads(
                re.findall(GET_TOPIC_NAME_REGEX, response, re.DOTALL)[0]
            )
            return str(response_json["topic_name"])
        except (IndexError, json.JSONDecodeError, KeyError):
            match = re.search(r'"topic_name"\s*:\s*"(.*?)"', response, re.DOTALL)
            if match:
                return match.group(1)
            raise ValueError(f"Failed to extract topic name from response: {response}")

    def extract_disambiguated_names(self, response: str) -> list[str]:
        try:
            response_json = json.loads(
                re.findall(GET_TOPIC_CLUSTER_NAMES_REGEX, response, re.DOTALL)[0]
            )
            mapping = response_json["new_topic_name_mapping"]
            return [
                mapping[key]
                for key in sorted(mapping, key=lambda key: int(str(key).rstrip(".")))
            ]
        except (IndexError, json.JSONDecodeError, KeyError, ValueError):
            mapping = re.findall(
                r'"new_topic_name_mapping":\s*\{(.*?)\}',
                response,
                re.DOTALL,
            )[0]
            new_names = [
                name
                for _, name in sorted(
                    (
                        (int(index), name)
                        for index, name in re.findall(
                            r'"\s*(\d+)\.?\s*":\s*"(.*?)",?',
                            mapping,
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

