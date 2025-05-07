import jinja2


SUMMARY_KINDS = [
    "domain expert level (8 to 15 word)",
    "very specific and detailed (6 to 12 word)",
    "specific and detailed (4 to 8 word)",
    "clear and concise (3 to 6 word)",
    "focussed and brief (2 to 5 word)",
    "essential and core (1 to 4 word)",
    "simple (1 or 2 word)",
]

GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*[\w.]+\s*\}'
GET_TOPIC_CLUSTER_NAMES_REGEX = r'\{\s*"new_topic_name_mapping":\s*.*?,\s*"topic_specificities": .*?\}'

PROMPT_TEMPLATES = {
    "layer": {
        "system": jinja2.Template(
            """
You are an expert at classifying {{document_type}} from {{corpus_description}} into topics.
Your task is to analyze information about a group of {{document_type}} and assign a {{summary_kind}} name to this group.
The response must be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>}
where NAME is the topic name you generate and SCORE is a float value between 0.0 and 1.0,
representing how specific and well-defined the topic name is given the input information.
A score of 1.0 means a perfectly descriptive and specific name, while 0.0 would be a completely generic or unrelated name.
{% if is_very_specific_summary %}
The topic name should be specific to the information given and sufficiently detailed to ensure
it can be distinguished from other similarly detailed topics.
{% elif is_general_summary %}
The topic name should be broad and simple enough to capture the overall sense of the
large and diverse range of {{document_type}} contained in it at a glance.
{% endif %}
{% if has_major_subtopics %}
You should primarily make use of the major and minor subtopics of this group to generate a name,
and ensure the topic name reflects the core essence of *all* major subtopics.
{% endif %}
Ensure your entire response is only the JSON object, with no other text before or after it.
"""
        ),
        "user": jinja2.Template(
            """
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
"""
        ),
        "combined": jinja2.Template(
        """
You are an expert of classifying {{document_type}} from {{corpus_description}} into topics.
Below is a information about a group of {{document_type}} from {{corpus_description}} that 
are all on the same topic and need to be given topic name.

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

You are to give a {{summary_kind}} name to this group of {{document_type}}.
{% if has_major_subtopics -%}
You should primarily make use of the major and minor subtopics of this group to generate a name, 
and ensure the topic name covers *all* of the major subtopics.
{%- endif %}
{% if is_very_specific_summary -%}
The topic name should be specific to the information given and sufficiently detailed to ensure 
it can be distinguished from other similar detailed topics.
{% elif is_general_summary -%}
The topic name should be broad and simple enough to capture of overall sense of the 
large and diverse range of {{document_type}} contained in it at a glance.
{%- endif %}
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} 
where SCORE is a value in the range 0 to 1.
"""
        ),
      },
    "disambiguate_topics": {
        "system": jinja2.Template(
            """
You are an expert in {{larger_topic}}. You have been asked to provide more specific and distinguishing names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Your task is to generate a new {{summary_kind}} name for each topic group presented.
You should make use of the relative relationships between these topics, their keywords, subtopic information, and sample {{document_type}} to generate new, distinct topic names.
The new names must be in the same order as the original topics are presented.
There should be no duplicate topic names in your final list of new names.

{% if is_very_specific_summary %}
Each new topic name should be specific to the information of that topic and sufficiently detailed to ensure it can be distinguished from all the other similar topics listed.
{% elif is_general_summary %}
Each topic name should be broad and simple enough to capture the overall sense of the large and diverse range of {{document_type}} contained in it at a glance, while still separating it from the other topics listed.
{% endif %}
{% if has_major_subtopics %}
For each topic, you should primarily make use of its major and minor subtopics to generate a name, and ensure the new topic name reflects the core essence of *all* of its major subtopics.
{% endif %}

The response must be formatted as a single JSON object in the format:
{"new_topic_name_mapping": {"1. OLD_NAME1": "NEW_NAME1", "2. OLD_NAME2": "NEW_NAME2", ... }, "topic_specificities": [NEW_TOPIC_SCORE1, NEW_TOPIC_SCORE2, ...]}
where SCORE is a float value between 0.0 and 1.0 representing the quality and specificity of the new name.
Ensure your entire response is only the JSON object, with no other text before or after it.
"""
        ),
        "user": jinja2.Template(
            """
Below are the auto-generated topic names, along with keywords, subtopics, and sample {{document_type}} for each topic area.

Original larger topic context: {{larger_topic}}
Corpus description: {{corpus_description}}

{% for topic in topics %}
"{{loop.index}}. {{topic}}":
{% if cluster_keywords[loop.index - 1] %}
- Keywords for this group include: {{", ".join(cluster_keywords[loop.index - 1])}}
{% endif %}
{%- if cluster_subtopics["major"][loop.index - 1] %}
- Major subtopics of this group are:
{%- for subtopic in cluster_subtopics["major"][loop.index - 1] %}
    * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["minor"][loop.index - 1] %}
- Minor subtopics of this group are:
{%- for subtopic in cluster_subtopics["minor"][loop.index - 1] %}
    * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["misc"][loop.index - 1] %}
- Other miscellaneous specific subtopics of this group in order of relevance (from most to least) include:
{%- for subtopic in cluster_subtopics["misc"][loop.index - 1] %}
    * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_sentences[loop.index - 1] %}
- Sample {{document_type}} from this group include:
{%- for sentence in cluster_sentences[loop.index - 1] %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}
{% endfor %}

Please provide new {{summary_kind}} names for each topic, following the JSON output format specified.
"""
        ),
        "combined": jinja2.Template(
        """
You are an expert in {{larger_topic}}, and have been asked to provide a more specific names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Below are the auto-generated topic names, along with some keywords associated to each topic, and a sampling of {{document_type}} from the topic area.

{% for topic in topics %}

"{{loop.index}}. {{topic}}":
{% if cluster_keywords[loop.index - 1] %}
 - Keywords for this group include: {{", ".join(cluster_keywords[loop.index - 1])}}
{% endif %}
{%- if cluster_subtopics["major"][loop.index - 1] %}
 - Major subtopics of this group are: 
{%- for subtopic in cluster_subtopics["major"][loop.index - 1] %}
      * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["minor"][loop.index - 1] %}
 - Minor subtopics of this group are:
{%- for subtopic in cluster_subtopics["minor"][loop.index - 1] %}
      * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_subtopics["misc"][loop.index - 1] %}
 - Other miscellaneous specific subtopics of this group in order of relevance (from most to least) include:
{%- for subtopic in cluster_subtopics["misc"][loop.index - 1] %}
      * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_sentences[loop.index - 1] %}
 - Sample {{document_type}} from this group include:
{%- for sentence in cluster_sentences[loop.index - 1] %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}
{% endfor %}

You are to give a new {{summary_kind}} name to each topic.
You should make use of the relative relationships between these topics as well as the keywords and {{self.document_type}} information and your expertise in {{larger_topic}} to generate new better and more distinguishing topic names. 
{% if cluster_subtopics["major"] -%}
You should primarily make use of the major and minor subtopics of each topic to generate a name, and ensure the new topic name covers *all* of the major subtopics.
{%- endif %}
{% if "very specific" in summary_kind -%}
Each new topic name should be specific to the information of that topic and sufficiently detailed to ensure  it can be distinguished from all the other similar topics listed.
{% elif "general" in summary_kind -%}
Each topic name should be broad and simple enough to capture of overall sense of the large and diverse range of {{document_type}} contained in it at a glance, while still separating it from the other topics listed.
{%- endif %}
The new names must be in the same order as presented above. There should be no duplicate topic names in the final list. The primary goal is to make each new topic name clearly distinguishable from the others in this list, based on the provided details.

The response should be formatted as JSON in the format 
    {"new_topic_name_mapping": {<1. OLD_NAME1>: <NEW_NAME1>, <2. OLD_NAME2>: <NEW_NAME2>, ... }, topic_specificities": [<NEW_TOPIC_SCORE1>, <NEW_TOPIC_SCORE2>, ...]}
where SCORE is a value in the range 0 to 1.
The response must contain only JSON with no preamble and must have one entry for each topic to be renamed.
"""
      ),
    },
}
