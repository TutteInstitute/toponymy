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

GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?, "topic_specificity":\s*\d+\.\d+\s*\}'
GET_TOPIC_CLUSTER_NAMES_REGEX = r'\{\s*"new_topic_name_mapping":\s*.*?, "topic_specificities": .*?\}'

PROMPT_TEMPLATES = {
    "layer": jinja2.Template(
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
 - Other miscellaneous detailed subtopics of this group include:
{%- for subtopic in cluster_subtopics["misc"] %}
      * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_sentences %}
 - Sample {{document_type}} from this group include:
{%- for sentence in cluster_sentences %}
      * "{{sentence}}"
{%- endfor %}
{%- endif %}

You are to give a {{summary_kind}} name to this group of {{document_type}}.
{% if cluster_subtopics["major"] -%}
You should primarily make use of the major and minor subtopics of this group to generate a name, 
and ensure the topic name covers *all* of the major subtopics.
{%- endif %}
{% if "very specific" in summary_kind -%}
The topic name should be specific to the information given and sufficiently detailed to ensure 
it can be distinguished from other similar detailed topics.
{% elif "general" in summary_kind -%}
The topic name should be broad and simple enough to capture of overall sense of the 
large and diverse range of {{document_type}} contained in it at a glance.
{%- endif %}
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} 
where SCORE is a value in the range 0 to 1.
"""
    ),
    "remedy": jinja2.Template(
        """
You are an expert in {{larger_topic}} and have been asked to provide a more specific name for a group of 
{{document_type}} from {{corpus_description}}. The group of {{document_type}} has been described as having a topic of one of 
{{attempted_topic_names}}. These topic names were not specific enough.

The other groups of {{document_type}} that can be confused with this topic are:

{% for topic in matching_topics  %}
{{topic}}:
{%- if matching_topic_keywords[topic] %}
 - Keywords: {{", ".join(matching_topic_keywords[topic])}}
{%- endif %}
{%- if matching_topic_subtopics[topic] %}
 - Subtopics: {{", ".join(matching_topic_subtopics[topic])}}
{%- endif %}
{%- if matching_topic_sentences[topic] %}
 - Sample {{document_type}}:
{%- for sentence in matching_topic_sentences[topic] %}
      * "{{sentence}}"
{%- endfor %}
{%- endif %}
{%- endfor %}

As an expert in {{larger_topic}}, you need to provide a more specific name for this group of {{document_type}}:
{%- if cluster_keywords %}
 - Keywords: {{", ".join(cluster_keywords)}}
{%- endif %}
{%- if cluster_subtopics %}
 - Subtopics: {{", ".join(cluster_subtopics)}}
{%- endif %}
{%- if cluster_sentences %}
 - Sample {{document_type}}:
{%- for sentence in cluster_sentences %}
      * "{{sentence}}"
{%- endfor %}
{%- endif %}

You should make use of the relative relationships between these topics as well as the keywords
and {{self.document_type}} information and your expertise in {{larger_topic}} to generate new 
better and more *specific* topic name.

The response should be only JSON with no preamble formatted as 
  {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} 
where SCORE is a value in the range 0 to 1.
The response must contain only JSON with no preamble.
"""
    ),
    "disambiguate_topics": jinja2.Template(
        """
You are an expert in {{larger_topic}}, and have been asked to provide a more specific names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Below are the auto-generated topic names, along with some keywords associated to each topic, and a sampling of {{self.document_type}} from the topic area.

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
 - Other miscellaneous specific subtopics of this group include:
{%- for subtopic in cluster_subtopics["misc"][loop.index - 1] %}
      * {{subtopic}}
{%- endfor %}
{%- endif %}
{%- if cluster_sentences[loop.index - 1] %}
 - Sample {{document_type}} from this group include:
{%- for sentence in cluster_sentences[loop.index - 1] %}
      * "{{sentence}}"
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
The new names must be in the same order as presented above. There should be no duplicate topic names in the final list.

The response should be formatted as JSON in the format 
    {"new_topic_name_mapping": {<1. OLD_NAME1>: <NEW_NAME1>, <2. OLD_NAME2>: <NEW_NAME2>, ... }, topic_specificities": [<NEW_TOPIC_SCORE1>, <NEW_TOPIC_SCORE2>, ...]}
where SCORE is a value in the range 0 to 1.
The response must contain only JSON with no preamble and must have one entry for each topic to be renamed.
"""
    ),
}
