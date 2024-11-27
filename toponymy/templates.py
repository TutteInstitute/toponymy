import jinja2


_PROMPT_TEMPLATES = {
    "remedy": jinja2.Template(
        """
You are an expert in {{larger_topic}} and have been asked to provide a more specific name for a group of 
{{document_type}} from {{corpus_description}}. The group of {{document_type}} has been described as having a topic of one of 
{{attempted_topic_names}}. These topic names were not specific enough.

The other groups of {{document_type}} that can be confused with this topic are:

{% for topic in matching_topics  %}
{{topic}}:
 - Keywords: {{", ".join(matching_topic_keywords[topic])}}
{% if matching_topic_subtopics[topic] %}
 - Subtopics: {{", ".join(matching_topic_subtopics[topic])}}
{% endif %}
 - Sample {{document_type}}:
{% for sentence in matching_topic_sentences[topic] %}
      - "{{sentence}}"
{% endfor %}
{% endfor %}

As an expert in {{larger_topic}}, you need to provide a more specific name for this group of {{document_type}}:
 - Keywords: {{", ".join(cluster_keywords)}}
{% if cluster_subtopics %}
 - Subtopics: {{", ".join(cluster_subtopics)}}
{% endif %}
 - Sample {{document_type}}:
{% for sentence in cluster_sentences %}
      - "{{sentence}}"
{% endfor %}

You should make use of the relative relationships between these topics as well as the keywords
and {{self.document_type}} information and your expertise in {{larger_topic}} to generate new 
better and more *specific* topic name.

The response should be only JSON with no preamble formatted as 
  {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} 
where SCORE is a value in the range 0 to 1.
The response must contain only JSON with no preamble.
"""
    ),
    "distinguish_base_layer_topics": jinja2.Template(
        """
You are an expert in {{larger_topic}} and have been asked to provide a more specific names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Below are the auto-generated topic names, along with some keywords associated to each topic, and a sampling of {self.document_type} from the topic area.

{% for topic, keywords, sentences in base_layer_topic_data %}
"{{topic}}":
 - Keywords: {{", ".join(keywords)}}
 - Sample {{document_type}}:
{% for sentence in sentences %}
      - "{{sentence}}"
{% endfor %}
{% endfor %}

Your should make use of the relative relationships between these topics as well as the keywords
and {{self.document_type}} information and your expertise in {{larger_topic}} to generate new
better and more distinguishing topic names.

The result should be formatted as JSON in the format 
  [{<OLD_TOPIC_NAME1>: <NEW_TOPIC_NAME>}, {<OLD_TOPIC_NAME2>: <NEW_TOPIC_NAME>}, ...]
The result must contain only JSON with no preamble and must have one entry for each topic to be renamed.
"""
    ),
}
