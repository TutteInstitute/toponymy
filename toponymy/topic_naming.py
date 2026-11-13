from dataclasses import dataclass, field
from typing import Any, Dict

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)


from toponymy.annotation import AnnotationTree, DescendantsInput
from toponymy.llm_wrappers import (
    TopicParseResult,
    parse_raw_response,
    LLMWrapper,
    AsyncLLMWrapper,
    _should_retry,
)
from toponymy.prompt_construction import topic_name_prompt_by_node
from toponymy.templates import PROMPT_TEMPLATES, SUMMARY_KINDS
import numpy as np


def topic_name_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(
            multiplier=1,
            min=4,
            max=10,
        ),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )


def summary_kind(node, tree, lowest_detail_level=0.0, highest_detail_level=1.0):
    detail_levels = np.linspace(
        lowest_detail_level, highest_detail_level, tree.n_layers
    )
    summary_level = int(round(detail_levels[node.layer] * (len(SUMMARY_KINDS) - 1)))
    summary_kind = SUMMARY_KINDS[summary_level]
    return summary_kind


@dataclass
class TopicNamerTemplate:
    object_description: str
    corpus_description: str
    tree: AnnotationTree
    exemplar_start_delimiter: str = ""
    exemplar_end_delimiter: str = ""
    prompt_template: Dict[str, Any] = field(
        default_factory=lambda: PROMPT_TEMPLATES.copy()
    )
    max_num_keyphrases: int = 32
    max_num_subtopics: int = 16
    max_num_exemplars: int = 128

    """
    TODO: 
    * Sort out how to handle summary kind since it requires the tree to determine the level. Could just take n_layers?
    """

    def render(
        self,
        node,
        *,
        major_subtopics=None,
        minor_subtopics=None,
        other_subtopics=None,
        exemplars=None,
        keyphrases=None,
    ) -> str:
        exemplars = exemplars or []
        keyphrases = keyphrases or []
        major_subtopics = major_subtopics or []
        minor_subtopics = minor_subtopics or []
        other_subtopics = other_subtopics or []

        prompt = topic_name_prompt_by_node(
            node=node,
            major_subtopics=major_subtopics,
            minor_subtopics=minor_subtopics,
            other_subtopics=other_subtopics,
            current_exemplars=exemplars,
            current_keyphrases=keyphrases,
            object_description=self.object_description,
            corpus_description=self.corpus_description,
            summary_kind=summary_kind(node, self.tree),
            exemplar_start_delimiter=self.exemplar_start_delimiter,
            exemplar_end_delimiter=self.exemplar_end_delimiter,
            prompt_template=self.prompt_template,
        )

        return prompt

    def parse(
        self,
        topic_info_raw: str,
        null_response_value: str = "",
    ) -> TopicParseResult:
        get_topic_name_regex = self.prompt_template["layer"].get("get_topic_name_regex")
        extract_topic_name_function = self.prompt_template["layer"].get(
            "extract_topic_name"
        )
        return parse_raw_response(
            topic_info_raw,
            extract_topic_name_function,
            get_topic_name_regex,
            null_response_value,
        )


class TopicNameAnnotator:
    """
    TODO IGNORE FOR NOW:
    * where does max_tokens (the kind of call?) and temperature (the llm?) come from?
    """

    inputs = []
    optional_inputs = [
        "exemplars",
        "output_topics",
        "subtopics",
        "keyphrases",
    ]  # use it if it's there. don't if not.
    outputs = ["draft_topics"]
    algorithm_input_type = {
        "exemplars": "node",
        "output_topics": DescendantsInput(from_output="draft_topics"),
        "subtopics": "node",
        "keyphrases": "node",
    }
    algorithm_output_type = "node"
    execution_mode: str

    def __init__(self, llm_namer=None, topic_name_template=None):
        self.llm_namer = llm_namer
        self.topic_name_template = topic_name_template
        if isinstance(self.llm_namer, LLMWrapper):
            self.execution_mode = "sync"
        elif isinstance(self.llm_namer, AsyncLLMWrapper):
            self.execution_mode = "async"
        else:
            raise TypeError("llm_namer must be an LLMWrapper or AsyncLLMWrapper")

    def _build_prompt(
        self,
        node,
        *,
        exemplars=None,
        output_topics=None,
        subtopics=None,
        keyphrases=None,
    ):
        if exemplars is None:
            exemplars = []
        if keyphrases is None:
            keyphrases = []
        if (output_topics is not None) and (
            len(output_topics) > 0
        ):  # slightly different logic here still
            major_subtopics = [
                output_topic
                for child, output_topic in output_topics.items()
                if child.layer == node.layer - 1
            ]
            minor_subtopics = [
                output_topic
                for child, output_topic in output_topics.items()
                if child.layer == node.layer - 2
            ]

            # promote minor_subtopics to major_subtopics if there are not enough children one layer down
            if len(major_subtopics) <= 1:
                major_subtopics = major_subtopics + minor_subtopics
                minor_subtopics = [
                    output_topic
                    for child, output_topic in output_topics.items()
                    if child.layer < node.layer - 2
                ]

            if node.layer > 1:
                other_subtopics = subtopics
            else:
                other_subtopics = []
        else:
            major_subtopics = []
            minor_subtopics = []
            other_subtopics = []

        prompt = self.topic_name_template.render(
            node=node,
            major_subtopics=major_subtopics,
            minor_subtopics=minor_subtopics,
            other_subtopics=other_subtopics,
            exemplars=exemplars,
            keyphrases=keyphrases,
        )
        return prompt

    def _implied_topic(self, output_topics):
        """
        If only has one child, use the child's topic.
        """
        if (output_topics is not None) and (
            len(output_topics) > 0
        ):  # slightly different logic here still
            if len(output_topics) == 1 and (list(output_topics.values())[0] != ""):
                return list(output_topics.values())[0]
        return None

    @topic_name_retry()
    def _generate_and_parse(self, prompt):
        """
        Effectively _generate_topic_name_with_raw_response
        """
        raw_response = self.llm_namer._call_llm_for_prompt(
            prompt,
            temperature=0.4,
            routine="TopicNameAnnotator.annotate",
            max_tokens=None,
        )
        return self.topic_name_template.parse(raw_response)

    @topic_name_retry()
    async def _agenerate_and_parse(self, prompt):
        """
        Effectively single version of _generate_topic_names_with_raw_response
        """
        raw_response = await self.llm_namer._acall_llm_for_prompt(
            prompt,
            temperature=0.4,
            routine="TopicNameAnnotator.aannotate",
            max_tokens=None,
        )
        return self.topic_name_template.parse(raw_response)

    def annotate(
        self,
        node,
        *,
        exemplars=None,
        output_topics=None,
        subtopics=None,
        keyphrases=None,
    ):
        if exemplars is None:
            exemplars = []
        if keyphrases is None:
            keyphrases = []

        implied = self._implied_topic(output_topics)
        if implied is not None:
            return {"draft_topics": implied}

        prompt = self._build_prompt(
            node,
            exemplars=exemplars,
            output_topics=output_topics,
            subtopics=subtopics,
            keyphrases=keyphrases,
        )

        if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
            topic_name = self._generate_and_parse(prompt).topic_payload
        else:
            topic_name = prompt.removeprefix("[!SKIP!]: ")

        if topic_name == "":
            raise ValueError("Failed to generate a valid topic name")

        return {"draft_topics": topic_name}  # keys have to match outputs

    async def aannotate(
        self,
        node,
        *,
        exemplars=None,
        output_topics=None,
        subtopics=None,
        keyphrases=None,
    ):
        if exemplars is None:
            exemplars = []
        if keyphrases is None:
            keyphrases = []

        implied = self._implied_topic(output_topics)
        if implied is not None:
            return {"draft_topics": implied}

        prompt = self._build_prompt(
            node,
            exemplars=exemplars,
            output_topics=output_topics,
            subtopics=subtopics,
            keyphrases=keyphrases,
        )

        if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
            result = await self._agenerate_and_parse(prompt)
            topic_name = result.topic_payload
        else:
            topic_name = prompt.removeprefix("[!SKIP!]: ")

        if topic_name == "":
            raise ValueError("Failed to generate a valid topic name")
        return {"draft_topics": topic_name}  # keys have to match outputs
