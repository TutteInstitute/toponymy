import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List
from warnings import warn

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)


from toponymy.annotation import (
    AnnotationTree,
    DescendantsInput,
    GroupDescendantsInput,
    LayerExecutionStrategy,
    MultiNodeInput,
    Order,
)
from toponymy.llm_wrappers import (
    TopicParseResult,
    llm_output_to_result,
    parse_raw_response,
    LLMWrapper,
    AsyncLLMWrapper,
    _should_retry,
    FailFastLLMError,
    InvalidLLMInputError,
)
from toponymy.prompt_construction import (
    cluster_topic_names_for_renaming,
    distinguish_topic_names_prompt_for_group,
    topic_name_prompt_by_node,
)
from toponymy.templates import (
    GET_TOPIC_CLUSTER_NAMES_REGEX,
    PROMPT_TEMPLATES,
    SUMMARY_KINDS,
    default_extract_topic_names,
)
import numpy as np

NULL_RESPONSE_VALUE = ""


def _topic_name_error_callback(retry_state, routine="generate_topic_name"):
    """Callback function for when all retries are exhausted. Logs the error and returns an empty string."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, (FailFastLLMError, InvalidLLMInputError)):
        raise exc
    raw_response = getattr(exc, "raw_response", None)
    warn(f"All retries exhausted for {routine}: {type(exc).__name__}: {exc}")
    return TopicParseResult(
        topic_payload=NULL_RESPONSE_VALUE, error=exc, raw_response=raw_response
    )


def topic_name_retry(routine="generate_topic_name"):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(
            multiplier=1,
            min=4,
            max=10,
        ),
        retry=retry_if_exception(_should_retry),
        reraise=True,
        retry_error_callback=partial(_topic_name_error_callback, routine=routine),
    )


def embed_topic_names(embedding_model, topic_names):
    """Embed a mapping of node to topic name with one encode call; return node to vector."""
    nodes = list(topic_names)
    if not nodes:
        return {}
    vectors = embedding_model.encode(
        [topic_names[node] for node in nodes], show_progress_bar=False
    )
    return dict(zip(nodes, vectors))


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
    lowest_detail_level: float = 0.0
    highest_detail_level: float = 1.0

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
            summary_kind=summary_kind(
                node,
                self.tree,
                lowest_detail_level=self.lowest_detail_level,
                highest_detail_level=self.highest_detail_level,
            ),
            exemplar_start_delimiter=self.exemplar_start_delimiter,
            exemplar_end_delimiter=self.exemplar_end_delimiter,
            prompt_template=self.prompt_template,
        )

        return prompt

    def parse(
        self,
        topic_info_raw: str,
        null_response_value: str = NULL_RESPONSE_VALUE,
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

    def render_disambiguation(
        self,
        group,
        *,
        topic_names,
        exemplars,
        keyphrases,
        major_subtopics,
        minor_subtopics,
        other_subtopics,
    ) -> Dict[str, str]:
        """Render the prompt that renames a group of topics, given one list entry per topic."""
        return distinguish_topic_names_prompt_for_group(
            topic_names=topic_names,
            keyphrases=keyphrases,
            exemplar_texts=exemplars,
            major_subtopics=major_subtopics,
            minor_subtopics=minor_subtopics,
            other_subtopics=other_subtopics,
            object_description=self.object_description,
            corpus_description=self.corpus_description,
            summary_kind=summary_kind(
                group[0],
                self.tree,
                lowest_detail_level=self.lowest_detail_level,
                highest_detail_level=self.highest_detail_level,
            ),
            max_num_keyphrases=self.max_num_keyphrases,
            max_num_subtopics=self.max_num_subtopics,
            max_num_exemplars=self.max_num_exemplars,
            exemplar_start_delimiter=self.exemplar_start_delimiter,
            exemplar_end_delimiter=self.exemplar_end_delimiter,
            prompt_template=self.prompt_template,
        )

    def parse_disambiguation(
        self, raw_response: str, old_names: List[str]
    ) -> List[str]:
        """Parse the new names, in the order of ``old_names``; raise if any is missing."""
        section = self.prompt_template["disambiguate_topics"]
        regex = section.get("get_topic_names_regex", GET_TOPIC_CLUSTER_NAMES_REGEX)
        extract = section.get("extract_topic_names", default_extract_topic_names)
        new_names = extract(
            llm_output_to_result(raw_response, regex), old_names, raw_response
        )
        if len(new_names) != len(old_names) or not all(
            isinstance(name, str) and name for name in new_names
        ):
            raise ValueError(
                f"Expected {len(old_names)} nonempty names for {old_names}, got {new_names}"
            )
        return list(new_names)


class TopicNameAnnotator:
    """
    TODO IGNORE FOR NOW:
    * where does/should temperature (the llm?) come from?
    """

    inputs = []
    optional_inputs = [
        "exemplars",
        "output_topics",
        "subtopics",
        "keyphrases",
    ]  # use it if it's there. don't if not.
    outputs = ["topic_names", "topic_name_embeddings"]
    algorithm_input_type = {
        "exemplars": "node",
        # The final names of the layers below, even when this annotator writes drafts
        "output_topics": DescendantsInput(source="topic_names"),
        "subtopics": "node",
        "keyphrases": "node",
    }
    algorithm_output_type = "node"
    execution_mode: str
    execution_strategy = LayerExecutionStrategy(Order.BOTTOM_UP)
    max_tokens: int | None = None
    temperature: int | None = None

    def __init__(self, llm_namer, topic_name_template, embedding_model):
        self.llm_namer = llm_namer
        self.topic_name_template = topic_name_template
        self.embedding_model = embedding_model
        if isinstance(self.llm_namer, LLMWrapper):
            self.execution_mode = "sync"
        elif isinstance(self.llm_namer, AsyncLLMWrapper):
            self.execution_mode = "async"
        else:
            raise TypeError("llm_namer must be an LLMWrapper or AsyncLLMWrapper")
        if self.max_tokens is None:
            self.max_tokens = 128
        if self.temperature is None:
            self.temperature = 0.4

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

    def _outputs(self, node, topic_name):
        if topic_name == "":
            raise ValueError("Failed to generate a valid topic name")
        embedding = embed_topic_names(self.embedding_model, {node: topic_name})[node]
        return {"topic_names": topic_name, "topic_name_embeddings": embedding}

    @topic_name_retry()
    def _generate_and_parse(self, prompt):
        """
        Effectively _generate_topic_name_with_raw_response
        """
        raw_response = self.llm_namer._call_llm_for_prompt(
            prompt,
            temperature=self.temperature,
            routine="TopicNameAnnotator.annotate",
            max_tokens=self.max_tokens,
        )
        try:
            return self.topic_name_template.parse(raw_response)
        except Exception as exc:
            exc.raw_response = raw_response
            raise

    @topic_name_retry()
    async def _agenerate_and_parse(self, prompt):
        """
        Effectively single version of _generate_topic_names_with_raw_response
        """
        raw_response = await self.llm_namer._acall_llm_for_prompt(
            prompt,
            temperature=self.temperature,
            routine="TopicNameAnnotator.aannotate",
            max_tokens=self.max_tokens,
        )
        try:
            return self.topic_name_template.parse(raw_response)
        except Exception as exc:
            exc.raw_response = raw_response
            raise

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
            return self._outputs(node, implied)

        prompt = self._build_prompt(
            node,
            exemplars=exemplars,
            output_topics=output_topics,
            subtopics=subtopics,
            keyphrases=keyphrases,
        )

        if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
            result = self._generate_and_parse(prompt)
            if not result.error:
                topic_name = result.topic_payload
            else:
                raise result.error
        else:
            topic_name = prompt.removeprefix("[!SKIP!]: ")

        return self._outputs(node, topic_name)

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
            return self._outputs(node, implied)

        prompt = self._build_prompt(
            node,
            exemplars=exemplars,
            output_topics=output_topics,
            subtopics=subtopics,
            keyphrases=keyphrases,
        )

        if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
            result = await self._agenerate_and_parse(prompt)
            if not result.error:
                topic_name = result.topic_payload
            else:
                raise result.error
        else:
            topic_name = prompt.removeprefix("[!SKIP!]: ")

        return self._outputs(node, topic_name)


class TopicDisambiguationAnnotator:
    """Turn draft topic names into names that are distinct within each layer.

    Each layer is one unit. Its topics are grouped by complete linkage clustering of
    their draft name embeddings, as the legacy layers do, and each group of up to
    ``max_topics_per_prompt`` topics is renamed with one LLM call. As in the legacy
    layers, a second pass runs if a name is still shared by more than two topics.
    Topics that do not clash keep their draft names.

    ``disambiguation`` records what happened to each topic: ``"renamed"``,
    ``"unchanged"``, or ``"failed: <reason>"`` when the call for its group failed,
    in which case it keeps its draft name.

    Parameters
    ----------
    llm_namer : LLMWrapper or AsyncLLMWrapper
        The LLM that renames the topics.
    topic_name_template : TopicNamerTemplate
        Renders the prompts and parses the responses.
    embedding_model : TextEmbedderProtocol
        Embeds the new names.
    max_topics_per_prompt : int, optional
        Larger groups are split into chunks of this size, by default 12.
    """

    inputs = ("draft_topic_names", "draft_topic_name_embeddings")
    optional_inputs = ("exemplars", "keyphrases", "subtopics", "child_topics")
    outputs = ("topic_names", "topic_name_embeddings", "disambiguation")
    algorithm_input_type = {
        "draft_topic_names": MultiNodeInput(groups="layers"),
        "draft_topic_name_embeddings": "multinode",
        "exemplars": "multinode",
        "keyphrases": "multinode",
        "subtopics": "multinode",
        "child_topics": GroupDescendantsInput(from_output="topic_names", depth=1),
    }
    algorithm_output_type = "multinode"
    temperature = 0.4
    max_passes = 2

    def __init__(
        self,
        llm_namer,
        topic_name_template,
        embedding_model,
        max_topics_per_prompt=12,
    ):
        if isinstance(llm_namer, LLMWrapper):
            self.execution_mode = "sync"
        elif isinstance(llm_namer, AsyncLLMWrapper):
            self.execution_mode = "async"
        else:
            raise TypeError("llm_namer must be an LLMWrapper or AsyncLLMWrapper")
        self.llm_namer = llm_namer
        self.topic_name_template = topic_name_template
        self.embedding_model = embedding_model
        self.max_topics_per_prompt = max_topics_per_prompt
        self.max_tokens = getattr(llm_namer, "max_tokens_cluster_names", 1024)

    def layers(self, draft_topic_names):
        """One unit per layer, of the topics that have a draft name."""
        by_layer = defaultdict(list)
        for node in draft_topic_names:
            by_layer[node.layer].append(node)
        return [tuple(nodes) for nodes in by_layer.values()]

    def _clashing_groups(self, vectors):
        """The groups of topics whose names embed close together."""
        nodes = list(vectors)
        if len(nodes) < 2:
            return []
        clusters, labels = cluster_topic_names_for_renaming(
            topic_names=[],
            topic_name_embeddings=np.asarray([vectors[node] for node in nodes]),
        )
        groups = []
        for cluster in clusters:
            members = [node for node, label in zip(nodes, labels) if label == cluster]
            for start in range(0, len(members), self.max_topics_per_prompt):
                chunk = members[start : start + self.max_topics_per_prompt]
                if len(chunk) > 1:
                    groups.append(tuple(chunk))
        return groups

    def _build_prompt(
        self,
        group,
        old_names,
        exemplars=None,
        keyphrases=None,
        subtopics=None,
        child_topics=None,
    ):
        def per_node(values):
            return [values[node] if values else [] for node in group]

        def children_in_layer(offset):
            return [
                [
                    name
                    for child, name in (child_topics or {}).get(node, {}).items()
                    if child.layer == node.layer - offset
                ]
                for node in group
            ]

        return self.topic_name_template.render_disambiguation(
            group,
            topic_names=old_names,
            exemplars=per_node(exemplars),
            keyphrases=per_node(keyphrases),
            major_subtopics=children_in_layer(1),
            minor_subtopics=children_in_layer(2),
            other_subtopics=per_node(subtopics),
        )

    def _requests(self, names, vectors, context):
        """The (group, old names, prompt) of each group of clashing topics."""
        requests = []
        for group in self._clashing_groups(vectors):
            old_names = [names[node] for node in group]
            prompt = self._build_prompt(group, old_names, **context)
            requests.append((group, old_names, prompt))
        return requests

    def _finish_pass(self, requests, results, names, vectors, failures):
        """Apply a pass's results; return whether a name is still shared by more than two topics."""
        renamed = {}
        for (group, old_names, _), result in zip(requests, results):
            if result.error:
                reason = f"failed: {type(result.error).__name__}: {result.error}"
                failures.update(dict.fromkeys(group, reason))
                continue
            for node, old, new in zip(group, old_names, result.topic_payload):
                if new != old:
                    renamed[node] = new
        names.update(renamed)
        vectors.update(embed_topic_names(self.embedding_model, renamed))
        return bool(requests) and max(Counter(names.values()).values()) > 2

    @staticmethod
    def _outputs(draft_names, names, vectors, failures):
        disambiguation = {
            node: (
                "renamed"
                if names[node] != draft_names[node]
                else failures.get(node, "unchanged")
            )
            for node in names
        }
        return {
            "topic_names": names,
            "topic_name_embeddings": vectors,
            "disambiguation": disambiguation,
        }

    def _parse(self, raw_response, old_names):
        try:
            new_names = self.topic_name_template.parse_disambiguation(
                raw_response, old_names
            )
        except Exception as exc:
            exc.raw_response = raw_response
            raise
        return TopicParseResult(
            topic_payload=new_names, raw_response=raw_response, error=None
        )

    @topic_name_retry("TopicDisambiguationAnnotator")
    def _generate_and_parse(self, prompt, old_names):
        raw_response = self.llm_namer._call_llm_for_prompt(
            prompt,
            temperature=self.temperature,
            routine="TopicDisambiguationAnnotator.annotate",
            max_tokens=self.max_tokens,
        )
        return self._parse(raw_response, old_names)

    @topic_name_retry("TopicDisambiguationAnnotator")
    async def _agenerate_and_parse(self, prompt, old_names):
        raw_response = await self.llm_namer._acall_llm_for_prompt(
            prompt,
            temperature=self.temperature,
            routine="TopicDisambiguationAnnotator.aannotate",
            max_tokens=self.max_tokens,
        )
        return self._parse(raw_response, old_names)

    def annotate(
        self, layer, *, draft_topic_names, draft_topic_name_embeddings, **context
    ):
        names, vectors, failures = (
            dict(draft_topic_names),
            dict(draft_topic_name_embeddings),
            {},
        )
        for _ in range(self.max_passes):
            requests = self._requests(names, vectors, context)
            results = [
                self._generate_and_parse(prompt, old_names)
                for _, old_names, prompt in requests
            ]
            if not self._finish_pass(requests, results, names, vectors, failures):
                break
        return self._outputs(draft_topic_names, names, vectors, failures)

    async def aannotate(
        self, layer, *, draft_topic_names, draft_topic_name_embeddings, **context
    ):
        names, vectors, failures = (
            dict(draft_topic_names),
            dict(draft_topic_name_embeddings),
            {},
        )
        for _ in range(self.max_passes):
            requests = self._requests(names, vectors, context)
            results = await asyncio.gather(
                *(
                    self._agenerate_and_parse(prompt, old_names)
                    for _, old_names, prompt in requests
                )
            )
            if not self._finish_pass(requests, results, names, vectors, failures):
                break
        return self._outputs(draft_topic_names, names, vectors, failures)
