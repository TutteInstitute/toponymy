import asyncio
import copy
import re
import pytest
from unittest.mock import AsyncMock, Mock
import numpy as np

from tenacity import wait_none

from toponymy import ToponymyClusterer
from toponymy.annotation import AnnotationStore, Executor, AnnotationState
from toponymy.exemplar_texts import DiverseExemplarAnnotator
from toponymy.llm_wrappers import (
    OllamaNamer,
    TopicParseResult,
    LLMWrapper,
    AsyncLLMWrapper,
)
from toponymy.templates import PROMPT_TEMPLATES
from toponymy.tests.helpers.dummy_wrappers import NameEmbedder
from toponymy.tools.notebook_data_load import load_small_newsgroups
from toponymy.topic_naming import TopicNameAnnotator, TopicNamerTemplate


@pytest.fixture(scope="module")
def sm_newsgroups_executor_factory():
    newsgroups_df = load_small_newsgroups()
    embeddings = np.stack(newsgroups_df["embedding"].values)
    document_map = np.stack(newsgroups_df["map"].values)

    clusterer = ToponymyClusterer(min_clusters=4, verbose=True)
    clusterer.fit(document_map, embeddings)
    objects = newsgroups_df["post"].str.strip().values
    embedding_vectors = embeddings

    def create_executor(use_exemplars=False):
        store = AnnotationStore.from_clusterer(clusterer, objects, embedding_vectors)
        executor = Executor(store)

        if use_exemplars:
            null_topic = np.mean(embeddings, axis=0)
            executor.run(DiverseExemplarAnnotator(null_topic=null_topic))

        return executor

    return create_executor


@pytest.fixture
def sm_newsgroups_executor(sm_newsgroups_executor_factory):
    return sm_newsgroups_executor_factory


def disable_retry_wait(monkeypatch, method):
    monkeypatch.setattr(method.retry, "wait", wait_none())


def test_executor_is_independent(sm_newsgroups_executor_factory):
    executor1 = sm_newsgroups_executor_factory(use_exemplars=False)
    executor2 = sm_newsgroups_executor_factory(use_exemplars=False)

    assert executor1 is not executor2
    assert executor1.store is not executor2.store

    # Modify one executor
    original_store = executor2.store
    executor1.store = None

    # Verify the other remains unchanged
    assert executor2.store is original_store


def make_topic_name_annotator(executor, namer):
    template = TopicNamerTemplate(
        object_description="newsgroup posts",
        corpus_description="a collection of newsgroup posts",
        tree=executor.tree,
        prompt_template=PROMPT_TEMPLATES.copy(),
    )

    return TopicNameAnnotator(
        llm_namer=namer,
        topic_name_template=template,
        embedding_model=NameEmbedder(),
    )


class TestTopicNamerTemplate:
    def test_topic_namer_template_render_passes_configuration(
        self, topic_store, monkeypatch
    ):
        import toponymy.topic_naming as topic_naming

        node = topic_store.tree.nodes[0]
        expected_prompt = "rendered prompt"
        captured = {}

        def fake_prompt(**kwargs):
            captured.update(kwargs)
            return expected_prompt

        monkeypatch.setattr(
            topic_naming,
            "topic_name_prompt_by_node",
            fake_prompt,
        )

        template = TopicNamerTemplate(
            object_description="posts",
            corpus_description="newsgroup posts",
            tree=topic_store.tree,
            exemplar_start_delimiter="<",
            exemplar_end_delimiter=">",
        )

        result = template.render(
            node,
            major_subtopics=["major"],
            minor_subtopics=["minor"],
            other_subtopics=["other"],
            exemplars=["example"],
            keyphrases=["phrase"],
        )

        assert result == expected_prompt
        assert captured["node"] == node
        assert captured["major_subtopics"] == ["major"]
        assert captured["minor_subtopics"] == ["minor"]
        assert captured["other_subtopics"] == ["other"]
        assert captured["current_exemplars"] == ["example"]
        assert captured["current_keyphrases"] == ["phrase"]
        assert captured["object_description"] == "posts"
        assert captured["corpus_description"] == "newsgroup posts"
        assert captured["exemplar_start_delimiter"] == "<"
        assert captured["exemplar_end_delimiter"] == ">"

    # The inputs are listed in the user and combined renderings, not in system
    @pytest.mark.parametrize("rendering", ["user", "combined"])
    def test_topic_namer_template_render_applies_input_limits(
        self, topic_store, rendering
    ):
        def items(kind):
            return [f"{kind}-{i}" for i in range(1, 6)]

        template = TopicNamerTemplate(
            object_description="posts",
            corpus_description="newsgroup posts",
            tree=topic_store.tree,
            max_num_keyphrases=2,
            max_num_subtopics=3,
            max_num_exemplars=4,
        )

        prompt = template.render(
            topic_store.tree.nodes[0],
            major_subtopics=items("major-subtopic"),
            minor_subtopics=items("minor-subtopic"),
            other_subtopics=items("other-subtopic"),
            exemplars=items("exemplar"),
            keyphrases=items("keyphrase"),
        )

        # Each subtopic kind is limited on its own
        expected = {
            "keyphrase": items("keyphrase")[:2],
            "major-subtopic": items("major-subtopic")[:3],
            "minor-subtopic": items("minor-subtopic")[:3],
            "other-subtopic": items("other-subtopic")[:3],
            "exemplar": items("exemplar")[:4],
        }
        # Whole items in the order they appear, so "keyphrase-1" never matches a
        # longer item
        shown = {
            kind: re.findall(rf"{kind}-\d+", prompt[rendering]) for kind in expected
        }
        assert shown == expected

    def test_topic_namer_template_render_defaults_to_empty_inputs(
        self, topic_store, monkeypatch
    ):
        import toponymy.topic_naming as topic_naming

        captured = {}

        def fake_prompt(**kwargs):
            captured.update(kwargs)
            return "prompt"

        monkeypatch.setattr(
            topic_naming,
            "topic_name_prompt_by_node",
            fake_prompt,
        )

        template = TopicNamerTemplate(
            object_description="posts",
            corpus_description="newsgroup posts",
            tree=topic_store.tree,
        )

        template.render(topic_store.tree.nodes[0])

        assert captured["major_subtopics"] == []
        assert captured["minor_subtopics"] == []
        assert captured["other_subtopics"] == []
        assert captured["current_exemplars"] == []
        assert captured["current_keyphrases"] == []

    def test_topic_namer_template_uses_layer_summary_kind(
        self, topic_store, monkeypatch
    ):
        import toponymy.topic_naming as topic_naming

        node = topic_store.tree.nodes[0]
        monkeypatch.setattr(
            topic_naming,
            "summary_kind",
            lambda *args, **kwargs: "expected summary kind",
        )

        captured = {}

        def fake_prompt(**kwargs):
            captured.update(kwargs)
            return "prompt"

        monkeypatch.setattr(
            topic_naming,
            "topic_name_prompt_by_node",
            fake_prompt,
        )

        template = TopicNamerTemplate(
            object_description="posts",
            corpus_description="newsgroup posts",
            tree=topic_store.tree,
        )

        template.render(node)

        assert captured["summary_kind"] == "expected summary kind"

    def test_topic_namer_template_parse_delegates_to_parser(
        self, topic_store, monkeypatch
    ):
        import toponymy.topic_naming as topic_naming

        expected = TopicParseResult(
            topic_payload="science",
            raw_response='{"topic_name": "science"}',
        )
        captured = {}

        def fake_parse(*args):
            captured["args"] = args
            return expected

        monkeypatch.setattr(topic_naming, "parse_raw_response", fake_parse)

        extract = Mock()
        regex = "topic regex"

        template = TopicNamerTemplate(
            object_description="posts",
            corpus_description="newsgroup posts",
            tree=topic_store.tree,
            prompt_template={
                "layer": {
                    "get_topic_name_regex": regex,
                    "extract_topic_name": extract,
                }
            },
        )

        result = template.parse("raw response", null_response_value="unknown")

        assert result is expected
        assert captured["args"] == (
            "raw response",
            extract,
            regex,
            "unknown",
        )

    def test_topic_namer_template_parse_uses_default_null_value(
        self, topic_store, monkeypatch
    ):
        import toponymy.topic_naming as topic_naming

        captured = {}

        def fake_parse(*args):
            captured["args"] = args
            return TopicParseResult(topic_payload="", raw_response="")

        monkeypatch.setattr(topic_naming, "parse_raw_response", fake_parse)

        template = TopicNamerTemplate(
            object_description="posts",
            corpus_description="newsgroup posts",
            tree=topic_store.tree,
        )

        template.parse("raw response")

        assert captured["args"][-1] == ""


class TestTopicNameAnnotatorEndtoEnd:
    def assert_topic_name_completion(self, executor, min_completion=0.7):
        names = executor.store["topic_names"]
        completed = sum(bool(name) for name in names.values())

        assert completed / len(names) >= min_completion

    @pytest.mark.parametrize("use_exemplars", [True, False])
    def test_topic_name_annotator_end_to_end_sync_ollama(
        self,
        ollama_test_model,
        sm_newsgroups_executor,
        use_exemplars,
    ):
        executor = sm_newsgroups_executor(use_exemplars=use_exemplars)

        namer = OllamaNamer(
            model=ollama_test_model,
            api_base="http://localhost:11434",
            temperature_override=0.0,
        )

        annotator = make_topic_name_annotator(executor, namer)

        failures = executor.run(annotator)

        self.assert_topic_name_completion(executor, min_completion=0.7)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_exemplars", [True, False])
    async def test_topic_name_annotator_end_to_end_async_ollama(
        self, sm_newsgroups_executor, async_llm, use_exemplars
    ):
        executor = sm_newsgroups_executor(use_exemplars=use_exemplars)

        annotator = make_topic_name_annotator(executor, async_llm)

        failures = await executor.arun(annotator)

        self.assert_topic_name_completion(executor, min_completion=0.7)


class FakeTopicNamerTemplate:
    def __init__(self, prompt="test prompt", parsed_name="test topic"):
        self.prompt = prompt
        self.parsed_name = parsed_name
        self.render_kwargs = None
        self.parse_calls = []

    def render(self, node, **kwargs):
        self.render_kwargs = {"node": node, **kwargs}
        return self.prompt

    def parse(self, raw_response):
        self.parse_calls.append(raw_response)
        return TopicParseResult(
            topic_payload=self.parsed_name,
            raw_response=raw_response,
        )


@pytest.fixture
def topic_namer_template():
    return FakeTopicNamerTemplate()


@pytest.fixture
def topic_name_annotator(dummy_namer, topic_namer_template):
    return TopicNameAnnotator(
        llm_namer=dummy_namer,
        topic_name_template=topic_namer_template,
        embedding_model=NameEmbedder(),
    )


def configure_sync_namer(namer, response='{"topic_name": "science"}'):
    namer._call_llm_for_prompt = Mock(return_value=response)


def configure_async_namer(namer, response='{"topic_name": "science"}'):
    namer._acall_llm_for_prompt = AsyncMock(return_value=response)


class TestTopicNameAnnotator:
    def test_topic_name_annotator_sets_execution_mode(
        self, dummy_namer, topic_namer_template
    ):
        annotator = TopicNameAnnotator(
            dummy_namer, topic_namer_template, NameEmbedder()
        )

        expected = (
            "async"
            if dummy_namer.__class__.__name__.startswith("DummyAsync")
            else "sync"
        )

        assert annotator.execution_mode == expected
        assert annotator.max_tokens == 128
        assert annotator.temperature == 0.4

    def test_topic_name_annotator_rejects_invalid_namer(topic_namer_template):
        with pytest.raises(TypeError, match="LLMWrapper or AsyncLLMWrapper"):
            TopicNameAnnotator(
                llm_namer=object(),
                topic_name_template=topic_namer_template,
                embedding_model=NameEmbedder(),
            )

    def test_topic_name_annotator_build_prompt_without_subtopics(
        self, topic_store, topic_name_annotator, topic_namer_template
    ):
        node = topic_store.tree.nodes[0]

        prompt = topic_name_annotator._build_prompt(
            node,
            exemplars=["example"],
            keyphrases=["phrase"],
        )

        assert prompt == "test prompt"
        assert topic_namer_template.render_kwargs == {
            "node": node,
            "major_subtopics": [],
            "minor_subtopics": [],
            "other_subtopics": [],
            "exemplars": ["example"],
            "keyphrases": ["phrase"],
        }

    def test_topic_name_annotator_build_prompt_defaults(
        self, topic_store, topic_name_annotator, topic_namer_template
    ):
        node = topic_store.tree.nodes[0]

        topic_name_annotator._build_prompt(node)

        assert topic_namer_template.render_kwargs["exemplars"] == []
        assert topic_namer_template.render_kwargs["keyphrases"] == []
        assert topic_namer_template.render_kwargs["major_subtopics"] == []
        assert topic_namer_template.render_kwargs["minor_subtopics"] == []
        assert topic_namer_template.render_kwargs["other_subtopics"] == []

    def test_topic_name_annotator_implied_topic(self, topic_name_annotator):
        assert topic_name_annotator._implied_topic({}) is None
        assert topic_name_annotator._implied_topic(None) is None
        assert topic_name_annotator._implied_topic({object(): ""}) is None
        assert topic_name_annotator._implied_topic({object(): "science"}) == "science"

    def test_topic_name_annotator_uses_implied_topic_without_llm(
        self, topic_store, topic_name_annotator, dummy_namer
    ):
        node = topic_store.tree.nodes[0]
        output_topics = {node: "implied topic"}

        result = topic_name_annotator.annotate(
            node,
            output_topics=output_topics,
        )

        assert result["topic_names"] == "implied topic"
        np.testing.assert_array_equal(
            result["topic_name_embeddings"],
            NameEmbedder().encode(["implied topic"])[0],
        )

    def test_topic_name_annotator_sync_calls_namer(
        self, topic_store, topic_name_annotator, dummy_namer, topic_namer_template
    ):
        if topic_name_annotator.execution_mode != "sync":
            pytest.skip("sync namer test")

        node = topic_store.tree.nodes[0]
        raw_response = '{"topic_name": "science"}'
        configure_sync_namer(dummy_namer, raw_response)

        result = topic_name_annotator.annotate(
            node,
            exemplars=["example"],
            keyphrases=["phrase"],
        )

        assert result["topic_names"] == "test topic"
        assert topic_namer_template.parse_calls == [raw_response]
        np.testing.assert_array_equal(
            result["topic_name_embeddings"], NameEmbedder().encode(["test topic"])[0]
        )

        dummy_namer._call_llm_for_prompt.assert_called_once_with(
            "test prompt",
            temperature=0.4,
            routine="TopicNameAnnotator.annotate",
            max_tokens=128,
        )

    @pytest.mark.asyncio
    async def test_topic_name_annotator_async_calls_namer(
        self, topic_store, topic_name_annotator, dummy_namer, topic_namer_template
    ):
        if topic_name_annotator.execution_mode != "async":
            pytest.skip("async namer test")

        node = topic_store.tree.nodes[0]
        raw_response = '{"topic_name": "science"}'
        configure_async_namer(dummy_namer, raw_response)

        result = await topic_name_annotator.aannotate(
            node,
            exemplars=["example"],
            keyphrases=["phrase"],
        )

        assert result["topic_names"] == "test topic"
        assert topic_namer_template.parse_calls == [raw_response]
        np.testing.assert_array_equal(
            result["topic_name_embeddings"], NameEmbedder().encode(["test topic"])[0]
        )

        dummy_namer._acall_llm_for_prompt.assert_awaited_once_with(
            "test prompt",
            temperature=0.4,
            routine="TopicNameAnnotator.aannotate",
            max_tokens=128,
        )

    def test_topic_name_annotator_groups_major_and_minor_subtopics(
        self, deep_topic_store, topic_name_annotator, topic_namer_template
    ):
        tree = deep_topic_store.tree
        node = next(n for n in tree.nodes if n.layer >= 2)

        descendants = {
            child: f"topic-{child.layer}-{child.cluster}"
            for child in tree.nodes
            if child.layer < node.layer
        }

        topic_name_annotator._build_prompt(
            node,
            output_topics=descendants,
            subtopics=["other topic"],
        )

        kwargs = topic_namer_template.render_kwargs

        expected_major = [
            value
            for child, value in descendants.items()
            if child.layer == node.layer - 1
        ]
        expected_minor = [
            value
            for child, value in descendants.items()
            if child.layer == node.layer - 2
        ]

        assert kwargs["major_subtopics"] == expected_major
        assert kwargs["minor_subtopics"] == expected_minor

    def test_topic_name_annotator_promotes_minor_subtopics_when_few_major(
        self, deep_topic_store, topic_name_annotator, topic_namer_template
    ):
        tree = deep_topic_store.tree

        # Find a node with at least two ancestor layers.
        node = next(n for n in tree.nodes if n.layer >= 2)

        descendants = {
            child: f"topic-{child.layer}-{child.cluster}"
            for child in tree.nodes
            if child.layer < node.layer
        }

        major = [
            value
            for child, value in descendants.items()
            if child.layer == node.layer - 1
        ]
        minor = [
            value
            for child, value in descendants.items()
            if child.layer == node.layer - 2
        ]

        if len(major) > 1:
            pytest.skip("This node has more than one major subtopic")

        topic_name_annotator._build_prompt(
            node,
            output_topics=descendants,
            subtopics=["other"],
        )

        kwargs = topic_namer_template.render_kwargs

        assert kwargs["major_subtopics"] == major + minor
        assert kwargs["minor_subtopics"] == []

    def test_topic_name_annotator_returns_skip_prompt_value(
        self, topic_store, topic_name_annotator, topic_namer_template
    ):
        node = topic_store.tree.nodes[0]
        topic_namer_template.prompt = "[!SKIP!]: skipped topic"

        result = topic_name_annotator.annotate(node)

        assert result["topic_names"] == "skipped topic"
        np.testing.assert_array_equal(
            result["topic_name_embeddings"],
            NameEmbedder().encode(["skipped topic"])[0],
        )

    @staticmethod
    async def annotate_in_mode(annotator, node, **kwargs):
        if annotator.execution_mode == "async":
            return await annotator.aannotate(node, **kwargs)
        return annotator.annotate(node, **kwargs)

    @pytest.mark.asyncio
    async def test_summary_annotator_writes_the_parsed_summary_and_explanation(
        self, topic_store, dummy_namer, topic_namer_template
    ):
        configure_sync_namer(dummy_namer)
        configure_async_namer(dummy_namer)
        topic_namer_template.parsed_name = ("science", "A summary.", "An analysis.")
        annotator = TopicNameAnnotator(
            dummy_namer, topic_namer_template, NameEmbedder(), summaries=True
        )

        result = await self.annotate_in_mode(annotator, topic_store.tree.nodes[0])

        assert result["topic_names"] == "science"
        assert result["topic_summaries"] == "A summary."
        assert result["topic_explanations"] == "An analysis."
        assert "topic_summaries" in annotator.outputs
        assert "topic_summaries" not in TopicNameAnnotator.outputs
        call = (
            dummy_namer._acall_llm_for_prompt
            if annotator.execution_mode == "async"
            else dummy_namer._call_llm_for_prompt
        )
        assert call.call_args.kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_annotator_without_summaries_names_from_a_summary_reply(
        self, topic_store, topic_name_annotator, dummy_namer, topic_namer_template
    ):
        configure_sync_namer(dummy_namer)
        configure_async_namer(dummy_namer)
        topic_namer_template.parsed_name = ("science", "A summary.", "An analysis.")

        result = await self.annotate_in_mode(
            topic_name_annotator, topic_store.tree.nodes[0]
        )

        assert result["topic_names"] == "science"
        assert "topic_summaries" not in result

    @pytest.mark.asyncio
    async def test_summary_annotator_shows_each_childs_summary_and_explanation(
        self, topic_store, dummy_namer, topic_namer_template
    ):
        configure_sync_namer(dummy_namer)
        configure_async_namer(dummy_namer)
        topic_namer_template.parsed_name = ("science", "A summary.", "An analysis.")
        annotator = TopicNameAnnotator(
            dummy_namer, topic_namer_template, NameEmbedder(), summaries=True
        )
        tree = topic_store.tree
        node = next(n for n in tree.nodes if n.layer >= 1)
        children = [c for c in tree.nodes if c.layer == node.layer - 1][:2]

        await self.annotate_in_mode(
            annotator,
            node,
            output_topics={child: f"name {i}" for i, child in enumerate(children)},
            output_summaries={children[0]: "summary 0"},
            output_explanations={children[0]: "analysis 0"},
        )

        assert topic_namer_template.render_kwargs["major_subtopics"] == [
            "name 0\n--\nsummary 0\n--\nanalysis 0",
            "name 1\n--\n\n--\n",
        ]

    @pytest.mark.asyncio
    async def test_summary_annotator_inherits_a_single_childs_summary(
        self, topic_store, dummy_namer, topic_namer_template
    ):
        configure_sync_namer(dummy_namer)
        configure_async_namer(dummy_namer)
        annotator = TopicNameAnnotator(
            dummy_namer, topic_namer_template, NameEmbedder(), summaries=True
        )
        tree = topic_store.tree
        node = next(n for n in tree.nodes if n.layer >= 1)
        child = next(c for c in tree.nodes if c.layer == node.layer - 1)

        result = await self.annotate_in_mode(
            annotator,
            node,
            output_topics={child: "only child"},
            output_summaries={child: "its summary"},
            output_explanations={child: "its analysis"},
        )

        assert result["topic_names"] == "only child"
        assert result["topic_summaries"] == "its summary"
        assert result["topic_explanations"] == "its analysis"
        dummy_namer._call_llm_for_prompt.assert_not_called()
        dummy_namer._acall_llm_for_prompt.assert_not_called()

    @pytest.mark.parametrize(
        "output_names",
        [
            None,
            {
                "topic_names": "draft_topic_names",
                "topic_name_embeddings": "draft_topic_name_embeddings",
            },
        ],
        ids=["final", "draft"],
    )
    def test_topic_name_annotator_writes_each_nodes_embedding_with_its_name(
        self, topic_store, dummy_namer, output_names
    ):
        class EchoTemplate(FakeTopicNamerTemplate):
            def render(self, node, **kwargs):
                return f"prompt for {node}"

            def parse(self, raw_response):
                return TopicParseResult(
                    topic_payload=raw_response.replace("prompt", "name"),
                    raw_response=raw_response,
                )

        def echo(prompt, **kwargs):
            return prompt

        dummy_namer._call_llm_for_prompt = Mock(side_effect=echo)
        dummy_namer._acall_llm_for_prompt = AsyncMock(side_effect=echo)
        annotator = TopicNameAnnotator(dummy_namer, EchoTemplate(), NameEmbedder())
        layer_0 = topic_store.tree.layer(0)

        failures = Executor(topic_store).run(
            annotator, nodes=layer_0, output_names=output_names
        )

        assert not failures
        wiring = output_names or {}
        names = topic_store[wiring.get("topic_names", "topic_names")]
        vectors = topic_store[
            wiring.get("topic_name_embeddings", "topic_name_embeddings")
        ]
        assert len({names[node] for node in layer_0}) == len(layer_0)
        for node in layer_0:
            assert names[node] == f"name for {node}"
            np.testing.assert_array_equal(
                vectors[node], NameEmbedder().encode([names[node]])[0]
            )

    def test_topic_name_annotator_raises_on_empty_topic_name(
        self, topic_store, topic_name_annotator, dummy_namer
    ):
        if topic_name_annotator.execution_mode == "sync":
            configure_sync_namer(dummy_namer)
            topic_name_annotator.topic_name_template.parsed_name = ""
            with pytest.raises(ValueError, match="Failed to generate"):
                topic_name_annotator.annotate(topic_store.tree.nodes[0])
        else:
            pytest.skip("sync namer test")

    @pytest.mark.asyncio
    async def test_topic_name_annotator_async_raises_on_empty_topic_name(
        self, topic_store, topic_name_annotator, dummy_namer
    ):
        if topic_name_annotator.execution_mode == "async":
            configure_async_namer(dummy_namer)
            topic_name_annotator.topic_name_template.parsed_name = ""
            with pytest.raises(ValueError, match="Failed to generate"):
                await topic_name_annotator.aannotate(topic_store.tree.nodes[0])
        else:
            pytest.skip("async namer test")

    def test_generate_and_parse_retries_after_parse_error(
        self, topic_name_annotator, dummy_namer, monkeypatch
    ):
        if not isinstance(dummy_namer, LLMWrapper):
            pytest.skip("Requires a synchronous namer")

        method = topic_name_annotator._generate_and_parse
        disable_retry_wait(monkeypatch, method)

        dummy_namer._call_llm_for_prompt = Mock(return_value="raw response")
        topic_name_annotator.topic_name_template.parse = Mock(
            side_effect=ValueError("invalid response")
        )

        with pytest.warns(UserWarning, match="All retries exhausted"):
            result = method("test prompt")

        assert dummy_namer._call_llm_for_prompt.call_count == 3
        assert topic_name_annotator.topic_name_template.parse.call_count == 3
        assert result.topic_payload == ""
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "invalid response"
        assert result.raw_response == "raw response"

    @pytest.mark.asyncio
    async def test_agenerate_and_parse_retries_after_parse_error(
        self, topic_name_annotator, dummy_namer, monkeypatch
    ):
        if not isinstance(dummy_namer, AsyncLLMWrapper):
            pytest.skip("Requires an asynchronous namer")

        method = topic_name_annotator._agenerate_and_parse
        disable_retry_wait(monkeypatch, method)

        dummy_namer._acall_llm_for_prompt = AsyncMock(return_value="raw response")
        topic_name_annotator.topic_name_template.parse = Mock(
            side_effect=ValueError("invalid response")
        )

        with pytest.warns(UserWarning, match="All retries exhausted"):
            result = await method("test prompt")

        assert dummy_namer._acall_llm_for_prompt.await_count == 3
        assert topic_name_annotator.topic_name_template.parse.call_count == 3
        assert result.topic_payload == ""
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "invalid response"
        assert result.raw_response == "raw response"
