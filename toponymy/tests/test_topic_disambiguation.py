import numpy as np
import pytest
from tenacity import wait_none

from toponymy.annotation import Annotation, AnnotationStore, AnnotationTree, Executor
from toponymy.clusterers import build_cluster_tree
from toponymy.tests.helpers.dummy_wrappers import (
    CountingNamer,
    DummyAsyncSingleWrapper,
    NameEmbedder,
)
from toponymy.topic_naming import TopicDisambiguationAnnotator, TopicNamerTemplate

# Six layer 0 clusters of three objects each, nested two to a layer 1 cluster
LAYER_0 = np.repeat(np.arange(6), 3)
LAYER_1 = LAYER_0 // 2
DRAFTS = {
    0: ["Same", "Same", "Same", "Other", "Third", "Fourth"],
    1: ["Broad", "Broad", "Wide"],
}
# The first three layer 0 drafts, and the first two layer 1 drafts, embed together
VECTORS = {
    0: np.eye(6)[[0, 0, 0, 1, 2, 3]],
    1: np.eye(6)[[0, 0, 4]],
}


def make_store():
    tree = AnnotationTree(build_cluster_tree([LAYER_0, LAYER_1]))
    store = AnnotationStore(tree)
    annotations = {
        name: Annotation(name, tree)
        for name in (
            "draft_topic_names",
            "draft_topic_name_embeddings",
            "exemplars",
            "keyphrases",
        )
    }
    for node in tree.nodes:
        annotations["draft_topic_names"][node] = DRAFTS[node.layer][node.cluster]
        annotations["draft_topic_name_embeddings"][node] = VECTORS[node.layer][
            node.cluster
        ]
        annotations["exemplars"][node] = [f"exemplar of {node}"]
        annotations["keyphrases"][node] = [f"keyphrase of {node}"]
    for annotation in annotations.values():
        store.add(annotation)
    return store


def make_annotator(namer, store, **kwargs):
    template = TopicNamerTemplate(
        object_description="sentences",
        corpus_description="a test corpus",
        tree=store.tree,
    )
    return TopicDisambiguationAnnotator(namer, template, NameEmbedder(), **kwargs)


def run_on_layer(store, annotator, layer):
    failures = Executor(store).run(annotator, nodes=store.tree.layer(layer))
    assert not failures
    nodes = store.tree.layer(layer)
    return [store.topic_names[node] for node in nodes], [
        store.disambiguation[node] for node in nodes
    ]


def test_each_layer_is_one_unit():
    store = make_store()
    annotator = make_annotator(CountingNamer(), store)

    units = annotator.layers(store.draft_topic_names.computed())

    assert units == [tuple(store.tree.layer(0)), tuple(store.tree.layer(1))]


def test_clashing_groups_split_large_groups_and_drop_single_leftovers():
    store = make_store()
    annotator = make_annotator(CountingNamer(), store, max_topics_per_prompt=2)

    groups = annotator._clashing_groups(
        store.draft_topic_name_embeddings.computed(store.tree.layer(0))
    )

    assert groups == [((0, 0), (0, 1))]


def test_clashing_topics_are_renamed_and_the_rest_pass_through():
    store = make_store()

    names, disambiguation = run_on_layer(
        store, make_annotator(CountingNamer(), store), 0
    )

    assert names == ["Topic 0.1", "Topic 0.2", "Topic 0.3"] + DRAFTS[0][3:]
    assert disambiguation == ["renamed"] * 3 + ["unchanged"] * 3
    for node, name in zip(store.tree.layer(0), names):
        expected = (
            NameEmbedder().encode([name])[0]
            if node.cluster < 3
            else VECTORS[0][node.cluster]
        )
        np.testing.assert_array_equal(store.topic_name_embeddings[node], expected)
    assert [store.draft_topic_names[n] for n in store.tree.layer(0)] == DRAFTS[0]


def test_a_layer_without_clashes_makes_no_llm_calls():
    store = make_store()
    for node in store.tree.layer(0):
        store.draft_topic_name_embeddings[node] = np.eye(6)[node.cluster]
    namer = CountingNamer()

    names, disambiguation = run_on_layer(store, make_annotator(namer, store), 0)

    assert not namer.prompts
    assert names == DRAFTS[0]
    assert disambiguation == ["unchanged"] * 6


@pytest.mark.parametrize("rename, passes", [(True, 1), (False, 2)])
def test_a_second_pass_runs_while_a_name_is_shared_by_more_than_two_topics(
    rename, passes
):
    store = make_store()
    namer = CountingNamer(name="Same", rename=rename)

    run_on_layer(store, make_annotator(namer, store), 0)

    assert len(namer.disambiguation_prompts) == passes


def test_a_failed_group_keeps_its_draft_names(monkeypatch):
    monkeypatch.setattr(
        TopicDisambiguationAnnotator._generate_and_parse.retry, "wait", wait_none()
    )
    store = make_store()
    # Three attempts in each of the two passes
    annotator = make_annotator(CountingNamer(fail_calls=set(range(6))), store)

    with pytest.warns(UserWarning, match="retries exhausted for TopicDisambiguation"):
        names, disambiguation = run_on_layer(store, annotator, 0)

    assert names == DRAFTS[0]
    assert all(status.startswith("failed: ValueError") for status in disambiguation[:3])
    assert disambiguation[3:] == ["unchanged"] * 3


def test_the_prompt_lists_each_topics_final_child_names_and_subtopics():
    store = make_store()
    subtopics = Annotation("subtopics", store.tree)
    for node in store.tree.layer(1):
        subtopics[node] = [f"subtopic of {node}"]
    store.add(subtopics)
    namer = CountingNamer()
    annotator = make_annotator(namer, store)

    run_on_layer(store, annotator, 0)
    run_on_layer(store, annotator, 1)

    user = namer.disambiguation_prompts[-1]["user"]
    assert '"1. Broad"' in user and '"2. Broad"' in user
    assert (
        "Major subtopics of this group are:\n    * Topic 0.1\n    * Topic 0.2" in user
    )
    assert "Major subtopics of this group are:\n    * Topic 0.3\n    * Other" in user
    assert "subtopic of (1, 0)" in user and "keyphrase of (1, 1)" in user
    assert "exemplar of (1, 0)" in user


@pytest.mark.parametrize(
    "response",
    [
        '{"new_topic_name_mapping": {"1.": "A"}, "topic_specificities": [0.9]}',
        '{"new_topic_name_mapping": {"1.": "A", "2.": ""}, "topic_specificities": [0.9, 0.9]}',
        # Qwen 0.5B sometimes nests a dict in each value, or answers with a list
        '{"new_topic_name_mapping": {"1.": {"new_name": "A"}, "2.": {"new_name": "B"}}, "topic_specificities": [0.9, 0.9]}',
        '{"new_topic_name_mapping": ["A"], "topic_specificities": [0.9]}',
        "no json here",
    ],
)
def test_parse_disambiguation_rejects_missing_or_empty_names(response):
    template = TopicNamerTemplate("sentences", "a test corpus", make_store().tree)

    with pytest.raises(ValueError):
        template.parse_disambiguation(response, ["Same", "Same"])


def test_rejects_a_namer_that_is_not_an_llm_wrapper():
    with pytest.raises(TypeError, match="LLMWrapper"):
        make_annotator(object(), make_store())


class AsyncCountingNamer(DummyAsyncSingleWrapper):
    def __init__(self):
        self.namer = CountingNamer()

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return self.namer._respond(prompt)

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens):
        return self.namer._respond(prompt)


def test_an_async_namer_renames_clashing_topics():
    store = make_store()
    annotator = make_annotator(AsyncCountingNamer(), store)

    names, disambiguation = run_on_layer(store, annotator, 0)

    assert annotator.execution_mode == "async"
    assert len(set(names)) == len(names)
    assert disambiguation == ["renamed"] * 3 + ["unchanged"] * 3
