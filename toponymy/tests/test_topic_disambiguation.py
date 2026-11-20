import numpy as np
import pytest
from tenacity import wait_none

from toponymy.annotation import Annotation, AnnotationStore, AnnotationTree, Executor
from toponymy.clusterers import build_cluster_tree
from toponymy.tests.helpers.dummy_wrappers import (
    CountingNamer,
    DummyAsyncSingleWrapper,
)
from toponymy.topic_naming import TopicDisambiguationAnnotator, TopicNamerTemplate

# Six layer 0 clusters of three objects each, nested two to a layer 1 cluster
LAYER_0 = np.repeat(np.arange(6), 3)
LAYER_1 = LAYER_0 // 2
NAMES = {
    0: ["Same", "Same", "Same", "Other", "Third", "Fourth"],
    1: ["Broad", "Broad", "Wide"],
}
# The first three layer 0 names, and the first two layer 1 names, embed together
VECTORS = {
    0: np.eye(6)[[0, 0, 0, 1, 2, 3]],
    1: np.eye(6)[[0, 0, 4]],
}


def make_store():
    tree = AnnotationTree(build_cluster_tree([LAYER_0, LAYER_1]))
    store = AnnotationStore(tree)
    annotations = {
        name: Annotation(name, tree)
        for name in ("topic_names", "topic_name_embeddings", "exemplars", "keyphrases")
    }
    for node in tree.nodes:
        annotations["topic_names"][node] = NAMES[node.layer][node.cluster]
        annotations["topic_name_embeddings"][node] = VECTORS[node.layer][node.cluster]
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
    return TopicDisambiguationAnnotator(namer, template, **kwargs)


def names_of(store, layer):
    return [store.topic_names[node] for node in store.tree.layer(layer)]


def test_clashing_groups_stay_within_a_layer():
    store = make_store()
    annotator = make_annotator(CountingNamer(), store)

    groups = annotator.clashing_groups(store.topic_name_embeddings.computed())

    assert sorted(groups) == [((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1))]


def test_clashing_groups_split_large_groups_and_drop_single_leftovers():
    store = make_store()
    annotator = make_annotator(CountingNamer(), store, max_topics_per_prompt=2)

    groups = annotator.clashing_groups(
        store.topic_name_embeddings.computed(store.tree.layer(0))
    )

    assert groups == [((0, 0), (0, 1))]


def test_annotate_renames_clashing_topics_in_place_and_records_the_old_names():
    store = make_store()
    annotator = make_annotator(CountingNamer(), store)

    failures = Executor(store).run(annotator, nodes=store.tree.layer(0))

    assert not failures
    assert names_of(store, 0) == ["Topic 0.1", "Topic 0.2", "Topic 0.3"] + NAMES[0][3:]
    assert store.replaced_topic_names.computed() == {
        (0, 0): "Same",
        (0, 1): "Same",
        (0, 2): "Same",
    }


def test_a_failed_group_keeps_its_names(monkeypatch):
    monkeypatch.setattr(
        TopicDisambiguationAnnotator._generate_and_parse.retry, "wait", wait_none()
    )
    store = make_store()
    annotator = make_annotator(CountingNamer(fail_calls={0, 1, 2}), store)

    with pytest.warns(UserWarning, match="All retries exhausted"):
        failures = Executor(store).run(annotator, nodes=store.tree.layer(0))

    assert list(failures) == [((0, 0), (0, 1), (0, 2))]
    assert names_of(store, 0) == NAMES[0]


def test_the_prompt_lists_each_topics_children_and_subtopics():
    store = make_store()
    subtopics = Annotation("subtopics", store.tree)
    for node in store.tree.layer(1):
        subtopics[node] = [f"subtopic of {node}"]
    store.add(subtopics)
    namer = CountingNamer()

    Executor(store).run(make_annotator(namer, store), nodes=store.tree.layer(1))

    (prompt,) = namer.disambiguation_prompts
    user = prompt["user"]
    assert '"1. Broad"' in user and '"2. Broad"' in user
    assert "Major subtopics of this group are:\n    * Same\n    * Same" in user
    assert "Major subtopics of this group are:\n    * Same\n    * Other" in user
    assert "subtopic of (1, 0)" in user and "keyphrase of (1, 1)" in user
    assert "exemplar of (1, 0)" in user


@pytest.mark.parametrize(
    "response",
    [
        '{"new_topic_name_mapping": {"1.": "A"}, "topic_specificities": [0.9]}',
        '{"new_topic_name_mapping": {"1.": "A", "2.": ""}, "topic_specificities": [0.9, 0.9]}',
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

    assert annotator.execution_mode == "async"
    assert not Executor(store).run(annotator, nodes=store.tree.layer(0))
    assert len(set(names_of(store, 0))) == len(NAMES[0])
    assert set(store.replaced_topic_names.computed().values()) == {"Same"}
