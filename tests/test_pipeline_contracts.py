import asyncio
import json

import numpy as np
import pytest

from toponymy import PrecomputedClusterer, Toponymy, TopicModel
from toponymy.feature_extraction import SubtopicExtractor, TextExemplarExtractor
from toponymy.templates import Prompt, SummaryTemplate


class RecordingNamer:
    def __init__(self, responses=None):
        self.responses = iter(responses or [])
        self.calls = []

    def generate_topic_name(self, prompt, *, response_parser):
        self.calls.append(prompt)
        value = next(self.responses, f"Topic {len(self.calls)}")
        if isinstance(value, Exception):
            raise value
        return response_parser(
            json.dumps(
                {
                    "topic_name": value,
                    "topic_specificity": 0.5,
                    "topic_summary": "A summary",
                    "topic_analysis": "Evidence",
                }
            )
        )

    def generate_topic_cluster_names(self, prompt, names, *, response_parser):
        self.calls.append(prompt)
        return response_parser(
            json.dumps(
                {
                    "new_topic_name_mapping": {
                        str(i + 1): f"{name} {i + 1}" for i, name in enumerate(names)
                    },
                    "topic_specificities": [0.5] * len(names),
                }
            )
        )


class AsyncRecordingNamer(RecordingNamer):
    async def generate_topic_names(
        self, prompts, *, response_parser, return_results=False
    ):
        from toponymy.llm_wrappers import CallResult

        names = [
            super(AsyncRecordingNamer, self).generate_topic_name(
                prompt, response_parser=response_parser
            )
            for prompt in prompts
        ]
        return [CallResult(value=name) for name in names] if return_results else names

    async def generate_topic_cluster_names(self, prompts, names, *, response_parser):
        return [
            super(AsyncRecordingNamer, self).generate_topic_cluster_names(
                prompt, group, response_parser=response_parser
            )
            for prompt, group in zip(prompts, names)
        ]


def make_model(namer=None, **kwargs):
    return Toponymy(
        namer or RecordingNamer(),
        clusterer=PrecomputedClusterer(
            [[7, 7, 90000, 90000, -1], [20, 20, 20, 20, -1]]
        ),
        feature_extractors=[
            TextExemplarExtractor(selection_method="random", n_exemplars=1),
            SubtopicExtractor(),
        ],
        **kwargs,
    )


OBJECTS = ["cats purr", "cats sleep", "stars shine", "stars orbit", "noise"]
VECTORS = np.arange(15, dtype=float).reshape(5, 3)


def test_prepare_inspects_prompts_without_requests_and_names_dependencies():
    model = make_model().prepare(OBJECTS, VECTORS, VECTORS[:, :2])
    assert model.llm_wrapper.calls == []
    assert set(model.topics_) == {(0, 7), (0, 90000), (1, 20)}
    assert all(isinstance(topic.prompt, Prompt) for topic in model.topics_.values())
    assert model.clusterable_vectors_.shape == (5, 2)
    assert model.embedding_vectors_.shape == (5, 3)
    assert not model.embedding_vectors_.flags.writeable
    assert VECTORS.flags.writeable
    model.name_topics()
    assert len(model.llm_wrapper.calls) == 3
    assert "Topic 1" in model.topics_[(1, 20)].prompt.user
    assert "Topic 2" in model.topics_[(1, 20)].prompt.user
    assert model.topic_names_ == [{7: "Topic 1", 90000: "Topic 2"}, {20: "Topic 3"}]
    assert model.topic_sizes_ == [{7: 2, 90000: 2}, {20: 4}]
    assert model.topic_name_vectors_[0].tolist() == [
        "Topic 1",
        "Topic 1",
        "Topic 2",
        "Topic 2",
        "Unlabelled",
    ]
    assert not hasattr(model.cluster_layers_[0][0], "features")
    model.name_topics()
    assert len(model.llm_wrapper.calls) == 3


def test_refit_and_shared_clusterer_keep_topic_state_independent():
    first = make_model().fit(OBJECTS, VECTORS)
    previous = first.topic_model_
    second = Toponymy(
        RecordingNamer(["Different", "Other", "Top"]),
        clusterer=first.clusterer,
        feature_extractors=[],
    ).fit(OBJECTS, VECTORS)
    assert first.topic_names_[0][7] == "Topic 1"
    assert second.topic_names_[0][7] == "Different"
    first.fit(OBJECTS, VECTORS)
    assert first.topic_model_ is not previous
    assert previous.topics[(0, 7)].name == "Topic 1"


def test_duplicate_disambiguation_is_enabled_and_counted():
    model = make_model(RecordingNamer(["Same", "same", "Parent"])).fit(OBJECTS, VECTORS)
    assert model.topic_names_[0] == {7: "Same 1", 90000: "same 2"}
    assert model.request_counts_ == {
        "naming": 3,
        "disambiguation": 1,
        "name_embeddings": 0,
    }


def test_disambiguation_switch_changes_behavior_explicitly():
    model = make_model(
        RecordingNamer(["Same", "Same", "Parent"]), disambiguate=False
    ).fit(OBJECTS, VECTORS)
    assert model.topic_names_[0] == {7: "Same", 90000: "Same"}
    assert len(model.llm_wrapper.calls) == 3


def test_partial_failure_keeps_prompts_and_propagates_programming_error():
    model = make_model(
        RecordingNamer(["First", TypeError("provider defect"), "Second", "Parent"])
    )
    with pytest.raises(TypeError, match="provider defect"):
        model.fit(OBJECTS, VECTORS)
    assert model.topics_[(0, 7)].name == "First"
    assert model.topics_[(0, 90000)].name is None
    assert len(model.llm_wrapper.calls) == 2
    model.name_topics()
    assert model.topics_[(0, 7)].name == "First"
    assert len(model.llm_wrapper.calls) == 4


@pytest.mark.asyncio
async def test_async_pipeline_matches_sync_and_uses_explicit_event_loop():
    model = make_model(AsyncRecordingNamer(["Same", "Same", "Parent"]))
    assert await model.fit_async(OBJECTS, VECTORS) is model
    assert model.topic_names_ == [{7: "Same 1", 90000: "Same 2"}, {20: "Parent"}]
    assert len(model.llm_wrapper.calls) == 4


@pytest.mark.asyncio
async def test_cancellation_propagates_with_inspectable_unfinished_state():
    class CancelledNamer:
        async def generate_topic_names(self, *args, **kwargs):
            raise asyncio.CancelledError

    model = make_model(CancelledNamer())
    with pytest.raises(asyncio.CancelledError):
        await model.fit_async(OBJECTS, VECTORS)
    assert all(topic.name is None for topic in model.topics_.values())


@pytest.mark.parametrize(
    "labels, objects, vectors",
    [([], [], np.empty((0, 3))), ([[-1, -1]], ["a", "b"], np.ones((2, 3)))],
)
def test_empty_and_noise_make_no_requests(labels, objects, vectors):
    model = Toponymy(RecordingNamer(), clusterer=PrecomputedClusterer(labels)).fit(
        objects, vectors
    )
    assert model.topics_ == {}
    assert model.llm_wrapper.calls == []


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones(5),
        np.ones((4, 2)),
        np.ones((5, 0)),
        np.full((5, 2), np.nan),
        np.full((5, 2), np.inf),
        np.ones((5, 2), dtype=complex),
    ],
)
def test_invalid_semantic_vectors_fail_before_requests(vectors):
    model = make_model()
    with pytest.raises(ValueError):
        model.fit(OBJECTS, vectors)
    assert model.llm_wrapper.calls == []


def test_summary_template_stores_three_outputs():
    model = make_model(prompt_template=SummaryTemplate("documents", "sample")).fit(
        OBJECTS, VECTORS
    )
    assert model.topics_[(0, 7)].summary == "A summary"
    assert model.topics_[(0, 7)].explanation == "Evidence"


def test_default_components_are_fresh():
    first, second = Toponymy(RecordingNamer()), Toponymy(RecordingNamer())
    assert first.clusterer is not second.clusterer
    assert first.feature_extractors[0] is not second.feature_extractors[0]
    assert first.prompt_template is not second.prompt_template


def test_invalid_name_does_not_poison_retry_state():
    from types import SimpleNamespace

    topic = SimpleNamespace(name=None, summary=None, explanation=None)
    with pytest.raises(ValueError, match="empty"):
        Toponymy._store_name(topic, ("  ", "summary", "explanation"))
    assert (topic.name, topic.summary, topic.explanation) == (None, None, None)


@pytest.mark.asyncio
async def test_async_partial_results_preserve_successful_topics_on_retry():
    from toponymy.llm_wrappers import CallResult

    class PartialNamer(AsyncRecordingNamer):
        async def generate_topic_names(
            self, prompts, *, response_parser, return_results=False
        ):
            if not self.calls:
                self.calls.extend(prompts)
                return [
                    CallResult(value="First"),
                    CallResult(error=TimeoutError("retry later")),
                ]
            return await super().generate_topic_names(
                prompts, response_parser=response_parser, return_results=return_results
            )

    model = make_model(PartialNamer())
    with pytest.raises(TimeoutError, match="retry later"):
        await model.fit_async(OBJECTS, VECTORS)
    assert model.topics_[(0, 7)].name == "First"
    assert model.topics_[(0, 90000)].name is None
    await model.name_topics_async()
    assert model.topics_[(0, 7)].name == "First"
    assert len(model.llm_wrapper.calls) == 4


def test_pipeline_accepts_precomputed_distance_graph_without_treating_it_as_embeddings(
    monkeypatch,
):
    import sys
    from types import SimpleNamespace
    from scipy import sparse
    from toponymy import PLSCANClusterer

    captured = []

    class ExternalPLSCAN:
        def __init__(self, **kwargs):
            assert kwargs["metric"] == "precomputed"

        def fit(self, data):
            captured.append(data)
            self.cluster_layers_ = [np.array([0, 0, 1, 1, -1])]

    monkeypatch.setitem(
        sys.modules, "fast_hdbscan", SimpleNamespace(PLSCAN=ExternalPLSCAN)
    )
    graph = sparse.csr_matrix(np.ones((5, 5)) - np.eye(5))
    model = Toponymy(
        RecordingNamer(),
        clusterer=PLSCANClusterer(metric="precomputed"),
        feature_extractors=[],
    ).fit(OBJECTS, VECTORS, graph)
    assert sparse.issparse(captured[0])
    assert graph.data.flags.writeable
    assert model.embedding_vectors_.shape == (5, 3)
    assert model.topic_model_.reduced_vectors is None
    assert model.topic_model_.clustering_graph.shape == (5, 5)


def test_explicit_keyphrase_inputs_do_not_require_an_embedding_provider():
    from scipy import sparse
    from toponymy.feature_extraction import TextKeyphraseExtractor

    model = Toponymy(
        RecordingNamer(),
        clusterer=PrecomputedClusterer([[7, 7, 9, 9, -1]]),
        feature_extractors=[TextKeyphraseExtractor("central", n_keyphrases=1)],
        feature_options={
            "cluster_keywords": {
                "object_x_keyphrase_matrix": sparse.csr_matrix(
                    [[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]]
                ),
                "keyphrase_list": ["cats", "stars"],
                "keyphrase_vectors": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        },
    ).fit(OBJECTS, VECTORS)
    assert model.topics_[(0, 7)].features["cluster_keywords"] == ["cats"]
    assert model.topics_[(0, 9)].features["cluster_keywords"] == ["stars"]


@pytest.mark.parametrize("previous_fit", [False, True])
def test_failed_prepare_cannot_be_named_and_can_be_prepared_again(previous_fit):
    from sklearn.exceptions import NotFittedError

    extractor = SubtopicExtractor()
    model = Toponymy(
        RecordingNamer(),
        clusterer=PrecomputedClusterer([[7, 7]]),
        feature_extractors=[extractor],
    )
    objects, vectors = ["apple", "orchard"], np.eye(2)
    if previous_fit:
        model.fit(objects, vectors)
    before = len(model.llm_wrapper.calls)
    extractor.source = None
    with pytest.raises(ValueError, match="source"):
        model.prepare(objects, vectors)
    with pytest.raises(NotFittedError):
        model.name_topics()
    assert len(model.llm_wrapper.calls) == before
    extractor.source = "name"
    model.prepare(objects, vectors).name_topics()
    assert len(model.llm_wrapper.calls) == before + 1


def test_precomputed_labels_accept_a_separate_sparse_clustering_graph():
    from scipy import sparse

    graph = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    model = Toponymy(
        RecordingNamer(),
        clusterer=PrecomputedClusterer([[7, 7]]),
        feature_extractors=[],
    ).fit(["apple", "orchard"], np.eye(2), graph)
    assert model.topic_names_ == [{7: "Topic 1"}]
    np.testing.assert_array_equal(
        model.topic_model_.clustering_graph.toarray(), graph.toarray()
    )
    assert graph.data.flags.writeable
