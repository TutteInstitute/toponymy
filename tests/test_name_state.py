import asyncio


from collections import Counter


from copy import deepcopy


import json


from types import SimpleNamespace


import numpy as np


import pytest


from toponymy import PrecomputedClusterer, Toponymy, TopicModel


from toponymy.llm_wrappers import CallResult


from toponymy.serialization import Topic


from toponymy.templates import Prompt, TextTemplate


def name_response(name):
    return json.dumps({"topic_name": name, "topic_specificity": 0.5})


def rename_response(names):
    return json.dumps(
        {
            "new_topic_name_mapping": {
                str(index + 1): name for index, name in enumerate(names)
            },
            "topic_specificities": [0.5] * len(names),
        }
    )


class SequenceNamer:
    def __init__(self, names):
        self.names = iter(names)
        self.name_calls = []
        self.rename_calls = []

    def generate_topic_name(self, prompt, *, response_parser):
        self.name_calls.append(prompt)
        return response_parser(name_response(next(self.names)))

    def generate_topic_cluster_names(self, prompt, names, *, response_parser):
        self.rename_calls.append((tuple(names), prompt._asdict()))
        return response_parser(rename_response(names))


class TwoGroupNamer(SequenceNamer):
    def __init__(self, failure_type):
        super().__init__(["Alpha", "Alpha", "Beta", "Beta"])
        self.failure_type = failure_type

    def _rename(self, prompt, names, response_parser):
        self.rename_calls.append((tuple(names), prompt._asdict()))
        if len(self.rename_calls) == 2:
            raise self.failure_type("second group interrupted")
        # Intentionally still duplicate within this successful group. These are
        # valid nonempty parser outputs. Recomputing groups on resume would
        # resubmit the prior successful group and incur another request.
        renamed = [name + " reviewed" for name in names]
        return response_parser(rename_response(renamed))

    def generate_topic_cluster_names(self, prompt, names, *, response_parser):
        return self._rename(prompt, names, response_parser)


class AsyncTwoGroupNamer(TwoGroupNamer):
    async def generate_topic_names(
        self, prompts, *, response_parser, return_results=False
    ):
        values = [
            SequenceNamer.generate_topic_name(
                self, prompt, response_parser=response_parser
            )
            for prompt in prompts
        ]
        return (
            [CallResult(value=value) for value in values] if return_results else values
        )

    async def generate_topic_cluster_names(self, prompts, names, *, response_parser):
        return [
            self._rename(prompt, group, response_parser)
            for prompt, group in zip(prompts, names)
        ]


def two_group_model(namer):
    return Toponymy(
        namer,
        clusterer=PrecomputedClusterer([[7, 70, 700, 7000]]),
        feature_extractors=[],
    ).prepare(["one", "two", "three", "four"], np.eye(4))


def assert_interrupted_groups(model, namer, failure_type):
    assert len(namer.name_calls) == 4
    assert len(namer.rename_calls) == 2
    history = model.topic_model_.disambiguation_history
    assert len(history) == 2
    expected_status = (
        "cancelled" if failure_type is asyncio.CancelledError else "failed"
    )
    assert Counter(record["status"] for record in history) == Counter(
        {"succeeded": 1, expected_status: 1}
    )
    assert {tuple(record["input_names"]) for record in history} == {
        ("Alpha", "Alpha"),
        ("Beta", "Beta"),
    }
    failed = next(record for record in history if record["status"] == expected_status)
    assert failed["output_names"] is None
    assert failed["attempts"] == 1
    assert failed["errors"][-1]["type"].endswith(failure_type.__name__)
    assert "second group interrupted" in failed["errors"][-1]["message"]
    assert failed["prompt"] == namer.rename_calls[1][1]
    return deepcopy(history)


def assert_resumed_groups(model, namer, before):
    assert len(namer.name_calls) == 4
    assert len(namer.rename_calls) == 3
    assert namer.rename_calls[2] == namer.rename_calls[1]
    assert namer.rename_calls[0][0] != namer.rename_calls[1][0]
    assert model.request_counts_ == {
        "naming": 4,
        "disambiguation": 3,
        "name_embeddings": 0,
    }
    assert model.topic_names_ == [
        {
            7: "Alpha reviewed",
            70: "Alpha reviewed",
            700: "Beta reviewed",
            7000: "Beta reviewed",
        }
    ]
    history = model.topic_model_.disambiguation_history
    assert len(history) == 2
    assert all(record["status"] == "succeeded" for record in history)
    assert sorted(record["attempts"] for record in history) == [1, 2]
    for previous in before:
        current = next(
            record for record in history if record["group"] == previous["group"]
        )
        assert current["input_names"] == previous["input_names"]
        assert current["prompt"] == previous["prompt"]
        if previous["status"] == "succeeded":
            assert current == previous
        else:
            assert current["errors"] == previous["errors"]


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
def test_sync_resume_never_resubmits_a_successful_disambiguation_group(failure_type):
    namer = TwoGroupNamer(failure_type)
    model = two_group_model(namer)
    with pytest.raises(failure_type, match="second group interrupted"):
        model.name_topics()
    before = assert_interrupted_groups(model, namer, failure_type)
    assert model.name_topics() is model
    assert_resumed_groups(model, namer, before)
    model.name_topics()
    assert len(namer.rename_calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_async_resume_never_resubmits_a_successful_disambiguation_group(
    failure_type,
):
    namer = AsyncTwoGroupNamer(failure_type)
    model = two_group_model(namer)
    with pytest.raises(failure_type, match="second group interrupted"):
        await model.name_topics_async()
    before = assert_interrupted_groups(model, namer, failure_type)
    assert await model.name_topics_async() is model
    assert_resumed_groups(model, namer, before)
    await model.name_topics_async()
    assert len(namer.rename_calls) == 3


def state_fixture():
    labels = [np.array([7, 70, 700, 7000, -1])]
    clusterer = PrecomputedClusterer(labels).fit(np.zeros((5, 2)))
    names = {7: "Orchard", 70: "Garden", 700: "Sky", 7000: "Sky"}
    vectors = {
        7: np.array([1.0, 0.0], dtype=np.float32),
        70: np.array([0.0, 1.0], dtype=np.float32),
        700: np.array([1.0, 1.0], dtype=np.float32),
        7000: np.array([1.0, 1.0], dtype=np.float32),
    }
    topics = {
        (0, cluster.label): Topic(
            0,
            cluster.label,
            cluster.members,
            name=names[cluster.label],
            name_embedding=vectors[cluster.label],
            embedded_name=names[cluster.label],
        )
        for cluster in clusterer.cluster_layers_[0]
    }
    topics[(0, 7)].prompt = Prompt(
        "Naming system",
        "Naming user",
        {"type": "object"},
        combined="独立 combined prompt",
    )
    model = TopicModel.from_topics(
        topics,
        clusterer.cluster_layers_,
        clusterer.cluster_tree_,
        np.arange(15, dtype=np.float64).reshape(5, 3),
        np.arange(5, dtype=np.float64).reshape(5, 1),
    )
    model.name_embedding_context = {
        "embedder_class": "local.FixedNameEmbedder",
        "model": "fixed-name-vectors-v1",
        "input_type": "topic_name",
        "scope": "prepared_fit",
    }
    model.disambiguation_history = [
        {
            "layer": 0,
            "group": 0,
            "topic_keys": [[0, 7], [0, 70]],
            "input_names": ["Plants", "Plants"],
            "input_name_embeddings": np.array(
                [[1.0, 0.1], [1.0, 0.1]], dtype=np.float32
            ).tolist(),
            "embedding_dtype": "float32",
            "prompt": Prompt("S0", "U0", combined="Rename plants")._asdict(),
            "status": "succeeded",
            "output_names": ["Orchard", "Garden"],
            "attempts": 1,
            "errors": [],
        },
        {
            "layer": 0,
            "group": 1,
            "topic_keys": [[0, 700], [0, 7000]],
            "input_names": ["Sky", "Sky"],
            "input_name_embeddings": [[1.0, 1.0], [1.0, 1.0]],
            "embedding_dtype": "float32",
            "prompt": Prompt("S1", "U1", combined="Rename sky")._asdict(),
            "status": "cancelled",
            "output_names": None,
            "attempts": 1,
            "errors": [
                {
                    "type": "asyncio.exceptions.CancelledError",
                    "message": "fixture cancellation",
                }
            ],
        },
    ]
    return model, names, vectors


def assert_state_snapshot(restored, original, names, vectors):
    assert restored.name_embedding_context == original.name_embedding_context
    assert restored.disambiguation_history == original.disambiguation_history
    assert restored.topics[(0, 7)].prompt.combined == "独立 combined prompt"
    assert restored.topics[(0, 70)].prompt is None
    np.testing.assert_array_equal(
        restored.embedding_vectors, original.embedding_vectors
    )
    np.testing.assert_array_equal(restored.reduced_vectors, original.reduced_vectors)
    assert restored.topic_name_vectors[0].tolist() == [
        "Orchard",
        "Garden",
        "Sky",
        "Sky",
        "Unlabelled",
    ]
    for label, name in names.items():
        topic = restored.topics[(0, label)]
        assert topic.name == topic.embedded_name == name
        assert topic.name_embedding.dtype == np.float32
        assert not topic.name_embedding.flags.writeable
        np.testing.assert_array_equal(topic.name_embedding, vectors[label])


@pytest.mark.parametrize("format", ["zip", "lance"])
def test_name_vectors_context_history_and_combined_prompts_round_trip(format, tmp_path):
    original, names, vectors = state_fixture()
    path = tmp_path / (
        "semantic-state.zip" if format == "zip" else "semantic-state.lance"
    )
    if format == "zip":
        original.to_file(path)
        restored = TopicModel.from_file(path)
    else:
        pytest.importorskip("lance")
        original.to_lance(path)
        restored = TopicModel.from_lance(path)
    assert_state_snapshot(restored, original, names, vectors)

    restored.name_embedding_context["model"] = "changed after load"
    restored.disambiguation_history[0]["input_names"][0] = "changed after load"
    assert original.name_embedding_context["model"] == "fixed-name-vectors-v1"
    assert original.disambiguation_history[0]["input_names"] == ["Plants", "Plants"]


def test_from_toponymy_preserves_independent_name_and_disambiguation_state():
    original, names, vectors = state_fixture()
    restored = TopicModel.from_toponymy(SimpleNamespace(topic_model_=original))
    assert_state_snapshot(restored, original, names, vectors)
    for key in original.topics:
        assert not np.shares_memory(
            restored.topics[key].name_embedding, original.topics[key].name_embedding
        )
    restored.name_embedding_context["model"] = "changed snapshot"
    restored.disambiguation_history[0]["input_names"][0] = "changed snapshot"
    assert original.name_embedding_context["model"] == "fixed-name-vectors-v1"
    assert original.disambiguation_history[0]["input_names"] == ["Plants", "Plants"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedded_name", "stale text"),
        ("embedded_name", None),
        ("name_embedding", None),
        ("name_embedding", np.ones((2, 2))),
        ("name_embedding", np.array([np.nan, 1.0])),
    ],
)
@pytest.mark.parametrize("format", ["zip", "lance"])
def test_invalid_edited_name_pair_preserves_existing_archive(
    field, value, format, tmp_path
):
    model, names, vectors = state_fixture()
    path = tmp_path / ("model.zip" if format == "zip" else "model.lance")
    if format == "lance":
        pytest.importorskip("lance")
    write = model.to_file if format == "zip" else model.to_lance
    read = TopicModel.from_file if format == "zip" else TopicModel.from_lance
    write(path)
    expected = read(path)
    setattr(model.topics[(0, 7)], field, value)
    with pytest.raises(ValueError, match="[Nn]ame|embedded"):
        write(path, **({"overwrite": True} if format == "lance" else {}))
    assert_state_snapshot(read(path), expected, names, vectors)


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([[0.5], [1.5]], "int64"),
        ([[256], [1]], "uint8"),
        ([[1.0], [2.0]], 0),
    ],
)
def test_history_dtype_must_preserve_recorded_values(values, dtype, tmp_path):
    model, _, _ = state_fixture()
    record = model.disambiguation_history[0]
    record["input_name_embeddings"] = values
    record["embedding_dtype"] = dtype
    with pytest.raises(ValueError, match="History embeddings"):
        model.to_file(tmp_path / "invalid.zip")


@pytest.mark.parametrize(
    "status,attempts,error_count",
    [
        ("pending", 0, 1),
        ("succeeded", 1, 1),
        ("failed", 1, 2),
    ],
)
def test_history_attempts_must_account_for_errors(
    status, attempts, error_count, tmp_path
):
    model, _, _ = state_fixture()
    record = model.disambiguation_history[0]
    record.update(
        status=status,
        attempts=attempts,
        errors=[{"type": "RuntimeError", "message": "failed"}] * error_count,
    )
    if status != "succeeded":
        record["output_names"] = None
    with pytest.raises(ValueError, match="History.*attempt"):
        model.to_file(tmp_path / "invalid.zip")


class RecordingEmbedder:
    model = "local-fixture"

    def __init__(self):
        self.calls = []
        self.response = None

    def encode(self, names):
        self.calls.append(tuple(names))
        if self.response is not None:
            return self.response
        return np.array([[len(name), 1.0] for name in names], dtype=np.float32)


def embedding_state_model():
    embedder = RecordingEmbedder()
    model = Toponymy(
        SequenceNamer([]),
        text_embedding_model=embedder,
        clusterer=PrecomputedClusterer([[7, 70, 700], [90000, 90000, 90000]]),
        feature_extractors=[],
        disambiguate=False,
    ).prepare(["one", "two", "three"], np.eye(3))
    for topic, name in zip(model.topics_.values(), ["Alpha", "Beta", "Beta", "Alpha"]):
        topic.name = name
    return model, embedder


def test_exact_names_reuse_owned_vectors_across_layers_and_invalidate_on_edit():
    model, embedder = embedding_state_model()
    first, second, duplicate, parent = model.topics_.values()
    model._ensure_name_embeddings([first, second, duplicate])
    assert embedder.calls == [("Alpha", "Beta")]
    old_vector = first.name_embedding
    first.name = "Alpha"
    assert first.name_embedding is old_vector
    model._ensure_name_embeddings([parent])
    assert embedder.calls == [("Alpha", "Beta")]
    assert not np.shares_memory(parent.name_embedding, first.name_embedding)
    assert not np.shares_memory(second.name_embedding, duplicate.name_embedding)
    first.name = "Gamma changed"
    assert first.name_embedding is first.embedded_name is None
    model._ensure_name_embeddings([first, second])
    assert embedder.calls == [("Alpha", "Beta"), ("Gamma changed",)]
    np.testing.assert_array_equal(parent.name_embedding, old_vector)
    assert first.embedded_name == "Gamma changed"
    assert not first.name_embedding.flags.writeable
    previous = model.topic_model_
    model.prepare(["one", "two", "three"], np.eye(3))
    assert all(topic.name_embedding is None for topic in model.topics_.values())
    assert previous.topics[(0, 7)].embedded_name == "Gamma changed"


@pytest.mark.parametrize(
    "response",
    [
        np.ones((1, 2)),
        np.ones((2, 3)),
        np.ones((2, 0)),
        np.array([[1.0, 2.0], [np.nan, 1.0]]),
        [[1.0], [1.0, 2.0]],
    ],
)
def test_bad_embedding_batch_installs_no_partial_vectors(response):
    model, embedder = embedding_state_model()
    first, second, third, _ = model.topics_.values()
    third.name = "Third"
    model._ensure_name_embeddings([first])
    previous = first.name_embedding
    embedder.response = response
    with pytest.raises(ValueError):
        model._ensure_name_embeddings([second, third])
    assert first.name_embedding is previous
    assert second.name_embedding is third.name_embedding is None
    assert model.request_counts_["name_embeddings"] == 2


def test_replacing_embedder_requires_prepare_but_keeps_results_inspectable():
    model, _ = embedding_state_model()
    model.embedding_model = RecordingEmbedder()
    with pytest.raises(ValueError, match="embedding model changed"):
        model.name_topics()
    assert model.topic_names_[0] == {7: "Alpha", 70: "Beta", 700: "Beta"}


def test_invalid_rename_response_cannot_partly_overwrite_names():
    model, _ = embedding_state_model()
    topics = list(model.topics_.values())[:2]
    model._ensure_name_embeddings(topics)
    vectors = [topic.name_embedding for topic in topics]
    with pytest.raises(ValueError):
        model._store_disambiguation(topics, ["Valid replacement", " "])
    assert [topic.name for topic in topics] == ["Alpha", "Beta"]
    assert all(topic.name_embedding is vector for topic, vector in zip(topics, vectors))
