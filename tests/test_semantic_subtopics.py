import numpy as np
import pytest
import json
from toponymy import Toponymy, PrecomputedClusterer
from toponymy.feature_extraction import SubtopicExtractor


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


class FixedNameEmbedder:
    model = "fixed-name-vectors-v1"
    input_type = "topic_name"

    def __init__(self):
        self.vectors = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.995, 0.1]),
            "C": np.array([0.0, 1.0]),
            "A revised": np.array([-1.0, 0.0]),
        }
        self.calls = []
        self.returned_arrays = []

    def encode(self, names, **kwargs):
        self.calls.append(tuple(names))
        # No vector exists for the original objects or terminal parent name:
        # an unnecessary embedding request is a fixture failure, not a fallback.
        result = np.vstack([self.vectors[name] for name in names])
        self.returned_arrays.append(result)
        return result


class RenameOneNamer(SequenceNamer):
    def generate_topic_cluster_names(self, prompt, names, *, response_parser):
        self.rename_calls.append((tuple(names), prompt._asdict()))
        assert tuple(names) == ("A", "B")
        return response_parser(rename_response(["A revised", "B"]))


def make_vector_consumer_model(*, semantic=True, disambiguate=True):
    embedder = FixedNameEmbedder()
    namer = RenameOneNamer(["A", "B", "C", "Parent"])
    extractor = SubtopicExtractor(
        n_subtopics=8, selection_method="central" if semantic else "size"
    )
    model = Toponymy(
        namer,
        text_embedding_model=embedder,
        clusterer=PrecomputedClusterer([[7, 70, 700, -1], [90000, 90000, 90000, -1]]),
        feature_extractors=[extractor],
        disambiguate=disambiguate,
        disambiguation_threshold=0.9,
    )
    return model, namer, embedder


def test_name_vectors_are_retained_reused_and_invalidated_only_for_changed_text():
    model, namer, embedder = make_vector_consumer_model()
    model.prepare(["first object", "second object", "third object", "noise"], np.eye(4))
    assert embedder.calls == []
    model.name_topics()

    assert embedder.calls == [("A", "B", "C"), ("A revised",)]
    assert len(namer.name_calls) == 4
    assert len(namer.rename_calls) == 1
    assert model.request_counts_ == {
        "naming": 4,
        "disambiguation": 1,
        "name_embeddings": 2,
    }
    expected_names = {7: "A revised", 70: "B", 700: "C"}
    for label, name in expected_names.items():
        topic = model.topics_[(0, label)]
        assert topic.name == topic.embedded_name == name
        np.testing.assert_array_equal(topic.name_embedding, embedder.vectors[name])
        assert not topic.name_embedding.flags.writeable
        assert all(
            not np.shares_memory(topic.name_embedding, returned)
            for returned in embedder.returned_arrays
        )

    parent = model.topics_[(1, 90000)]
    assert parent.name_embedding is None
    assert parent.embedded_name is None
    assert set(parent.features["cluster_subtopics"]["misc"]) == set(
        expected_names.values()
    )
    assert parent.features["cluster_subtopics"]["major"] == []
    assert parent.features["cluster_subtopics"]["minor"] == []
    assert model.topic_model_.name_embedding_context["model"] == embedder.model

    history = model.topic_model_.disambiguation_history
    assert len(history) == 1
    assert history[0]["input_names"] == ["A", "B"]
    assert history[0]["output_names"] == ["A revised", "B"]
    assert history[0]["status"] == "succeeded"
    np.testing.assert_array_equal(
        history[0]["input_name_embeddings"], [[1.0, 0.0], [0.995, 0.1]]
    )

    for returned in embedder.returned_arrays:
        returned[:] = 12345
    for label, name in expected_names.items():
        np.testing.assert_array_equal(
            model.topics_[(0, label)].name_embedding, embedder.vectors[name]
        )
    np.testing.assert_array_equal(
        history[0]["input_name_embeddings"], [[1.0, 0.0], [0.995, 0.1]]
    )

    model.name_topics()
    assert embedder.calls == [("A", "B", "C"), ("A revised",)]
    assert len(namer.name_calls) == 4
    assert len(namer.rename_calls) == 1

    previous = model.topic_model_
    model.prepare(["next one", "next two", "next three", "noise"], np.eye(4))
    assert model.topic_model_ is not previous
    assert model.topic_model_.disambiguation_history == []
    assert all(topic.name_embedding is None for topic in model.topics_.values())
    assert previous.topics[(0, 7)].embedded_name == "A revised"


def test_semantic_consumer_embeds_names_even_with_disambiguation_disabled():
    model, namer, embedder = make_vector_consumer_model(disambiguate=False)
    model.fit(["first object", "second object", "third object", "noise"], np.eye(4))
    assert embedder.calls == [("A", "B", "C")]
    assert namer.rename_calls == []
    assert model.request_counts_ == {
        "naming": 4,
        "disambiguation": 0,
        "name_embeddings": 1,
    }
    assert model.topic_model_.disambiguation_history == []


def test_no_semantic_consumer_and_disabled_disambiguation_do_not_encode_names():
    model, namer, embedder = make_vector_consumer_model(
        semantic=False, disambiguate=False
    )
    model.fit(["first object", "second object", "third object", "noise"], np.eye(4))
    assert embedder.calls == []
    assert namer.rename_calls == []
    assert model.request_counts_["name_embeddings"] == 0
    assert all(topic.name_embedding is None for topic in model.topics_.values())


@pytest.mark.parametrize(
    "method",
    ["central", "information_weighted", "facility_location", "saturated_coverage"],
)
def test_semantic_work_requires_embedder_before_any_naming(method):
    namer = SequenceNamer([])
    model = Toponymy(
        namer,
        clusterer=PrecomputedClusterer([[7, 70], [900, 900]]),
        feature_extractors=[SubtopicExtractor(selection_method=method)],
    )
    with pytest.raises(ValueError, match="embedding model"):
        model.prepare(["one", "two"], np.eye(2))
    assert namer.name_calls == []


@pytest.mark.parametrize(
    "labels", [[[7, 70]], [[-1, -1], [-1, -1]], [[7, 70], [-1, -1]]]
)
def test_no_semantic_work_does_not_require_a_text_model(labels):
    Toponymy(
        SequenceNamer([]),
        clusterer=PrecomputedClusterer(labels),
        feature_extractors=[SubtopicExtractor(selection_method="central")],
    ).prepare(["one", "two"], np.eye(2))


def test_information_crossing_is_rejected_during_prepare():
    namer = SequenceNamer([])
    embedder = FixedNameEmbedder()
    model = Toponymy(
        namer,
        text_embedding_model=embedder,
        clusterer=PrecomputedClusterer([[7, 7], [70, 700]]),
        feature_extractors=[SubtopicExtractor(selection_method="information_weighted")],
    )
    with pytest.raises(ValueError, match="multiple parents"):
        model.prepare(["one", "two"], np.eye(2))
    assert namer.name_calls == embedder.calls == []


@pytest.mark.parametrize(
    "method",
    ["central", "information_weighted", "facility_location", "saturated_coverage"],
)
def test_semantic_source_uses_selected_field_and_original_ids(method):
    from toponymy.serialization import Topic

    clusterer = PrecomputedClusterer([[7, 70, 700, -1], [90000, 90000, 90000, -1]]).fit(
        np.eye(4)
    )
    topics = {
        (i, cluster.label): Topic(
            i,
            cluster.label,
            cluster.members,
            name=f"Name {cluster.label}",
            summary=f"Résumé {cluster.label}",
        )
        for i, layer in enumerate(clusterer)
        for cluster in layer
    }
    vectors = {(0, label): vector for label, vector in zip([7, 70, 700], np.eye(3))}
    extractor = SubtopicExtractor(
        n_subtopics=8, source="summary", selection_method=method
    ).fit([], clusterer)
    features = extractor.extract_layer(
        1, topics, clusterer, topic_name_embeddings=vectors
    )
    assert features == [
        {"major": [], "minor": [], "misc": ["Résumé 7", "Résumé 70", "Résumé 700"]}
    ]
    topics[(0, 70)].summary = None
    with pytest.raises(ValueError, match="summary"):
        extractor.extract_layer(1, topics, clusterer, topic_name_embeddings=vectors)
