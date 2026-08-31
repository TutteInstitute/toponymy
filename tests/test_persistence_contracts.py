import zipfile

import numpy as np
import pytest

from toponymy import TopicModel
from toponymy.serialization import topic_uid, uid_to_ints
from test_pipeline_contracts import OBJECTS, VECTORS, make_model


def test_round_trip_preserves_sparse_ids_features_prompts_and_metadata(tmp_path):
    pipeline = make_model().fit(OBJECTS, VECTORS, VECTORS[:, :2])
    model = pipeline.topic_model_
    path = tmp_path / "topics.tm.zip"
    model.to_file(path)
    loaded = TopicModel.from_file(path)
    assert set(loaded.topics) == {(0, 7), (0, 90000), (1, 20)}
    assert loaded.topic_names == model.topic_names
    assert loaded.topic_sizes == model.topic_sizes
    assert loaded.metadata == model.metadata
    assert loaded.cluster_tree == model.cluster_tree
    for key in model.topics:
        assert loaded.topics[key].prompt == model.topics[key].prompt
        assert loaded.topics[key].features == model.topics[key].features
        np.testing.assert_array_equal(
            loaded.topics[key].members, model.topics[key].members
        )
    np.testing.assert_array_equal(loaded.embedding_vectors, VECTORS)
    np.testing.assert_array_equal(loaded.reduced_vectors, VECTORS[:, :2])
    assert [matrix.shape for matrix in loaded.cluster_layers] == [(5, 2), (5, 1)]


def test_from_toponymy_is_a_snapshot_of_names_without_copying_embeddings():
    pipeline = make_model().fit(OBJECTS, VECTORS)
    snapshot = TopicModel.from_toponymy(pipeline)
    pipeline.topics_[(0, 7)].name = "Edited"
    assert snapshot.topics[(0, 7)].name == "Topic 1"
    assert np.shares_memory(snapshot.embedding_vectors, pipeline.embedding_vectors_)


def test_dataframe_is_a_materialized_view_and_topics_are_editable():
    model = make_model().fit(OBJECTS, VECTORS).topic_model_
    table = model.topic_df
    table.loc[0, "name"] = "Table edit"
    assert model.topics[(0, 7)].name == "Topic 1"
    model.topics[(0, 7)].name = "Topic edit"
    assert model.topic_df.loc[0, "name"] == "Topic edit"


@pytest.mark.parametrize(
    "key",
    [(0, -1), (0, 0), (1, 7), (0, 1022), (0, 1023), (4, 90000), (100000, 1000000)],
)
def test_uid_round_trip_has_no_large_label_collisions(key):
    assert uid_to_ints(topic_uid(key)) == key
    assert topic_uid((0, 1023)) != topic_uid((1, -1))


@pytest.mark.parametrize("labels, count", [([], 0), ([[-1, -1]], 2)])
def test_empty_and_all_noise_persistence(tmp_path, labels, count):
    from toponymy import Toponymy, PrecomputedClusterer
    from test_pipeline_contracts import RecordingNamer

    pipeline = Toponymy(RecordingNamer(), clusterer=PrecomputedClusterer(labels)).fit(
        ["noise"] * count, np.ones((count, 3))
    )
    path = tmp_path / "empty.zip"
    pipeline.topic_model_.to_file(path)
    loaded = TopicModel.from_file(path)
    assert loaded.topics == {}
    assert len(loaded.topic_name_vectors) == len(labels)
    assert loaded.embedding_vectors.shape == (count, 3)


def test_load_actual_legacy_fixture():
    from pathlib import Path

    fixture = Path(__file__).parent / "data/mock-20ng.tm.zip"
    model = TopicModel.from_file(fixture)
    assert len(model.topics) > 0
    assert all(len(topic.members) > 0 for topic in model.topics.values())
    assert len(model.topic_name_vectors[0]) == len(model.embedding_vectors)


def test_loading_rejects_archive_path_escape(tmp_path):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.txt", "malformed archive")
    with pytest.raises(ValueError, match="archive"):
        TopicModel.from_file(path)
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("empty", [False, True])
def test_lance_preserves_runtime_state_and_vector_precision(tmp_path, empty):
    from toponymy import PrecomputedClusterer, Toponymy
    from test_pipeline_contracts import RecordingNamer

    if empty:
        pipeline = Toponymy(RecordingNamer(), clusterer=PrecomputedClusterer([])).fit(
            [], np.empty((0, 3), dtype=np.float64)
        )
    else:
        vectors = VECTORS + np.finfo(float).eps
        pipeline = make_model().fit(OBJECTS, vectors, vectors[:, :2])
    model = pipeline.topic_model_
    path = tmp_path / "topics.lance"
    model.to_lance(path)
    loaded = TopicModel.from_lance(path)
    assert loaded.topic_names == model.topic_names
    assert loaded.topic_sizes == model.topic_sizes
    assert loaded.metadata == model.metadata
    assert loaded.embedding_vectors.dtype == model.embedding_vectors.dtype
    np.testing.assert_array_equal(loaded.embedding_vectors, model.embedding_vectors)
    for key in model.topics:
        assert loaded.topics[key].features == model.topics[key].features
        assert loaded.topics[key].prompt == model.topics[key].prompt
    with pytest.raises(FileExistsError):
        model.to_lance(path)


@pytest.mark.parametrize("format", ["zip", "lance"])
def test_sparse_clustering_graph_round_trip(tmp_path, format):
    from scipy import sparse

    model = make_model().fit(OBJECTS, VECTORS).topic_model_
    model.clustering_graph = sparse.csr_matrix(
        np.ones((5, 5), dtype=np.float32) - np.eye(5, dtype=np.float32)
    )
    model.reduced_vectors = None
    path = tmp_path / ("graph." + format)
    if format == "zip":
        model.to_file(path)
        loaded = TopicModel.from_file(path)
    else:
        model.to_lance(path)
        loaded = TopicModel.from_lance(path)
    assert loaded.clustering_graph.dtype == np.float32
    assert loaded.clustering_graph.shape == (5, 5)
    assert (loaded.clustering_graph != model.clustering_graph).nnz == 0
    assert loaded.reduced_vectors is None
