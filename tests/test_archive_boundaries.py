"""Reject ambiguous archive structure while retaining portable topic identities."""

import json
import zipfile

import numpy as np
import pytest
from scipy import sparse

from toponymy.serialization import Topic, TopicModel, topic_uid, uid_to_ints


@pytest.fixture
def topic_archive(tmp_path):
    vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    topics = {
        (0, 7): Topic(0, 7, np.array([0, 1]), name="Orchard"),
        (0, 90): Topic(0, 90, np.array([2, 3]), name="River"),
    }
    membership = sparse.csr_matrix(
        np.array([[255, 0], [255, 0], [0, 255], [0, 255]], dtype=np.uint8)
    )
    model = TopicModel(
        None,
        {(1, 0): [(0, 7), (0, 90)]},
        [membership],
        vectors,
        reduced_vectors=vectors[:, :1],
        topics=topics,
    )
    path = tmp_path / "valid.zip"
    model.to_file(path)
    return path


def rewrite_archive(source, destination, *, omit=(), replace=None, append=()):
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(destination, "w") as changed,
    ):
        for member in original.infolist():
            if member.filename not in omit:
                changed.writestr(
                    member, (replace or {}).get(member.filename, original.read(member))
                )
        for name, content in append:
            changed.writestr(name, content)


def test_current_archive_control_retains_sparse_topic_members(topic_archive):
    loaded = TopicModel.from_file(topic_archive)
    assert loaded.topic_names == [{7: "Orchard", 90: "River"}]
    assert loaded.topics[(0, 7)].members.tolist() == [0, 1]
    assert loaded.topics[(0, 90)].members.tolist() == [2, 3]
    np.testing.assert_array_equal(loaded.reduced_vectors, [[1.0], [0.8], [0.0], [0.2]])


def test_archive_rejects_missing_declared_cluster_layer(topic_archive, tmp_path):
    malformed = tmp_path / "missing-layer.zip"
    rewrite_archive(topic_archive, malformed, omit=("cluster_matrices/layer_0.npz",))
    with pytest.raises(ValueError):
        TopicModel.from_file(malformed)


def test_archive_rejects_declared_layer_count_mismatch(topic_archive, tmp_path):
    with zipfile.ZipFile(topic_archive) as original:
        metadata = json.loads(original.read("metadata.json"))
    metadata["n_layers"] = 2
    malformed = tmp_path / "layer-count.zip"
    rewrite_archive(
        topic_archive, malformed, replace={"metadata.json": json.dumps(metadata)}
    )
    with pytest.raises(ValueError):
        TopicModel.from_file(malformed)


def test_archive_rejects_duplicate_required_member(topic_archive, tmp_path):
    with zipfile.ZipFile(topic_archive) as original:
        metadata = original.read("metadata.json")
    malformed = tmp_path / "duplicate-metadata.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        rewrite_archive(topic_archive, malformed, append=(("metadata.json", metadata),))
    with pytest.raises(ValueError):
        TopicModel.from_file(malformed)


@pytest.mark.parametrize("identifier", ["", "!!!!", "AAAB!", "AAAAAA"])
def test_compact_uid_rejects_malformed_identity(identifier):
    with pytest.raises(ValueError):
        uid_to_ints(identifier)


@pytest.mark.parametrize("key", [(0, -1), (0, 0), (12, 1022), (2, 90000)])
def test_strict_uid_validation_preserves_legacy_and_large_ids(key):
    assert uid_to_ints(topic_uid(key)) == key
