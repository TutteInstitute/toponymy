"""Persistence identity and materialized-view ownership at public load boundaries."""

from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from toponymy.serialization import Topic, TopicModel, topic_uid
from toponymy.templates import Prompt


@pytest.fixture
def topic_model():
    topics = {
        (0, 0): Topic(
            0,
            0,
            np.array([0]),
            features={"cluster_keywords": ["café"]},
            prompt=Prompt("System", "User"),
            name="Café",
        ),
        (0, 90000): Topic(
            0,
            90000,
            np.array([1]),
            features={"cluster_keywords": ["猫"]},
            name="猫",
        ),
    }
    return TopicModel(
        None,
        {(1, 0): [(0, 0), (0, 90000)]},
        [sparse.csr_matrix(np.eye(2, dtype=np.uint8) * 255)],
        np.eye(2, dtype=np.float64),
        topics=topics,
        metadata={"source": "local control"},
    )


@pytest.fixture(params=["zip", "lance"])
def stored_model(request, topic_model, tmp_path):
    format_name = request.param
    path = tmp_path / ("model." + format_name)
    if format_name == "zip":
        topic_model.to_file(path)
    else:
        pytest.importorskip("lance")
        topic_model.to_lance(path)
    return format_name, path


def load_model(format_name, path):
    if format_name == "zip":
        return TopicModel.from_file(path)
    return TopicModel.from_lance(path)


def replace_zip_member(path, member_name, payload):
    changed = path.with_name("changed.zip")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(changed, "w") as target:
        assert member_name in source.namelist()
        for member in source.infolist():
            target.writestr(
                member,
                payload if member.filename == member_name else source.read(member),
            )
    return changed


def replace_topic_table(format_name, path, mutate):
    if format_name == "zip":
        with zipfile.ZipFile(path) as archive:
            table = pd.read_parquet(BytesIO(archive.read("topic_df.parquet")))
        mutate(table)
        payload = BytesIO()
        table.to_parquet(payload)
        return replace_zip_member(path, "topic_df.parquet", payload.getvalue())

    import lance
    import pyarrow as pa

    table_path = str(path / "topics.lance")
    original = lance.dataset(table_path).to_table()
    table = original.to_pandas()
    mutate(table)
    changed = pa.Table.from_pandas(table, schema=original.schema, preserve_index=False)
    lance.write_dataset(changed, table_path, mode="overwrite")
    return path


def replace_tree(format_name, path, tree_json):
    if format_name == "zip":
        return replace_zip_member(path, "cluster_tree.json", tree_json.encode("utf8"))

    import lance
    import pyarrow as pa

    table_path = str(path / "config.lance")
    original = lance.dataset(table_path).to_table()
    config = original.to_pydict()
    config["cluster_tree"] = [tree_json]
    lance.write_dataset(
        pa.table(config, schema=original.schema), table_path, mode="overwrite"
    )
    return path


def test_valid_identity_control_preserves_unicode_membership_and_metadata(
    stored_model, topic_model
):
    loaded = load_model(*stored_model)
    assert loaded.topic_names == [{0: "Café", 90000: "猫"}]
    assert loaded.cluster_tree == topic_model.cluster_tree
    assert loaded.metadata == topic_model.metadata
    assert set(loaded.topics) == set(topic_model.topics)
    for key, expected in topic_model.topics.items():
        np.testing.assert_array_equal(loaded.topics[key].members, expected.members)
        assert loaded.topics[key].features == expected.features
    np.testing.assert_array_equal(
        loaded.embedding_vectors, topic_model.embedding_vectors
    )


def test_loader_rejects_duplicate_topic_identity_before_returning(stored_model):
    format_name, path = stored_model

    def duplicate_identity(table):
        assert table["cluster"].tolist() == [0, 90000]
        table.loc[1, ["uid", "layer", "cluster"]] = table.loc[
            0, ["uid", "layer", "cluster"]
        ].to_numpy()

    changed = replace_topic_table(format_name, path, duplicate_identity)
    with pytest.raises(ValueError):
        load_model(format_name, changed)


@pytest.mark.parametrize("collision", ["repeated-json-key", "compact-v2-alias"])
def test_loader_rejects_ambiguous_parent_identity(stored_model, collision):
    format_name, path = stored_model
    parent = topic_uid((1, 0))
    alias = parent if collision == "repeated-json-key" else "v2:1:0"
    first = json.dumps(parent) + ":" + json.dumps([topic_uid((0, 0))])
    second = json.dumps(alias) + ":" + json.dumps([topic_uid((0, 90000))])
    changed = replace_tree(format_name, path, "{" + first + "," + second + "}")
    with pytest.raises(ValueError):
        load_model(format_name, changed)


def test_editing_materialized_keywords_does_not_mutate_topic_features(topic_model):
    table = topic_model.topic_df
    table.at[0, "keyphrases"].append("table-only")
    assert topic_model.topics[(0, 0)].features["cluster_keywords"] == ["café"]


def test_topic_edit_does_not_change_already_materialized_keywords(topic_model):
    table = topic_model.topic_df
    topic_model.topics[(0, 0)].features["cluster_keywords"].append("topic-only")
    assert table.at[0, "keyphrases"] == ["café"]
    assert topic_model.topic_df.at[0, "keyphrases"] == ["café", "topic-only"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer", 1),
        ("cluster", -1),
        ("cluster", 0.5),
        ("cluster", True),
        ("cluster", "0"),
        ("uid", "v2:0:123"),
    ],
)
def test_topic_table_rejects_invalid_keys_without_partial_materialization(
    topic_model, field, value
):
    table = topic_model.topic_df
    table[field] = table[field].astype(object)
    table.at[0, field] = value
    loaded = TopicModel(
        table,
        topic_model.cluster_tree,
        topic_model.cluster_layers,
        topic_model.embedding_vectors,
    )
    with pytest.raises(ValueError):
        loaded.topics
    assert loaded._topics is None


@pytest.mark.parametrize("missing", [0, 1])
def test_topic_table_cannot_omit_populated_membership_columns(topic_model, missing):
    table = topic_model.topic_df.drop(index=missing)
    loaded = TopicModel(
        table, {}, topic_model.cluster_layers, topic_model.embedding_vectors
    )
    with pytest.raises(ValueError, match="does not describe"):
        loaded.topics


@pytest.mark.parametrize(
    "tree",
    [
        {"v2:1:0": ["v2:0:0", "AAAB"]},
        {"v2:1:0": "AAAB"},
        {"v2:1:0": ["v2:0:999"]},
        {"v2:1:0": ["v2:0:0"], "v2:2:0": ["v2:0:90000"]},
        {"v2:0:0": ["v2:0:90000"]},
        {"v2:1:0": ["v2:0:0"]},
        [],
    ],
)
def test_loader_rejects_invalid_tree_structure(stored_model, tree):
    format_name, path = stored_model
    changed = replace_tree(format_name, path, json.dumps(tree))
    with pytest.raises(ValueError):
        load_model(format_name, changed)


@pytest.mark.parametrize(
    "metadata",
    [
        '{"serial_version":"0.2","serial_version":"0.1","n_layers":1,"has_reduced":false}',
        '{"serial_version":[],"n_layers":1,"has_reduced":false}',
        '{"serial_version":"0.2","n_layers":true,"has_reduced":false}',
        '{"serial_version":"0.2","n_layers":1,"has_reduced":"false"}',
        "[]",
    ],
)
def test_zip_metadata_has_one_typed_interpretation(topic_model, tmp_path, metadata):
    path = tmp_path / "model.zip"
    topic_model.to_file(path)
    changed = replace_zip_member(path, "metadata.json", metadata)
    with pytest.raises(ValueError):
        TopicModel.from_file(changed)


@pytest.mark.parametrize(
    "matrix_type", [sparse.csr_matrix, sparse.csr_array, sparse.coo_matrix]
)
def test_zip_sparse_membership_representations(topic_model, tmp_path, matrix_type):
    topic_model.cluster_layers[0] = matrix_type(topic_model.cluster_layers[0])
    path = tmp_path / "sparse.zip"
    topic_model.to_file(path)
    assert TopicModel.from_file(path).topics[(0, 0)].members.tolist() == [0]


@pytest.mark.parametrize("defect", ["duplicate-coordinate", "multiple-topics"])
def test_zip_membership_must_be_a_partition(topic_model, tmp_path, defect):
    rows = [0, 1, 0]
    columns = [0, 1, 0 if defect == "duplicate-coordinate" else 1]
    topic_model.cluster_layers[0] = sparse.coo_matrix(
        (np.full(3, 255, dtype=np.uint8), (rows, columns)), shape=(2, 2)
    )
    path = tmp_path / "ambiguous-membership.zip"
    topic_model.to_file(path)
    with pytest.raises(ValueError):
        TopicModel.from_file(path)


def test_materializing_borrowed_csr_preserves_array_identity(topic_model):
    matrix = topic_model.cluster_layers[0]
    arrays = (matrix.data, matrix.indices, matrix.indptr)
    for array in arrays:
        array.flags.writeable = False
    loaded = TopicModel(
        topic_model.topic_df,
        topic_model.cluster_tree,
        [matrix],
        topic_model.embedding_vectors,
    )
    assert loaded.topics[(0, 0)].members.tolist() == [0]
    assert all(
        actual is expected
        for actual, expected in zip(
            (matrix.data, matrix.indices, matrix.indptr), arrays
        )
    )
    assert all(not array.flags.writeable for array in arrays)


@pytest.mark.parametrize(
    "prompt",
    [{"system": 5, "user": []}, {"system": "s", "user": "u", "json_schema": []}],
)
def test_archive_prompt_fields_are_typed(stored_model, prompt):
    format_name, path = stored_model

    def invalid_prompt(table):
        table.loc[0, "prompt_json"] = json.dumps(prompt)

    changed = replace_topic_table(format_name, path, invalid_prompt)
    with pytest.raises(ValueError, match="prompt fields"):
        load_model(format_name, changed)


@pytest.mark.parametrize("column", ["features_json", "prompt_json"])
@pytest.mark.parametrize("value", ["", 0, False])
def test_present_invalid_json_cell_is_not_treated_as_absent(topic_model, column, value):
    table = topic_model.topic_df
    table[column] = table[column].astype(object)
    table.at[0, column] = value
    loaded = TopicModel(
        table,
        topic_model.cluster_tree,
        topic_model.cluster_layers,
        topic_model.embedding_vectors,
    )
    with pytest.raises(ValueError, match="Invalid topic"):
        loaded.topics
