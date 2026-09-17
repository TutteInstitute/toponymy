"""Lance integrity checks using tiny current-writer datasets.

These are synthetic controls, not historical Lance fixtures. No model or network
calls are made. Numeric float controls cover vectors/graph, not membership weights.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from toponymy.serialization import Topic, TopicModel

lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")


@pytest.fixture
def model():
    vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    vectors[0, 0] += np.finfo(np.float64).eps
    return TopicModel(
        None,
        {(1, 0): [(0, 7), (0, 90000)]},
        [
            sparse.csr_matrix(
                np.array([[255, 0], [255, 0], [0, 255], [0, 255]], dtype=np.uint8)
            )
        ],
        vectors,
        reduced_vectors=vectors[:, :1],
        topics={
            (0, 7): Topic(0, 7, np.array([0, 1]), name="Café"),
            (0, 90000): Topic(0, 90000, np.array([2, 3]), name="猫"),
        },
    )


@pytest.fixture
def stored_model(model, tmp_path):
    path = tmp_path / "model.lance"
    model.to_lance(path)
    return path


def replace_column(table, name, values, arrow_type=None):
    index = table.schema.get_field_index(name)
    assert index >= 0
    field = table.schema.field(index)
    if arrow_type is not None:
        field = pa.field(
            name, arrow_type, nullable=field.nullable, metadata=field.metadata
        )
    return table.set_column(index, field, pa.array(values, type=field.type))


def rewrite_table(path, name, transform):
    location = str(path / (name + ".lance"))
    original = lance.dataset(location).to_table()
    lance.write_dataset(transform(original), location, mode="overwrite")


def test_lance_float_vector_and_graph_control(model, tmp_path):
    model.clustering_graph = sparse.csr_matrix(
        (np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)) * 0.125
    )
    path = tmp_path / "float-control.lance"
    model.to_lance(path)
    loaded = TopicModel.from_lance(path)
    assert loaded.embedding_vectors.dtype == np.float64
    assert loaded.reduced_vectors.dtype == np.float64
    np.testing.assert_array_equal(loaded.embedding_vectors, model.embedding_vectors)
    np.testing.assert_array_equal(loaded.reduced_vectors, model.reduced_vectors)
    assert loaded.clustering_graph.dtype == np.float32
    assert (loaded.clustering_graph != model.clustering_graph).nnz == 0
    assert loaded.topic_names == model.topic_names
    for key, topic in model.topics.items():
        np.testing.assert_array_equal(loaded.topics[key].members, topic.members)


def test_lance_empty_control_preserves_vector_width(tmp_path):
    model = TopicModel(None, {}, [], np.empty((0, 3), dtype=np.float64), topics={})
    path = tmp_path / "empty-control.lance"
    model.to_lance(path)
    loaded = TopicModel.from_lance(path)
    assert loaded.topics == {}
    assert loaded.cluster_layers == []
    assert loaded.cluster_tree == {}
    assert loaded.embedding_vectors.shape == (0, 3)
    assert loaded.embedding_vectors.dtype == np.float64


def test_lance_synthetic_label_indexed_gap_control(tmp_path):
    table = pd.DataFrame(
        {"layer": [0, 0], "cluster": [4, 1], "name": ["Second", "First"]}
    )
    membership = sparse.csr_matrix(
        (np.array([255, 255, 255], dtype=np.uint8), ([0, 1, 2], [1, 1, 4])),
        shape=(3, 5),
    )
    model = TopicModel(table, {(1, 0): [(0, 1), (0, 4)]}, [membership], np.eye(3))
    path = tmp_path / "synthetic-gap-control.lance"
    model.to_lance(path)
    loaded = TopicModel.from_lance(path)
    assert loaded.topic_names == [{1: "First", 4: "Second"}]
    assert loaded.cluster_layers[0].shape == (3, 5)
    assert loaded.topics[(0, 1)].members.tolist() == [0, 1]
    assert loaded.topics[(0, 4)].members.tolist() == [2]
    assert (loaded.cluster_layers[0] != membership).nnz == 0


def test_lance_accepts_legacy_coordinate_integer_widths(stored_model):
    def narrower_integer_columns(table):
        for name, dtype in (
            ("layer", pa.int16()),
            ("row_idx", pa.int32()),
            ("col_idx", pa.int16()),
        ):
            table = replace_column(table, name, table[name].to_pylist(), dtype)
        return table

    rewrite_table(stored_model, "clusters", narrower_integer_columns)
    loaded = TopicModel.from_lance(stored_model)
    assert loaded.topics[(0, 7)].members.tolist() == [0, 1]
    assert loaded.topics[(0, 90000)].members.tolist() == [2, 3]


@pytest.mark.parametrize(
    "field, upper_bound", [("layer", 1), ("row_idx", 4), ("col_idx", 2)]
)
@pytest.mark.parametrize("invalid_kind", ["negative", "out-of-range", "fractional"])
def test_lance_rejects_invalid_membership_coordinates(
    stored_model, field, upper_bound, invalid_kind
):
    invalid = {"negative": -1, "out-of-range": upper_bound, "fractional": 0.5}[
        invalid_kind
    ]

    def corrupt_coordinate(table):
        values = table[field].to_pylist()
        values[0] = invalid
        # Preserve schema for integer values; fractional values need an actual
        # float column so Arrow cannot truncate the malformed input for us.
        dtype = pa.float64() if invalid_kind == "fractional" else None
        return replace_column(table, field, values, dtype)

    rewrite_table(stored_model, "clusters", corrupt_coordinate)
    with pytest.raises(ValueError):
        TopicModel.from_lance(stored_model)


def test_lance_rejects_multiple_conflicting_config_rows(stored_model):
    def extra_config_row(table):
        conflicting = replace_column(table, "n_layers", [2])
        return pa.concat_tables([table, conflicting])

    rewrite_table(stored_model, "config", extra_config_row)
    with pytest.raises(ValueError):
        TopicModel.from_lance(stored_model)


@pytest.mark.parametrize(
    "value, kind",
    [(-1, "integer"), (0, "integer"), (1.5, "float"), ("1", "string"), (True, "bool")],
)
def test_lance_rejects_invalid_or_inconsistent_layer_count(stored_model, value, kind):
    types = {"float": pa.float64(), "string": pa.string(), "bool": pa.bool_()}
    rewrite_table(
        stored_model,
        "config",
        lambda table: replace_column(table, "n_layers", [value], types.get(kind)),
    )
    with pytest.raises(ValueError):
        TopicModel.from_lance(stored_model)


def test_lance_rejects_duplicate_membership_coordinates_before_summing(stored_model):
    rewrite_table(
        stored_model,
        "clusters",
        lambda table: pa.concat_tables([table, table.slice(0, 1)]),
    )
    with pytest.raises(ValueError):
        TopicModel.from_lance(stored_model)


@pytest.mark.parametrize("value", [-1, 256, 511, 0.5])
def test_lance_rejects_lossy_membership_value_conversion(stored_model, value):
    def corrupt_value(table):
        values = table["value"].to_pylist()
        values[0] = value
        dtype = pa.float64() if isinstance(value, float) else None
        return replace_column(table, "value", values, dtype)

    rewrite_table(stored_model, "clusters", corrupt_value)
    with pytest.raises(ValueError):
        TopicModel.from_lance(stored_model)


@pytest.mark.parametrize("widths", [[], [1], [2, 2]])
def test_lance_rejects_declared_width_inventory_mismatch(stored_model, widths):
    rewrite_table(
        stored_model,
        "config",
        lambda table: replace_column(table, "layer_columns", [widths]),
    )
    with pytest.raises(ValueError):
        TopicModel.from_lance(stored_model)


@pytest.mark.parametrize("value", [0.5, -1.0, 256.0, np.nan, np.inf])
def test_lance_writer_rejects_lossy_values_before_replacing_destination(
    model, tmp_path, value
):
    path = tmp_path / "existing.lance"
    model.to_lance(path)
    model.cluster_layers[0] = model.cluster_layers[0].astype(float)
    model.cluster_layers[0].data[0] = value
    with pytest.raises(ValueError, match="membership values"):
        model.to_lance(path, overwrite=True)
    assert TopicModel.from_lance(path).topics[(0, 7)].members.tolist() == [0, 1]


def test_document_metadata_need_not_have_item_num_column(model, tmp_path):
    model.document_df = pd.DataFrame({"text": ["a", None, "c", "d"]})
    path = tmp_path / "custom-documents.lance"
    model.to_lance(path)
    loaded = TopicModel.from_lance(path)
    pd.testing.assert_frame_equal(loaded.document_df, model.document_df)


@pytest.mark.parametrize("field", ["embedding_dtype", "reduced_dtype"])
def test_lance_dtype_metadata_cannot_truncate_stored_vectors(stored_model, field):
    rewrite_table(
        stored_model, "config", lambda table: replace_column(table, field, ["int32"])
    )
    with pytest.raises(ValueError, match="dtype"):
        TopicModel.from_lance(stored_model)


def test_lance_dtype_metadata_cannot_truncate_graph_distances(model, tmp_path):
    model.clustering_graph = sparse.csr_matrix(np.eye(4, dtype=np.float32) * 0.125)
    path = tmp_path / "graph-dtype.lance"
    model.to_lance(path)
    rewrite_table(
        path, "config", lambda table: replace_column(table, "graph_dtype", ["int32"])
    )
    with pytest.raises(ValueError, match="dtype"):
        TopicModel.from_lance(path)
