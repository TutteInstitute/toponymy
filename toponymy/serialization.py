import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from copy import copy, deepcopy
from dataclasses import dataclass, field
import base64

import scipy.sparse as sp
import pandas as pd
import numpy as np
import shutil

from toponymy.topic_tree import TopicTree

_SERIAL_VERSION = "0.2"
_READABLE_SERIAL_VERSIONS = {"0.1", "0.2"}


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} in topic archive")
        result[key] = value
    return result


def _load_json(value, context):
    try:
        return json.loads(value, object_pairs_hook=_unique_json_pairs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {context}: {error}") from error


def _missing_cell(value):
    # Nullable pandas string columns use NaN; absent legacy columns use None.
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and np.isnan(value))
    )


def _optional_text(value, context):
    if _missing_cell(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value


def _nonnegative_integer(value, context):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{context} must be a nonnegative integer")
    return int(value)


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Archive configuration must be an object")
    version = config.get("serial_version")
    if not isinstance(version, str) or version not in _READABLE_SERIAL_VERSIONS:
        raise ValueError(f"Unsupported topic archive serial version: {version!r}")
    _nonnegative_integer(config.get("n_layers"), "Archive n_layers")
    for key in ("has_reduced", "has_clustering_graph"):
        if not isinstance(config.get(key, False), bool):
            raise ValueError(f"Archive {key} must be a boolean")


def _coo_integers(values, context):
    array = np.asarray(values)
    if array.ndim != 1 or (
        array.size
        and (
            array.dtype.kind not in "iu"
            or (array < 0).any()
            or (array > np.iinfo(np.int64).max).any()
        )
    ):
        raise ValueError(f"{context} must contain nonnegative integer values")
    return array.astype(np.int64, copy=False)


def _duplicate_coordinates(*coordinates):
    order = np.lexsort(coordinates[::-1])
    repeated = np.ones(max(0, len(order) - 1), dtype=bool)
    for values in coordinates:
        ordered = values[order]
        repeated &= ordered[1:] == ordered[:-1]
    return repeated.any()


def _lance_dtype(declared, stored, context):
    try:
        actual = np.dtype(stored.to_pandas_dtype())
        dtype = actual if declared is None else np.dtype(declared)
        if dtype.kind not in "fiu" or dtype.newbyteorder("=") != actual.newbyteorder(
            "="
        ):
            raise ValueError("declared dtype does not match the stored numeric type")
        return dtype
    except (TypeError, ValueError, NotImplementedError) as error:
        raise ValueError(f"Invalid Lance {context} dtype: {error}") from error


def _lance_vectors(rows, dtype, dimension, context):
    try:
        dtype = np.dtype(dtype)
        if dtype.kind not in "fiu":
            raise ValueError("expected a real numeric dtype")
        vectors = np.asarray(rows, dtype=dtype)
        if dimension is not None:
            width = _nonnegative_integer(dimension, f"{context} dimension")
            vectors = vectors.reshape(len(rows), width)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("expected a finite real matrix")
        return vectors
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid Lance {context}: {error}") from error


def _decode_tree(value):
    raw_tree = _load_json(value, "cluster tree")
    if not isinstance(raw_tree, dict):
        raise ValueError("Archive cluster tree must be an object")
    tree = {}
    for key, values in raw_tree.items():
        parent = uid_to_ints(key)
        if parent in tree:
            raise ValueError(f"Duplicate decoded tree parent {parent}")
        if not isinstance(values, list):
            raise ValueError(f"Tree children for {parent} must be a list")
        children = [uid_to_ints(child) for child in values]
        if len(set(children)) != len(children):
            raise ValueError(f"Duplicate decoded tree child for {parent}")
        tree[parent] = children
    return tree


def topic_uid(tup) -> str:
    a, b = tup
    a = int(a)
    b = (
        int(b) + 1
    )  # Because unclustered is -1 and we can't convert negative to unsigned.
    if a < 0 or b < 0:
        raise ValueError(
            "Topic identifiers require nonnegative layers and labels >= -1"
        )
    if a >= (1 << 14) or b >= (1 << 10):
        return f"v2:{a}:{b - 1}"
    combined = (a << 10) | b  # retain existing compact identifiers when representable
    return base64.urlsafe_b64encode(combined.to_bytes(3, "big")).rstrip(b"=").decode()


def uid_to_ints(s: str):
    """Returns (layer, cluster_number)"""
    if not isinstance(s, str):
        raise ValueError("Invalid topic identifier")
    if s.startswith("v2:"):
        _, layer, cluster = s.split(":")
        values = int(layer), int(cluster)
        if values[0] < 0 or values[1] < -1:
            raise ValueError("Invalid topic identifier")
        return values
    if len(s) != 4:
        raise ValueError("Invalid compact topic identifier")
    decoded = base64.b64decode(s, altchars=b"-_", validate=True)
    if len(decoded) != 3 or base64.urlsafe_b64encode(decoded).decode() != s:
        raise ValueError("Invalid compact topic identifier")
    combined = int.from_bytes(decoded, "big")
    return combined >> 10, (combined & 0x3FF) - 1


def _pandas_col_to_arrow(series: pd.Series):
    """Infer a PyArrow type from a pandas Series for schema construction."""
    import pyarrow as pa

    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return pa.int64()
    if pd.api.types.is_float_dtype(dtype):
        return pa.float64()
    if pd.api.types.is_bool_dtype(dtype):
        return pa.bool_()
    return pa.string()


@dataclass
class Topic:
    """Learned topic state, keyed by layer and original clustering label."""

    layer: int
    label: int
    members: np.ndarray
    features: dict = field(default_factory=dict)
    prompt: object = None
    name: str | None = None
    summary: str | None = None
    explanation: str | None = None

    @property
    def key(self):
        return self.layer, self.label


class TopicModel:
    """Topic-centric fitted results and the existing portable storage interface.

    Topics own mutable naming state. Matrices are borrowed; membership matrices
    use columns in sorted original-ID order. topic_df is a materialized view,
    so changing it does not update names: edit a Topic explicitly instead.
    """

    def __init__(
        self,
        topic_df,
        cluster_tree,
        cluster_layers,
        embedding_vectors,
        reduced_vectors=None,
        document_df=None,
        *,
        topics=None,
        metadata=None,
        clustering_graph=None,
    ):
        self._topic_df = topic_df.copy() if topic_df is not None else None
        self._topics = topics
        self.cluster_tree = {
            key: list(children) for key, children in cluster_tree.items()
        }
        self.cluster_layers = list(cluster_layers)
        self.embedding_vectors = embedding_vectors
        self.reduced_vectors = reduced_vectors
        self.clustering_graph = clustering_graph
        self.document_df = (
            pd.DataFrame({"item_num": range(len(embedding_vectors))})
            if document_df is None
            else document_df.copy()
        )
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Runtime metadata must be an object")
        self.metadata = {} if metadata is None else dict(metadata)

    def __repr__(self):
        return f"TopicModel(n_samples={len(self.embedding_vectors)}, n_topics={len(self.topics)})"

    @classmethod
    def from_topics(cls, topics, layers, tree, embedding_vectors, reduced_vectors=None):
        matrices = []
        for layer in layers:
            rows = (
                np.concatenate([cluster.members for cluster in layer])
                if len(layer)
                else np.empty(0, dtype=np.int64)
            )
            cols = np.repeat(
                np.arange(len(layer)), [len(cluster.members) for cluster in layer]
            )
            matrices.append(
                sp.csr_matrix(
                    (np.full(len(rows), 255, dtype=np.uint8), (rows, cols)),
                    shape=(len(embedding_vectors), len(layer)),
                )
            )
        return cls(
            None, tree, matrices, embedding_vectors, reduced_vectors, topics=topics
        )

    @property
    def topics(self):
        if self._topics is None:
            from .templates import Prompt

            topics = {}
            table = self._topic_df
            if (
                not isinstance(self.embedding_vectors, np.ndarray)
                or self.embedding_vectors.ndim != 2
            ):
                raise ValueError("Archive embedding_vectors must be a matrix")
            n_documents = len(self.embedding_vectors)
            for name, vectors in (
                ("embedding_vectors", self.embedding_vectors),
                ("reduced_vectors", self.reduced_vectors),
            ):
                if vectors is not None and (
                    not isinstance(vectors, np.ndarray)
                    or vectors.ndim != 2
                    or len(vectors) != n_documents
                    or vectors.dtype.kind not in "fiu"
                    or not np.isfinite(vectors).all()
                ):
                    raise ValueError(
                        f"Archive {name} must be a finite real matrix aligned with documents"
                    )
            if len(self.document_df) != n_documents:
                raise ValueError("Archive document and embedding row counts differ")
            if self.clustering_graph is not None and (
                not sp.issparse(self.clustering_graph)
                or self.clustering_graph.shape != (n_documents, n_documents)
                or self.clustering_graph.dtype.kind not in "fiu"
                or not np.isfinite(self.clustering_graph.data).all()
                or (self.clustering_graph.data < 0).any()
            ):
                raise ValueError(
                    "Clustering graph must be a finite nonnegative square matrix aligned with documents"
                )
            for layer_index, matrix in enumerate(self.cluster_layers):
                if (
                    not sp.issparse(matrix)
                    or matrix.ndim != 2
                    or matrix.shape[0] != n_documents
                ):
                    raise ValueError(
                        f"Cluster matrix {layer_index} must align with documents"
                    )
                if (
                    matrix.dtype.kind not in "fiu"
                    or not np.isfinite(matrix.data).all()
                    or (matrix.data < 0).any()
                ):
                    raise ValueError(
                        f"Cluster matrix {layer_index} has invalid membership values"
                    )
                if hasattr(matrix, "check_format"):
                    # SciPy validation rebinds/prunes arrays on its receiver.
                    # A shallow container copy preserves the borrowed matrix.
                    copy(matrix).check_format(full_check=True)
                coo = matrix.tocoo()
                if _duplicate_coordinates(coo.row, coo.col):
                    raise ValueError(
                        f"Cluster matrix {layer_index} has duplicate coordinates"
                    )
                if np.bincount(coo.row[coo.data != 0]).max(initial=0) > 1:
                    raise ValueError(
                        f"Cluster matrix {layer_index} assigns a document to multiple topics"
                    )
            if table is not None:
                if not {"layer", "cluster"}.issubset(table.columns):
                    raise ValueError("Topic table requires layer and cluster columns")
                keys = set()
                for row in table.to_dict("records"):
                    layer = _nonnegative_integer(row["layer"], "Topic layer")
                    label = _nonnegative_integer(row["cluster"], "Topic cluster")
                    key = (layer, label)
                    if layer >= len(self.cluster_layers):
                        raise ValueError(f"Topic {key} has no declared cluster layer")
                    if key in keys:
                        raise ValueError(f"Duplicate topic identity {key}")
                    keys.add(key)
                    if "uid" in row and uid_to_ints(row["uid"]) != key:
                        raise ValueError(f"Topic UID does not match {key}")
                for layer_index, matrix in enumerate(self.cluster_layers):
                    matrix = matrix.tocsr(copy=False)
                    rows = table[table["layer"] == layer_index].sort_values("cluster")
                    if len(rows) != matrix.shape[1]:
                        labels = set(rows["cluster"])
                        if any(label >= matrix.shape[1] for label in labels) or not set(
                            matrix.nonzero()[1]
                        ).issubset(labels):
                            raise ValueError(
                                f"Topic table does not describe cluster matrix {layer_index}"
                            )
                    for ordinal, row in enumerate(rows.to_dict("records")):
                        label = int(row["cluster"])
                        column = ordinal if len(rows) == matrix.shape[1] else label
                        members = matrix[:, [column]].nonzero()[0]
                        members.flags.writeable = False
                        features = (
                            _load_json(row["features_json"], "topic features")
                            if not _missing_cell(row.get("features_json"))
                            else {"cluster_keywords": list(row.get("keyphrases", []))}
                        )
                        prompt_data = (
                            _load_json(row["prompt_json"], "topic prompt")
                            if not _missing_cell(row.get("prompt_json"))
                            else None
                        )
                        if not isinstance(features, dict) or (
                            prompt_data is not None
                            and not isinstance(prompt_data, dict)
                        ):
                            raise ValueError(
                                f"Topic {(layer_index, label)} features and prompt must be objects"
                            )
                        if prompt_data is not None and (
                            any(
                                not isinstance(prompt_data.get(field), str)
                                for field in ("system", "user")
                            )
                            or (
                                prompt_data.get("json_schema") is not None
                                and not isinstance(prompt_data["json_schema"], dict)
                            )
                        ):
                            raise ValueError(
                                f"Invalid prompt fields for topic {(layer_index, label)}"
                            )
                        try:
                            prompt = (
                                Prompt(**prompt_data)
                                if prompt_data is not None
                                else None
                            )
                        except (TypeError, ValueError) as error:
                            raise ValueError(
                                f"Invalid prompt for topic {(layer_index, label)}"
                            ) from error
                        topics[(layer_index, label)] = Topic(
                            layer_index,
                            label,
                            members,
                            features,
                            prompt,
                            _optional_text(row.get("name"), "Topic name"),
                            _optional_text(row.get("summary"), "Topic summary"),
                            _optional_text(row.get("explanation"), "Topic explanation"),
                        )
            elif any(matrix.nnz for matrix in self.cluster_layers):
                raise ValueError("Populated cluster matrices require a topic table")
            parents = {}
            for parent, children in self.cluster_tree.items():
                if parent not in topics and parent != (len(self.cluster_layers), 0):
                    raise ValueError(f"Unknown tree parent {parent}")
                for child in children:
                    if child not in topics or child[0] >= parent[0]:
                        raise ValueError(
                            f"Invalid descending tree edge {parent} -> {child}"
                        )
                    if child in parents:
                        raise ValueError(f"Duplicate tree parent for child {child}")
                    parents[child] = parent
                    if (
                        parent in topics
                        and not np.isin(
                            topics[child].members,
                            topics[parent].members,
                            assume_unique=True,
                        ).all()
                    ):
                        raise ValueError(
                            f"Tree parent {parent} does not contain child {child}"
                        )
            if (len(self.cluster_layers), 0) in self.cluster_tree and set(
                parents
            ) != set(topics):
                raise ValueError("Rooted cluster tree omits one or more topics")
            self._topics = topics
        return self._topics

    @property
    def topic_df(self):
        rows = []
        for key, topic in sorted(self.topics.items()):
            prompt = topic.prompt._asdict() if topic.prompt is not None else None
            rows.append(
                {
                    "uid": topic_uid(key),
                    "layer": key[0],
                    "cluster": key[1],
                    "name": topic.name,
                    "size": len(topic.members),
                    "keyphrases": list(topic.features.get("cluster_keywords", [])),
                    "features_json": json.dumps(topic.features, ensure_ascii=False),
                    "prompt_json": (
                        json.dumps(prompt, ensure_ascii=False) if prompt else None
                    ),
                    "summary": topic.summary,
                    "explanation": topic.explanation,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "uid",
                "layer",
                "cluster",
                "name",
                "size",
                "keyphrases",
                "features_json",
                "prompt_json",
                "summary",
                "explanation",
            ],
        )

    @property
    def topic_sizes(self):
        return [
            {
                label: len(topic.members)
                for (index, label), topic in self.topics.items()
                if index == layer
            }
            for layer in range(len(self.cluster_layers))
        ]

    @classmethod
    def from_toponymy(cls, toponymy, document_df=None):
        if hasattr(toponymy, "topic_model_"):
            model = toponymy.topic_model_
            return cls(
                model.topic_df,
                model.cluster_tree,
                model.cluster_layers,
                model.embedding_vectors,
                model.reduced_vectors,
                model.document_df if document_df is None else document_df,
                metadata=model.metadata,
                clustering_graph=model.clustering_graph,
            )
        # Read the stable pre-0.6 fitted representation for migration.
        topics, matrices = {}, []
        for layer_index, layer in enumerate(toponymy.cluster_layers_):
            labels = np.asarray(layer.cluster_labels)
            ids = np.unique(labels[labels >= 0])
            rows, cols = [], []
            for ordinal, label in enumerate(ids):
                members = np.flatnonzero(labels == label)
                rows.extend(members)
                cols.extend([ordinal] * len(members))
                topics[(layer_index, int(label))] = Topic(
                    layer_index,
                    int(label),
                    members,
                    {"cluster_keywords": list(layer.keyphrases[label])},
                    name=toponymy.topic_names_[layer_index][label],
                )
            matrices.append(
                sp.csr_matrix(
                    (np.full(len(rows), 255, dtype=np.uint8), (rows, cols)),
                    shape=(len(labels), len(ids)),
                )
            )
        return cls(
            None,
            toponymy.cluster_tree_,
            matrices,
            toponymy.embedding_vectors_,
            toponymy.clusterable_vectors_,
            document_df,
            topics=topics,
        )

    @classmethod
    def from_file(cls, path: str):
        """Load a trusted topic archive in the current or legacy format.

        Paths, duplicate members and layer counts are checked; NumPy object
        arrays are not loaded. Extraction and native array/Parquet readers have
        no resource quota, so callers must bound archive size and trust its
        source before loading.
        """
        path = Path(path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with zipfile.ZipFile(path) as z:
                members = set()
                for entry in z.infolist():
                    member = PurePosixPath(entry.filename.replace("\\", "/"))
                    if (
                        member.is_absolute()
                        or ".." in member.parts
                        or ":" in entry.filename
                    ):
                        raise ValueError("Invalid path in topic archive")
                    normalized = str(member).casefold()
                    if normalized in members:
                        raise ValueError("Duplicate member in topic archive")
                    members.add(normalized)
                z.extractall(root)

            if not (root / "metadata.json").is_file():
                raise ValueError("Topic archive is missing metadata.json")
            metadata = _load_json(
                (root / "metadata.json").read_text(encoding="utf8"), "archive metadata"
            )
            _validate_config(metadata)
            has_reduced = metadata.get("has_reduced", False)
            n_layers = metadata["n_layers"]
            layer_files = [
                root / "cluster_matrices" / f"layer_{index}.npz"
                for index in range(n_layers)
            ]
            if set((root / "cluster_matrices").glob("*")) != set(layer_files):
                raise ValueError("Archive cluster matrices must match n_layers")
            expected = {
                "metadata.json",
                "cluster_tree.json",
                "document_df.parquet",
                "topic_df.parquet",
                "embedding_vectors.npy",
            }
            expected.update(
                f"cluster_matrices/layer_{index}.npz" for index in range(n_layers)
            )
            if has_reduced:
                expected.add("reduced_vectors.npy")
            if metadata.get("has_clustering_graph", False):
                expected.add("clustering_graph.npz")
            if {
                file.relative_to(root).as_posix()
                for file in root.rglob("*")
                if file.is_file()
            } != expected:
                raise ValueError("Topic archive files do not match declared inventory")

            # --- DataFrames ---
            document_df = pd.read_parquet(root / "document_df.parquet")
            topic_df = pd.read_parquet(root / "topic_df.parquet")

            # --- Vectors ---
            embedding_vectors = np.load(root / "embedding_vectors.npy")
            reduced_vectors = None
            if has_reduced:
                reduced_vectors = np.load(
                    root / "reduced_vectors.npy"
                )  # bugfix: was loading from cwd

            # --- Sparse cluster matrices ---
            matrices = [sp.load_npz(f) for f in layer_files]

            # --- Cluster tree topology ---
            cluster_tree = _decode_tree(
                (root / "cluster_tree.json").read_text(encoding="utf8")
            )

            model = cls(
                embedding_vectors=embedding_vectors,
                reduced_vectors=reduced_vectors,
                document_df=document_df,
                topic_df=topic_df,
                cluster_tree=cluster_tree,
                cluster_layers=matrices,
                metadata=metadata.get("runtime_metadata", {}),
                clustering_graph=(
                    sp.load_npz(root / "clustering_graph.npz")
                    if metadata.get("has_clustering_graph", False)
                    else None
                ),
            )
            model.topics
            return model

    def to_file(self, path: str):
        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "topic_model"
            matrices_dir = root / "cluster_matrices"
            root.mkdir()
            matrices_dir.mkdir()

            self.document_df.to_parquet(root / "document_df.parquet")
            topic_df = deepcopy(self.topic_df)
            topic_df.to_parquet(root / "topic_df.parquet")

            np.save(root / "embedding_vectors.npy", self.embedding_vectors)
            has_reduced = False
            if self.reduced_vectors is not None:
                np.save(root / "reduced_vectors.npy", self.reduced_vectors)
                has_reduced = True

            if self.clustering_graph is not None:
                sp.save_npz(root / "clustering_graph.npz", self.clustering_graph)

            for i, matrix in enumerate(self.cluster_layers):
                sp.save_npz(matrices_dir / f"layer_{i}.npz", matrix)

            uid_tree = {
                topic_uid(k): [topic_uid(c) for c in v]
                for k, v in self.cluster_tree.items()
            }
            with open(root / "cluster_tree.json", "w") as f:
                json.dump(uid_tree, f)

            metadata = {
                "serial_version": _SERIAL_VERSION,
                "n_layers": len(self.cluster_layers),
                "has_reduced": has_reduced,
                "runtime_metadata": self.metadata,
                "has_clustering_graph": self.clustering_graph is not None,
            }
            with open(root / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # --- Bundle into zip ---
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for file in root.rglob("*"):
                    z.write(file, file.relative_to(root))

    @classmethod
    def from_lance(cls, path: str):
        """Load trusted local Lance tables; callers must bound input resources."""
        import lance

        path = Path(path)

        # --- config ---
        config_table = lance.dataset(str(path / "config.lance")).to_table()
        if config_table.num_rows != 1:
            raise ValueError("Lance configuration must have exactly one row")
        config = {key: values[0] for key, values in config_table.to_pydict().items()}
        _validate_config(config)
        n_layers = config["n_layers"]
        has_reduced = config.get("has_reduced", False)
        cluster_tree = _decode_tree(config.get("cluster_tree"))

        document_table = lance.dataset(str(path / "documents.lance")).to_table()
        n_docs = document_table.num_rows
        doc_table = document_table.to_pydict()
        import pyarrow as pa

        def vector_dtype(column, config_key):
            if column not in document_table.column_names:
                raise ValueError(f"Lance documents are missing {column}")
            stored = document_table.schema.field(column).type
            if not (
                pa.types.is_list(stored)
                or pa.types.is_large_list(stored)
                or pa.types.is_fixed_size_list(stored)
            ):
                raise ValueError(f"Lance {column} must contain vector rows")
            return _lance_dtype(config.get(config_key), stored.value_type, column)

        embedding_vectors = _lance_vectors(
            doc_table.pop("embedding", None),
            vector_dtype("embedding", "embedding_dtype"),
            config.get("embedding_dim"),
            "embedding vectors",
        )
        reduced_vectors = None
        if has_reduced:
            reduced_vectors = _lance_vectors(
                doc_table.pop("reduced_embedding", None),
                vector_dtype("reduced_embedding", "reduced_dtype"),
                config.get("reduced_dim"),
                "reduced vectors",
            )
        document_df = pd.DataFrame(doc_table, index=range(n_docs))

        topic_dict = lance.dataset(str(path / "topics.lance")).to_table().to_pydict()
        topic_df = pd.DataFrame(topic_dict)

        coo_dict = lance.dataset(str(path / "clusters.lance")).to_table().to_pydict()
        if not {"layer", "row_idx", "col_idx", "value"}.issubset(coo_dict):
            raise ValueError("Lance cluster table is missing COO columns")
        layers_arr = _coo_integers(coo_dict["layer"], "COO layer")
        rows_arr = _coo_integers(coo_dict["row_idx"], "COO row")
        cols_arr = _coo_integers(coo_dict["col_idx"], "COO column")
        vals_arr = _coo_integers(coo_dict["value"], "COO membership")
        if (
            (layers_arr >= n_layers).any()
            or (rows_arr >= n_docs).any()
            or (vals_arr > 255).any()
        ):
            raise ValueError(
                "Lance COO rows exceed declared layers, documents or membership range 0..255"
            )
        if _duplicate_coordinates(layers_arr, rows_arr, cols_arr):
            raise ValueError("Duplicate Lance COO coordinate")
        vals_arr = vals_arr.astype(np.uint8)
        widths = config.get("layer_columns")
        if widths is not None:
            if not isinstance(widths, list) or len(widths) != n_layers:
                raise ValueError("Lance layer_columns must match n_layers")
            widths = [_nonnegative_integer(width, "Layer width") for width in widths]

        matrices = []
        for layer_idx in range(n_layers):
            mask = layers_arr == layer_idx
            n_cols = (
                widths[layer_idx]
                if widths is not None
                else int(cols_arr[mask].max()) + 1 if mask.any() else 0
            )
            if (cols_arr[mask] >= n_cols).any():
                raise ValueError(f"Lance COO columns exceed layer {layer_idx} width")
            csr = sp.coo_matrix(
                (vals_arr[mask], (rows_arr[mask], cols_arr[mask])),
                shape=(n_docs, n_cols),
                dtype=np.uint8,
            ).tocsr()
            matrices.append(csr)

        clustering_graph = None
        if config.get("has_clustering_graph", False):
            edge_table = lance.dataset(str(path / "graph.lance")).to_table()
            edges = edge_table.to_pydict()
            graph_rows = _coo_integers(edges.get("row"), "Graph row")
            graph_cols = _coo_integers(edges.get("column"), "Graph column")
            if (
                (graph_rows >= n_docs).any()
                or (graph_cols >= n_docs).any()
                or _duplicate_coordinates(graph_rows, graph_cols)
            ):
                raise ValueError("Invalid Lance graph coordinates")
            clustering_graph = sp.csr_matrix(
                (
                    np.asarray(
                        edges["distance"],
                        dtype=_lance_dtype(
                            config.get("graph_dtype"),
                            edge_table.schema.field("distance").type,
                            "graph distances",
                        ),
                    ),
                    (graph_rows, graph_cols),
                ),
                shape=(n_docs, n_docs),
            )
        model = cls(
            embedding_vectors=embedding_vectors,
            reduced_vectors=reduced_vectors,
            document_df=document_df,
            topic_df=topic_df,
            cluster_tree=cluster_tree,
            cluster_layers=matrices,
            metadata=_load_json(
                config.get("runtime_metadata", "{}"), "runtime metadata"
            ),
            clustering_graph=clustering_graph,
        )
        model.topics
        return model

    def to_lance(self, path: str, overwrite: bool = False):

        import lance
        import pyarrow as pa

        path = Path(path)
        # Validate before replacing an existing artifact or narrowing membership.
        topic_df = self.topic_df
        for matrix in self.cluster_layers:
            values = matrix.data
            if (
                values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 255).any()
                or (values != np.floor(values)).any()
            ):
                raise ValueError("Lance membership values must be integers in 0..255")

        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists. Remove it first or choose a different path."
                )
            shutil.rmtree(path)

        path.mkdir(parents=True)

        # --- documents.lance ---
        doc_dict = {
            col: self.document_df[col].tolist() for col in self.document_df.columns
        }

        emb_dim = self.embedding_vectors.shape[1]
        doc_dict["embedding"] = self.embedding_vectors.tolist()
        schema_fields = [
            *[
                pa.field(col, _pandas_col_to_arrow(self.document_df[col]))
                for col in self.document_df.columns
            ],
            pa.field(
                "embedding",
                pa.list_(pa.from_numpy_dtype(self.embedding_vectors.dtype), emb_dim),
            ),
        ]

        has_reduced = self.reduced_vectors is not None
        if has_reduced:
            red_dim = self.reduced_vectors.shape[1]
            doc_dict["reduced_embedding"] = self.reduced_vectors.tolist()
            schema_fields.append(
                pa.field(
                    "reduced_embedding",
                    pa.list_(pa.from_numpy_dtype(self.reduced_vectors.dtype), red_dim),
                )
            )

        doc_schema = pa.schema(schema_fields)
        doc_table = pa.table(
            {
                column: pa.array(
                    values, type=doc_schema.field(column).type, from_pandas=True
                )
                for column, values in doc_dict.items()
            },
            schema=doc_schema,
        )
        lance.write_dataset(doc_table, str(path / "documents.lance"))

        # --- topics.lance ---
        topic_table = pa.Table.from_pandas(topic_df, preserve_index=False)
        lance.write_dataset(topic_table, str(path / "topics.lance"))

        # --- clusters.lance ---
        # Flatten all sparse layers to COO and tag each row with its layer index.
        # Lance has no uint8 column type, so values are stored as int32.
        coo_layers, coo_rows, coo_cols, coo_vals = [], [], [], []
        for layer_idx, matrix in enumerate(self.cluster_layers):
            coo = matrix.tocoo()
            n = len(coo.data)
            coo_layers.append(np.full(n, layer_idx, dtype=np.int64))
            coo_rows.append(coo.row.astype(np.int64))
            coo_cols.append(coo.col.astype(np.int64))
            coo_vals.append(coo.data.astype(np.int32))

        clusters_table = pa.table(
            {
                "layer": pa.array(
                    (np.concatenate(coo_layers) if coo_layers else []), type=pa.int64()
                ),
                "row_idx": pa.array(
                    (np.concatenate(coo_rows) if coo_rows else []), type=pa.int64()
                ),
                "col_idx": pa.array(
                    (np.concatenate(coo_cols) if coo_cols else []), type=pa.int64()
                ),
                "value": pa.array(
                    (np.concatenate(coo_vals) if coo_vals else []), type=pa.int32()
                ),
            }
        )
        lance.write_dataset(clusters_table, str(path / "clusters.lance"))

        uid_tree = {
            topic_uid(k): [topic_uid(c) for c in v]
            for k, v in self.cluster_tree.items()
        }

        if self.clustering_graph is not None:
            edges = self.clustering_graph.tocoo()
            graph_table = pa.table(
                {
                    "row": pa.array(edges.row, type=pa.int64()),
                    "column": pa.array(edges.col, type=pa.int64()),
                    "distance": pa.array(edges.data),
                }
            )
            lance.write_dataset(graph_table, str(path / "graph.lance"))

        # --- config.lance ---
        config_table = pa.table(
            {
                "serial_version": pa.array([_SERIAL_VERSION], type=pa.string()),
                "n_layers": pa.array([len(self.cluster_layers)], type=pa.int32()),
                "has_reduced": pa.array([has_reduced], type=pa.bool_()),
                "embedding_dtype": [str(self.embedding_vectors.dtype)],
                "embedding_dim": [self.embedding_vectors.shape[1]],
                "reduced_dtype": [
                    str(self.reduced_vectors.dtype) if has_reduced else None
                ],
                "reduced_dim": [self.reduced_vectors.shape[1] if has_reduced else None],
                "layer_columns": [[matrix.shape[1] for matrix in self.cluster_layers]],
                "runtime_metadata": [json.dumps(self.metadata)],
                "has_clustering_graph": [self.clustering_graph is not None],
                "graph_dtype": [
                    (
                        str(self.clustering_graph.dtype)
                        if self.clustering_graph is not None
                        else None
                    )
                ],
                "cluster_tree": pa.array(
                    [json.dumps(uid_tree)],
                    type=pa.string(),
                ),
            }
        )
        lance.write_dataset(config_table, str(path / "config.lance"))

    @property
    def topic_name_vectors(self):
        vectors = [
            np.full(len(self.embedding_vectors), "Unlabelled", dtype=object)
            for _ in self.cluster_layers
        ]
        for (layer, _), topic in self.topics.items():
            vectors[layer][topic.members] = (
                topic.name if topic.name is not None else "Unlabelled"
            )
        return vectors

    @property
    def topic_names(self):
        return [
            {
                label: topic.name
                for (index, label), topic in self.topics.items()
                if index == layer
            }
            for layer in range(len(self.cluster_layers))
        ]

    def topic_tree(self, prune_duplicates=True, **kwargs):
        """
        Returns the topic tree with configurable options.

        Parameters
        ----------
        prune_duplicates : bool, optional (default=True)
            If True, prune duplicate children from the tree.
        **kwargs
            Additional keyword arguments to pass to TopicTree constructor.

        Returns
        -------
        TopicTree
            A representation of the topic tree (either html or string).
        """
        return TopicTree(
            self.cluster_tree,
            self.topic_names,
            self.topic_sizes,
            self.embedding_vectors.shape[0],
            prune_duplicates=prune_duplicates,
            **kwargs,
        )
