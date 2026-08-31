import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from copy import deepcopy
from dataclasses import dataclass, field
import base64

import scipy.sparse as sp
import pandas as pd
import numpy as np
import shutil

from toponymy.topic_tree import TopicTree

_SERIAL_VERSION = "0.2"
_READABLE_SERIAL_VERSIONS = {"0.1", "0.2"}


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
    if s.startswith("v2:"):
        _, layer, cluster = s.split(":")
        values = int(layer), int(cluster)
        if values[0] < 0 or values[1] < -1:
            raise ValueError("Invalid topic identifier")
        return values
    padded = s + "=" * (-len(s) % 4)
    combined = int.from_bytes(base64.urlsafe_b64decode(padded), "big")
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
            if table is not None:
                for layer_index, matrix in enumerate(self.cluster_layers):
                    rows = table[table["layer"] == layer_index].sort_values("cluster")
                    for ordinal, row in enumerate(rows.to_dict("records")):
                        label = int(row["cluster"])
                        column = ordinal if len(rows) == matrix.shape[1] else label
                        members = matrix.getcol(column).nonzero()[0]
                        members.flags.writeable = False
                        features = (
                            json.loads(row["features_json"])
                            if row.get("features_json")
                            else {"cluster_keywords": list(row.get("keyphrases", []))}
                        )
                        prompt_data = (
                            json.loads(row["prompt_json"])
                            if row.get("prompt_json")
                            else None
                        )
                        prompt = Prompt(**prompt_data) if prompt_data else None
                        topics[(layer_index, label)] = Topic(
                            layer_index,
                            label,
                            members,
                            features,
                            prompt,
                            row.get("name"),
                            row.get("summary"),
                            row.get("explanation"),
                        )
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
                    "keyphrases": topic.features.get("cluster_keywords", []),
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
        path = Path(path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with zipfile.ZipFile(path) as z:
                for entry in z.infolist():
                    member = PurePosixPath(entry.filename.replace("\\", "/"))
                    if (
                        member.is_absolute()
                        or ".." in member.parts
                        or ":" in entry.filename
                    ):
                        raise ValueError("Invalid path in topic archive")
                z.extractall(root)

            with open(root / "metadata.json", encoding="utf8") as f:
                metadata = json.load(f)

            serial_version = metadata["serial_version"]
            if serial_version not in _READABLE_SERIAL_VERSIONS:
                raise ValueError(
                    f"The file's serial version ({serial_version}) does not match "
                    f"the current version ({_SERIAL_VERSION})."
                )

            has_reduced = metadata["has_reduced"]

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
            matrices_dir = root / "cluster_matrices"
            layer_files = sorted(
                matrices_dir.glob("layer_*.npz"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            matrices = [sp.load_npz(f) for f in layer_files]

            # --- Cluster tree topology ---
            with open(root / "cluster_tree.json") as f:
                raw_tree = json.load(f)

            cluster_tree = {
                uid_to_ints(k): [uid_to_ints(child) for child in v]
                for k, v in raw_tree.items()
            }

            return cls(
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
        import lance

        path = Path(path)

        # --- config ---
        config = lance.dataset(str(path / "config.lance")).to_table().to_pydict()
        serial_version = config["serial_version"][0]
        if serial_version not in _READABLE_SERIAL_VERSIONS:
            raise ValueError(
                f"The file's serial version ({serial_version}) does not match "
                f"the current version ({_SERIAL_VERSION})."
            )
        n_layers = config["n_layers"][0]
        has_reduced = config["has_reduced"][0]
        raw_tree = json.loads(config["cluster_tree"][0])

        cluster_tree = {
            uid_to_ints(k): [uid_to_ints(child) for child in v]
            for k, v in raw_tree.items()
        }

        doc_table = lance.dataset(str(path / "documents.lance")).to_table().to_pydict()
        embedding_vectors = np.array(
            doc_table.pop("embedding"),
            dtype=config.get("embedding_dtype", ["float32"])[0],
        )
        if "embedding_dim" in config:
            embedding_vectors = embedding_vectors.reshape(
                -1, config["embedding_dim"][0]
            )
        reduced_vectors = None
        if has_reduced:
            reduced_vectors = np.array(
                doc_table.pop("reduced_embedding"),
                dtype=config.get("reduced_dtype", ["float32"])[0],
            )
            if "reduced_dim" in config:
                reduced_vectors = reduced_vectors.reshape(-1, config["reduced_dim"][0])
        document_df = pd.DataFrame(doc_table)

        topic_dict = lance.dataset(str(path / "topics.lance")).to_table().to_pydict()
        topic_df = pd.DataFrame(topic_dict)

        coo_dict = lance.dataset(str(path / "clusters.lance")).to_table().to_pydict()
        layers_arr = np.array(coo_dict["layer"], dtype=np.int64)
        rows_arr = np.array(coo_dict["row_idx"], dtype=np.int64)
        cols_arr = np.array(coo_dict["col_idx"], dtype=np.int64)
        vals_arr = np.array(coo_dict["value"], dtype=np.uint8)  # safe: values are 0-255
        n_docs = len(document_df)

        matrices = []
        for layer_idx in range(n_layers):
            mask = layers_arr == layer_idx
            n_cols = (
                config["layer_columns"][0][layer_idx]
                if "layer_columns" in config
                else int(cols_arr[mask].max()) + 1 if mask.any() else 0
            )
            csr = sp.coo_matrix(
                (vals_arr[mask], (rows_arr[mask], cols_arr[mask])),
                shape=(n_docs, n_cols),
                dtype=np.uint8,
            ).tocsr()
            matrices.append(csr)

        clustering_graph = None
        if config.get("has_clustering_graph", [False])[0]:
            edges = lance.dataset(str(path / "graph.lance")).to_table().to_pydict()
            clustering_graph = sp.csr_matrix(
                (
                    np.asarray(edges["distance"], dtype=config["graph_dtype"][0]),
                    (edges["row"], edges["column"]),
                ),
                shape=(n_docs, n_docs),
            )
        return cls(
            embedding_vectors=embedding_vectors,
            reduced_vectors=reduced_vectors,
            document_df=document_df,
            topic_df=topic_df,
            cluster_tree=cluster_tree,
            cluster_layers=matrices,
            metadata=json.loads(config.get("runtime_metadata", ["{}"])[0]),
            clustering_graph=clustering_graph,
        )

    def to_lance(self, path: str, overwrite: bool = False):

        import lance
        import pyarrow as pa

        path = Path(path)

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
        doc_table = pa.table(doc_dict, schema=doc_schema)
        lance.write_dataset(doc_table, str(path / "documents.lance"))

        # --- topics.lance ---
        topic_df = deepcopy(self.topic_df)
        topic_dict = {col: topic_df[col].tolist() for col in topic_df.columns}
        topic_table = pa.table(topic_dict)
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
