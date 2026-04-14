import json
import tempfile
import zipfile
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
import base64

import scipy.sparse as sp
import pandas as pd
import numpy as np

_SERIAL_VERSION = "0.1"


def topic_uid(tup) -> str:
    a, b = tup
    a = int(a)
    b = (
        int(b) + 1
    )  # Because unclustered is -1 and we can't convert negative to unsigned.
    combined = (a << 10) | b  # pack into 20 bits
    return base64.urlsafe_b64encode(combined.to_bytes(3, "big")).rstrip(b"=").decode()


def uid_to_ints(s: str):
    """Returns (layer, cluster_number)"""
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
class TopicModel:
    """Storage class for the data of a fitted Toponymy."""

    topic_df: pd.DataFrame
    cluster_tree: dict
    cluster_layers: list
    embedding_vectors: np.ndarray
    reduced_vectors: np.ndarray = None
    document_df: pd.DataFrame = None

    def __repr__(self):
        n_samples = self.embedding_vectors.shape[0]
        n_topics = len(self.topic_df)
        s = f"TopicModel(n_samples={n_samples},"
        s += f" n_topics={n_topics})"
        return s

    @classmethod
    def from_toponymy(cls, toponymy, document_df=None):
        cluster_layers = []
        for layer_idx, layer in enumerate(toponymy.cluster_layers_):
            labels = layer.cluster_labels
            unique_labels = np.unique(labels)
            n_clusters = int((unique_labels[unique_labels >= 0]).max()) + 1

            matrix = np.zeros((len(labels), n_clusters), dtype=np.uint8)
            for doc_idx, label in enumerate(labels):
                if label >= 0:  # skip noise points (label == -1)
                    matrix[doc_idx, label] = 255

            cluster_layers.append(sp.csr_matrix(matrix))

        # --- Topic metadata ---
        rows = []
        for layer_idx, layer in enumerate(toponymy.cluster_layers_):
            unique_labels = np.unique(layer.cluster_labels)
            for cluster in unique_labels[unique_labels >= 0]:
                rows.append(
                    {
                        "uid": topic_uid((layer_idx, int(cluster))),
                        "layer": layer_idx,
                        "cluster": cluster,
                        "name": toponymy.topic_names_[layer_idx][cluster],
                        "keyphrases": toponymy.cluster_layers_[layer_idx].keyphrases[
                            cluster
                        ],
                    }
                )
        topic_df = pd.DataFrame(rows)
        if document_df is None:
            n_samples = toponymy.embedding_vectors_.shape[0]
            document_df = pd.DataFrame({"item_num": range(n_samples)})

        return cls(
            cluster_layers=cluster_layers,
            cluster_tree=toponymy.cluster_tree_,
            topic_df=topic_df,
            embedding_vectors=toponymy.embedding_vectors_,
            reduced_vectors=toponymy.clusterable_vectors_,
            document_df=document_df,
        )

    @classmethod
    def from_file(cls, path: str):
        path = Path(path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with zipfile.ZipFile(path) as z:
                z.extractall(root)

            with open(root / "metadata.json") as f:
                metadata = json.load(f)

            serial_version = metadata["serial_version"]
            if serial_version != _SERIAL_VERSION:
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
        if serial_version != _SERIAL_VERSION:
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
        embedding_vectors = np.array(doc_table.pop("embedding"), dtype=np.float32)
        reduced_vectors = None
        if has_reduced:
            reduced_vectors = np.array(
                doc_table.pop("reduced_embedding"), dtype=np.float32
            )
        document_df = pd.DataFrame(doc_table)

        topic_dict = lance.dataset(str(path / "topics.lance")).to_table().to_pydict()
        topic_df = pd.DataFrame(topic_dict)

        coo_dict = lance.dataset(str(path / "clusters.lance")).to_table().to_pydict()
        layers_arr = np.array(coo_dict["layer"], dtype=np.int16)
        rows_arr = np.array(coo_dict["row_idx"], dtype=np.int32)
        cols_arr = np.array(coo_dict["col_idx"], dtype=np.int16)
        vals_arr = np.array(coo_dict["value"], dtype=np.uint8)  # safe: values are 0-255
        n_docs = len(document_df)

        matrices = []
        for layer_idx in range(n_layers):
            mask = layers_arr == layer_idx
            n_cols = int(cols_arr[mask].max()) + 1 if mask.any() else 0
            csr = sp.coo_matrix(
                (vals_arr[mask], (rows_arr[mask], cols_arr[mask])),
                shape=(n_docs, n_cols),
                dtype=np.uint8,
            ).tocsr()
            matrices.append(csr)

        return cls(
            embedding_vectors=embedding_vectors,
            reduced_vectors=reduced_vectors,
            document_df=document_df,
            topic_df=topic_df,
            cluster_tree=cluster_tree,
            cluster_layers=matrices,
        )

    def to_lance(self, path: str):

        import lance
        import pyarrow as pa

        path = Path(path)
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Remove it first or choose a different path."
            )
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
            pa.field("embedding", pa.list_(pa.float32(), emb_dim)),
        ]

        has_reduced = self.reduced_vectors is not None
        if has_reduced:
            red_dim = self.reduced_vectors.shape[1]
            doc_dict["reduced_embedding"] = self.reduced_vectors.tolist()
            schema_fields.append(
                pa.field("reduced_embedding", pa.list_(pa.float32(), red_dim))
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
            coo_layers.append(np.full(n, layer_idx, dtype=np.int16))
            coo_rows.append(coo.row.astype(np.int32))
            coo_cols.append(coo.col.astype(np.int16))
            coo_vals.append(coo.data.astype(np.int32))

        clusters_table = pa.table(
            {
                "layer": pa.array(np.concatenate(coo_layers), type=pa.int16()),
                "row_idx": pa.array(np.concatenate(coo_rows), type=pa.int32()),
                "col_idx": pa.array(np.concatenate(coo_cols), type=pa.int16()),
                "value": pa.array(np.concatenate(coo_vals), type=pa.int32()),
            }
        )
        lance.write_dataset(clusters_table, str(path / "clusters.lance"))

        uid_tree = {
            topic_uid(k): [topic_uid(c) for c in v]
            for k, v in self.cluster_tree.items()
        }

        # --- config.lance ---
        config_table = pa.table(
            {
                "serial_version": pa.array([_SERIAL_VERSION], type=pa.string()),
                "n_layers": pa.array([len(self.cluster_layers)], type=pa.int32()),
                "has_reduced": pa.array([has_reduced], type=pa.bool_()),
                "cluster_tree": pa.array(
                    [json.dumps(uid_tree)],
                    type=pa.string(),
                ),
            }
        )
        lance.write_dataset(config_table, str(path / "config.lance"))

    @property
    def topic_name_vectors(self):
        vectors = []
        max_len = max([len(x) for x in self.topic_df["name"].values])
        for layer, matrix in enumerate(self.cluster_layers):
            matrix = matrix.todense()
            vector_layer = np.full(matrix.shape[0], "Unlabelled", dtype=f"<U{max_len}")
            for cluster in range(matrix.shape[1]):
                cluster_uid = topic_uid((layer, cluster))
                cluster_name = self.topic_df[self.topic_df["uid"] == cluster_uid][
                    "name"
                ].values[0]
                column = matrix[:, cluster]
                cluster_index = (column == 255).nonzero()[0]
                vector_layer[cluster_index] = cluster_name
            vectors.append(vector_layer)
        return vectors

    @property
    def topic_names(self):
        all_names = []
        for layer, matrix in enumerate(self.cluster_layers):
            layer_names = []
            for cluster in range(matrix.shape[1]):
                cluster_uid = topic_uid((layer, cluster))
                cluster_name = self.topic_df[self.topic_df["uid"] == cluster_uid][
                    "name"
                ].values[0]
                layer_names.append(cluster_name)
            all_names.append(layer_names)
        return all_names
