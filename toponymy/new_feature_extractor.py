"""Cluster features, independent of clustering and topic naming state.

Results are indexed by layer, then sorted cluster ordinal, then feature.
Original cluster labels are preserved on the clusterer; dense IDs are used only
when calling the existing numerical extraction helpers.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class FeatureExtractorBase(ABC, BaseEstimator):
    """Fit-scoped feature extraction. Calling fit_predict always recomputes."""

    feature_key = ""
    feature_return_type = str
    layer_dependent = False

    @property
    def features(self) -> list[list[list[str]]]:
        return getattr(self, "features_", None)

    def __sklearn_is_fitted__(self):
        return self.features is not None

    def can_fit_from_objects(self):
        return True

    @abstractmethod
    def fit(self, objects, clusterer, **configuration):
        """Compute features for the supplied objects and clustering."""

    def predict(self):
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(f"{type(self).__name__} has not been fitted")
        return self.features_

    def fit_predict(self, objects, clusterer, **configuration):
        self.fit(objects, clusterer, **configuration)
        return self.predict()


def _dense_labels(labels, n_objects):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n_objects:
        raise ValueError("Cluster labels must match the number of objects")
    if labels.dtype.kind not in "iu" or np.any(labels < -1):
        raise ValueError("Cluster labels must be integers with only -1 for noise")
    present = labels >= 0
    dense = np.full(n_objects, -1, dtype=np.int64)
    ids, dense[present] = np.unique(labels[present], return_inverse=True)
    return dense, ids


def _vectors(vectors, n_objects):
    if vectors is None:
        raise ValueError("This selection method requires embedding_vectors")
    result = np.asarray(vectors)
    if result.ndim != 2 or result.shape[0] != n_objects or not result.shape[1]:
        raise ValueError("Vectors must have shape (number of objects, dimensions)")
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError("Vectors must contain finite real numbers")
    return np.ascontiguousarray(result, dtype=np.float64)


class TextExemplarExtractor(FeatureExtractorBase):
    """Represent clusters using central, random, or submodular exemplars."""

    feature_key = "cluster_sentences"
    supported_selection_methods = (
        "central", "random", "facility_location", "saturated_coverage"
    )

    def __init__(self, selection_method="central", n_exemplars=4,
                 diversify_alpha=1.0, random_state=0,
                 object_to_text_function: Optional[Callable] = None):
        self.selection_method = selection_method
        self.n_exemplars = n_exemplars
        self.diversify_alpha = diversify_alpha
        self.random_state = random_state
        self.object_to_text_function = object_to_text_function

    def fit(self, objects, clusterer, selection_method=None, object_vectors=None,
            *, embedding_vectors=None, **configuration):
        from .exemplar_texts import (
            diverse_exemplars, random_exemplars, submodular_selection_exemplars,
        )

        self.features_ = None
        method = self.selection_method if selection_method is None else selection_method
        if method not in self.supported_selection_methods:
            raise ValueError(f"Unsupported selection method: {method}")
        parameters = dict(configuration)
        parameters.setdefault("n_exemplars", self.n_exemplars)
        parameters.setdefault("object_to_text_function", self.object_to_text_function)
        n_exemplars = parameters["n_exemplars"]
        if not isinstance(n_exemplars, (int, np.integer)) or n_exemplars < 1:
            raise ValueError("n_exemplars must be a positive integer")
        parameters.setdefault("random_state", self.random_state)
        if method == "central":
            parameters.setdefault("diversify_alpha", self.diversify_alpha)
        layers = list(clusterer)
        labels_and_ids = [_dense_labels(layer.labels, len(objects)) for layer in layers]
        vectors = embedding_vectors if embedding_vectors is not None else object_vectors
        if method != "random" and any(len(ids) for _, ids in labels_and_ids):
            vectors = _vectors(vectors, len(objects))
        features, indices = [], []
        for labels, ids in labels_and_ids:
            if not len(ids):
                layer_features, layer_indices = [], []
            elif method == "random":
                layer_features, layer_indices = random_exemplars(
                    labels, objects, **parameters
                )
            elif method == "central":
                layer_features, layer_indices = diverse_exemplars(
                    labels, objects, vectors, **parameters,
                )
            else:
                layer_features, layer_indices = submodular_selection_exemplars(
                    labels, objects, vectors, submodular_function=method, **parameters
                )
            features.append([list(values) for values in layer_features])
            indices.append([list(map(int, values)) for values in layer_indices])
        self.features_, self.indices_ = features, indices
        return self


class TextKeyphraseExtractor(FeatureExtractorBase):
    """Optional keyphrase extraction using the established numerical strategies.

    Provide an embedder or an explicit count matrix, vocabulary and keyphrase
    vectors. Explicit inputs belong to this fit; they are never reused implicitly.
    """

    feature_key = "cluster_keywords"
    supported_selection_methods = (
        "information_weighted", "central", "bm25", "saturated_coverage",
        "facility_location", "graph_cut",
    )

    def __init__(self, selection_method="information_weighted", n_keyphrases=16,
                 keyphrase_builder=None):
        self.selection_method = selection_method
        self.n_keyphrases = n_keyphrases
        self.keyphrase_builder = keyphrase_builder

    def fit(self, objects, clusterer, selection_method=None, *, embedder=None,
            embedding_model=None, object_x_keyphrase_matrix=None,
            keyphrase_list=None, keyphrase_vectors=None, **configuration):
        from . import keyphrases

        self.features_ = None
        method = self.selection_method if selection_method is None else selection_method
        if method not in self.supported_selection_methods:
            raise ValueError(f"Unsupported selection method: {method}")
        layers = [_dense_labels(layer.labels, len(objects)) for layer in clusterer]
        if not any(len(ids) for _, ids in layers):
            self.features_ = [[] for _ in layers]
            return self
        model = embedder if embedder is not None else embedding_model
        parameters = dict(configuration)
        # Semantic document vectors are pipeline inputs, not keyphrase vectors.
        parameters.pop("object_vectors", None)
        parameters.pop("embedding_vectors", None)
        builder_keys = (
            "object_to_text", "ngram_range", "tokenizer", "token_pattern",
            "max_features", "min_occurrences", "stop_words", "n_jobs",
        )
        builder_options = {key: parameters.pop(key) for key in builder_keys
                           if key in parameters}
        builder_options.setdefault("n_jobs", 1)
        if object_x_keyphrase_matrix is None:
            if keyphrase_list is not None or keyphrase_vectors is not None:
                raise ValueError("Explicit keyphrase vectors require a count matrix and vocabulary")
            builder = self.keyphrase_builder
            if builder is None:
                builder = keyphrases.KeyphraseBuilder(embedder=model, **builder_options)
            matrix, vocabulary, vectors = builder.fit_transform(objects)
        else:
            matrix, vocabulary, vectors = (
                object_x_keyphrase_matrix, keyphrase_list, keyphrase_vectors
            )
        matrix = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
        if matrix.shape[0] != len(objects) or vocabulary is None or matrix.shape[1] != len(vocabulary):
            raise ValueError("Count matrix must align with objects and keyphrase vocabulary")
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Keyphrase counts must be finite and nonnegative")
        vocabulary = list(vocabulary)
        if not all(isinstance(phrase, str) for phrase in vocabulary):
            raise TypeError("Keyphrases must be strings")
        n_keyphrases = parameters.pop("n_keyphrases", self.n_keyphrases)
        if not isinstance(n_keyphrases, (int, np.integer)) or n_keyphrases < 1:
            raise ValueError("n_keyphrases must be a positive integer")
        if not vocabulary or not matrix.nnz:
            self.features_ = [[[] for _ in ids] for _, ids in layers]
            return self
        if vectors is None:
            if model is None:
                raise ValueError("Keyphrase extraction requires an embedder or keyphrase_vectors")
            vectors = model.encode(vocabulary, show_progress_bar=False)
        # Existing selectors fill missing embeddings in place; own this small array.
        vectors = _vectors(vectors, len(vocabulary)).copy()
        helpers = {
            "information_weighted": keyphrases.information_weighted_keyphrases,
            "central": keyphrases.central_keyphrases,
            "bm25": keyphrases.bm25_keyphrases,
        }
        helper = helpers.get(method, keyphrases.submodular_selection_information_keyphrases)
        if method not in helpers:
            parameters["submodular_function"] = method
        features = []
        for labels, ids in layers:
            if not len(ids):
                values = []
            elif not matrix[labels >= 0].nnz:
                values = [[] for _ in ids]
            else:
                values = helper(labels, matrix, vocabulary, vectors, model,
                                n_keyphrases=n_keyphrases, **parameters)
            features.append([list(value) for value in values])
        self.object_x_keyphrase_matrix_ = matrix
        self.keyphrase_list_, self.keyphrase_vectors_ = vocabulary, vectors
        self.features_ = features
        return self


# The shorter name is convenient when mixing text and layer-dependent extractors.
KeyphraseExtractor = TextKeyphraseExtractor
