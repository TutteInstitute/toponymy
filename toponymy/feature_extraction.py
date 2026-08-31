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

ClusterFeatures = list[str] | dict[str, list[str]]


class FeatureExtractorBase(ABC, BaseEstimator):
    """Fit-scoped feature extraction. Calling fit_predict always recomputes."""

    feature_key = ""
    feature_return_type = str
    layer_dependent = False

    @property
    def features(self) -> list[list[ClusterFeatures]]:
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
    if labels.size and (
        labels.dtype.kind not in "iu"
        or np.any(labels < -1)
        or np.any(labels > np.iinfo(np.int64).max)
    ):
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
    # Numerical helpers accept strided float32/float64 inputs without copying.
    return (
        result
        if result.dtype in (np.dtype("float32"), np.dtype("float64"))
        else result.astype(np.float64)
    )


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class TextExemplarExtractor(FeatureExtractorBase):
    """Represent clusters using central, random, or submodular exemplars."""

    feature_key = "cluster_sentences"
    supported_selection_methods = (
        "central",
        "random",
        "facility_location",
        "saturated_coverage",
    )

    def __init__(
        self,
        selection_method="central",
        n_exemplars=4,
        diversify_alpha=1.0,
        random_state=0,
        object_to_text_function: Optional[Callable] = None,
    ):
        self.selection_method = selection_method
        self.n_exemplars = n_exemplars
        self.diversify_alpha = diversify_alpha
        self.random_state = random_state
        self.object_to_text_function = object_to_text_function

    def fit(
        self,
        objects,
        clusterer,
        selection_method=None,
        object_vectors=None,
        *,
        embedding_vectors=None,
        **configuration,
    ):
        from .exemplar_texts import (
            diverse_exemplars,
            random_exemplars,
            submodular_selection_exemplars,
        )

        self.features_ = None
        self.indices_ = None
        method = self.selection_method if selection_method is None else selection_method
        if method not in self.supported_selection_methods:
            raise ValueError(f"Unsupported selection method: {method}")
        parameters = dict(configuration)
        parameters.setdefault("n_exemplars", self.n_exemplars)
        parameters.setdefault("object_to_text_function", self.object_to_text_function)
        n_exemplars = parameters["n_exemplars"]
        _positive_integer(n_exemplars, "n_exemplars")
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
                    labels,
                    objects,
                    vectors,
                    **parameters,
                )
            else:
                layer_features, layer_indices = submodular_selection_exemplars(
                    labels, objects, vectors, submodular_function=method, **parameters
                )
            layer_features = [list(values) for values in layer_features]
            if any(
                not all(isinstance(value, str) for value in values)
                or len(values) != len(selected)
                for values, selected in zip(layer_features, layer_indices)
            ):
                raise TypeError(
                    "Exemplar conversion must return one string per selected object"
                )
            features.append(layer_features)
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
        "information_weighted",
        "central",
        "bm25",
        "saturated_coverage",
        "facility_location",
        "graph_cut",
    )

    def __init__(
        self,
        selection_method="information_weighted",
        n_keyphrases=16,
        keyphrase_builder=None,
    ):
        self.selection_method = selection_method
        self.n_keyphrases = n_keyphrases
        self.keyphrase_builder = keyphrase_builder

    def fit(
        self,
        objects,
        clusterer,
        selection_method=None,
        *,
        embedder=None,
        embedding_model=None,
        object_x_keyphrase_matrix=None,
        keyphrase_list=None,
        keyphrase_vectors=None,
        **configuration,
    ):
        from . import keyphrases

        self.features_ = None
        self.object_x_keyphrase_matrix_ = None
        self.keyphrase_list_, self.keyphrase_vectors_ = [], None
        method = self.selection_method if selection_method is None else selection_method
        if method not in self.supported_selection_methods:
            raise ValueError(f"Unsupported selection method: {method}")
        layers = [_dense_labels(layer.labels, len(objects)) for layer in clusterer]
        n_keyphrases = _positive_integer(
            configuration.get("n_keyphrases", self.n_keyphrases), "n_keyphrases"
        )
        if not any(len(ids) for _, ids in layers):
            self.features_ = [[] for _ in layers]
            return self
        model = embedder if embedder is not None else embedding_model
        parameters = dict(configuration)
        parameters.pop("n_keyphrases", None)
        # Semantic document vectors are pipeline inputs, not keyphrase vectors.
        parameters.pop("object_vectors", None)
        parameters.pop("embedding_vectors", None)
        builder_keys = (
            "object_to_text",
            "ngram_range",
            "tokenizer",
            "token_pattern",
            "max_features",
            "min_occurrences",
            "stop_words",
            "n_jobs",
        )
        builder_options = {
            key: parameters.pop(key) for key in builder_keys if key in parameters
        }
        builder_options.setdefault("n_jobs", 1)
        if object_x_keyphrase_matrix is None:
            if keyphrase_list is not None or keyphrase_vectors is not None:
                raise ValueError(
                    "Explicit keyphrase vectors require a count matrix and vocabulary"
                )
            builder = self.keyphrase_builder
            if builder is None:
                builder = keyphrases.KeyphraseBuilder(embedder=model, **builder_options)
            matrix, vocabulary, vectors = builder.fit_transform(objects)
        else:
            matrix, vocabulary, vectors = (
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
            )
        matrix = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
        if (
            matrix.shape[0] != len(objects)
            or vocabulary is None
            or matrix.shape[1] != len(vocabulary)
        ):
            raise ValueError(
                "Count matrix must align with objects and keyphrase vocabulary"
            )
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Keyphrase counts must be finite and nonnegative")
        vocabulary = list(vocabulary)
        if not all(isinstance(phrase, str) for phrase in vocabulary):
            raise TypeError("Keyphrases must be strings")
        if not vocabulary or not matrix.nnz:
            self.features_ = [[[] for _ in ids] for _, ids in layers]
            return self
        if vectors is None:
            if model is None:
                raise ValueError(
                    "Keyphrase extraction requires an embedder or keyphrase_vectors"
                )
            vectors = model.encode(vocabulary, show_progress_bar=False)
        # Existing selectors fill missing embeddings in place; own this small array.
        vectors = _vectors(vectors, len(vocabulary)).copy()
        helpers = {
            "information_weighted": keyphrases.information_weighted_keyphrases,
            "central": keyphrases.central_keyphrases,
            "bm25": keyphrases.bm25_keyphrases,
        }
        helper = helpers.get(
            method, keyphrases.submodular_selection_information_keyphrases
        )
        if method not in helpers:
            parameters["submodular_function"] = method
        features = []
        for labels, ids in layers:
            if not len(ids):
                values = []
            elif not matrix[labels >= 0].nnz:
                values = [[] for _ in ids]
            else:
                values = helper(
                    labels,
                    matrix,
                    vocabulary,
                    vectors,
                    model,
                    n_keyphrases=n_keyphrases,
                    **parameters,
                )
            features.append([list(value) for value in values])
        self.object_x_keyphrase_matrix_ = matrix
        self.keyphrase_list_, self.keyphrase_vectors_ = vocabulary, vectors
        self.features_ = features
        return self


# The shorter name is convenient when mixing text and layer-dependent extractors.
KeyphraseExtractor = TextKeyphraseExtractor


class SubtopicExtractor(FeatureExtractorBase):
    """Use direct children's evidence, ordered by size and stable key.

    The validated containment tree supplies children even when crossing
    partitions cause an edge to skip a layer. Naming the lower layers first is
    required; cluster objects themselves never acquire topic names.
    ``source`` selects each child's ``name`` (the default), ``summary`` or
    ``explanation``. The selected field must be available and nonempty;
    summary evidence therefore requires a summary-producing naming template.
    """

    feature_key = "cluster_subtopics"
    feature_return_type = dict
    layer_dependent = True

    def __init__(self, n_subtopics=64, *, source="name"):
        self.n_subtopics = n_subtopics
        self.source = source

    def _validate_configuration(self):
        _positive_integer(self.n_subtopics, "n_subtopics")
        if self.source not in ("name", "summary", "explanation"):
            raise ValueError("source must be 'name', 'summary' or 'explanation'")

    def fit(self, objects, clusterer, **configuration):
        self._validate_configuration()
        self.features_ = [
            [{"major": [], "minor": [], "misc": []} for _ in layer]
            for layer in clusterer
        ]
        return self

    def extract_layer(self, layer_index, topics, clusterer):
        self._validate_configuration()
        layers = list(clusterer)
        if not 0 <= layer_index < len(layers):
            raise ValueError("Invalid layer index")
        tree = clusterer.cluster_tree_
        cluster_lookup = {
            (i, cluster.label): cluster
            for i, layer in enumerate(layers)
            for cluster in layer
        }
        features = []
        for cluster in layers[layer_index]:
            children = tree.get((layer_index, cluster.label), [])
            children = sorted(
                children, key=lambda key: (-len(cluster_lookup[key].members), key)
            )
            values = []
            for key in children:
                if key[0] >= layer_index:
                    raise ValueError("Subtopics must come from lower layers")
                topic = topics.get(key)
                name = (
                    topic.get("name")
                    if isinstance(topic, dict)
                    else getattr(topic, "name", None)
                )
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"Subtopic {key} must be named before extracting layer {layer_index}"
                    )
                value = (
                    topic.get(self.source)
                    if isinstance(topic, dict)
                    else getattr(topic, self.source, None)
                )
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Subtopic {key} requires a nonempty {self.source} "
                        f"before extracting layer {layer_index}"
                    )
                if value not in values:
                    values.append(value)
                if len(values) == self.n_subtopics:
                    break
            features.append({"major": values, "minor": [], "misc": []})
        if self.features is None or len(self.features_) != len(layers):
            self.fit([], layers)
        self.features_[layer_index] = features
        return features


class TreeSHAPKeyphraseExtractor(FeatureExtractorBase):
    """Opt-in contrastive phrases explained with exact TreeSHAP.

    For each cluster, a small binary ExtraTrees classifier contrasts its
    documents with other non-noise documents in that layer. Positive mean
    TreeSHAP values for terms present in the cluster rank the phrases. Training
    and explanation samples are bounded per class. These explain the fitted
    local classifier; they are not a validation of topic quality or causality.

    Requires the ``treeshap`` extra. No embedding model or network is used.
    """

    feature_key = "cluster_keywords"

    def __init__(
        self,
        n_keyphrases=16,
        max_features=512,
        max_samples_per_class=128,
        n_estimators=64,
        random_state=0,
        object_to_text: Optional[Callable[[Any], str]] = None,
    ):
        self.n_keyphrases = n_keyphrases
        self.max_features = max_features
        self.max_samples_per_class = max_samples_per_class
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.object_to_text = object_to_text

    def fit(
        self,
        objects,
        clusterer,
        *,
        object_x_keyphrase_matrix=None,
        keyphrase_list=None,
        object_vectors=None,
        embedding_vectors=None,
        embedder=None,
        embedding_model=None,
    ):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_extraction.text import CountVectorizer

        self.features_ = None
        self.attribution_scores_, self.keyphrase_list_ = [], []
        self.classifier_count_ = 0
        for key in (
            "n_keyphrases",
            "max_features",
            "max_samples_per_class",
            "n_estimators",
        ):
            _positive_integer(getattr(self, key), key)
        layers = [_dense_labels(layer.labels, len(objects)) for layer in clusterer]
        if not any(len(ids) > 1 for _, ids in layers):
            self.features_ = [[[] for _ in ids] for _, ids in layers]
            self.attribution_scores_ = [[] for _ in layers]
            self.classifier_count_ = 0
            return self
        try:
            from shap import TreeExplainer
        except ImportError as error:
            raise ImportError(
                "TreeSHAP extraction requires toponymy[treeshap]"
            ) from error
        if object_x_keyphrase_matrix is None:
            if keyphrase_list is not None:
                raise ValueError("A supplied vocabulary requires a count matrix")
            texts = (
                list(objects)
                if self.object_to_text is None
                else [self.object_to_text(obj) for obj in objects]
            )
            if not all(isinstance(text, str) for text in texts):
                raise TypeError("TreeSHAP extraction requires text or object_to_text")
            vectorizer = CountVectorizer(
                max_features=self.max_features, ngram_range=(1, 2), stop_words="english"
            )
            matrix = vectorizer.fit_transform(texts)
            vocabulary = vectorizer.get_feature_names_out().tolist()
        else:
            matrix = sp.csr_matrix(object_x_keyphrase_matrix, dtype=np.float64)
            if keyphrase_list is None:
                raise ValueError("A count matrix requires a keyphrase vocabulary")
            vocabulary = list(keyphrase_list)
        if matrix.shape != (len(objects), len(vocabulary)):
            raise ValueError("Count matrix must align with objects and vocabulary")
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Keyphrase counts must be finite and nonnegative")
        if not all(isinstance(phrase, str) for phrase in vocabulary):
            raise TypeError("Keyphrases must be strings")
        # Bound dense allocations before converting any document rows.
        if len(vocabulary) > self.max_features:
            totals = np.asarray(matrix.sum(axis=0)).ravel()
            keep = np.argsort(-totals, kind="stable")[: self.max_features]
            matrix = matrix[:, keep]
            vocabulary = [vocabulary[i] for i in keep]
        features, attribution_scores = [], []
        self.classifier_count_ = 0
        rng = np.random.default_rng(self.random_state)
        for labels, ids in layers:
            values, scores = [], []
            for ordinal in range(len(ids)):
                positive = np.flatnonzero(labels == ordinal)
                negative = np.flatnonzero((labels >= 0) & (labels != ordinal))
                if not len(negative) or not vocabulary:
                    values.append([])
                    scores.append(np.zeros(len(vocabulary)))
                    continue
                positive = rng.choice(
                    positive,
                    min(len(positive), self.max_samples_per_class),
                    replace=False,
                )
                negative = rng.choice(
                    negative,
                    min(len(negative), self.max_samples_per_class),
                    replace=False,
                )
                training = matrix[np.concatenate((negative, positive))].toarray()
                target = np.concatenate(
                    (np.zeros(len(negative)), np.ones(len(positive)))
                )
                classifier = ExtraTreesClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=1,
                ).fit(training, target)
                explainer = TreeExplainer(
                    classifier, feature_perturbation="tree_path_dependent"
                )
                explained = training[len(negative) :]
                shap_values = explainer.shap_values(explained, approximate=False)
                # SHAP >= 0.46 has (samples, features, class outputs) shape.
                shap_values = np.asarray(shap_values)
                if shap_values.shape != (len(positive), len(vocabulary), 2):
                    raise ValueError("Unexpected binary TreeSHAP output shape")
                score = shap_values[:, :, 1].mean(axis=0)
                present = np.asarray(matrix[labels == ordinal].sum(axis=0)).ravel() > 0
                candidates = np.flatnonzero((score > 0) & present)
                chosen = candidates[
                    np.argsort(-score[candidates], kind="stable")[: self.n_keyphrases]
                ]
                values.append([vocabulary[i] for i in chosen])
                scores.append(score)
                self.classifier_count_ += 1
            features.append(values)
            attribution_scores.append(scores)
        self.keyphrase_list_ = vocabulary
        self.attribution_scores_, self.features_ = attribution_scores, features
        return self
