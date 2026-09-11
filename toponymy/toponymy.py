"""The staged topic modelling pipeline."""

from copy import copy, deepcopy
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.exceptions import NotFittedError

from .clustering import PLSCANClusterer, validate_cluster_tree
from .feature_extraction import TextExemplarExtractor, TextKeyphraseExtractor
from .serialization import Topic, TopicModel
from .templates import TextTemplate
from .utility_functions import _normalize_rows


def _matrix(values, rows, name):
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (number of objects, dimensions)")
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    view = result.view()
    view.flags.writeable = False
    return view


class Toponymy:
    """Cluster objects, extract evidence, then name topics layer by layer.

    Call prepare to inspect features and initial prompts without provider
    requests. name_topics completes naming, including dependent layers.
    Each fit owns fresh topic state; matrices are borrowed read-only views.
    Callers must not mutate those matrices while fitting or using the result.
    Feature extractors use semantic embeddings; clusterable_vectors can have
    a different dimension and are used only for clustering.
    """

    def __init__(
        self,
        llm_wrapper,
        text_embedding_model=None,
        clusterer=None,
        *,
        feature_extractors=None,
        feature_options=None,
        prompt_template=None,
        object_description="objects",
        corpus_description="collection of objects",
        lowest_detail_level=0.0,
        highest_detail_level=1.0,
        disambiguate=True,
        disambiguation_threshold=0.9,
        max_disambiguation_group_size=4,
        verbose=False,
    ):
        if not 0 <= lowest_detail_level <= highest_detail_level <= 1:
            raise ValueError("Detail levels must satisfy 0 <= lowest <= highest <= 1")
        if not 0 <= disambiguation_threshold <= 1:
            raise ValueError("disambiguation_threshold must lie in [0, 1]")
        if (
            isinstance(max_disambiguation_group_size, bool)
            or not isinstance(max_disambiguation_group_size, int)
            or max_disambiguation_group_size < 2
        ):
            raise ValueError(
                "max_disambiguation_group_size must be an integer of at least 2"
            )
        self.llm_wrapper = llm_wrapper
        self.embedding_model = text_embedding_model
        self.clusterer = PLSCANClusterer() if clusterer is None else clusterer
        self.feature_extractors = (
            [TextExemplarExtractor()]
            if feature_extractors is None
            else list(feature_extractors)
        )
        keys = [extractor.feature_key for extractor in self.feature_extractors]
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Every feature extractor needs a nonempty feature_key")
        if len(keys) != len(set(keys)):
            raise ValueError("Feature extractors must have distinct feature_key values")
        self.feature_options = (
            {}
            if feature_options is None
            else {key: dict(options) for key, options in feature_options.items()}
        )
        if set(self.feature_options) - set(keys):
            raise ValueError("feature_options keys must identify configured extractors")
        self.prompt_template = (
            TextTemplate(object_description, corpus_description)
            if prompt_template is None
            else prompt_template
        )
        self.object_description = object_description
        self.corpus_description = corpus_description
        self.lowest_detail_level = lowest_detail_level
        self.highest_detail_level = highest_detail_level
        self.disambiguate = disambiguate
        self.disambiguation_threshold = disambiguation_threshold
        self.max_disambiguation_group_size = max_disambiguation_group_size
        self.verbose = verbose

    def _require_prepared(self):
        if not getattr(self, "_prepared", False):
            raise NotFittedError("Call prepare or fit before accessing topic results")

    def prepare(
        self,
        objects,
        embedding_vectors=None,
        clusterable_vectors=None,
        *,
        object_vectors=None,
    ):
        """Build inspectable topic evidence without calling the naming provider."""
        self._prepared = False
        if embedding_vectors is not None and object_vectors is not None:
            raise ValueError("Supply embedding_vectors or object_vectors, not both")
        if embedding_vectors is None:
            embedding_vectors = object_vectors
        self.__dict__.pop("topic_model_", None)
        self.objects_ = tuple(objects)
        n_objects = len(self.objects_)
        self.embedding_vectors_ = _matrix(
            embedding_vectors, n_objects, "embedding_vectors"
        )
        clustering_graph = None
        if sparse.issparse(clusterable_vectors):
            graph = clusterable_vectors.tocsr(copy=False)
            if (
                graph.shape != (n_objects, n_objects)
                or not np.isfinite(graph.data).all()
                or np.any(graph.data < 0)
            ):
                raise ValueError(
                    "A clustering distance graph must be square, finite and nonnegative"
                )
            clustering_graph = sparse.csr_matrix(
                (graph.data.view(), graph.indices.view(), graph.indptr.view()),
                shape=graph.shape,
            )
            for array in (
                clustering_graph.data,
                clustering_graph.indices,
                clustering_graph.indptr,
            ):
                array.flags.writeable = False
            self.clusterable_vectors_ = clustering_graph
        else:
            self.clusterable_vectors_ = (
                self.embedding_vectors_
                if clusterable_vectors is None
                else _matrix(clusterable_vectors, n_objects, "clusterable_vectors")
            )
        self.stage_timings_ = {}
        started = perf_counter()
        self.clusterer.fit(self.clusterable_vectors_)
        self.cluster_layers_ = tuple(self.clusterer.cluster_layers_)
        if any(len(layer.labels) != n_objects for layer in self.cluster_layers_):
            raise ValueError("Clustering output does not match the number of objects")
        validate_cluster_tree(self.clusterer.cluster_tree_, self.cluster_layers_)
        self.cluster_tree_ = {
            key: list(children)
            for key, children in self.clusterer.cluster_tree_.items()
        }
        # Retain this fit's hierarchy when a caller later refits the estimator.
        self._fitted_clusterer = copy(self.clusterer)
        self._fitted_clusterer.cluster_layers_ = self.cluster_layers_
        self._fitted_clusterer.cluster_tree_ = self.cluster_tree_
        self.stage_timings_["clustering"] = perf_counter() - started
        topics = {
            (layer.layer_index, cluster.label): Topic(
                layer.layer_index, cluster.label, cluster.members
            )
            for layer in self.cluster_layers_
            for cluster in layer
        }
        self.topic_model_ = TopicModel.from_topics(
            topics,
            self.cluster_layers_,
            self.cluster_tree_,
            self.embedding_vectors_,
            None if clustering_graph is not None else self.clusterable_vectors_,
        )
        self.topic_model_.clustering_graph = clustering_graph
        self.topic_model_.metadata.update(
            object_description=self.object_description,
            corpus_description=self.corpus_description,
            disambiguation=self.disambiguate,
            semantic_quality="NOT VALIDATED",
        )
        started = perf_counter()
        for extractor in self.feature_extractors:
            if extractor.layer_dependent:
                if extractor.can_fit_from_objects():
                    extractor.fit(self.objects_, self._fitted_clusterer)
                else:
                    extractor.predict()
                continue
            options = {"embedding_vectors": self.embedding_vectors_}
            if isinstance(extractor, TextKeyphraseExtractor):
                options = {"embedder": self.embedding_model}
            options.update(self.feature_options.get(extractor.feature_key, {}))
            features = (
                extractor.fit_predict(self.objects_, self._fitted_clusterer, **options)
                if extractor.can_fit_from_objects()
                else extractor.predict()
            )
            if len(features) != len(self.cluster_layers_):
                raise ValueError("Extractor output must have one entry per layer")
            for layer, values in zip(self.cluster_layers_, features):
                self._assign_features(extractor.feature_key, layer, values)
                if hasattr(extractor, "indices_"):
                    for cluster, indices in zip(
                        layer, extractor.indices_[layer.layer_index]
                    ):
                        topics[(layer.layer_index, cluster.label)].features[
                            "exemplar_indices"
                        ] = list(indices)
        self.stage_timings_["extraction"] = perf_counter() - started
        self.request_counts_ = {"naming": 0, "disambiguation": 0, "name_embeddings": 0}
        self._named_layers = set()
        for layer in self.cluster_layers_:
            self._make_prompts(layer)
        self._prepared = True
        return self

    def _assign_features(self, key, layer, values):
        if len(values) != len(layer):
            raise ValueError("Extractor output must align with sorted cluster IDs")
        for cluster, value in zip(layer, values):
            self.topic_model_.topics[(layer.layer_index, cluster.label)].features[
                key
            ] = deepcopy(value)

    def _name_kind(self, layer_index):
        count = len(self.cluster_layers_)
        detail = (
            self.lowest_detail_level
            if count < 2
            else self.lowest_detail_level
            + layer_index
            / (count - 1)
            * (self.highest_detail_level - self.lowest_detail_level)
        )
        return (
            "specific"
            if detail < 0.34
            else "general" if detail > 0.66 else "descriptive"
        )

    def _make_prompts(self, layer):
        for cluster in layer:
            topic = self.topic_model_.topics[(layer.layer_index, cluster.label)]
            topic.prompt = self.prompt_template.cluster_prompt(
                topic.features, self._name_kind(layer.layer_index)
            )

    def _prepare_layer(self, layer):
        if layer.layer_index:
            for extractor in self.feature_extractors:
                if extractor.layer_dependent:
                    values = extractor.extract_layer(
                        layer.layer_index, self.topics_, self._fitted_clusterer
                    )
                    self._assign_features(extractor.feature_key, layer, values)
        self._make_prompts(layer)
        return [self.topics_[(layer.layer_index, cluster.label)] for cluster in layer]

    @staticmethod
    def _store_name(topic, value):
        if isinstance(value, tuple):
            if len(value) != 3 or not all(isinstance(item, str) for item in value):
                raise ValueError(
                    "Summary parsers must return (name, summary, explanation)"
                )
            name, summary, explanation = value
        elif isinstance(value, str):
            name, summary, explanation = value, None, None
        else:
            raise ValueError("Name parsers must return a string or summary tuple")
        if not name.strip():
            raise ValueError("A naming provider returned an empty topic name")
        topic.name, topic.summary, topic.explanation = name, summary, explanation

    def _similar_groups(self, topics):
        n = len(topics)
        if n < 2:
            return []
        names = [topic.name for topic in topics]
        normalized = [name.strip().casefold() for name in names]
        duplicates = np.array([[a == b for b in normalized] for a in normalized])
        distances = np.ones((n, n), dtype=float)
        if self.embedding_model is not None:
            self.request_counts_["name_embeddings"] += 1
            vectors = _matrix(
                self.embedding_model.encode(names), n, "topic name embeddings"
            )
            scaled = _normalize_rows(vectors)
            distances = np.clip(1.0 - scaled @ scaled.T, 0, 2)
        distances[duplicates] = 0.0
        np.fill_diagonal(distances, 0.0)
        # Complete linkage prevents a chain of neighbours from grouping names
        # whose endpoints are unrelated. Bound each provider request as well.
        labels = fcluster(
            linkage(squareform(distances, checks=False), method="complete"),
            t=1.0 - self.disambiguation_threshold,
            criterion="distance",
        )
        groups = []
        for label in dict.fromkeys(labels):
            members = np.flatnonzero(labels == label)
            if len(members) < 2:
                continue
            count = (
                len(members) + self.max_disambiguation_group_size - 1
            ) // self.max_disambiguation_group_size
            groups.extend(
                part.tolist()
                for part in np.array_split(members, count)
                if len(part) > 1
            )
        return groups

    def _disambiguation_prompt(self, topics, group, layer):
        selected = [topics[index] for index in group]
        names = [topic.name for topic in selected]
        prompt = self.prompt_template.disambiguate_prompt(
            names,
            [topic.features for topic in selected],
            self._name_kind(layer.layer_index),
        )
        return selected, names, prompt

    @staticmethod
    def _store_disambiguation(topics, names):
        if len(names) != len(topics) or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError(
                "Disambiguation results must align with every requested topic"
            )
        for topic, name in zip(topics, names):
            topic.name = name

    def name_topics(self):
        """Name prepared layers synchronously; errors leave inspectable partial state."""
        self._require_prepared()
        started = perf_counter()
        for layer in self.cluster_layers_:
            if layer.layer_index in self._named_layers:
                continue
            topics = self._prepare_layer(layer)
            for topic in topics:
                if topic.name is None:
                    self.request_counts_["naming"] += 1
                    self._store_name(
                        topic,
                        self.llm_wrapper.generate_topic_name(
                            topic.prompt,
                            response_parser=self.prompt_template.extract_name,
                        ),
                    )
            if self.disambiguate:
                for group in self._similar_groups(topics):
                    selected, names, prompt = self._disambiguation_prompt(
                        topics, group, layer
                    )
                    self.request_counts_["disambiguation"] += 1
                    renamed = self.llm_wrapper.generate_topic_cluster_names(
                        prompt,
                        names,
                        response_parser=self.prompt_template.extract_disambiguated_names,
                    )
                    self._store_disambiguation(selected, renamed)
            self._named_layers.add(layer.layer_index)
        self.stage_timings_["naming"] = perf_counter() - started
        return self

    async def name_topics_async(self):
        """Name each layer through the asynchronous wrapper's bounded batch API."""
        self._require_prepared()
        started = perf_counter()
        for layer in self.cluster_layers_:
            if layer.layer_index in self._named_layers:
                continue
            topics = self._prepare_layer(layer)
            pending = [topic for topic in topics if topic.name is None]
            if pending:
                self.request_counts_["naming"] += len(pending)
                names = await self.llm_wrapper.generate_topic_names(
                    [topic.prompt for topic in pending],
                    response_parser=self.prompt_template.extract_name,
                    return_results=True,
                )
                if len(names) != len(pending):
                    raise ValueError("Naming results do not align with prompts")
                failures = []
                for topic, result in zip(pending, names):
                    if result.error is None:
                        self._store_name(topic, result.value)
                    else:
                        failures.append(result.error)
                if failures:
                    raise failures[0]
            if self.disambiguate:
                for group in self._similar_groups(topics):
                    selected, names, prompt = self._disambiguation_prompt(
                        topics, group, layer
                    )
                    self.request_counts_["disambiguation"] += 1
                    renamed = await self.llm_wrapper.generate_topic_cluster_names(
                        [prompt],
                        [names],
                        response_parser=self.prompt_template.extract_disambiguated_names,
                    )
                    if len(renamed) != 1:
                        raise ValueError("Disambiguation batch results do not align")
                    self._store_disambiguation(selected, renamed[0])
            self._named_layers.add(layer.layer_index)
        self.stage_timings_["naming"] = perf_counter() - started
        return self

    def fit(
        self,
        objects,
        embedding_vectors=None,
        clusterable_vectors=None,
        *,
        object_vectors=None,
    ):
        return self.prepare(
            objects,
            embedding_vectors,
            clusterable_vectors,
            object_vectors=object_vectors,
        ).name_topics()

    async def fit_async(
        self,
        objects,
        embedding_vectors=None,
        clusterable_vectors=None,
        *,
        object_vectors=None,
    ):
        self.prepare(
            objects,
            embedding_vectors,
            clusterable_vectors,
            object_vectors=object_vectors,
        )
        return await self.name_topics_async()

    def fit_predict(
        self,
        objects,
        embedding_vectors=None,
        clusterable_vectors=None,
        *,
        object_vectors=None,
    ):
        self.fit(
            objects,
            embedding_vectors,
            clusterable_vectors,
            object_vectors=object_vectors,
        )
        return self.topic_name_vectors_

    @property
    def topics_(self):
        self._require_prepared()
        return self.topic_model_.topics

    @property
    def topic_names_(self):
        self._require_prepared()
        return self.topic_model_.topic_names

    @property
    def topic_sizes_(self):
        self._require_prepared()
        return self.topic_model_.topic_sizes

    @property
    def topic_name_vectors_(self):
        self._require_prepared()
        return self.topic_model_.topic_name_vectors

    @property
    def topic_tree_(self):
        self._require_prepared()
        return self.topic_model_.topic_tree()
