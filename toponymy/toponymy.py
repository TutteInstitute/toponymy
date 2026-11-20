import copy
from collections import Counter
from types import SimpleNamespace
from typing import List, Any, Optional, Type, Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm


from toponymy.annotation import Annotator, AnnotationStore, Executor
from toponymy.clustering import ToponymyClusterer, Clusterer
from toponymy.embedding_wrappers import TextEmbedderProtocol
from toponymy.exemplar_texts import exemplar_annotator
from toponymy.keyphrases import InformationWeightedKeyphraseAnnotator, KeyphraseBuilder
from toponymy.subtopics import (
    CentralSubtopicAnnotator,
    SubmodularSubtopicAnnotator,
    TopicNameEmbeddingAnnotator,
)
from toponymy.cluster_layer import (
    ClusterLayer,
    ClusterLayerSummaryText,
    ClusterLayerText,
)
from toponymy.llm_wrappers import LLMWrapper
from toponymy.templates import PROMPT_TEMPLATES, SUMMARY_PROMPT_TEMPLATES
from toponymy.topic_naming import (
    TopicDisambiguationAnnotator,
    TopicNameAnnotator,
    TopicNamerTemplate,
)
from toponymy.topic_tree import TopicTree
from toponymy._annotation_bridge import (
    copy_exemplars_to_layers,
    has_stock_exemplars,
    copy_keyphrases_to_layers,
    copy_layers_to_store,
    has_stock_keyphrases,
    copy_topics_to_layers,
    copy_subtopics_to_layers,
)
from toponymy._utils import handle_verbose_params


def _exemplar_annotator_of(annotators):
    for annotator in annotators:
        if {"exemplars", "exemplar_original_indices"} <= set(annotator.outputs):
            return annotator
    raise ValueError(
        "annotators must include one that outputs 'exemplars' and 'exemplar_original_indices'"
    )


def _with_null_topic(annotator, null_topic):
    # TODO: drop once the store supports global annotations; the corpus mean is shared state.
    is_exemplar = getattr(annotator, "kind", None) == "exemplar"
    if (
        is_exemplar
        and hasattr(annotator, "null_topic")
        and annotator.null_topic is None
    ):
        annotator = copy.copy(annotator)
        annotator.null_topic = null_topic
    return annotator


_SUBTOPIC_METHODS = ("central", "facility_location", "saturated_coverage")


def _run_annotators(executor, runs, **wiring):
    """Runs each (annotator, nodes) pair, raising on the first that has failed nodes."""
    for annotator, nodes in runs:
        failures = executor.run(annotator, nodes=nodes, **wiring)
        if failures:
            node, reason = next(iter(failures.items()))
            raise RuntimeError(
                f"{type(annotator).__name__} failed for {len(failures)} node(s); "
                f"first, {node}: {reason}"
            )


class Toponymy:
    """
    A class for generating topic names for vector based topic modeling.

    Parameters:
    -----------
    llm_wrapper: class
        A llm_wrapper class.  These should be objects that inherit from the LlmWrapper base classs.
    text_embedding_model: TextEmbedderProtocol
        a function with an encode used to vectorize the objects that we are topic modeling.
    clusterer: Clusterer
        The clusterer to use for clustering the objects. This should be a clusterer that inherits from the Clusterer base class.
    layer_class: Type[Any]
        The class to used for creating layers from our objects. Default is ClusterLayerText.
    keyphrase_builder: KeyphraseBuilder
        The keyphrase builder to use for building keyphrases from the objects.
    object_description: str
        A description of the objects being topic modeled.
    corpus_description: str
        A description of the collection of objects being topic modeled.
    lowest_detail_level: float
        The lowest detail level to use for the topic names. This should be a value between 0 (finest grained detail) and 1 (very high level).
    highest_detail_level: float
        The highest detail level to use for the topic names. This should be a value between 0 (finest grained detail) and 1 (very high level).
    exemplar_delimiters: List[str]
        A list of strings that represent the delimiters for the exemplar texts. Default is ["    *\"", "\"\n"].
    verbose: bool
        Whether to show progress bars and verbose output. If True, shows all output. If False, suppresses all output.
    show_progress_bars: bool, deprecated
        Deprecated. Use verbose instead.
    annotators: List[Annotator], optional
        The annotators to run, in order, over the whole cluster tree. They must include one that outputs
        ``exemplars`` and ``exemplar_original_indices``. An annotator with ``kind = "exemplar"`` and an unset
        ``null_topic`` gets the mean of the embedding vectors. If None, the exemplars are made with the exemplar_method given to fit, using the
        settings of the cluster layers.

    Attributes:
    -----------
    llm_wrapper: class
        A llm_wrapper class.  These should be objects that inherit from the LlmWrapper base classs.
    embedding_model: callable
        a function with an encode used to vectorize the objects that we are topic modeling.
    clusterer: Clusterer
        The clusterer to use for clustering the objects. This should be a clusterer that inherits from the Clusterer base class.
    layer_class: Type[Any]
        The class to used for creating layers from our objects.
    keyphrase_builder: KeyphraseBuilder
        The keyphrase builder to use for building keyphrases from the objects.
    object_description: str
        A description of the objects being topic modeled.
    corpus_description: str
        A description of the collection of objects being topic modeled.
    lowest_detail_level: float
        The lowest detail level to use for the topic names. This should be a value between 0 (finest grained detail) and 1 (very high level).
    highest_detail_level: float
        The highest detail level to use for the topic names. This should be a value between 0 (finest grained detail) and 1 (very high level).
    exemplar_delimiters: List[str]
        A list of strings that represent the delimiters for the exemplar texts.
    verbose: bool
        Whether to show progress bars and verbose output.
    clusterable_vectors_: np.array
        A numpy array of shape=(number_of_objects, clustering_dimension) used for clustering.
    embedding_vectors_: np.array
        A numpy array of shape=(number_of_objects, embedding_dimension) used for vectorizing.
    cluster_layers_: List[ClusterLayer]
        A list of ClusterLayer objects that represent the layers of the topic model.
    cluster_tree_: dict
        A dictionary that represents the tree of clusters.
    store_: AnnotationStore
        The annotations computed for the nodes of the cluster tree.
    object_x_keyphrase_matrix_: np.array
        A numpy array of shape=(number_of_objects, number_of_keyphrases) that represents the objects and their keyphrases.
    keyphrase_list_: List[str]
        A list of keyphrases.
    keyphrase_vectors_: np.array
        A numpy array of shape=(number_of_keyphrases, embedding_dimension) that represents the keyphrase vectors.
    topic_names_: List[List[str]]
        A list of lists of strings that represent the topic names at each layer of the topic model.
    topic_name_vectors_: List[np.array]
        A list of numpy arrays of shape=(number_of_topics, embedding_dimension) that represent the topic names of each object
        at each layer of the topic model.
    topic_sizes_: List[List[int]]
        A list of lists where topic_sizes_[i][j] is the number of objects in cluster j at layer i.

    """

    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        text_embedding_model: TextEmbedderProtocol,
        clusterer: Clusterer = ToponymyClusterer(),
        layer_class: Type[ClusterLayer] = ClusterLayerText,
        prompt_template: Dict[str, Any] = PROMPT_TEMPLATES,
        keyphrase_builder: KeyphraseBuilder = KeyphraseBuilder(),
        object_description: str = "objects",
        corpus_description: str = "collection of objects",
        lowest_detail_level: float = 0.0,
        highest_detail_level: float = 1.0,
        exemplar_delimiters: List[str] = ['    * "', '"\n'],
        verbose: Optional[bool] = None,
        show_progress_bars: Optional[bool] = None,
        annotators: Optional[List[Annotator]] = None,
    ):
        self.llm_wrapper = llm_wrapper
        self.embedding_model = text_embedding_model
        self.layer_class = layer_class
        self.clusterer = clusterer
        self.keyphrase_builder = keyphrase_builder
        self.object_description = object_description
        self.corpus_description = corpus_description
        self.lowest_detail_level = lowest_detail_level
        self.highest_detail_level = highest_detail_level
        self.exemplar_delimiters = exemplar_delimiters
        self.annotators = annotators
        self.prompt_template = prompt_template

        # Handle verbose parameters
        self.show_progress_bars, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bars=show_progress_bars, default_verbose=True
        )

        # If the default prompt template is used, but the layer class is ClusterLayerSummaryText, it is
        # reasonable to switch to the summary prompt templates, if not, the user may be passing their own.
        base = getattr(layer_class, "func", layer_class)
        if (
            isinstance(base, type)
            and issubclass(base, ClusterLayerSummaryText)
            and prompt_template == PROMPT_TEMPLATES
        ):
            self.prompt_template = SUMMARY_PROMPT_TEMPLATES

    def __sklearn_tags__(self):
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.requires_fit = True
        tags.non_deterministic = True
        tags.input_tags.one_d_array = False
        tags.input_tags.two_d_array = False
        tags.input_tags.string = True

        return tags

    def _sync_layer_runtime_config(self) -> None:
        """
        Refresh wrapper-sensitive / run-sensitive settings on all cluster layers.

        This ensures that reused pre-fit cluster layers respect the current
        llm_wrapper and prompt configuration for this run.
        """
        for layer in self.cluster_layers_:
            layer.exemplar_delimiters = self.exemplar_delimiters
            layer.show_progress_bar = self.show_progress_bars
            layer.verbose = self.verbose

    def _compute_topic_sizes(self) -> List[List[int]]:
        """
        Compute the size of each cluster in each layer.

        Returns
        -------
        List[List[int]]
            A list of lists where topic_sizes[i][j] is the size of cluster j in layer i.
        """

        def cluster_size(cluster_label_array):
            if cluster_label_array.min() < 0:
                return np.bincount(cluster_label_array - cluster_label_array.min())[
                    -cluster_label_array.min() :
                ].tolist()
            else:
                return np.bincount(cluster_label_array).tolist()

        return [cluster_size(layer.cluster_labels) for layer in self.cluster_layers_]

    def _object_to_text_function(self):
        """The function that makes exemplar text from objects; None for text objects."""
        if self.annotators is None:
            source = self.cluster_layers_[0]
        else:
            source = _exemplar_annotator_of(self.annotators)
        return getattr(source, "object_to_text_function", None)

    def _annotator_runs(self, embedding_vectors):
        """The (annotator, nodes) pairs for the given annotators, with None for all nodes."""
        if not all(has_stock_exemplars(layer) for layer in self.cluster_layers_):
            raise ValueError(
                "annotators cannot be used with layer classes that override make_exemplar_texts"
            )
        null_topic = np.mean(embedding_vectors, axis=0)
        return [(_with_null_topic(a, null_topic), None) for a in self.annotators]

    def _exemplar_runs(
        self, objects, embedding_vectors, exemplar_method, non_text_objects
    ):
        """The (annotator, nodes) pairs that make each layer's exemplars from the layer settings."""
        null_topic = np.mean(embedding_vectors, axis=0)
        exemplar_method = exemplar_method or "central"
        runs = []
        for i, layer in enumerate(self.cluster_layers_):
            # TODO: non-text layer 0 ignores exemplar_method because its exemplars also
            # define the keyphrase corpus; give the corpus its own annotation.
            method = (
                "facility_location" if i == 0 and non_text_objects else exemplar_method
            )
            if not has_stock_exemplars(layer):
                layer.make_exemplar_texts(objects, embedding_vectors, method=method)
                continue
            annotator = exemplar_annotator(
                method,
                layer.n_exemplars,
                layer.object_to_text_function,
                layer.exemplars_diversify_alpha,
                null_topic,
            )
            runs.append((annotator, self.store_.tree.layer(i)))
        return runs

    def _keyphrase_runs(self, keyphrase_method):
        """The (annotator, nodes) pairs that make each layer's information weighted keyphrases."""
        if keyphrase_method != "information_weighted":
            return []
        # The vectors fit embedded itself, so the annotator does not embed them again
        fitted_keyphrases = SimpleNamespace(
            object_x_keyphrase_matrix_=self.object_x_keyphrase_matrix_,
            keyphrase_list_=self.keyphrase_list_,
            keyphrase_vectors_=self.keyphrase_vectors_,
        )
        base = InformationWeightedKeyphraseAnnotator(
            fitted_keyphrases, self.embedding_model
        )
        runs = []
        for i, layer in enumerate(self.cluster_layers_):
            if not has_stock_keyphrases(layer):
                continue
            # Copies share the keyphrase vectors across layers, as the legacy layers do
            annotator = copy.copy(base)
            annotator.n_keyphrases = layer.n_keyphrases
            annotator.max_alpha = layer.keyphrase_diversify_alpha
            runs.append((annotator, self.store_.tree.layer(i)))
        return runs

    def _embed_topic_names(self, executor, nodes):
        """Embeds the names of the given nodes that have a name but no embedding."""
        names = self.store_.topic_names
        embedded = self.store_.get("topic_name_embeddings", ())
        pending = [node for node in nodes if node in names and node not in embedded]
        if pending:
            _run_annotators(
                executor, [(TopicNameEmbeddingAnnotator(self.embedding_model), pending)]
            )

    def _disambiguate_topic_names(self, executor, disambiguator, nodes):
        """Renames the clashing topics of one layer and re-embeds the renamed ones."""
        names = self.store_.topic_names
        self._embed_topic_names(executor, nodes)
        # As the legacy layers do, run again if a name is still shared by more than two topics
        for _ in range(2):
            before = names.computed(nodes)
            if not before:
                return
            executor.run(disambiguator, nodes=nodes)
            for node, name in before.items():
                if names[node] != name:
                    self.store_.topic_name_embeddings.reset(node)
            self._embed_topic_names(executor, nodes)
            if max(Counter(names.computed(nodes).values()).values()) <= 2:
                return

    def _annotate_subtopics(self, executor, subtopic_method):
        """Stores the subtopics of the nodes above layer 0, chosen from the layer 0 topic names."""
        tree, layers = self.store_.tree, self.cluster_layers_
        if len(layers) < 2 or "topic_name_embeddings" not in self.store_:
            return
        embeddings = self.store_.topic_name_embeddings.computed(tree.layer(0))
        if not embeddings:
            return
        null_topic = np.mean(list(embeddings.values()), axis=0)
        runs = []
        for i, layer in enumerate(layers[1:], 1):
            if subtopic_method == "central":
                annotator = CentralSubtopicAnnotator(
                    n_subtopics=layer.n_subtopics,
                    diversify_alpha=layer.subtopic_diversify_alpha,
                    null_topic=null_topic,
                )
            else:
                annotator = SubmodularSubtopicAnnotator(
                    n_subtopics=layer.n_subtopics,
                    null_topic=null_topic,
                    submodular_function=subtopic_method,
                )
            runs.append((annotator, tree.layer(i)))
        _run_annotators(executor, runs)
        copy_subtopics_to_layers(self.store_, layers)

    def fit(
        self,
        objects: List[Any],
        embedding_vectors: np.ndarray,
        clusterable_vectors: np.ndarray,
        exemplar_method: Optional[str] = None,
        keyphrase_method: str = "information_weighted",
        subtopic_method: str = "central",
        **clusterer_kwd_args,
    ):
        """
        Vectorizes using the classes embedding_model and constructs a low dimension data map with UMAP if object_vectors and object_map aren't spec.

        Parameters:
        -----------
        objects: Object
            The objects over which to perform topic modeling.  These are often text documents or images.
        embedding_vectors: np.array
            An numpy array of shape=(number_of_objects, embedding_dimension) created with the same embedding_model specified in the constructor.
        clusterable_vectors: np.array
            A numpy array of shape=(number_of_objects, clustering_dimension).  It is recommended that the clustering_dimension should be low enough
            for density based clustering to be efficient (2-25).
        exemplar_method: str, Optional
            The method to use for generating exemplars. Default is "central". Other options are "facility_location", "saturated_coverage", and "random".
            Cannot be combined with the annotators given to the constructor.
        keyphrase_method: str, Optional
            The method to use for generating keyphrases. Default is "information_weighted". Other options are "saturated_coverage", "facility_location", "graph_cut", "central" and "bm25".
        subtopic_method: str, Optional
            The method to use for choosing the subtopics of the topics above layer 0. Default is "central". Other options are "facility_location" and "saturated_coverage".

        Returns:
        --------
        self: object
            Returns the instance of the class.
        """
        self.clusterable_vectors_ = clusterable_vectors
        self.embedding_vectors_ = embedding_vectors

        if self.annotators is not None and exemplar_method is not None:
            raise ValueError("exemplar_method cannot be combined with annotators")
        if subtopic_method not in _SUBTOPIC_METHODS:
            raise ValueError(
                f"Unknown subtopic_method {subtopic_method!r}; use one of {_SUBTOPIC_METHODS}"
            )

        # Build our layers and cluster tree
        if hasattr(self.clusterer, "cluster_layers_") and hasattr(
            self.clusterer, "cluster_tree_"
        ):
            # If the clusterer has already been fit, we can skip this step
            prefit_layer = self.clusterer.cluster_layers_[0]
            requested_class = getattr(self.layer_class, "func", self.layer_class)
            if isinstance(requested_class, type) and not isinstance(
                prefit_layer, requested_class
            ):
                raise ValueError(
                    f"Prefit {type(prefit_layer).__name__} layers are incompatible with "
                    f"layer_class={requested_class.__name__}. Pass a matching "
                    "layer_class or a clusterer that has not been fit."
                )
            self.cluster_layers_ = self.clusterer.cluster_layers_
            self.cluster_tree_ = self.clusterer.cluster_tree_
            if self.prompt_template not in (PROMPT_TEMPLATES, SUMMARY_PROMPT_TEMPLATES):
                for layer in self.cluster_layers_:
                    layer.prompt_template = self.prompt_template
        else:
            clusterer_kwd_args = {
                "verbose": self.verbose,
                "show_progress_bar": self.show_progress_bars,
                "exemplar_delimiters": self.exemplar_delimiters,
                "prompt_template": self.prompt_template,
                **clusterer_kwd_args,
            }
            self.cluster_layers_, self.cluster_tree_ = self.clusterer.fit_predict(
                clusterable_vectors,
                embedding_vectors,
                self.layer_class,
                **clusterer_kwd_args,
            )

        self._sync_layer_runtime_config()

        self.store_ = AnnotationStore.from_clusterer(self, objects, embedding_vectors)
        executor = Executor(self.store_)

        # Initialize other data structures
        self.topic_names_: List[List[str]] = [[]] * len(self.cluster_layers_)
        self.topic_name_vectors_: List[np.ndarray] = [np.array([])] * len(
            self.cluster_layers_
        )
        if isinstance(self.cluster_layers_[0], ClusterLayerSummaryText):
            self.topic_summaries_: List[List[str]] = [[]] * len(self.cluster_layers_)
            self.topic_explanations_: List[List[str]] = [[]] * len(self.cluster_layers_)
            _summarize_topics = True
        else:
            _summarize_topics = False

        detail_levels = np.linspace(
            self.lowest_detail_level,
            self.highest_detail_level,
            len(self.cluster_layers_),
        )

        non_text_objects = self._object_to_text_function() is not None
        if self.annotators is not None:
            runs = self._annotator_runs(embedding_vectors)
        else:
            runs = self._exemplar_runs(
                objects, embedding_vectors, exemplar_method, non_text_objects
            )
        _run_annotators(executor, runs)
        copy_exemplars_to_layers(self.store_, self.cluster_layers_)
        copy_layers_to_store(
            self.store_, self.cluster_layers_, "exemplars", "exemplars"
        )
        copy_layers_to_store(
            self.store_,
            self.cluster_layers_,
            "exemplar_original_indices",
            "exemplar_indices",
        )

        # Build keyphrase matrix
        if non_text_objects:
            # Non-text objects: use exemplars to build keyphrase matrix
            exemplars = self.cluster_layers_[0].exemplars
            exemplar_indices = self.cluster_layers_[0].exemplar_indices

            # Create aligned text list
            aligned_texts = [""] * len(objects)  # Empty strings for non-exemplars
            for cluster_idx, cluster_exemplars in enumerate(exemplars):
                for exemplar_idx, exemplar_text in zip(
                    exemplar_indices[cluster_idx], cluster_exemplars
                ):
                    aligned_texts[exemplar_idx] = exemplar_text

            # Build keyphrase matrix from aligned texts
            (
                self.object_x_keyphrase_matrix_,
                self.keyphrase_list_,
                self.keyphrase_vectors_,
            ) = self.keyphrase_builder.fit_transform(aligned_texts)
        else:
            # Text objects: build keyphrase matrix directly from objects
            (
                self.object_x_keyphrase_matrix_,
                self.keyphrase_list_,
                self.keyphrase_vectors_,
            ) = self.keyphrase_builder.fit_transform(objects)

        if self.keyphrase_vectors_ is None:
            # If the keyphrase vectors are None, we need to generate them
            self.keyphrase_vectors_ = self.embedding_model.encode(
                self.keyphrase_list_,
                show_progress_bar=self.show_progress_bars,
            )

        keyphrase_runs = self._keyphrase_runs(keyphrase_method)
        _run_annotators(executor, keyphrase_runs)
        if keyphrase_runs:
            copy_keyphrases_to_layers(self.store_, self.cluster_layers_)
        for layer in self.cluster_layers_:
            if not keyphrase_runs or not has_stock_keyphrases(layer):
                layer.make_keyphrases(
                    self.keyphrase_list_,
                    self.object_x_keyphrase_matrix_,
                    self.keyphrase_vectors_,
                    self.embedding_model,
                    method=keyphrase_method,
                )
        copy_layers_to_store(
            self.store_, self.cluster_layers_, "keyphrases", "keyphrases"
        )

        template = TopicNamerTemplate(
            object_description=self.object_description,
            corpus_description=self.corpus_description,
            tree=self.store_.tree,
            exemplar_start_delimiter=self.exemplar_delimiters[0],
            exemplar_end_delimiter=self.exemplar_delimiters[1],
            prompt_template=self.prompt_template,
            lowest_detail_level=self.lowest_detail_level,
            highest_detail_level=self.highest_detail_level,
        )
        topic_annotator = TopicNameAnnotator(
            llm_namer=self.llm_wrapper,
            topic_name_template=template,
        )
        disambiguator = TopicDisambiguationAnnotator(self.llm_wrapper, template)

        # Each layer is named from the final names of the layers below it
        tree = self.store_.tree
        for i in range(len(self.cluster_layers_)):
            nodes = tree.layer(i)
            executor.run(
                topic_annotator,
                nodes=nodes,
                output_names={"draft_topics": "topic_names"},
            )
            self._disambiguate_topic_names(executor, disambiguator, nodes)
            if i == 0:
                self._annotate_subtopics(executor, subtopic_method)

        copy_topics_to_layers(self.store_, self.cluster_layers_)

        for i, layer in enumerate(self.cluster_layers_):
            self.topic_names_[i] = layer.topic_names
            self.topic_name_vectors_[i] = layer.make_topic_name_vector()

            # Compute and cache topic sizes for efficient access and serialization
            self.topic_sizes_ = self._compute_topic_sizes()

        return self

    def fit_predict(
        self,
        objects: List[Any],
        object_vectors: np.ndarray,
        clusterable_vectors: np.ndarray,
        exemplar_method: Optional[str] = None,
        keyphrase_method: str = "information_weighted",
        subtopic_method: str = "facility_location",
    ) -> List[np.ndarray]:
        """
        Fit the model with objects and return the topic names.

        Parameters:
        -----------
        objects: List[Any]
            A list of objects to perform topic modeling over.
        object_vectors: np.array
            An array of shape=(number_of_objects, embedding_dimension) created with the same embedding_model specified in the constructor.
        object_map: np.array
            An array of shape=(number_of_objects, clustering_dimension).  It is recommended that the clustering_dimension should be low enough
            for density based clustering to be efficient (2-25).

        Returns:
        --------
        topic_name_vectors: List[np.array]
            A list of numpy arrays of shape=(number_of_topics, embedding_dimension) that represent the topic names of each object
            at each layer of the topic model.
        """
        self.fit(
            objects,
            object_vectors,
            clusterable_vectors,
            exemplar_method=exemplar_method,
            keyphrase_method=keyphrase_method,
            subtopic_method=subtopic_method,
        )
        return self.topic_name_vectors_

    @property
    def topic_tree_(self) -> TopicTree:
        """
        Returns the topic tree.

        Returns:
        --------
        TopicTree
            A representation of the topic tree (either html or string).
        """
        check_is_fitted(
            self,
            ["cluster_tree_", "topic_names_", "topic_name_vectors_", "topic_sizes_"],
        )

        return TopicTree(
            self.cluster_tree_,
            self.topic_names_,
            self.topic_sizes_,
            self.embedding_vectors_.shape[0],
        )
