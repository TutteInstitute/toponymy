from toponymy.clustering import ToponymyClusterer, Clusterer
from toponymy._annotation_bridge import copy_exemplars_to_layers, has_stock_exemplars
from toponymy.annotation import Annotator, AnnotationStore, Executor
from toponymy.exemplar_texts import exemplar_annotator
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.cluster_layer import (
    ClusterLayer,
    ClusterLayerSummaryText,
    ClusterLayerText,
)
from toponymy.topic_tree import TopicTree
from toponymy.llm_wrappers import LLMWrapper
from toponymy.embedding_wrappers import TextEmbedderProtocol
from toponymy.templates import PROMPT_TEMPLATES, SUMMARY_PROMPT_TEMPLATES
from toponymy._utils import handle_verbose_params

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import copy
import numpy as np

from tqdm.auto import tqdm

from typing import List, Any, Optional, Type, Dict, Tuple


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

    def fit(
        self,
        objects: List[Any],
        embedding_vectors: np.ndarray,
        clusterable_vectors: np.ndarray,
        exemplar_method: Optional[str] = None,
        keyphrase_method: str = "information_weighted",
        subtopic_method: str = "central",
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
            The method to use for generating subtopics. Default is "facility_location". Other options are "information_weighted".

        Returns:
        --------
        self: object
            Returns the instance of the class.
        """
        self.clusterable_vectors_ = clusterable_vectors
        self.embedding_vectors_ = embedding_vectors

        if self.annotators is not None and exemplar_method is not None:
            raise ValueError("exemplar_method cannot be combined with annotators")

        # Build our layers and cluster tree
        if hasattr(self.clusterer, "cluster_layers_") and hasattr(
            self.clusterer, "cluster_tree_"
        ):
            # If the clusterer has already been fit, we can skip this step
            self.cluster_layers_ = self.clusterer.cluster_layers_
            self.cluster_tree_ = self.clusterer.cluster_tree_
        else:
            self.cluster_layers_, self.cluster_tree_ = self.clusterer.fit_predict(
                clusterable_vectors,
                embedding_vectors,
                self.layer_class,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bars,
                exemplar_delimiters=self.exemplar_delimiters,
                prompt_template=self.prompt_template,
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
        for annotator, nodes in runs:
            failures = executor.run(annotator, nodes=nodes)
            if failures:
                node, reason = next(iter(failures.items()))
                raise RuntimeError(
                    f"{type(annotator).__name__} failed for {len(failures)} node(s); "
                    f"first, {node}: {reason}"
                )
        copy_exemplars_to_layers(self.store_, self.cluster_layers_)

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

        # Iterate through the layers and build the topic names
        for i, layer in tqdm(
            enumerate(self.cluster_layers_),
            desc=f"Building topic names by layer",
            disable=not self.show_progress_bars,
            total=len(self.cluster_layers_),
            unit="layer",
        ):
            layer.make_keyphrases(
                self.keyphrase_list_,
                self.object_x_keyphrase_matrix_,
                self.keyphrase_vectors_,
                self.embedding_model,
                method=keyphrase_method,
            )

            if i > 0:
                if not hasattr(self.cluster_layers_[0], "topic_name_embeddings"):
                    self.cluster_layers_[0].embed_topic_names(self.embedding_model)

            if _summarize_topics:
                if i > 0:
                    layer.make_subtopics(
                        self.topic_names_[0],
                        self.cluster_layers_[0].cluster_labels,
                        self.cluster_layers_[0].topic_name_embeddings,
                        self.embedding_model,
                        method=subtopic_method,
                        topic_summaries=self.topic_summaries_[0],
                        topic_explanations=self.topic_explanations_[0],
                    )
                layer.make_prompts(
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                    None,
                    None,
                    self.topic_summaries_,
                    self.topic_explanations_,
                )
                (
                    self.topic_names_[i],
                    self.topic_summaries_[i],
                    self.topic_explanations_[i],
                ) = layer.name_topics(
                    self.llm_wrapper,
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                    self.embedding_model,
                    self.topic_summaries_,
                    self.topic_explanations_,
                )
            else:
                if i > 0:
                    layer.make_subtopics(
                        self.topic_names_[0],
                        self.cluster_layers_[0].cluster_labels,
                        self.cluster_layers_[0].topic_name_embeddings,
                        self.embedding_model,
                        method=subtopic_method,
                    )
                layer.make_prompts(
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                )
                self.topic_names_[i] = layer.name_topics(
                    self.llm_wrapper,
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                    self.embedding_model,
                )
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
