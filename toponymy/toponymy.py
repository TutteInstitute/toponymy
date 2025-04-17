from toponymy.clustering import ToponymyClusterer, Clusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.cluster_layer import ClusterLayer, ClusterLayerText
from toponymy.topic_tree import TopicTree

from sentence_transformers import SentenceTransformer
from sklearn.utils.validation import check_is_fitted
import numpy as np

from tqdm.auto import tqdm

from typing import List, Any, Type


class Toponymy:
    """
    A class for generating topic names for vector based topic modeling.

    Parameters:
    -----------
    llm_wrapper: class
        A llm_wrapper class.  These should be objects that inherit from the LlmWrapper base classs.
    embedding_model: callable
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
    show_progress_bars: bool
        Whether to show progress bars or not.
    
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
    show_progress_bars: bool
        Whether to show progress bars or not.
    clusterable_vectors_: np.array
        A numpy array of shape=(number_of_objects, clustering_dimension) used for clustering.
    embedding_vectors_: np.array
        A numpy array of shape=(number_of_objects, embedding_dimension) used for vectorizing.
    cluster_layers_: List[ClusterLayer]
        A list of ClusterLayer objects that represent the layers of the topic model.
    cluster_tree_: dict
        A dictionary that represents the tree of clusters.
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

    """

    def __init__(
        self,
        llm_wrapper,
        text_embedding_model: SentenceTransformer,
        clusterer: Clusterer = ToponymyClusterer(),
        layer_class: Type[ClusterLayer] = ClusterLayerText,
        keyphrase_builder: KeyphraseBuilder = KeyphraseBuilder(),
        object_description: str = "objects",
        corpus_description: str = "collection of objects",
        lowest_detail_level: float = 0.0,
        highest_detail_level: float = 1.0,
        exemplar_delimiters: List[str] = ["    *\"", "\"\n"],
        show_progress_bars: bool = True,
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
        self.show_progress_bars = show_progress_bars


    def fit(
        self,
        objects: List[Any],
        embedding_vectors: np.array,
        clusterable_vectors: np.array,
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

        Returns:
        --------
        self: object
            Returns the instance of the class.
        """
        self.clusterable_vectors_ = clusterable_vectors
        self.embedding_vectors_ = embedding_vectors

        # Build our layers and cluster tree
        if hasattr(self.clusterer, "cluster_layers_") and hasattr(self.clusterer, "cluster_tree_"):
            # If the clusterer has already been fit, we can skip this step
            self.cluster_layers_ = self.clusterer.cluster_layers_
            self.cluster_tree_ = self.clusterer.cluster_tree_
        else:
            self.cluster_layers_, self.cluster_tree_ = self.clusterer.fit_predict(
                clusterable_vectors,
                embedding_vectors,
                self.layer_class,
                show_progress_bar=self.show_progress_bars,
                exemplar_delimiters=self.exemplar_delimiters,
            )

        # Initialize other data structures
        self.topic_names_ = [[]] * len(self.cluster_layers_)
        self.topic_name_vectors_ = [np.array([])] * len(self.cluster_layers_)
        detail_levels = np.linspace(
            self.lowest_detail_level,
            self.highest_detail_level,
            len(self.cluster_layers_),
        )

        # Get exemplars for layer 0 first and build keyphrase matrix
        if hasattr(self.cluster_layers_[0], 'object_to_text_function') and \
        self.cluster_layers_[0].object_to_text_function is not None:
            # Non-text objects: use exemplars to build keyphrase matrix
            exemplars, exemplar_indices = self.cluster_layers_[0].make_exemplar_texts(
                objects,
                embedding_vectors,
            )
            
            # Create aligned text list
            aligned_texts = [''] * len(objects)  # Empty strings for non-exemplars
            for cluster_idx, cluster_exemplars in enumerate(exemplars):
                for exemplar_idx, exemplar_text in zip(exemplar_indices[cluster_idx], cluster_exemplars):
                    aligned_texts[exemplar_idx] = exemplar_text
                    
            # Build keyphrase matrix from aligned texts
            self.object_x_keyphrase_matrix_, self.keyphrase_list_ = (
                self.keyphrase_builder.fit_transform(aligned_texts)
            )
        else:
            # Text objects: build keyphrase matrix directly from objects
            self.object_x_keyphrase_matrix_, self.keyphrase_list_ = (
                self.keyphrase_builder.fit_transform(objects)
            )
            # Still need to generate exemplars for layer 0
            self.cluster_layers_[0].make_exemplar_texts(
                objects,
                embedding_vectors,
            )
        
        # Generate keyphrase vectors
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
            if i > 0:  # Skip layer 0 exemplars as we already did them
                layer.make_exemplar_texts(
                    objects,
                    embedding_vectors,
                )
                
            layer.make_keyphrases(
                self.keyphrase_list_,
                self.object_x_keyphrase_matrix_,
                self.keyphrase_vectors_,
            )
            
            if i > 0:
                if not hasattr(self.cluster_layers_[0], "topic_name_embeddings"):
                    self.cluster_layers_[0].embed_topic_names(self.embedding_model)
                    
                layer.make_subtopics(
                    self.topic_names_[0],
                    self.cluster_layers_[0].cluster_labels,
                    self.cluster_layers_[0].topic_name_embeddings,
                    self.embedding_model,
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

        return self

    def fit_predict(self, objects: List[Any], object_vectors: np.array, clusterable_vectors: np.array) -> List[np.array]:
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
        self.fit(objects, object_vectors, clusterable_vectors)
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
        check_is_fitted(self, ["cluster_tree_", "topic_names_", "topic_name_vectors_"])
        def cluster_size(cluster_label_array):
            if cluster_label_array.min() < 0:
                return np.bincount(cluster_label_array - cluster_label_array.min())[-cluster_label_array.min():].tolist()
            else:
                return np.bincount(cluster_label_array).tolist()
        topic_sizes = [
            cluster_size(layer.cluster_labels) for layer in self.cluster_layers_
        ]
        return TopicTree(
            self.cluster_tree_,
            self.topic_names_,
            topic_sizes,
            self.embedding_vectors_.shape[0],
        )

