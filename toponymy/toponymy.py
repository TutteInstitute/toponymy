from toponymy.clustering import ToponymyClusterer, Clusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.cluster_layer import ClusterLayer
from sentence_transformers import SentenceTransformer
import numpy as np

from tqdm.auto import tqdm

from typing import List, Any, Type


class Toponymy:
    """
    A class for generating topic names for vector based topic modeling.

    Attributes:
    -----------
    llm_wrapper: class
        A llm_wrapper class.  These should be objects that inherit from the LlmWrapper base classs.
    embedding_model: callable
        a function with an encode used to vectorize the objects that we are topic modeling.
    layer_class: Type[Any]
        The class to used for creating layers from our objects.

    """

    def __init__(
        self,
        llm_wrapper,
        text_embedding_model: SentenceTransformer,
        layer_class: Type[ClusterLayer],
        clusterer: Clusterer = ToponymyClusterer(),
        keyphrase_builder: KeyphraseBuilder = KeyphraseBuilder(),
        object_description: str = "objects",
        corpus_description: str = "collection of objects",
        lowest_detail_level: float = 0.0,
        highest_detail_level: float = 1.0,
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
        self.show_progress_bars = show_progress_bars

    def fit(
        self,
        objects: List[Any],
        embedding_vectors: np.array,
        clusterable_vectors: np.array,
    ):
        """
        Vectorizes using the classes embedding_model and constructs a low dimension data map with UMAP if object_vectors and object_map aren't specc.
        Attributes:
        -----------
        objects: Object
            The objects over which to perform topic modeling.  These are often text documents or images.
        embedding_vectors: np.array
            An numpy array of shape=(number_of_objects, embedding_dimension) created with the same embedding_model specified in the constructor.
        clusterable_vectors: np.array
            A numpy array of shape=(number_of_objects, clustering_dimension).  It is recommended that the clustering_dimension should be low enough
            for density based clustering to be efficient (2-25).
        """
        self.clusterable_vectors_ = clusterable_vectors
        self.embedding_vectors_ = embedding_vectors

        # Build our layers and cluster tree
        self.cluster_layers_, self.cluster_tree_ = self.clusterer.fit_predict(
            clusterable_vectors,
            embedding_vectors,
            self.layer_class,
            show_progress_bar=self.show_progress_bars,
        )

        # Build keyphrase information
        self.object_x_keyphrase_matrix_, self.keyphrase_list_ = (
            self.keyphrase_builder.fit_transform(objects)
        )
        self.keyphrase_vectors_ = self.embedding_model.encode(self.keyphrase_list_, show_progress_bar=self.show_progress_bars, )

        # Initialize other data structures
        self.topic_names_ = [[]] * len(self.cluster_layers_)
        detail_levels = np.linspace(
            self.lowest_detail_level,
            self.highest_detail_level,
            len(self.cluster_layers_),
        )

        # Iterate through the layers and build the topic names
        for i, layer in tqdm(
            enumerate(self.cluster_layers_),
            desc=f"Building topic names by layer",
            disable=not self.show_progress_bars,
            total=len(self.cluster_layers_),
            unit="layer",
        ):
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
                layer.make_subtopics(
                    self.topic_names_[0],
                    self.cluster_layers_[0].cluster_labels,
                    self.cluster_layers_[0].centroid_vectors,
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

    def fit_predict(self, objects, object_vectors=None, object_map=None):
        self.fit(objects, object_vectors, object_map)
        return self.topic_names_, [layer.label_vector for layer in self.cluster_layers_]
