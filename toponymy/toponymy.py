from toponymy.cluster_layer import ClusterLayer
from toponymy.clustering import create_cluster_layers
import numpy as np

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
            embedding_model, 
            layer_class,
            min_clusters: int = 6,
            min_samples: int = 5,
            base_min_cluster_size: int = 10,
            next_cluster_size_quantile: float = 0.8,
            ):
        self.llm_wrapper = llm_wrapper
        self.embedding_model = embedding_model
        self.layer_class = layer_class
        self.min_clusters = min_clusters
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.next_cluster_size_quantile = next_cluster_size_quantile

    def fit(self, objects, embedding_vectors: np.array, clusterable_vectors: np.array):
        """
        Vecotrizes using the classes embedding_model and constructs a low dimension data map with UMAP if object_vectors and object_map aren't specc.
        Attributes:
        -----------
        objects: Object
            The objects over which to perform topic modeling.  These are often text documents or images.
        embedding_vectors: np.array
            An numpy array of shape=(number_of_objects, embedding_dimension) created with the same embedding_model specified in the constructor.  
        object_map: np.array
            A numpy array of shape=(number_of_objects, clustering_dimension).  It is recommended that the clustering_dimension should be low enough 
            for density based clustering to be efficient (2-5).  
        """

        #Buid our layers and cluster tree
        layers, cluster_tree = create_cluster_layers(
            self.layer_class,
            clusterable_vectors = self.clusterable_vectors_,
            embedding_vectors = self.embedding_vectors_,
            min_clusters = self.min_clusters,
            min_samples = self.min_samples,
            base_min_cluster_size = self.base_min_cluster_size,
            next_cluster_size_quantile = self.next_cluster_size_quantile,
        )

        for i, layer in enumerate(layers):
            # How to determine which summary information to use in template
            # Use ClusterLayer to create necessary summary info
            # Use jinja template to construct prompt
            # LLM Wrapper around prompt to create topic layer using i to determine depth
            pass
     
        

    def fit_predict(self, objects, object_vectors=None, object_map=None):
        pass

