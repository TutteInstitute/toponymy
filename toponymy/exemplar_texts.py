import random
import numpy as np
from typing import List, Tuple, FrozenSet, Dict, Callable
from sklearn.metrics import pairwise_distances
from toponymy.utility_functions import diversify_max_alpha as diversify



def random_sample_exemplar(
    cluster_label_vector: np.ndarray,
    objects: List[str],
    n_exemplars: int = 4,
) -> List[List[str]]:
    """Generates a list of exemplar texts for each cluster in a cluster layer.
    These exemplars are randomly sampled from each cluster.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.  Cluster labels below zero are ignored and cluster labels are
        explected to be integers ranging from 0 to the number_of_clusters.
    objects : List[str]
        A list of objects; these are text objects a sample of which are returned as exemplars for each cluster.
    n_samples : int, optional
        The number of exemplars to sample for each cluster, by default 4.

    Returns
    -------
    exemplars List[List[str]]
        A list of lists of exemplar text for each cluster.
    """

    results = []
    for cluster_num in range(cluster_label_vector.max() + 1):
        #Grab the vectors associated with the objects in this cluster
        cluster_objects = np.array(objects)[cluster_label_vector==cluster_num]
        #If there is an empty cluster emit the empty list of exemplars
        if(len(cluster_objects)==0):
            results.append([])
            continue
        # Randomly permute the index to create a random selection
        exemplar_order = np.random.permutation(len(cluster_objects))[:n_exemplars]
        chosen_exemplars = [cluster_objects[i] for i in exemplar_order]
        results.append(chosen_exemplars)
    return results
    
def centroid_sample_exemplar(
    cluster_label_vector: np.ndarray,
    objects: List[str],
    object_vectors: np.ndarray,
    centroid_vectors: np.ndarray,
    n_exemplars: int = 4,
    diversify_alpha: float = 1.0,
) -> List[List[str]]:
    """Generates a list of exemplar text for each cluster in a cluster layer.  
    These exemplars are selected to be the closest vectors to the cluster centroid while retaining 
    sufficient diversity.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.
    objects : List[str]
        A list of objects; these are text objects a sample of which are returned as exemplars for each cluster.
    object_vectors = np.ndarray
        An ndarray of topic vectors for each object.
    centroid_vectors : np.ndarray
        An ndarray of centroid vectors for each cluster.
    n_exemplars : int, optional
        The number of exemplars to sample for each cluster, by default 4.
    diversify_alpha : float, optional
        The alpha parameter for diversifying the keyphrase selection, by default 1.0.

    Returns
    -------
    exemplars List[List[str]]
        A list of lists of exemplar text for each cluster.
    """
    results = []
    for cluster_num in range(cluster_label_vector.max() + 1):
        #Grab the vectors associated with the objects in this cluster
        cluster_objects = np.array(objects)[cluster_label_vector==cluster_num]
        #If there is an empty cluster emit the empty list of exemplars
        if(len(cluster_objects)==0):
            results.append([])
            continue
        cluster_object_vectors = object_vectors[cluster_label_vector==cluster_num]

        # Select the central exemplars as the objects to each centroid
        exemplar_distances = pairwise_distances(
            centroid_vectors[cluster_num].reshape(1, -1), cluster_object_vectors, metric="cosine"
        )
        exemplar_order = np.argsort(exemplar_distances.flatten())
        # I'm uncertain about selecting up to n_exemplar**2 candidates
        exemplar_candidates = [cluster_objects[i] for i in exemplar_order[ : n_exemplars **2]]
        candidate_vectors = np.asarray([cluster_object_vectors[i] for i in exemplar_order[ : n_exemplars **2]])
        chosen_indices = diversify(
            centroid_vectors[cluster_num], candidate_vectors, n_exemplars, max_alpha=diversify_alpha
        )[:n_exemplars]
        chosen_exemplars = [exemplar_candidates[i] for i in chosen_indices]
        results.append(chosen_exemplars)
    return results

def random_diverse_exemplar(
    cluster_label_vector: np.ndarray,
    objects: List[str],
    object_vectors: np.ndarray,
    centroid_vectors: np.ndarray,
    n_exemplars: int = 4,
    diversify_alpha: float = 1.0,
) -> List[List[str]]:
    """Generates a list of exemplar text for each cluster in a cluster layer.  
    These exemplars are randomly while still requiring sufficient diversity.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.
    objects : List[str]
        A list of objects; these are text objects a sample of which are returned as exemplars for each cluster.
    object_vectors = np.ndarray
        An ndarray of topic vectors for each object.
    centroid_vectors : np.ndarray
        An ndarray of centroid vectors for each cluster.
    n_exemplars : int, optional
        The number of exemplars to sample for each cluster, by default 4.
    diversify_alpha : float, optional
        The alpha parameter for diversifying the keyphrase selection, by default 1.0.

    Returns
    -------
    exemplars List[List[str]]
        A list of lists of exemplar text for each cluster.
    """
    results = []
    for cluster_num in range(cluster_label_vector.max() + 1):
        #Grab the vectors associated with the objects in this cluster
        cluster_objects = np.array(objects)[cluster_label_vector==cluster_num]
        #If there is an empty cluster emit the empty list of exemplars
        if(len(cluster_objects)==0):
            results.append([])
            continue
        cluster_object_vectors = object_vectors[cluster_label_vector==cluster_num]

        # Randomly permute the index to create a random selection
        exemplar_order = np.random.permutation(len(cluster_objects))
        # I'm uncertain about selecting up to n_exemplar**2 candidates
        exemplar_candidates = [cluster_objects[i] for i in exemplar_order[ : n_exemplars **2]]
        candidate_vectors = np.asarray([cluster_object_vectors[i] for i in exemplar_order[ : n_exemplars **2]])
        chosen_indices = diversify(
            centroid_vectors[cluster_num], candidate_vectors, n_exemplars, max_alpha=diversify_alpha
        )[:n_exemplars]
        chosen_exemplars = [exemplar_candidates[i] for i in chosen_indices]
        results.append(chosen_exemplars)
    return results