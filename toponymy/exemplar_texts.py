import random
import numpy as np
from typing import List, Tuple, FrozenSet, Dict, Callable


def random_sample_exemplar(
    cluster_label_vector: np.ndarray,
    object_list: List[str],
    n_samples: int = 4,
) -> List[List[str]]:
    """Generates a list of exemplar texts for each cluster in a cluster layer.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.
    n_samples : int, optional
        The number of exemplars to sample for each cluster, by default 4.

    Returns
    -------
    keyphrases List[List[str]]
        A list of lists of keyphrases for each cluster.
    """

    exemplars = []
    for i in np.unique(cluster_label_vector):
        indices = np.where(cluster_label_vector==i)[0]
        if(len(indices)>n_samples):
            sample_indices = np.random.choice(indices, n_samples, replace=False)
            sample_paragraphs = object_list[sample_indices] 
        else:
            sample_paragraphs = object_list[indices]
        exemplars.append(sample_paragraphs)

    return exemplars
    