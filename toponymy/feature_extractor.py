from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, TypeVar

from apricot import SaturatedCoverageSelection
import numba
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from toponymy.new_clustering import Clusterer
from toponymy.new_utility_functions import centroids_from_labels, diversify_max_alpha
from toponymy._new_utils import FacilityLocationSelection

FeatureReturnType = TypeVar("FeatureReturnType")


class FeatureExtractorBase(ABC, BaseEstimator):
    """
    Abstract base class for a feature extractor.

    A feature extractor is a class that can build features for objects
    and then extract features to represent clusters.
    """

    def __init__(self, *args, **kwargs):
        self.features_ = None

    def __sklearn_is_fitted__(self):
        return self.features is not None

    def can_fit_from_objects(self) -> bool:
        """
        If True, enables the FeatureExtractor to be fitted on the fly.

        If False, specifies that the FeatureExtractor must be pre-fitted.
        """
        return False

    @property
    @abstractmethod
    def feature_return_type(self) -> FeatureReturnType:
        """
        The return type of features extracted by the FeatureExtractor.

        Must be implemented in any subclass.
        """
        raise NotImplemented

    @property
    def features(self) -> List[List[List[FeatureReturnType]]]:
        """
        A list of features: at each cluster layer, list the features for each cluster in the layer.
        """
        return self.features_

    @abstractmethod
    def fit(
        self,
        objects: List[Any],
        clusterer: Clusterer,
        *args,
        **kwargs,
    ):
        """
        An abstract method to fit a collection of features to a set of objects
        and assign features to represent each cluster across layers.

        Must be defined in any subclass.

        Parameters
        ----------
        objects: List[Any]
            The objects to fit the FeatureExtractor to.
        clusterer: Clusterer
            A fitted Clusterer with a cluster layer structure.

        Notes
        -----
        Any implementation of this method should update self.features.
        """
        raise NotImplemented

    def predict(
        self,
    ) -> List[List[List[str]]]:
        """
        Returns the list of features for each cluster per cluster layer.
        """
        return self.features

    def fit_predict(
        self,
        objects: List[Any],
        clusterer: Clusterer,
        *args,
        **kwargs,
    ) -> List[List[List[str]]]:
        """
        Checks to see if the
        """
        try:
            check_is_fitted(self)
        except NotFittedError as err:
            if self.can_fit_from_objects():
                self.fit(objects, clusterer, *args, **kwargs)
            else:
                raise NotFittedError(
                    f"Cannot fit {self.__class__.__name__} from objects, please fit manually."
                )
        return self.predict()


class TextExemplarExtractor(FeatureExtractorBase):
    """
    Selects exemplar texts from a collection of objects to represent clusters.

    Attributes
    ----------
    supported_selection_methods: List[str]
        A list of selection methods which are supported by `.get_cluster_features()`.

    Notes
    -----
    The feature extractor should be first called with `.fit()`.

    At each layer, exemplars can be extracted using `.predict()`.
    """

    def __init__(
        self,
    ):
        super(TextExemplarExtractor, self).__init__()

    @property
    def supported_selection_methods(self):
        return [
            "facility_location",
            "saturated_coverage",
            "random",
            "central",
        ]

    @property
    def feature_return_type(self):
        return str

    def can_fit_from_objects(self):
        return True

    def fit(
        self,
        objects: List[Any],
        clusterer: Clusterer,
        selection_method: str,
        object_vectors: np.ndarray | None,
    ):
        """
        Extracts exemplars for each cluster, layer by layer.

        Parameters
        ----------
        objects: List[Any]
            A list of the objects within the clusters.
        clusterer: Clusterer
            A fitted Clusterer with cluster layers.
        selection_method: str
            The method used to extract exemplars.
            Choose from 'facility_location', 'saturated_coverage', 'random' or 'central'.
        object_vectors: np.ndarray or None
            High-dimensional vectors representing each of the objects.
        **kwargs
            Additional parameters relevant to the particular selection method.

        See Also
        --------
        diverse_exemplars
        random_exemplars
        submodular_selection_exemplars
        """
        self.features = []

        for l, layer in enumerate(clusterer):
            cluster_label_vector = layer.labels

            if selection_method == "facility_location":
                exemplars, indices = submodular_selection_exemplars(
                    cluster_label_vector,
                    objects,
                    object_vectors,
                    submodular_function=selection_method,
                    **kwargs,
                )
            elif selection_method == "saturated_coverage":
                exemplars, indices = submodular_selection_exemplars(
                    cluster_label_vector,
                    objects,
                    object_vectors,
                    submodular_function=selection_method,
                    **kwargs,
                )
            elif selection_method == "random":
                exemplars, indices = random_exemplars(
                    cluster_label_vector, objects, object_vectors, **kwargs
                )
            elif selection_method == "central":
                exemplars, indices = diverse_exemplars(
                    cluster_label_vector, objects, object_vectors, **kwargs
                )
            else:
                raise ValueError(
                    f"Unsupported selection method: {selection_method}. Please use one of the currently supported selection methods: {self.supported_selection_methods}"
                )

            for c, cluster in enumerate(layer):
                cluster.features = exemplars[c]

            self.features.append(exemplars)

    @staticmethod
    def submodular_selection_exemplars(
        cluster_label_vector: np.ndarray,
        objects: List[str],
        object_vectors: np.ndarray,
        n_exemplars: int = 4,
        object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
        submodular_function: str = "facility_location",
        verbose: bool = False,
    ) -> Tuple[List[List[str]], List[List[int]]]:
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
        n_exemplars : int, optional
            The number of exemplars to sample for each cluster, by default 4.
        diversify_alpha : float, optional
            The alpha parameter for diversifying the keyphrase selection, by default 1.0.
        object_to_text_function: Callable[List[Any], List[str]]
            A function which takes an object and returns an exemplar string, by default for strings it is lambda x: x
        submodular_function : str, optional
            The sampling method for selecting exemplars 'facility_location' or 'saturated_coverage', by default it is 'facility_location'.
        verbose : bool, optional
            Whether to print verbose output and show progress bars, by default False.

        Returns
        -------
        Tuple[List[List[str]], List[List[int]]]
            A tuple containing:
            - A list of lists of exemplar texts for each cluster
            - A list of lists of indices indicating the position of each exemplar in the original object list
        """
        results = []
        indices = []

        null_topic_vector = np.mean(object_vectors, axis=0)
        if submodular_function == "facility_location":
            selector = FacilityLocationSelection(
                n_exemplars, metric="cosine", optimizer="lazy"
            )
        elif submodular_function == "saturated_coverage":
            selector = SaturatedCoverageSelection(
                n_exemplars, metric="cosine", optimizer="lazy"
            )
        else:
            raise ValueError(
                f"selection_function={submodular_function} is not a valid selection. Please choose one of (facility_location,saturated_coverage)"
            )

        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc=f"Selecting {submodular_function} exemplars",
            disable=not verbose,
            unit="cluster",
            leave=False,
            position=1,
        ):
            # Get mask for current cluster
            cluster_mask = cluster_label_vector == cluster_num
            # subsample if it is too large
            if np.sum(cluster_mask) > 16384:
                cluster_mask = np.random.choice(
                    np.where(cluster_mask)[0], size=16384, replace=False
                )
                cluster_mask = np.isin(
                    np.arange(len(cluster_label_vector)), cluster_mask
                )
            # Get the objects in this cluster

            # Store original indices for this cluster
            original_indices = np.where(cluster_mask)[0]

            # Index objects by integer position — no np.array(objects) needed
            cluster_objects = [objects[i] for i in original_indices]

            # If there is an empty cluster emit empty lists
            if len(cluster_objects) == 0:
                results.append([])
                indices.append([])
                continue

            cluster_object_vectors = object_vectors[cluster_mask] - null_topic_vector
            cluster_indices = np.arange(cluster_object_vectors.shape[0])

            if cluster_object_vectors.shape[0] >= n_exemplars:
                _, candidate_indices = selector.fit_transform(
                    cluster_object_vectors, y=cluster_indices
                )
            else:
                candidate_indices = cluster_indices

            chosen_exemplars = object_to_text_function(
                [cluster_objects[i] for i in candidate_indices]
            )

            # Map chosen indices back to original object list indices
            chosen_original_indices = [original_indices[i] for i in candidate_indices]

            results.append(chosen_exemplars)
            indices.append(chosen_original_indices)

        return results, indices

    @staticmethod
    def random_exemplars(
        cluster_label_vector: np.ndarray,
        objects: List[str],
        n_exemplars: int = 4,
        object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
        verbose: bool = False,
    ) -> Tuple[List[List[str]], List[List[int]]]:
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
        object_to_text_function: Callable[List[Any], List[str]]
            A function which takes an object and returns an exemplar string, by default for strings it is lambda x: x
        verbose : bool, optional
            Whether to print verbose output and show progress bars, by default False.

        Returns
        -------
        Tuple[List[List[str]], List[List[int]]]
            A tuple containing:
            - A list of lists of exemplar texts for each cluster
            - A list of lists of indices indicating the position of each exemplar in the original object list
        """
        results = []
        indices = []
        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc="Selecting random exemplars",
            disable=not verbose,
            unit="cluster",
            leave=False,
            position=1,
        ):
            # Get mask for current cluster
            cluster_mask = cluster_label_vector == cluster_num

            # Store original indices for this cluster
            original_indices = np.where(cluster_mask)[0]

            # Index objects by integer position — no np.array(objects) needed
            cluster_objects = [objects[i] for i in original_indices]

            # If there is an empty cluster emit empty lists
            if len(cluster_objects) == 0:
                results.append([])
                indices.append([])
                continue

            # Randomly permute the index to create a random selection
            exemplar_order = np.random.permutation(len(cluster_objects))[:n_exemplars]
            chosen_exemplars = object_to_text_function(
                [cluster_objects[i] for i in exemplar_order]
            )

            # Map chosen indices back to original object list indices
            chosen_original_indices = [original_indices[i] for i in exemplar_order]

            results.append(chosen_exemplars)
            indices.append(chosen_original_indices)

        return results, indices

    @staticmethod
    def diverse_exemplars(
        cluster_label_vector: np.ndarray,
        objects: List[str],
        object_vectors: np.ndarray,
        n_exemplars: int = 4,
        diversify_alpha: float = 1.0,
        object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
        sampling_method: str = "centroid",
        verbose: bool = False,
    ) -> Tuple[List[List[str]], List[List[int]]]:
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
        n_exemplars : int, optional
            The number of exemplars to sample for each cluster, by default 4.
        diversify_alpha : float, optional
            The alpha parameter for diversifying the keyphrase selection, by default 1.0.
        object_to_text_function: Callable[List[Any], List[str]]
            A function which takes an object and returns an exemplar string, by default for strings it is lambda x: x
        sampling_method : str, optional
            The sampling method for selecting exemplars 'centroid' or 'random', by default it is 'centroid'.
        verbose : bool, optional
            Whether to print verbose output and show progress bars, by default False.

        Returns
        -------
        Tuple[List[List[str]], List[List[int]]]
            A tuple containing:
            - A list of lists of exemplar texts for each cluster
            - A list of lists of indices indicating the position of each exemplar in the original object list
        """
        # Compute centroid vectors
        if verbose:
            print("Computing centroid vectors")
        centroid_vectors = centroids_from_labels(cluster_label_vector, object_vectors)

        results = []
        indices = []
        null_topic = np.mean(object_vectors, axis=0)

        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc="Selecting central exemplars",
            disable=not verbose,
            unit="cluster",
            leave=False,
            position=1,
        ):

            # Get mask for current cluster
            cluster_mask = cluster_label_vector == cluster_num

            # Store original indices for this cluster
            original_indices = np.where(cluster_mask)[0]

            # Index objects by integer position — no np.array(objects) needed
            cluster_objects = [objects[i] for i in original_indices]

            # If there is an empty cluster emit empty lists
            if len(cluster_objects) == 0:
                results.append([])
                indices.append([])
                continue

            cluster_object_vectors = object_vectors[cluster_mask] - null_topic
            if sampling_method == "centroid":
                # Select the central exemplars as the objects to each centroid
                exemplar_distances = pairwise_distances(
                    centroid_vectors[cluster_num].reshape(1, -1) - null_topic,
                    cluster_object_vectors,
                    metric="cosine",
                )
                exemplar_order = np.argsort(exemplar_distances.flatten())
            elif sampling_method == "random":
                exemplar_order = np.random.permutation(len(cluster_objects))
            else:
                raise ValueError(
                    f"sampling_method={sampling_method} is not a valid selection. Please choose one of (centroid,random)"
                )

            # We need more exemplars than we want in case we drop some via diversify
            n_exemplars_to_take = max((n_exemplars**2), 16)
            exemplar_candidates = [
                cluster_objects[i] for i in exemplar_order[:n_exemplars_to_take]
            ]
            candidate_vectors = np.asarray(
                [
                    cluster_object_vectors[i]
                    for i in exemplar_order[:n_exemplars_to_take]
                ]
            )

            chosen_indices = diversify_max_alpha(
                centroid_vectors[cluster_num] - null_topic,
                candidate_vectors,
                n_exemplars,
                max_alpha=diversify_alpha,
                tolerance=0.01,
            )[:n_exemplars]

            chosen_exemplars = object_to_text_function(
                [exemplar_candidates[i] for i in chosen_indices]
            )

            # Map chosen indices back to original object list indices
            chosen_original_indices = [
                original_indices[exemplar_order[i]] for i in chosen_indices
            ]

            results.append(chosen_exemplars)
            indices.append(chosen_original_indices)

        return results, indices
