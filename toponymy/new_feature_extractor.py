from abc import ABC, abstractmethod
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)
from warnings import warn

from apricot import GraphCutSelection, SaturatedCoverageSelection
from joblib import delayed, effective_n_jobs, Parallel
import numba
import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from vectorizers.transformers import InformationWeightTransformer

from toponymy.new_clustering import Clusterer
from toponymy.new_types import TextEmbedderProtocol
from toponymy.new_utility_functions import centroids_from_labels, diversify_max_alpha
from toponymy._new_utils import FacilityLocationSelection

FeatureReturnType = TypeVar("FeatureReturnType")
Ngrammer = Callable[[str], List[str]]


# Define a protocol for objects that behave like Tokenizers
class TokenizerLike(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> Any: ...

    def decode(self, ids: Any, *args: Any, **kwargs: Any) -> str: ...


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
        raise NotImplementedError

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
        *args, **kwargs
            Additional parameters needed for the fit function.

        Notes
        -----
        Any implementation of this method should update self.features.
        """
        raise NotImplementedError

    def predict(
        self,
    ) -> List[List[List[str]]]:
        """
        Returns the list of features for each cluster per cluster layer.

        Returns
        -------
        List[List[List[str]]]
            A list of features for each cluster in a cluster layer, across all cluster layers.
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
        Checks to see if the FeatureExtractor has already been fit.
        If not, and it is possible to do so, it will fit the FeatureExtractor and returns the features.

        Parameters
        ----------
        objects: List[Any]
            The objects to fit the FeatureExtractor to.
        clusterer: Clusterer
            A fitted Clusterer with a cluster layer structure.
        *args, **kwargs
            Additional parameters needed for the fit function.

        Returns
        -------
        List[List[List[str]]]
            A list of features for each cluster in a cluster layer, across all cluster layers.
        """
        try:
            check_is_fitted(self)
        except NotFittedError as err:
            if self.can_fit_from_objects():
                self.fit(objects, clusterer, *args, **kwargs)
            else:
                raise NotFittedError(
                    f"Cannot fit {self.__class__.__name__} from objects, please fit manually."
                ) from err
        return self.predict()


class TextExemplarExtractor(FeatureExtractorBase):
    """
    Selects exemplar texts from a collection of objects to represent clusters.

    Attributes
    ----------
    supported_selection_methods: List[str]
        A list of selection methods which are supported by `.fit()`.
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
        **kwargs,
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

        Raises
        ------
        ValueError
            If a selection method that is not in `self.supported_selection_methods` is supplied.

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
                exemplars, indices = TextExemplarExtractor.submodular_selection_exemplars(
                    cluster_label_vector,
                    objects,
                    object_vectors,
                    submodular_function=selection_method,
                    **kwargs,
                )
            elif selection_method == "saturated_coverage":
                exemplars, indices = TextExemplarExtractor.submodular_selection_exemplars(
                    cluster_label_vector,
                    objects,
                    object_vectors,
                    submodular_function=selection_method,
                    **kwargs,
                )
            elif selection_method == "random":
                exemplars, indices = TextExemplarExtractor.random_exemplars(
                    cluster_label_vector, objects, object_vectors, **kwargs
                )
            elif selection_method == "central":
                exemplars, indices = TextExemplarExtractor.diverse_exemplars(
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


class TextKeyphraseExtractor(FeatureExtractorBase):
    """
    Selects keyphrases from string representations of objects in each cluster.

    Attributes
    ----------
    keyphrases: List[str]
        A list of keyphrases across the corpus of objects.
    object_x_keyphrase_matrix: scipy.sparse.spmatrix:
        A sparse matrix showing which keyphrases are associated to which object.
    supported_selection_methods: List[str]
        A list of selection methods which are supported by `.fit()`.
    """

    def __init__(
        self,
    ):
        super(TextKeyphraseExtractor, self).__init__()
        self.keyphrases = None
        self.object_x_keyphrase_matrix = None

    def can_fit_from_objects(self):
        return True

    @property
    def feature_return_type(self):
        return str

    @property
    def supported_selection_methods(self):
        return [
            "information_weighted",
            "central",
            "bm25",
            "saturated_coverage",
            "facility_location",
            "graph_cut",
        ]

    def _convert_objects_to_text(
        self,
        objects: List[Any],
        object_to_text_function: Optional[Callable[[Any], str]] = None,
    ) -> List[str]:
        """
        Converts objects into string representations of those objects, using the supplied self.object_to_text function.

        Parameters
        ----------
        objects: List[Any]
            A list of objects.

        Returns
        -------
        List[str]
            A list of string representations of objects.
        """
        if object_to_text_function is None:
            object_texts = objects
        else:
            object_texts = [object_to_text_function(obj) for obj in objects]
        return object_texts

    def _create_ngrammer(
        self,
        ngram_range: Tuple[int, int] = (1, 4),
        tokenizer: Optional[TokenizerLike] = None,
        token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
    ) -> Ngrammer:
        """
        Creates an ngrammer function that uses a tokenizer to tokenize the text and then generates n-grams.

        If no tokenizer is supplied, then a CountVectorizer is used by default.

        Returns
        -------
        Ngrammer
            A function that takes a string and returns a list of n-grams.
        """
        if tokenizer is None:
            # use CountVectorizer to build an ngram analyzer to ensure compatibility
            cv = CountVectorizer(
                lowercase=True,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
            )
            ngrammer = cv.build_analyzer()
        else:
            ngrammer = TextKeyphraseExtractor._create_tokenizers_ngrammer(tokenizer, ngram_range=ngram_range)
        return ngrammer

    def build_keyphrase_vocabulary(
        self,
        objects: List[str],
        ngrammer: Ngrammer,
        max_features: int = 50_000,
        min_occurrences: int = 1,
        stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
        n_jobs: int = -1,
        min_chunk_size: int = 20_000,
        verbose: bool = False,
    ) -> List[str]:
        """
        Builds a keyphrase vocabulary from a list of objects.

        Parameters
        ----------
        objects : List[str]
            A list of objects; for use in building a vocabulary this should be string representations of the objects.
        ngrammer : Ngrammer
            A function that takes a string and returns a list of n-grams.
        max_features : int, optional
            The maximum number of features to consider, by default 50_000.
        min_occurrences : int, optional
            The minimum number of occurrences for a keyphrase to be included, by default 1.
        stop_words : FrozenSet[str], optional
            The set of stop words to use, by default sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.
        n_jobs : int, optional
            The number of jobs to use in parallel processing, by default -1. If -1, all available cores are used.
        min_chunk_size : int, optional
            The minimum chunk size for parallel processing, by default 20_000.
        verbose : bool, optional
            Whether to show progress bars and verbose output. If True, shows all output. If False, suppresses all output.

        Returns
        -------
        List[str]
            A keyphrase list of the most commonly occurring keyphrases.
        """
        # count ngrams in parallel with joblib
        n_chunks = effective_n_jobs(n_jobs)
        chunk_size = max((len(objects) // n_chunks) + 1, min_chunk_size)
        n_chunks = len(objects) // chunk_size + 1
        if verbose:
            print(
                f"Chunking into {n_chunks} chunks of size {chunk_size} for keyphrase identification."
            )
        chunked_count_dicts = Parallel(n_jobs=n_chunks)(
            delayed(TextKeyphraseExtractor._count_docs_ngrams)(
                objects[i : i + chunk_size],
                ngrammer,
                stop_words,
                max_ngrams=max_features * 10,
            )
            for i in range(0, len(objects), chunk_size)
        )

        if verbose:
            print("Combining count dictionaries ...")
        # Combine dictionaries and count the most common ngrams
        # all_vocab_counts = reduce(combine_dicts, chunked_count_dicts, {})
        all_vocab_counts = TextKeyphraseExtractor._tree_combine_dicts(
            chunked_count_dicts, max_ngrams=max_features * 10
        )
        vocab_counter = Counter(all_vocab_counts)
        result = [
            ngram
            for ngram, occurrences in vocab_counter.most_common(max_features)
            if occurrences >= min_occurrences
        ]
        if len(result) == 0:
            raise ValueError(
                "No keyphrases found. Try increasing the max_features parameter or check that there are any re-occuring sections of text."
            )

        return result

    def build_keyphrase_count_matrix(
        self,
        objects: List[str],
        keyphrases: Dict[str, int],
        ngrammer: Ngrammer,
        n_jobs: int = -1,
        min_chunk_size: int = 20_000,
        verbose: bool = False,
    ) -> scipy.sparse.spmatrix:
        """
        Builds a count matrix of keyphrases in a list of objects.

        Parameters
        ----------
        objects : List[str]
            A list of objects; for use in building a count matrix this should be string representations of the objects.
        keyphrases : Dict[str, int]
            A dictionary where keys are keyphrases to count in the objects and values are their respective indices.
        ngrammer : Ngrammer
            A function that takes a string and returns a list of n-grams.
        n_jobs : int, optional
            The number of jobs to use in parallel processing, by default -1. If -1, all available cores are used.
        min_chunk_size : int, optional
            The minimum chunk size for parallel processing, by default 20_000.
        verbose : bool, optional
            Whether to show progress bars and verbose output. If True, shows all output. If False, suppresses all output.


        Returns
        -------
        scipy.sparse.spmatrix
            A sparse count matrix of keyphrases in the objects.
        """
        # count ngrams in parallel with joblib
        n_chunks = effective_n_jobs(n_jobs)
        chunk_size = max((len(objects) // n_chunks) + 1, min_chunk_size)
        n_chunks = (len(objects) // chunk_size) + 1
        if verbose:
            print(
                f"Chunking into {n_chunks} chunks of size {chunk_size} for keyphrase count construction."
            )
        chunked_count_matrices = Parallel(n_jobs=n_chunks)(
            delayed(TextKeyphraseExtractor.build_count_matrix)(
                objects[i : i + chunk_size], keyphrases, ngrammer
            )
            for i in range(0, len(objects), chunk_size)
        )
        if verbose:
            print("Combining count matrix chunks ...")

        # stack the count matrices
        result = scipy.sparse.vstack(chunked_count_matrices)

        return result

    def build_object_x_keyphrase_matrix(
        self,
        objects: List[str],
        ngram_range: Tuple[int, int] = (1, 4),
        tokenizer: Optional[TokenizerLike] = None,
        token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
        max_features: int = 50_000,
        min_occurrences: int = 1,
        stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
        n_jobs: int = -1,
        min_chunk_size: int = 20_000,
        verbose: bool = False,
    ) -> scipy.sparse.spmatrix:
        """
        Builds a count matrix of keyphrases in a list of objects.

        Parameters
        ----------
        objects : List[str]
            A list of objects; for use in building a count matrix this should be string representations of the objects.
        ngram_range : Tuple[int, int], optional
            The range of n-grams to consider, by default (1, 4).
        tokenizer : Optional[TokenizerLike], optional
            A tokenizer object that has encode and decode methods, by default None. If None, a CountVectorizer is used.
        token_pattern : str, optional
            The regular expression pattern to use for tokenization, by default "(?u)\\b\\w[-'\\w]+\\b".
        max_features : int, optional
            The maximum number of features to consider, by default 50_000.
        min_occurrences : int, optional
            The minimum number of occurrences for a keyphrase to be included, by default 1.
        stop_words : FrozenSet[str], optional
            The set of stop words to use, by default sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.
        n_jobs : int, optional
            The number of jobs to use in parallel processing, by default -1. If -1, all available cores are used.
        min_chunk_size : int, optional
            The minimum chunk size for parallel processing, by default 20_000.
        verbose : bool, optional
            Whether to show progress bars and verbose output. If True, shows all output. If False, suppresses all output.

        Returns
        -------
        scipy.sparse.spmatrix
            A sparse count matrix of keyphrases in the objects.
        """
        ngrammer = self._create_ngrammer(ngram_range, tokenizer, token_pattern)

        keyphrases = self.build_keyphrase_vocabulary(
            objects,
            ngrammer=ngrammer,
            max_features=max_features,
            min_occurrences=min_occurrences,
            stop_words=stop_words,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if verbose:
            print(f"Found {len(keyphrases)} keyphrases.")

        keyphrase_dict = {keyphrase: i for i, keyphrase in enumerate(keyphrases)}
        result = self.build_keyphrase_count_matrix(
            objects,
            keyphrase_dict,
            ngrammer=ngrammer,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.keyphrases = keyphrases
        self.object_x_keyphrase_matrix = result

        return result, keyphrases

    def build_keyphrase_vectors(
        self,
        keyphrases: List[str],
        embedder: Optional[TextEmbedderProtocol] = None,
        verbose: bool = False,
    ) -> np.typing.NDArray[np.floating] | None:
        if self.embedder is not None:
            if verbose:
                print("Building keyphrase vectors ... ")

            keyphrase_vectors = embedder.encode(
                keyphrases,
                verbose=verbose,
            )
        else:
            keyphrase_vectors = None

        self.keyphrase_vectors = keyphrase_vectors
        return keyphrase_vectors

    def _check_is_prefitted(self) -> bool:
        return (
            (self.keyphrases is not None)
            and (self.object_x_keyphrase_matrix is not None)
            and hasattr(self.keyphrase_vectors)
        )

    def fit(
        self,
        objects: List[Any],
        clusterer: Clusterer,
        selection_method: str,
        *,
        ngram_range: Tuple[int, int] = (1, 4),
        tokenizer: Optional[TokenizerLike] = None,
        token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
        max_features: int = 50_000,
        min_occurrences: int = 1,
        stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
        n_jobs: int = -1,
        min_chunk_size: int = 20_000,
        embedder: Optional[TextEmbedderProtocol] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Extracts keyphrases for each cluster, layer by layer.

        Parameters
        ----------
        objects: List[Any]
            A list of the objects within the clusters.
        clusterer: Clusterer
            A fitted Clusterer with cluster layers.
        selection_method: str
            The method used to extract keyphrases.
            Choose from 'information_weighted', 'central', 'bm25', 'saturated_coverage', 'facility_location' or 'graph_cut'.
        **kwargs
            Additional parameters relevant to the particular selection method.

        Raises
        ------
        ValueError
            If a selection method that is not in `self.supported_selection_methods` is supplied.

        See Also
        --------
        diverse_exemplars
        random_exemplars
        submodular_selection_exemplars
        """
        object_texts = self._convert_objects_to_text(objects)

        if not self._check_is_prefitted():
            object_x_feature_matrix, keyphrases = self.build_object_x_keyphrase_matrix(
                object_texts,
                ngram_range=ngram_range,
                tokenizer=tokenizer,
                token_pattern=token_pattern,
                max_features=max_features,
                min_occurrences=min_occurrences,
                stop_words=stop_words,
                n_jobs=n_jobs,
                min_chunk_size=min_chunk_size,
                verbose=verbose,
            )
            keyphrase_vectors = self.build_keyphrase_vectors(keyphrases, embedder)

        if embedder is None:
            warn(
                "No embedding model has been supplied. At present no keyphrase selection method exists which does not require an embedding model. Therefore keyphrase selection will not take place and this FeatureExtractor will not be correctly fitted. To fix this warning, please supply an embedding model with the keyword argument `embedder`.",
                category=RuntimeWarning,
            )
            return

        self.features = []

        for l, layer in enumerate(clusterer):
            cluster_label_vector = layer.labels

            if selection_method == "information_weighted":
                keyphrases_per_cluster = TextKeyphrasExtractor.information_weighted_keyphrases(
                    cluster_label_vector=cluster_label_vector,
                    object_x_keyphrase_matrix=self.object_x_feature_matrix,
                    keyphrase_list=self.keyphrases,
                    keyphrase_vectors=self.keyphrase_vectors,
                    embedding_model=embedder,
                    **kwargs,
                )
            elif selection_method == "central":
                keyphrases_per_cluster = TextKeyphraseExtractor.central_keyphrases(
                    cluster_label_vector=cluster_label_vector,
                    object_x_keyphrase_matrix=self.object_x_feature_matrix,
                    keyphrase_list=self.keyphrases,
                    keyphrase_vectors=self.keyphrase_vectors,
                    embedding_model=embedder,
                    **kwargs,
                )
            elif selection_method == "bm25":
                keyphrases_per_cluster = TextKeyphraseExtractor.bm25_keyphrases(
                    cluster_label_vector=cluster_label_vector,
                    object_x_keyphrase_matrix=self.object_x_feature_matrix,
                    keyphrase_list=self.keyphrases,
                    keyphrase_vectors=self.keyphrase_vectors,
                    embedding_model=embedder,
                    **kwargs,
                )
            elif selection_method in [
                "saturated_coverage",
                "facility_location",
                "graph_cut",
            ]:
                keyphrases_per_cluster = TextKeyphraseExtractor.submodular_selection_information_keyphrases(
                    cluster_label_vector=cluster_label_vector,
                    object_x_keyphrase_matrix=self.object_x_feature_matrix,
                    keyphrase_list=self.keyphrases,
                    keyphrase_vectors=self.keyphrase_vectors,
                    embedding_model=embedder,
                    submodular_function=selection_method,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported selection method: {selection_method}. Please use one of the currently supported selection methods: {self.supported_selection_methods}"
                )

            for c, cluster in enumerate(layer):
                cluster.features = keyphrases_per_cluster[c]

            self.features.append(keyphrases_per_cluster)

    @staticmethod
    def _create_tokenizers_ngrammer(
        tokenizer: TokenizerLike,
        ngram_range: Tuple[int, int] = (1, 4),
    ) -> Ngrammer:
        """
        Creates an ngrammer function that uses a tokenizer to tokenize the text and then generates n-grams.

        Parameters
        ----------
        tokenizer : TokenizerLike
            A tokenizer object that has encode and decode methods.
        ngram_range : Tuple[int, int], optional
            The range of n-grams to consider, by default (1, 4).

        Returns
        -------
        Ngrammer
            A function that takes a string and returns a list of n-grams.
        """

        def ngrammer(text: str) -> List[str]:
            encoded = tokenizer.encode(text)
            if isinstance(encoded, list):
                tokens = encoded
            else:
                tokens = encoded.ids
            return [
                tokenizer.decode(tokens[i : i + n])
                for n in range(ngram_range[0], ngram_range[1] + 1)
                for i in range(len(tokens) - n)
            ]

        return ngrammer

    @staticmethod
    def _count_docs_ngrams(
        docs: List[str],
        ngrammer: Ngrammer,
        stop_words: FrozenSet[str],
        max_ngrams: int = 250_000,
    ) -> Dict[str, int]:
        result = {}
        for doc in docs:
            for gram in ngrammer(doc):
                split_gram = gram.split()
                if len(split_gram) > 0 and (
                    split_gram[0] in stop_words or split_gram[-1] in stop_words
                ):
                    continue
                if gram in result:
                    result[gram] += 1
                else:
                    result[gram] = 1

        if len(result) > max_ngrams:
            trim_value = np.sort(list(result.values()))[-max_ngrams]
            result = {key: value for key, value in result.items() if value >= trim_value}

        return result

    @staticmethod
    def _combine_dicts(
        dict1: Dict[str, int], dict2: Dict[str, int], max_ngrams: int = 250_000
    ) -> Dict[str, int]:
        result = dict1
        for key in dict2:
            if key in result:
                result[key] += dict2[key]
            else:
                result[key] = dict2[key]

        if len(result) > max_ngrams:
            trim_value = np.sort(list(result.values()))[-max_ngrams]
            result = {
                key: value for key, value in result.items() if value >= trim_value
            }
        return result

    @staticmethod
    def _combine_tree_layer(
        dict_list: List[Dict[str, int]], max_ngrams: int = 250_000
    ) -> List[Dict[str, int]]:
        result = []
        for i in range(0, len(dict_list) - 1, 2):
            result.append(
                TextKeyphraseExtractor._combine_dicts(dict_list[i], dict_list[i + 1], max_ngrams=max_ngrams)
            )
        if len(dict_list) % 2 == 1:
            result.append(dict_list[-1])
        return result

    @staticmethod
    def _tree_combine_dicts(
        dict_list: List[Dict[str, int]], max_ngrams: int = 250_000
    ) -> Dict[str, int]:
        while len(dict_list) > 1:
            dict_list = TextKeyphraseExtractor._combine_tree_layer(dict_list, max_ngrams=max_ngrams)
        return dict_list[0]

    @staticmethod
    def build_count_matrix(
        docs: List[str], vocab: Dict[str, int], ngrammer: Ngrammer
    ) -> scipy.sparse.csr_matrix:
        col_indices = []
        indptr = [0]
        data = []
        for doc in docs:
            ngram_counter = {}
            for gram in ngrammer(doc):
                try:
                    ngram_idx = vocab[gram]
                    if ngram_idx in ngram_counter:
                        ngram_counter[ngram_idx] += 1
                    else:
                        ngram_counter[ngram_idx] = 1
                except KeyError:
                    continue

            col_indices.extend(ngram_counter.keys())
            data.extend(ngram_counter.values())
            indptr.append(len(col_indices))

        col_indices = np.asarray(col_indices, dtype=np.int32)
        data = np.asarray(data, dtype=np.int32)
        indptr = np.asarray(indptr, dtype=np.int32)

        X = scipy.sparse.csr_matrix(
            (data, col_indices, indptr), shape=(len(docs), len(vocab))
        )
        X.sort_indices()
        return X

    @staticmethod
    @numba.njit()
    def longest_keyphrases(candidate_keyphrases):  # pragma: no cover
        """
        Builds a list of keyphrases that are not substrings of other keyphrases.
        """
        result = []
        for i, phrase in enumerate(candidate_keyphrases):
            for other in candidate_keyphrases:
                if f" {phrase}" in other or f"{phrase} " in other:
                    phrase = other

            if phrase not in result:
                candidate_keyphrases[i] = phrase
                result.append(phrase)

        return result

    @staticmethod
    def subset_matrix_and_class_labels(
        cluster_label_vector: np.ndarray,
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
    ) -> Tuple[scipy.sparse.spmatrix, np.ndarray, np.ndarray]:
        # Mask out noise points, and then columns and rows that then have no entries
        count_matrix = object_x_keyphrase_matrix[cluster_label_vector >= 0, :]
        column_mask = np.squeeze(np.asarray(count_matrix.sum(axis=0))) > 0.0
        count_matrix = count_matrix[:, column_mask]
        column_map = np.arange(object_x_keyphrase_matrix.shape[1])[column_mask]
        row_mask = np.squeeze(np.asarray(count_matrix.sum(axis=1))) > 0.0
        count_matrix = count_matrix[row_mask, :]

        # Make a label vector contracted to the appropriate space
        class_labels = cluster_label_vector[cluster_label_vector >= 0][row_mask]

        return count_matrix, class_labels, column_map

    @staticmethod
    def information_weighted_keyphrases(
        cluster_label_vector: np.ndarray,
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_list: List[str],
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol],
        n_keyphrases: int = 16,
        prior_strength: float = 0.1,
        weight_power: float = 2.0,
        max_alpha: float = 1.0,
        min_alpha: float = 0.5,
        alpha_tolerance: float = 0.1,
        verbose: bool = False,
    ) -> List[List[str]]:
        """Generates a list of keyphrases for each cluster in a cluster layer.

        Parameters
        ----------
        cluster_label_vector : np.ndarray
            A vector of cluster labels for each object.
        object_x_keyphrase_matrix : scipy.sparse.spmatrix
            A sparse matrix of keyphrase counts for each object.
        keyphrase_list : List[str]
            A list of keyphrases in the same order as columns in object_x_keyphrase_matrix.
        keyphrase_vectors : np.ndarray
            An ndarray of keyphrase vectors in the same order as columns in object_x_keyphrase_matrix.
        embedding_model : TextEmbedderProtocol
            A text embedding model for embedding keyphrases.
        n_keyphrases : int, optional
            The number of keyphrases to generate for each cluster, by default 16.
        prior_strength : float, optional
            The strength of the prior for the information weighting, by default 0.1.
        weight_power : float, optional
            The power to raise the information weights to, by default 2.0.
        max_alpha : float, optional
            The alpha parameter for diversifying the keyphrase selection, by default 1.0.
        min_alpha : float, optional
            The minimum alpha parameter for diversifying the keyphrase selection, by default 0.5.
        alpha_tolerance : float, optional
            The tolerance for the alpha parameter when diversifying keyphrases, by default 0.1.
        show_progress_bar : bool, optional
            Whether to show a progress bar for the computation, by default False.

        Returns
        -------
        keyphrases List[List[str]]
            A list of lists of keyphrases for each cluster.
        """
        keyphrase_vector_mapping = {
            keyphrase: vector
            for keyphrase, vector in zip(keyphrase_list, keyphrase_vectors)
            if not np.all(vector == 0.0)
        }
        count_matrix, class_labels, column_map = TextKeyphraseExtractor.subset_matrix_and_class_labels(
            cluster_label_vector, object_x_keyphrase_matrix
        )

        iwt = InformationWeightTransformer(
            prior_strength=prior_strength, weight_power=weight_power
        ).fit(count_matrix, class_labels)
        count_matrix.data = np.log(count_matrix.data + 1)
        count_matrix.eliminate_zeros()
        weighted_matrix = iwt.transform(count_matrix)

        result = []
        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc="Generating informative keyphrases",
            disable=not verbose,
            leave=False,
            unit="cluster",
            position=1,
        ):
            # Sum over the cluster; get the top scoring indices
            contrastive_scores = np.squeeze(
                np.asarray(weighted_matrix[class_labels == cluster_num].sum(axis=0))
            )
            if sum(contrastive_scores) == 0:
                result.append(["No notable keyphrases"])
                continue

            keyphrases_present_indices = np.where(contrastive_scores > 0)[0]
            keyphrase_weights = contrastive_scores[keyphrases_present_indices]
            keyphrases_present = [
                keyphrase_list[column_map[j]] for j in keyphrases_present_indices
            ]
            # Update keyphrase mapping with present keyphrases it is missing
            missing_keyphrases = [
                keyphrase
                for keyphrase in keyphrases_present
                if keyphrase not in keyphrase_vector_mapping
            ]
            if len(missing_keyphrases) > 0:
                if embedding_model is None:
                    raise ValueError(
                        "On demand keyphrase vectorization was requested but no embedding model provided. Please provide an embedding model."
                    )
                missing_keyphrase_vectors = embedding_model.encode(
                    missing_keyphrases, show_progress_bar=False
                )
                for keyphrase, vector in zip(
                    missing_keyphrases, missing_keyphrase_vectors
                ):
                    keyphrase_vector_mapping[keyphrase] = vector

            # Compute the centroid of the keyphrases present in the cluster
            centroid_vector = np.average(
                [
                    keyphrase_vector_mapping[keyphrase]
                    for keyphrase in keyphrases_present
                ],
                weights=keyphrase_weights,
                axis=0,
            )

            chosen_indices = np.argsort(contrastive_scores)[
                -max((n_keyphrases * 4), 16) :
            ]

            # Map the indices back to the original vocabulary
            chosen_keyphrases = [
                keyphrase_list[column_map[j]]
                for j in reversed(chosen_indices)
                if j in keyphrases_present_indices
            ]

            # Extract the longest keyphrases, then diversify the selection
            chosen_keyphrases = TextKeyphraseExtractor.longest_keyphrases(chosen_keyphrases)
            chosen_vectors = np.asarray(
                [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
            )
            chosen_indices = diversify_max_alpha(
                centroid_vector,
                chosen_vectors,
                n_keyphrases,
                max_alpha=max_alpha,
                min_alpha=min_alpha,
                tolerance=alpha_tolerance,
            )[:n_keyphrases]
            chosen_keyphrases = [chosen_keyphrases[j] for j in chosen_indices]

            result.append(chosen_keyphrases)

        # Update keyphrase vectors with vectors from the mapping
        for i, keyphrase in enumerate(keyphrase_list):
            if keyphrase in keyphrase_vector_mapping:
                keyphrase_vectors[i] = keyphrase_vector_mapping[keyphrase]

        return result

    @staticmethod
    def central_keyphrases(
        cluster_label_vector: np.ndarray,
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_list: List[str],
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol],
        n_keyphrases: int = 16,
        diversify_alpha: float = 1.0,
        verbose: bool = False,
    ):
        """
        Generates a list of keyphrases for each cluster in a cluster layer using the central keyphrase method.

        Parameters
        ----------
        cluster_label_vector : np.ndarray
            A vector of cluster labels for each object.
        object_x_keyphrase_matrix : scipy.sparse.spmatrix
            A sparse matrix of keyphrase counts for each object.
        keyphrase_list : List[str]
            A list of keyphrases in the same order as columns in object_x_keyphrase_matrix.
        keyphrase_vectors : np.ndarray
            An ndarray of keyphrase vectors in the same order as columns in object_x_keyphrase_matrix.
        embedding_model : TextEmbedderProtocol
            A text embedding model for embedding keyphrases.
        n_keyphrases : int, optional
            The number of keyphrases to generate for each cluster, by default 16.
        diversify_alpha : float, optional
            The alpha parameter for diversifying the keyphrase selection, by default 1.0.
        show_progress_bar : bool, optional
            Whether to show a progress bar for the computation, by default False.

        Returns
        -------
        keyphrases : List[List[str]]
            A list of lists of keyphrases for each cluster.
        """
        keyphrase_vector_mapping = {
            keyphrase: vector
            for keyphrase, vector in zip(keyphrase_list, keyphrase_vectors)
            if not np.all(vector == 0.0)
        }

        count_matrix, class_labels, column_map = TextKeyphraseExtractor.subset_matrix_and_class_labels(
            cluster_label_vector, object_x_keyphrase_matrix
        )

        result = []
        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc="Generating central keyphrases",
            disable=not verbose,
            leave=False,
            unit="cluster",
            position=1,
        ):
            # Sum over the cluster; get the non-zero indices
            base_candidate_indices = np.where(
                np.squeeze(
                    np.asarray(count_matrix[class_labels == cluster_num].sum(axis=0))
                )
                > 0
            )[0]

            null_topic = np.mean(keyphrase_vectors, axis=0)

            # Map the indices back to the original vocabulary
            base_candidates = [
                keyphrase_list[column_map[j]] for j in base_candidate_indices
            ]
            # Update keyphrase mapping with present keyphrases it is missing
            missing_keyphrases = [
                keyphrase
                for keyphrase in base_candidates
                if keyphrase not in keyphrase_vector_mapping
            ]
            if len(missing_keyphrases) > 0:
                if embedding_model is None:
                    raise ValueError(
                        "On demand keyphrase vectorization was requested but no embedding model provided. Please provide an embedding model."
                    )
                missing_keyphrase_vectors = embedding_model.encode(
                    missing_keyphrases, show_progress_bar=False
                )
                for keyphrase, vector in zip(
                    missing_keyphrases, missing_keyphrase_vectors
                ):
                    keyphrase_vector_mapping[keyphrase] = vector

            base_vectors = (
                np.asarray(
                    [keyphrase_vector_mapping[phrase] for phrase in base_candidates]
                )
                - null_topic
            )
            base_weights = np.squeeze(
                np.asarray(count_matrix[class_labels == cluster_num].sum(axis=0))
            )[base_candidate_indices]
            centroid = np.average(base_vectors, axis=0, weights=base_weights)

            # Select the central keyphrases as the closest samples to the centroid
            base_distances = pairwise_distances(
                centroid.reshape(1, -1), base_vectors, metric="cosine"
            )
            base_order = np.argsort(base_distances.flatten())

            chosen_keyphrases = [
                base_candidates[i] for i in base_order[: n_keyphrases**2]
            ]

            # Extract the longest keyphrases, then diversify the selection
            chosen_keyphrases = TextKeyphraseExtractor.longest_keyphrases(chosen_keyphrases)
            chosen_vectors = np.asarray(
                [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
            )
            chosen_indices = diversify_max_alpha(
                centroid,
                chosen_vectors,
                n_keyphrases,
                max_alpha=diversify_alpha,
            )[:n_keyphrases]
            chosen_keyphrases = [chosen_keyphrases[j] for j in chosen_indices]

            result.append(chosen_keyphrases)

        # Update keyphrase vectors with vectors from the mapping
        for i, keyphrase in enumerate(keyphrase_list):
            if keyphrase in keyphrase_vector_mapping:
                keyphrase_vectors[i] = keyphrase_vector_mapping[keyphrase]

        return result

    @staticmethod
    def bm25_keyphrases(
        cluster_label_vector: np.ndarray,
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_list: List[str],
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol],
        n_keyphrases: int = 16,
        k1: float = 1.5,
        b: float = 0.75,
        diversify_alpha: float = 1.0,
        verbose: bool = False,
    ) -> List[List[str]]:
        """Generates a list of keyphrases for each cluster in a cluster layer using BM25 for scoring.

        Parameters
        ----------
        cluster_label_vector : np.ndarray
            A vector of cluster labels for each object.
        object_x_keyphrase_matrix : scipy.sparse.spmatrix
            A sparse matrix of keyphrase counts for each object.
        keyphrase_list : List[str]
            A list of keyphrases in the same order as columns in object_x_keyphrase_matrix.
        keyphrase_vectors : np.ndarray
            An ndarray of keyphrase vectors in the same order as columns in object_x_keyphrase_matrix.
        embedding_model : TextEmbedderProtocol
            A text embedding model for embedding keyphrases.
        n_keyphrases : int, optional
            The number of keyphrases to generate for each cluster, by default 16.
        k1 : float, optional
            The k1 parameter for BM25, by default 1.5.
        b : float, optional
            The b parameter for BM25, by default 0.75.
        diversify_alpha : float, optional
            The alpha parameter for diversifying the keyphrase selection, by default 1.0.

        Returns
        -------
        keyphrases : List[List[str]]
            A list of lists of keyphrases for each cluster.
        """
        keyphrase_vector_mapping = {
            keyphrase: vector
            for keyphrase, vector in zip(keyphrase_list, keyphrase_vectors)
            if not np.all(vector == 0.0)
        }

        count_matrix, class_labels, column_map = TextKeyphraseExtractor.subset_matrix_and_class_labels(
            cluster_label_vector, object_x_keyphrase_matrix
        )

        # Build a class based count matrix
        groupby_matrix = scipy.sparse.csr_matrix(
            (
                np.ones(class_labels.shape[0]),
                (class_labels, np.arange(class_labels.shape[0])),
            ),
            shape=(class_labels.max() + 1, class_labels.shape[0]),
        )
        class_count_matrix = groupby_matrix @ count_matrix

        # Compute BM25 scores for every entry in the matrix
        N = class_count_matrix.shape[0]
        df = (class_count_matrix > 0).sum(axis=0)
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5))

        doc_lengths = count_matrix.sum(axis=1)
        avg_doc_length = doc_lengths.mean()

        for i in range(class_count_matrix.shape[0]):
            tf_array = class_count_matrix.data[
                class_count_matrix.indptr[i] : class_count_matrix.indptr[i + 1]
            ]
            tf_score = tf_array / (
                k1 * ((1 - b) + b * doc_lengths[i] / avg_doc_length) + tf_array
            )
            class_count_matrix.data[
                class_count_matrix.indptr[i] : class_count_matrix.indptr[i + 1]
            ] = tf_score

        bm25_matrix = class_count_matrix.multiply(idf).tocsr()

        # Select the top scoring keyphrases for each cluster based on BM25 scores for the cluster
        result = []
        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc="Generating bm25 keyphrases",
            disable=not verbose,
            leave=False,
            unit="meta-cluster",
            position=1,
        ):
            # Sum over the cluster; get the top scoring indices
            contrastive_scores = bm25_matrix[cluster_num].toarray().flatten()
            if sum(contrastive_scores) == 0:
                result.append(["No notable keyphrases"])
                continue

            keyphrases_present_indices = np.where(contrastive_scores > 0)[0]
            keyphrase_weights = contrastive_scores[keyphrases_present_indices]
            keyphrases_present = [
                keyphrase_list[column_map[j]] for j in keyphrases_present_indices
            ]
            # Update keyphrase mapping with present keyphrases it is missing
            missing_keyphrases = [
                keyphrase
                for keyphrase in keyphrases_present
                if keyphrase not in keyphrase_vector_mapping
            ]
            if len(missing_keyphrases) > 0:
                if embedding_model is None:
                    raise ValueError(
                        "On demand keyphrase vectorization was requested but no embedding model provided. Please provide an embedding model."
                    )
                missing_keyphrase_vectors = embedding_model.encode(
                    missing_keyphrases, show_progress_bar=False
                )
                for keyphrase, vector in zip(
                    missing_keyphrases, missing_keyphrase_vectors
                ):
                    keyphrase_vector_mapping[keyphrase] = vector

            # Compute the centroid of the keyphrases present in the cluster
            centroid_vector = np.average(
                [
                    keyphrase_vector_mapping[keyphrase]
                    for keyphrase in keyphrases_present
                ],
                weights=keyphrase_weights,
                axis=0,
            )

            chosen_indices = np.argsort(contrastive_scores)[-(n_keyphrases**2) :]

            # Map the indices back to the original vocabulary
            chosen_keyphrases = [
                keyphrase_list[column_map[j]]
                for j in reversed(chosen_indices)
                if j in keyphrases_present_indices
            ]

            # Extract the longest keyphrases, then diversify the selection
            chosen_keyphrases = TextKeyphraseExtractor.longest_keyphrases(chosen_keyphrases)
            chosen_vectors = np.asarray(
                [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
            )
            chosen_indices = diversify_max_alpha(
                centroid_vector,
                chosen_vectors,
                n_keyphrases,
                max_alpha=diversify_alpha,
            )[:n_keyphrases]
            chosen_keyphrases = [chosen_keyphrases[j] for j in chosen_indices]

            result.append(chosen_keyphrases)

        # Update keyphrase vectors with vectors from the mapping
        for i, keyphrase in enumerate(keyphrase_list):
            if keyphrase in keyphrase_vector_mapping:
                keyphrase_vectors[i] = keyphrase_vector_mapping[keyphrase]

        return result

    @staticmethod
    def submodular_selection_information_keyphrases(
        cluster_label_vector: np.ndarray,
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_list: List[str],
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol],
        n_keyphrases: int = 16,
        prior_strength: float = 0.01,
        weight_power: float = 2.0,
        submodular_function: str = "saturated_coverage",
        verbose: bool = False,
    ) -> List[List[str]]:
        """Generates a list of keyphrases for each cluster in a cluster layer using saturated coverage information.

        Parameters
        ----------
        cluster_label_vector : np.ndarray
            A vector of cluster labels for each object.
        object_x_keyphrase_matrix : scipy.sparse.spmatrix
            A sparse matrix of keyphrase counts for each object.
        keyphrase_list : List[str]
            A list of keyphrases in the same order as columns in object_x_keyphrase_matrix.
        keyphrase_vectors : np.ndarray
            An ndarray of keyphrase vectors in the same order as columns in object_x_keyphrase_matrix.
        embedding_model : TextEmbedderProtocol
            A text embedding model for embedding keyphrases.
        n_keyphrases : int, optional
            The number of keyphrases to generate for each cluster, by default 16.
        prior_strength : float, optional
            The strength of the prior for the information weighting, by default 0.1.
        weight_power : float, optional
            The power to raise the information weights to, by default 2.0.
        submodular_function : str, optional
            The submodular function to use for keyphrase selection, by default "saturated_coverage".
            Can be one of "facility_location", "saturated_coverage", or "graph_cut".
        show_progress_bar : bool, optional
            Whether to show a progress bar for the computation, by default False.

        Returns
        -------
        keyphrases : List[List[str]]
            A list of lists of keyphrases for each cluster.
        """
        keyphrase_vector_mapping = {
            keyphrase: vector
            for keyphrase, vector in zip(keyphrase_list, keyphrase_vectors)
            if not np.all(vector == 0.0)
        }
        count_matrix, class_labels, column_map = TextKeyphraseExtractor.subset_matrix_and_class_labels(
            cluster_label_vector, object_x_keyphrase_matrix
        )
        central_vector = keyphrase_vectors.mean(axis=0)

        iwt = InformationWeightTransformer(
            prior_strength=prior_strength, weight_power=weight_power
        ).fit(count_matrix, class_labels)
        count_matrix.data = np.log(count_matrix.data + 1)
        count_matrix.eliminate_zeros()
        weighted_matrix = iwt.transform(count_matrix)
        if submodular_function == "facility_location":
            selector = FacilityLocationSelection(
                n_keyphrases, metric="cosine", optimizer="lazy"
            )
        elif submodular_function == "graph_cut":
            selector = GraphCutSelection(
                n_keyphrases, metric="cosine", optimizer="lazy"
            )
        elif submodular_function == "saturated_coverage":
            selector = SaturatedCoverageSelection(
                n_keyphrases, metric="cosine", optimizer="lazy"
            )
        else:
            raise ValueError(
                f"Unknown submodular function {submodular_function}. Must be one of 'facility_location', 'saturated_coverage', or 'graph_cut'."
            )

        result = []
        for cluster_num in tqdm(
            range(cluster_label_vector.max() + 1),
            desc="Generating saturated coverage keyphrases",
            disable=not verbose,
            leave=False,
            unit="cluster",
            position=1,
        ):
            # Sum over the cluster; get the top scoring indices
            contrastive_scores = np.squeeze(
                np.asarray(weighted_matrix[class_labels == cluster_num].sum(axis=0))
            )
            if sum(contrastive_scores) == 0:
                result.append(["No notable keyphrases"])
                continue

            keyphrases_present_indices = np.where(contrastive_scores > 0)[0]
            keyphrase_weights = contrastive_scores[keyphrases_present_indices]
            candidate_keyphrases = np.asarray(
                [keyphrase_list[column_map[j]] for j in keyphrases_present_indices]
            )
            # Update keyphrase mapping with present keyphrases it is missing
            missing_keyphrases = [
                keyphrase
                for keyphrase in candidate_keyphrases
                if keyphrase not in keyphrase_vector_mapping
            ]
            if len(missing_keyphrases) > 0:
                if embedding_model is None:
                    raise ValueError(
                        "On demand keyphrase vectorization was requested but no embedding model provided. Please provide an embedding model."
                    )
                missing_keyphrase_vectors = embedding_model.encode(
                    missing_keyphrases, show_progress_bar=False
                )
                for keyphrase, vector in zip(
                    missing_keyphrases, missing_keyphrase_vectors
                ):
                    keyphrase_vector_mapping[keyphrase] = vector

            if len(candidate_keyphrases) >= n_keyphrases:
                keyphrase_costs = 1.0 - (
                    keyphrase_weights / (0.01 + keyphrase_weights.max())
                )
                candidate_vectors = (
                    np.asarray(
                        [
                            keyphrase_vector_mapping[phrase]
                            for phrase in candidate_keyphrases
                        ]
                    )
                    - central_vector
                )

                _, chosen_keyphrases = selector.fit_transform(
                    X=candidate_vectors,
                    y=candidate_keyphrases,
                    sample_cost=keyphrase_costs,
                )
                chosen_keyphrases = chosen_keyphrases[:n_keyphrases]
            else:
                # If there are not enough keyphrases, just take the top n_keyphrases
                chosen_indices = np.argsort(contrastive_scores[contrastive_scores > 0])
                chosen_keyphrases = [
                    candidate_keyphrases[j] for j in reversed(chosen_indices)
                ]

            result.append(chosen_keyphrases)

        # Update keyphrase vectors with vectors from the mapping
        for i, keyphrase in enumerate(keyphrase_list):
            if keyphrase in keyphrase_vector_mapping:
                keyphrase_vectors[i] = keyphrase_vector_mapping[keyphrase]

        return result
