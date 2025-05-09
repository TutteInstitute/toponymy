import numpy as np

from typing import List, Tuple, FrozenSet, Dict, Callable, Any, Optional, Protocol
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from joblib import Parallel, delayed, effective_n_jobs
from functools import reduce
from collections import Counter
from toponymy.utility_functions import diversify_max_alpha as diversify
from vectorizers.transformers import InformationWeightTransformer
from sklearn.metrics import pairwise_distances

from sentence_transformers import SentenceTransformer

import scipy.sparse
import numba

from tqdm.auto import tqdm

Ngrammer = Callable[[str], List[str]]

from typing import Union, overload, TypeVar, cast


# Define a protocol for objects that behave like Tokenizers
class TokenizerLike(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> Any: ...

    def decode(self, ids: Any, *args: Any, **kwargs: Any) -> str: ...


def create_tokenizers_ngrammer(
    tokenizer: TokenizerLike, ngram_range: Tuple[int, int] = (1, 4)
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


def count_docs_ngrams(
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


def combine_dicts(
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
        result = {key: value for key, value in result.items() if value >= trim_value}
    return result


def _combine_tree_layer(
    dict_list: List[Dict[str, int]], max_ngrams: int = 250_000
) -> List[Dict[str, int]]:
    result = []
    for i in range(0, len(dict_list) - 1, 2):
        result.append(
            combine_dicts(dict_list[i], dict_list[i + 1], max_ngrams=max_ngrams)
        )
    if len(dict_list) % 2 == 1:
        result.append(dict_list[-1])
    return result


def tree_combine_dicts(
    dict_list: List[Dict[str, int]], max_ngrams: int = 250_000
) -> Dict[str, int]:
    while len(dict_list) > 1:
        dict_list = _combine_tree_layer(dict_list, max_ngrams=max_ngrams)
    return dict_list[0]


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


def build_keyphrase_vocabulary(
    objects: List[str],
    ngrammer: Ngrammer,
    max_features: int = 50_000,
    min_occurrences: int = 1,
    stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
    n_jobs: int = -1,
    verbose: bool = False,
) -> List[str]:
    """
    Builds a keyphrase vocabulary from a list of objects.

    Parameters
    ----------
    objects : List[str]
        A list of objects; for use in building avocabulary this should be string representations of the objects.
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
    verbose : bool, optional
        Whether to print out progress information, by default False.

    Returns
    -------
    List[str]
        A keyphrase list of the most commonly occurring keyphrases.
    """
    # count ngrams in parallel with joblib
    n_chunks = effective_n_jobs(n_jobs)
    chunk_size = max((len(objects) // n_chunks) + 1, 20_000)
    n_chunks = len(objects) // chunk_size + 1
    if verbose:
        print(
            f"Chunking into {n_chunks} chunks of size {chunk_size} for keyphrase identification."
        )
    chunked_count_dicts = Parallel(n_jobs=n_chunks)(
        delayed(count_docs_ngrams)(
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
    all_vocab_counts = tree_combine_dicts(
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
    objects: List[str],
    keyphrases: Dict[str, int],
    ngrammer: Ngrammer,
    n_jobs: int = -1,
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
    verbose : bool, optional
        Whether to print out progress information, by default False.


    Returns
    -------
    scipy.sparse.spmatrix
        A sparse count matrix of keyphrases in the objects.
    """
    # count ngrams in parallel with joblib
    n_chunks = effective_n_jobs(n_jobs)
    chunk_size = max((len(objects) // n_chunks) + 1, 20_000)
    n_chunks = (len(objects) // chunk_size) + 1
    if verbose:
        print(
            f"Chunking into {n_chunks} chunks of size {chunk_size} for keyphrase count construction."
        )
    chunked_count_matrices = Parallel(n_jobs=n_chunks)(
        delayed(build_count_matrix)(objects[i : i + chunk_size], keyphrases, ngrammer)
        for i in range(0, len(objects), chunk_size)
    )
    if verbose:
        print("Combining count matrix chunks ...")

    # stack the count matrices
    result = scipy.sparse.vstack(chunked_count_matrices)

    return result


def build_object_x_keyphrase_matrix(
    objects: List[str],
    ngram_range: Tuple[int, int] = (1, 4),
    tokenizer: Optional[TokenizerLike] = None,
    token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
    max_features: int = 50_000,
    min_occurrences: int = 1,
    stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
    n_jobs: int = -1,
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
    verbose : bool, optional
        Whether to print out progress information, by default False.

    Returns
    -------
    scipy.sparse.spmatrix
        A sparse count matrix of keyphrases in the objects.
    """
    if tokenizer is None:
        # use Countvectorizer to build an ngram analyzer to ensure compatiability
        cv = CountVectorizer(
            lowercase=True,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
        )
        ngrammer = cv.build_analyzer()
    else:
        ngrammer = create_tokenizers_ngrammer(tokenizer, ngram_range=ngram_range)

    keyphrases = build_keyphrase_vocabulary(
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
    result = build_keyphrase_count_matrix(
        objects,
        keyphrase_dict,
        ngrammer=ngrammer,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return result, keyphrases


class KeyphraseBuilder:
    """
    A class for building keyphrase count matrices from a list of objects. This can be useful
    as keyphrases can be a more specific way of helping prompt an LLM for a topic name. To
    make use of keyphrases you need to be able to convert objects to text. For basic
    short-text topic modeling, you can use the default settings, which simply assumes
    objects are already short texts. For other kinds of topic modeling you may
    need to provide a function that converts objects to text.

    Parameters
    ----------
    object_to_text : Optional[Callable[[Any], str]], optional
        A function that converts objects to text, by default None. If None, it is assumed that the objects are strings.
        An example of another case would be if objects were images and this function was a
        zero-short image captioning model.

    ngram_range : Tuple[int, int], optional
        The range of n-grams to consider, by default (1, 4).

    tokenizer : Optional[TokenizerLike], optional
        A tokenizer object that has encode and decode methods, by default None. If None, a CountVectorizer is used.

    token_pattern : str, optional
        The regular expression pattern to use for tokenization, by default "(?u)\\b\\w[-'\\w]+\\b".

    max_features : int, optional
        The maximum number of features to consider, by default 50_000.

    min_occurrences : int, optional
        The minimum number of occurrences for a keyphrase to be included, by default 2, so keyphrases have to re-occur.

    stop_words : FrozenSet[str], optional
        The set of stop words to use, by default sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.

    n_jobs : int, optional
        The number of jobs to use in parallel processing, by default -1. If -1, all available cores are used.

    Attributes
    ----------

    object_x_keyphrase_matrix_ : scipy.sparse.spmatrix
        A sparse count matrix of keyphrases in the objects.

    keyphrase_list_ : List[str]
        A list of keyphrases in the same order as columns in object_x_keyphrase_matrix.

    """

    def __init__(
        self,
        object_to_text: Optional[Callable[[Any], str]] = None,
        ngram_range: Tuple[int, int] = (1, 4),
        tokenizer: Optional[TokenizerLike] = None,
        token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
        max_features: int = 50_000,
        min_occurrences: int = 2,
        stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
        n_jobs: int = -1,
        embedder: Optional[SentenceTransformer] = None,
        verbose: bool = False,
    ):
        self.object_to_text = object_to_text
        self.ngram_range = ngram_range
        self.tokenizer = tokenizer
        self.token_pattern = token_pattern
        self.max_features = max_features
        self.min_occurrences = min_occurrences
        self.stop_words = stop_words
        self.n_jobs = n_jobs
        self.embedder = embedder
        self.verbose = verbose

    def fit(self, objects: List[Any]):
        if self.object_to_text is None:
            object_texts = objects
        else:
            object_texts = [self.object_to_text(obj) for obj in objects]

        if self.verbose:
            print("Building keyphrase matrix ... ")

        self.object_x_keyphrase_matrix_, self.keyphrase_list_ = (
            build_object_x_keyphrase_matrix(
                object_texts,
                ngram_range=self.ngram_range,
                tokenizer=self.tokenizer,
                token_pattern=self.token_pattern,
                max_features=self.max_features,
                min_occurrences=self.min_occurrences,
                stop_words=self.stop_words,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        )
        
        if self.verbose and self.embedder is not None:
            print("Building keyphrase vectors ... ")
            self.keyphrase_vectors_ = self.embedder.encode(
                self.keyphrase_list_, show_progress_bar=self.verbose,
            )
        else:
            self.keyphrase_vectors_ = None

        return self

    def fit_transform(self, objects: List[Any]) -> Tuple[scipy.sparse.spmatrix, List[str], Optional[np.ndarray]]:
        """
        Fits the KeyphraseBuilder to the objects and returns the object x keyphrase matrix, keyphrase list, and keyphrase vectors.
        """
        self.fit(objects)
        return self.object_x_keyphrase_matrix_, self.keyphrase_list_, self.keyphrase_vectors_


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


def information_weighted_keyphrases(
    cluster_label_vector: np.ndarray,
    object_x_keyphrase_matrix: scipy.sparse.spmatrix,
    keyphrase_list: List[str],
    keyphrase_vectors: np.ndarray,
    embedding_model: SentenceTransformer,
    n_keyphrases: int = 16,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    diversify_alpha: float = 1.0,
    show_progress_bar: bool = False,
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
    embedding_model : SentenceTransformer
        A SentenceTransformer model for embedding keyphrases.
    n_keyphrases : int, optional
        The number of keyphrases to generate for each cluster, by default 16.
    prior_strength : float, optional
        The strength of the prior for the information weighting, by default 0.1.
    weight_power : float, optional
        The power to raise the information weights to, by default 2.0.
    diversify_alpha : float, optional
        The alpha parameter for diversifying the keyphrase selection, by default 1.0.
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
    count_matrix, class_labels, column_map = subset_matrix_and_class_labels(
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
        disable=not show_progress_bar,
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
            for keyphrase, vector in zip(missing_keyphrases, missing_keyphrase_vectors):
                keyphrase_vector_mapping[keyphrase] = vector

        # Compute the centroid of the keyphrases present in the cluster
        centroid_vector = np.average(
            [keyphrase_vector_mapping[keyphrase] for keyphrase in keyphrases_present],
            weights=keyphrase_weights,
            axis=0,
        )

        chosen_indices = np.argsort(contrastive_scores)[-max((n_keyphrases * 4), 16) :]

        # Map the indices back to the original vocabulary
        chosen_keyphrases = [
            keyphrase_list[column_map[j]]
            for j in reversed(chosen_indices)
            if j in keyphrases_present_indices
        ]

        # Extract the longest keyphrases, then diversify the selection
        chosen_keyphrases = longest_keyphrases(chosen_keyphrases)
        chosen_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
        )
        chosen_indices = diversify(
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


def central_keyphrases(
    cluster_label_vector: np.ndarray,
    object_x_keyphrase_matrix: scipy.sparse.spmatrix,
    keyphrase_list: List[str],
    keyphrase_vectors: np.ndarray,
    embedding_model: SentenceTransformer,
    n_keyphrases: int = 16,
    diversify_alpha: float = 1.0,
    show_progress_bar: bool = False,
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
    embedding_model : SentenceTransformer
        A SentenceTransformer model for embedding keyphrases.
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

    count_matrix, class_labels, column_map = subset_matrix_and_class_labels(
        cluster_label_vector, object_x_keyphrase_matrix
    )

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Generating central keyphrases",
        disable=not show_progress_bar,
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
            for keyphrase, vector in zip(missing_keyphrases, missing_keyphrase_vectors):
                keyphrase_vector_mapping[keyphrase] = vector

        base_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in base_candidates]
        )
        centroid = np.average(base_vectors, axis=0)

        # Select the central keyphrases as the closest samples to the centroid
        base_distances = pairwise_distances(
            centroid.reshape(1, -1), base_vectors, metric="cosine"
        )
        base_order = np.argsort(base_distances.flatten())

        chosen_keyphrases = [base_candidates[i] for i in base_order[: n_keyphrases**2]]

        # Extract the longest keyphrases, then diversify the selection
        chosen_keyphrases = longest_keyphrases(chosen_keyphrases)
        chosen_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
        )
        chosen_indices = diversify(
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


def bm25_keyphrases(
    cluster_label_vector: np.ndarray,
    object_x_keyphrase_matrix: scipy.sparse.spmatrix,
    keyphrase_list: List[str],
    keyphrase_vectors: np.ndarray,
    embedding_model: SentenceTransformer,
    n_keyphrases: int = 16,
    k1: float = 1.5,
    b: float = 0.75,
    diversify_alpha: float = 1.0,
    show_progress_bar: bool = False,
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
    embedding_model : SentenceTransformer
        A SentenceTransformer model for embedding keyphrases.
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

    count_matrix, class_labels, column_map = subset_matrix_and_class_labels(
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
        disable=not show_progress_bar,
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
            for keyphrase, vector in zip(missing_keyphrases, missing_keyphrase_vectors):
                keyphrase_vector_mapping[keyphrase] = vector

        # Compute the centroid of the keyphrases present in the cluster
        centroid_vector = np.average(
            [keyphrase_vector_mapping[keyphrase] for keyphrase in keyphrases_present],
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
        chosen_keyphrases = longest_keyphrases(chosen_keyphrases)
        chosen_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
        )
        chosen_indices = diversify(
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
