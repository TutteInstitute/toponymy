import numpy as np

from typing import List, Tuple, FrozenSet, Dict, Callable
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from joblib import Parallel, delayed, effective_n_jobs, cpu_count
from functools import reduce
from collections import Counter
from toponymy.utility_functions import diversify_max_alpha as diversify
from vectorizers.transformers import InformationWeightTransformer
from sklearn.metrics import pairwise_distances

import scipy.sparse


def count_docs_ngrams(
    docs: List[str], ngrammer, stop_words: FrozenSet[str]
) -> Dict[str, int]:
    result = {}
    for doc in docs:
        for gram in ngrammer(doc):
            split_gram = gram.split()
            if split_gram[0] in stop_words or split_gram[-1] in stop_words:
                continue
            if gram in result:
                result[gram] += 1
            else:
                result[gram] = 1

    return result


def combine_dicts(dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, int]:
    result = dict1
    for key in dict2:
        if key in result:
            result[key] += dict2[key]
        else:
            result[key] = dict2[key]
    return result


def _combine_tree_layer(
    dict_list: List[Dict[str, int]], n_jobs: int = -1
) -> List[Dict[str, int]]:
    result = []
    for i in range(0, len(dict_list) - 1, 2):
        result.append(combine_dicts(dict_list[i], dict_list[i + 1]))
    if len(dict_list) % 2 == 1:
        result.append(dict_list[-1])
    return result


def tree_combine_dicts(dict_list: List[Dict[str, int]]) -> Dict[str, int]:
    while len(dict_list) > 1:
        dict_list = _combine_tree_layer(dict_list)
    return dict_list[0]


def build_count_matrix(
    docs: List[str], vocab: Dict[str, int], ngrammer: Callable[[str], List[str]]
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
    ngram_range: Tuple[int, int] = (1, 4),
    token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
    max_features: int = 50_000,
    stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
    n_jobs: int = -1,
) -> List[str]:
    """
    Builds a keyphrase vocabulary from a list of objects.

    Parameters
    ----------
    objects : List[str]
        A list of objects; for use in building avocabulary this should be string representations of the objects.
    ngram_range : Tuple[int, int], optional
        The range of n-grams to consider, by default (1, 4).
    token_pattern : str, optional
        The regular expression pattern to use for tokenization, by default "(?u)\\b\\w[-'\\w]+\\b".
    max_features : int, optional
        The maximum number of features to consider, by default 50_000.
    stop_words : FrozenSet[str], optional
        The set of stop words to use, by default sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.

    Returns
    -------
    List[str]
        A keyphrase list of the most commonly occurring keyphrases.
    """
    # use Countvectorizer to build an ngram analyzer to ensure compatiability
    cv = CountVectorizer(
        lowercase=True,
        token_pattern=token_pattern,
        ngram_range=ngram_range,
    )
    ngrammer = cv.build_analyzer()

    # count ngrams in parallel with joblib
    n_chunks = effective_n_jobs(n_jobs)
    chunk_size = (len(objects) // n_chunks) + 1
    chunked_count_dicts = Parallel(n_jobs=n_chunks)(
        delayed(count_docs_ngrams)(objects[i : i + chunk_size], ngrammer, stop_words)
        for i in range(0, len(objects), chunk_size)
    )

    # Combine dictionaries and count the most common ngrams
    # all_vocab_counts = reduce(combine_dicts, chunked_count_dicts, {})
    all_vocab_counts = tree_combine_dicts(chunked_count_dicts)
    vocab_counter = Counter(all_vocab_counts)
    result = [ngram for ngram, _ in vocab_counter.most_common(max_features)]

    return result


def build_keyphrase_count_matrix(
    objects: List[str],
    keyphrases: Dict[str, int],
    ngram_range: Tuple[int, int] = (1, 4),
    token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
    n_jobs: int = -1,
) -> scipy.sparse.spmatrix:
    """
    Builds a count matrix of keyphrases in a list of objects.

    Parameters
    ----------
    objects : List[str]
        A list of objects; for use in building a count matrix this should be string representations of the objects.
    keyphrases : Dict[str, int]
        A dictionary where keys are keyphrases to count in the objects and values are their respective indices.


    Returns
    -------
    scipy.sparse.spmatrix
        A sparse count matrix of keyphrases in the objects.
    """
    # use Countvectorizer to build an ngram analyzer to ensure compatiability
    cv = CountVectorizer(
        lowercase=True,
        token_pattern=token_pattern,
        ngram_range=ngram_range,
    )
    ngrammer = cv.build_analyzer()

    # count ngrams in parallel with joblib
    n_chunks = effective_n_jobs(n_jobs)
    chunk_size = max((len(objects) // n_chunks) + 1, 10_000)
    n_chunks = len(objects) // chunk_size + 1
    chunked_count_matrices = Parallel(n_jobs=n_chunks)(
        delayed(build_count_matrix)(objects[i : i + chunk_size], keyphrases, ngrammer)
        for i in range(0, len(objects), chunk_size)
    )

    # stack the count matrices
    result = scipy.sparse.vstack(chunked_count_matrices)

    return result


def build_object_x_keyphrase_matrix(
    objects: List[str],
    ngram_range: Tuple[int, int] = (1, 4),
    token_pattern: str = "(?u)\\b\\w[-'\\w]+\\b",
    max_features: int = 50_000,
    stop_words: FrozenSet[str] = ENGLISH_STOP_WORDS,
    n_jobs: int = -1,
) -> scipy.sparse.spmatrix:
    """
    Builds a count matrix of keyphrases in a list of objects.

    Parameters
    ----------
    objects : List[str]
        A list of objects; for use in building a count matrix this should be string representations of the objects.
    ngram_range : Tuple[int, int], optional
        The range of n-grams to consider, by default (1, 4).
    token_pattern : str, optional
        The regular expression pattern to use for tokenization, by default "(?u)\\b\\w[-'\\w]+\\b".
    max_features : int, optional
        The maximum number of features to consider, by default 50_000.
    stop_words : FrozenSet[str], optional
        The set of stop words to use, by default sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.


    Returns
    -------
    scipy.sparse.spmatrix
        A sparse count matrix of keyphrases in the objects.
    """
    keyphrases = build_keyphrase_vocabulary(
        objects,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        max_features=max_features,
        stop_words=stop_words,
        n_jobs=n_jobs,
    )

    keyphrase_dict = {keyphrase: i for i, keyphrase in enumerate(keyphrases)}
    result = build_keyphrase_count_matrix(
        objects,
        keyphrase_dict,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        n_jobs=n_jobs,
    )

    return result, keyphrases


def longest_keyphrases(candidate_keyphrases):
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
    centroid_vectors: np.ndarray,
    n_keyphrases: int = 16,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    diversify_alpha: float = 1.0,
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
    centroid_vectors : np.ndarray
        An ndarray of centroid vectors for each cluster.
    n_keyphrases : int, optional
        The number of keyphrases to generate for each cluster, by default 16.
    prior_strength : float, optional
        The strength of the prior for the information weighting, by default 0.1.
    weight_power : float, optional
        The power to raise the information weights to, by default 2.0.
    diversify_alpha : float, optional
        The alpha parameter for diversifying the keyphrase selection, by default 1.0.

    Returns
    -------
    keyphrases List[List[str]]
        A list of lists of keyphrases for each cluster.
    """
    keyphrase_vector_mapping = {
        keyphrase: vector
        for keyphrase, vector in zip(keyphrase_list, keyphrase_vectors)
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
    for cluster_num in range(cluster_label_vector.max() + 1):
        # Sum over the cluster; get the top scoring indices
        contrastive_scores = np.squeeze(
            np.asarray(weighted_matrix[class_labels == cluster_num].sum(axis=0))
        )
        if sum(contrastive_scores) == 0:
            result.append(["No notable keyphrases"])
            continue

        chosen_indices = np.argsort(contrastive_scores)[-n_keyphrases ** 2 :]

        # Map the indices back to the original vocabulary
        chosen_keyphrases = [
            keyphrase_list[column_map[j]] for j in reversed(chosen_indices)
        ]

        # Extract the longest keyphrases, then diversify the selection
        chosen_keyphrases = longest_keyphrases(chosen_keyphrases)
        chosen_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
        )
        chosen_indices = diversify(
            centroid_vectors[cluster_num], chosen_vectors, n_keyphrases, max_alpha=diversify_alpha
        )[:n_keyphrases]
        chosen_keyphrases = [chosen_keyphrases[j] for j in chosen_indices]

        result.append(chosen_keyphrases)

    return result


def central_keyphrases(
    cluster_label_vector: np.ndarray,
    object_x_keyphrase_matrix: scipy.sparse.spmatrix,
    keyphrase_list: List[str],
    keyphrase_vectors: np.ndarray,
    centroid_vectors: np.ndarray,
    n_keyphrases: int = 16,
    diversify_alpha: float = 1.0,
):
    keyphrase_vector_mapping = {
        keyphrase: vector
        for keyphrase, vector in zip(keyphrase_list, keyphrase_vectors)
    }

    count_matrix, class_labels, column_map = subset_matrix_and_class_labels(
        cluster_label_vector, object_x_keyphrase_matrix
    )

    result = []
    for cluster_num in range(cluster_label_vector.max() + 1):
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
        base_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in base_candidates]
        )

        # Select the central keyphrases as the closest samples to the centroid
        base_distances = pairwise_distances(
            centroid_vectors[cluster_num].reshape(1, -1), base_vectors, metric="cosine"
        )
        base_order = np.argsort(base_distances.flatten())

        chosen_keyphrases = [base_candidates[i] for i in base_order[: n_keyphrases ** 2]]

        # Extract the longest keyphrases, then diversify the selection
        chosen_keyphrases = longest_keyphrases(chosen_keyphrases)
        chosen_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
        )
        chosen_indices = diversify(
            centroid_vectors[cluster_num], chosen_vectors, n_keyphrases, max_alpha=diversify_alpha
        )[:n_keyphrases]
        chosen_keyphrases = [chosen_keyphrases[j] for j in chosen_indices]

        result.append(chosen_keyphrases)

    return result


def bm25_keyphrases(
    cluster_label_vector: np.ndarray,
    object_x_keyphrase_matrix: scipy.sparse.spmatrix,
    keyphrase_list: List[str],
    keyphrase_vectors: np.ndarray,
    centroid_vectors: np.ndarray,
    n_keyphrases: int = 16,
    k1: float = 1.5,
    b: float = 0.75,
    diversify_alpha: float = 1.0,
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
    centroid_vectors : np.ndarray
        An ndarray of centroid vectors for each cluster.
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
        tf_array = class_count_matrix.data[class_count_matrix.indptr[i] : class_count_matrix.indptr[i + 1]]
        tf_score = tf_array / (k1 * ((1 - b) + b * doc_lengths[i] / avg_doc_length) + tf_array)
        class_count_matrix.data[class_count_matrix.indptr[i] : class_count_matrix.indptr[i + 1]] = tf_score

    bm25_matrix = class_count_matrix.multiply(idf).tocsr()

    # Select the top scoring keyphrases for each cluster based on BM25 scores for the cluster
    result = []
    for cluster_num in range(cluster_label_vector.max() + 1):
        # Sum over the cluster; get the top scoring indices
        contrastive_scores = bm25_matrix[cluster_num].toarray().flatten()
        if sum(contrastive_scores) == 0:
            result.append(["No notable keyphrases"])
            continue

        chosen_indices = np.argsort(contrastive_scores)[-n_keyphrases ** 2 :]

        # Map the indices back to the original vocabulary
        chosen_keyphrases = [
            keyphrase_list[column_map[j]] for j in reversed(chosen_indices)
        ]

        # Extract the longest keyphrases, then diversify the selection
        chosen_keyphrases = longest_keyphrases(chosen_keyphrases)
        chosen_vectors = np.asarray(
            [keyphrase_vector_mapping[phrase] for phrase in chosen_keyphrases]
        )
        chosen_indices = diversify(
            centroid_vectors[cluster_num], chosen_vectors, n_keyphrases, max_alpha=diversify_alpha
        )[:n_keyphrases]
        chosen_keyphrases = [chosen_keyphrases[j] for j in chosen_indices]

        result.append(chosen_keyphrases)

    return result
