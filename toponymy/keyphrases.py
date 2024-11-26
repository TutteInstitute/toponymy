import numpy as np

from typing import List, Tuple, FrozenSet, Dict, Callable
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from joblib import Parallel, delayed, effective_n_jobs, cpu_count
from functools import reduce
from collections import Counter

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
    chunk_size = (len(objects) // n_chunks) + 1
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

    return result
