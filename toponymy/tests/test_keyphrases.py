from toponymy.keyphrases import (
    create_tokenizers_ngrammer,
    build_object_x_keyphrase_matrix,
    build_keyphrase_vocabulary,
    build_keyphrase_count_matrix,
    information_weighted_keyphrases,
    central_keyphrases,
    bm25_keyphrases,
    subset_matrix_and_class_labels,
)
from toponymy.clustering import (
    centroids_from_labels,
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
from pathlib import Path
from vectorizers.transformers import InformationWeightTransformer

import numpy as np

import pytest
import json
import bm25s
import sentence_transformers

def create_ngrammer(ngram_range, token_pattern=r"(?u)\b\w\w+\b"):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        token_pattern=token_pattern,
    )
    return vectorizer.build_analyzer()

EMBEDDER = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

BASE_NGRAMMER = create_ngrammer((1, 1))
TEST_OBJECTS = json.load(open(Path(__file__).parent / "test_objects.json", "r"))
TOPIC_OBJECTS = json.load(open(Path(__file__).parent / "topic_objects.json", "r"))
ALL_TOPIC_OBJECTS = sum([x["paragraphs"] for x in TOPIC_OBJECTS], [])
CLUSTER_LAYER = np.concatenate([np.arange(10).repeat(10), np.full(10, -1)])
MATRIX, KEYPHRASES = build_object_x_keyphrase_matrix(
    ALL_TOPIC_OBJECTS, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), min_occurrences=1
)
TOPIC_VECTORS = EMBEDDER.encode(ALL_TOPIC_OBJECTS)
KEYPHRASE_VECTORS = EMBEDDER.encode(KEYPHRASES)
CENTROID_VECTORS = centroids_from_labels(CLUSTER_LAYER, TOPIC_VECTORS)



@pytest.mark.parametrize("max_features", [900, 300])
@pytest.mark.parametrize("ngram_range", [4, 3, 2, 1])
def test_vocabulary_building(max_features, ngram_range):
    ngrammer = create_ngrammer((1, ngram_range))
    vocabulary = build_keyphrase_vocabulary(
        TEST_OBJECTS, n_jobs=4, max_features=max_features, ngrammer=ngrammer
    )
    assert len(vocabulary) <= max_features
    assert "the" not in vocabulary
    assert (" ".join(["quick", "brown", "fox", "jumps"][:ngram_range])) in vocabulary
    assert (" ".join(["sleeping", "dogs", "lie"][:ngram_range])) in vocabulary


@pytest.mark.parametrize("max_features", [900, 450])
@pytest.mark.parametrize("ngram_range", [4, 3, 2, 1])
def test_tokenizer_vocabulary_building(max_features, ngram_range):
    ngrammer = create_tokenizers_ngrammer(EMBEDDER.tokenizer, (1, ngram_range))
    vocabulary = build_keyphrase_vocabulary(
        TEST_OBJECTS, n_jobs=4, max_features=max_features, ngrammer=ngrammer
    )
    assert len(vocabulary) <= max_features
    assert "the" not in vocabulary
    assert (" ".join(["quick", "brown", "fox", "jumps"][:ngram_range])) in vocabulary
    assert (" ".join(["sleeping", "dogs", "lie"][:ngram_range])) in vocabulary

@pytest.mark.parametrize("ngram_range", [4, 3, 2, 1])
def test_count_matrix_building(ngram_range):
    ngrammer = create_ngrammer((1, ngram_range))
    vocabulary = build_keyphrase_vocabulary(
        TEST_OBJECTS, n_jobs=2, max_features=1000, ngrammer=ngrammer
    )
    vocabulary_map = {word: i for i, word in enumerate(vocabulary)}
    count_matrix = build_keyphrase_count_matrix(
        TEST_OBJECTS, vocabulary_map, n_jobs=4, ngrammer=ngrammer
    )
    assert count_matrix.shape[0] == len(TEST_OBJECTS)
    assert count_matrix.shape[1] == len(vocabulary)
    assert count_matrix.nnz > 0
    assert (
        count_matrix[
            0,
            vocabulary_map[" ".join(["quick", "brown", "fox", "jumps"][:ngram_range])],
        ]
        == 1
    )
    assert (
        count_matrix[
            -1,
            vocabulary_map[" ".join(["quick", "brown", "fox", "jumps"][-ngram_range:])],
        ]
        == 1
    )


@pytest.mark.parametrize("ngram_range", [3, 2])
@pytest.mark.parametrize("token_pattern", [r"(?u)\b\w[-'\w]+\b", r"(?u)\b\w\w+\b"])
@pytest.mark.parametrize("max_features", [1000, 100])
def test_matching_sklearn(ngram_range, token_pattern, max_features):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    ngrammer = create_ngrammer((1, ngram_range), token_pattern=token_pattern)
    vocabulary = build_keyphrase_vocabulary(
        TEST_OBJECTS,
        n_jobs=1,
        max_features=max_features,
        min_occurrences=1,
        ngrammer=ngrammer,
        stop_words=ENGLISH_STOP_WORDS
    )
    vocabulary_map = {word: i for i, word in enumerate(sorted(vocabulary))}
    count_matrix = build_keyphrase_count_matrix(
        TEST_OBJECTS,
        vocabulary_map,
        n_jobs=1,
        ngrammer=ngrammer,
    )

    vectorizer = CountVectorizer(
        ngram_range=(1, ngram_range),
        token_pattern=token_pattern,
        stop_words=None,
    ).fit(TEST_OBJECTS)
    matrix = vectorizer.transform(TEST_OBJECTS)

    vocab_subset = sorted(
        [
            x
            for x in vectorizer.get_feature_names_out()
            if x.split()[0] not in ENGLISH_STOP_WORDS
            and x.split()[-1] not in ENGLISH_STOP_WORDS
        ],
        key=lambda x: vocabulary_map[x] if x in vocabulary_map else len(vocabulary_map) + 1,
    )
    all_counts = np.squeeze(np.asarray(matrix.sum(axis=0)))
    vocab_counts = np.asarray(
        [all_counts[vectorizer.vocabulary_[x]] for x in vocab_subset]
    )
    vocab_counter = Counter(dict(zip(vocab_subset, vocab_counts)))
    vocab_subset = [x[0] for x in vocab_counter.most_common(max_features)]
    assert len(vocab_subset) == len(vocabulary)
    assert set(vocab_subset) == set(vocabulary)

    sklearn_matrix = CountVectorizer(
        ngram_range=(1, ngram_range),
        token_pattern=token_pattern,
        vocabulary=vocabulary_map,
    ).fit_transform(TEST_OBJECTS)

    assert count_matrix.shape == sklearn_matrix.shape
    assert np.all(count_matrix.toarray() == sklearn_matrix.toarray())


@pytest.mark.parametrize("n_keyphrases", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_central_keyphrases_result_sizes(n_keyphrases, diversify_alpha):
    central_keyphrases_results = central_keyphrases(
        CLUSTER_LAYER,
        MATRIX,
        KEYPHRASES,
        KEYPHRASE_VECTORS,
        diversify_alpha=diversify_alpha,
        n_keyphrases=n_keyphrases,
    )
    assert len(central_keyphrases_results) == len(np.unique(CLUSTER_LAYER)) - 1
    if diversify_alpha == 1.0:
        print(central_keyphrases_results)
    assert all([len(x) == n_keyphrases for x in central_keyphrases_results]) and central_keyphrases_results
    assert all([len(set(x)) == n_keyphrases for x in central_keyphrases_results])


def test_central_keyphrases():
    central_results = central_keyphrases(
        CLUSTER_LAYER,
        MATRIX,
        KEYPHRASES,
        KEYPHRASE_VECTORS,
        diversify_alpha=0.0,
        n_keyphrases=10,
    )
    for cluster_num, keyphrases in enumerate(central_results):
        phrases_indices_in_cluster = np.unique(MATRIX[CLUSTER_LAYER == cluster_num].indices)
        keyphrase_centroid = np.mean(
            KEYPHRASE_VECTORS[phrases_indices_in_cluster], axis=0
        )
        assert keyphrases == sorted(
            keyphrases,
            key=lambda x: 1.0 - (
                KEYPHRASE_VECTORS[KEYPHRASES.index(x)] @ keyphrase_centroid
            ),
        )
        worst_keyphrase_match_similarity = min(
            [
                KEYPHRASE_VECTORS[KEYPHRASES.index(x)]
                @ keyphrase_centroid
                for x in keyphrases
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                KEYPHRASE_VECTORS[i] @ keyphrase_centroid
                for i in phrases_indices_in_cluster
                if KEYPHRASES[i] not in keyphrases
            ]
        )


@pytest.mark.parametrize("n_keyphrases", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_bm25_keyphrases_result_sizes(n_keyphrases, diversify_alpha):
    bm25_keyphrases_results = bm25_keyphrases(
        CLUSTER_LAYER,
        MATRIX,
        KEYPHRASES,
        KEYPHRASE_VECTORS,
        diversify_alpha=diversify_alpha,
        n_keyphrases=n_keyphrases,
    )
    assert len(bm25_keyphrases_results) == len(np.unique(CLUSTER_LAYER)) - 1
    assert all([len(x) == n_keyphrases for x in bm25_keyphrases_results])
    assert all([len(set(x)) == n_keyphrases for x in bm25_keyphrases_results])


def test_bm25_keyphrases():
    bm25_results = bm25_keyphrases(
        CLUSTER_LAYER,
        MATRIX,
        KEYPHRASES,
        KEYPHRASE_VECTORS,
        diversify_alpha=0.0,
    )
    bm25_objects = [
        "\n".join(x["paragraphs"]) for x in TOPIC_OBJECTS if x["topic"] != "No topic"
    ]
    retriever = bm25s.BM25()
    corpus_tokens = bm25s.tokenize(bm25_objects)
    retriever.index(corpus_tokens)
    scoring = [
        [retriever.get_scores([x]) for x in bm25_results[i]]
        for i in range(len(bm25_results))
    ]

    for i in range(len(scoring)):
        score_for_cluster = [x[i] for x in scoring[i]]
        assert sorted(score_for_cluster, reverse=True) == score_for_cluster
        for j in range(len(scoring)):
            if i == j:
                continue
            assert min(score_for_cluster) >= max(
                [
                    x[i]
                    for n, x in enumerate(scoring[j])
                    if bm25_results[j][n] not in bm25_results[i]
                ]
            )


@pytest.mark.parametrize("n_keyphrases", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_information_weighted_keyphrases_result_sizes(n_keyphrases, diversify_alpha):
    bm25_keyphrases_results = information_weighted_keyphrases(
        CLUSTER_LAYER,
        MATRIX,
        KEYPHRASES,
        KEYPHRASE_VECTORS,
        diversify_alpha=diversify_alpha,
        n_keyphrases=n_keyphrases,
    )
    assert len(bm25_keyphrases_results) == len(np.unique(CLUSTER_LAYER)) - 1
    assert all([len(x) == n_keyphrases for x in bm25_keyphrases_results])
    assert all([len(set(x)) == n_keyphrases for x in bm25_keyphrases_results])

def test_information_weighted_keyphrases():
    iwt_results = information_weighted_keyphrases(
        CLUSTER_LAYER,
        MATRIX,
        KEYPHRASES,
        KEYPHRASE_VECTORS,
        diversify_alpha=0.0,
    )
    sub_matrix, class_layer, column_map = subset_matrix_and_class_labels(CLUSTER_LAYER, MATRIX)
    iwt_transformer = InformationWeightTransformer(weight_power=2.0, prior_strength=0.1)
    class_layer = CLUSTER_LAYER[CLUSTER_LAYER >= 0]
    iwt_transformer.fit(sub_matrix, class_layer)
    iwt_weights = iwt_transformer.information_weights_
    iwt_matrix = np.log1p(sub_matrix.toarray()) * iwt_weights
    scoring = [
        np.asarray([iwt_matrix[class_layer == j][:,[column_map[KEYPHRASES.index(x)] for x in iwt_results[i]]].sum(axis=0) for j in range(CLUSTER_LAYER.max() + 1)]).T
        for i in range(len(iwt_results))
    ]
    for i in range(len(scoring)):
        score_for_cluster = [x[i] for x in scoring[i]]
        assert sorted(score_for_cluster, reverse=True) == score_for_cluster
        for j in range(len(scoring)):
            if i == j:
                continue
            assert min(score_for_cluster) >= max(
                [
                    x[i]
                    for n, x in enumerate(scoring[j])
                    if iwt_results[j][n] not in iwt_results[i]
                ]
            )