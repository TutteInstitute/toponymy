from toponymy.topic_naming import Toponymy
import numpy as np
from toponymy.topic_naming import find_threshold_for_max_cluster_size
from toponymy.topic_naming import (
    Toponymy,
    create_distinguish_base_layer_topics_prompt,
    longest_keyphrases,
    topical_sentences_for_cluster,
    diversify,
    _SUMMARY_KINDS,
)


def test_find_threshold_for_max_cluster_size():
    # Test case 1: Simple case with a small distance matrix
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ]
    )
    threshold = find_threshold_for_max_cluster_size(distances, max_cluster_size=2)
    assert threshold > 0, "Threshold should be greater than 0 for a non-trivial case"

    # Test case 2: Larger distance matrix with a different max_cluster_size
    distances = np.random.rand(10, 10)
    distances = (distances + distances.T) / 2  # Make it symmetric
    np.fill_diagonal(distances, 0)  # Zero diagonal
    threshold = find_threshold_for_max_cluster_size(distances, max_cluster_size=3)
    assert threshold > 0, "Threshold should be greater than 0 for a non-trivial case"

    # Test case 3: Distance matrix with all zeros
    distances = np.zeros((5, 5))
    threshold = find_threshold_for_max_cluster_size(distances, max_cluster_size=2)
    assert threshold == 0, "Threshold should be 0 for a zero distance matrix"

    # Test case 4: Distance matrix with large values
    distances = np.full((5, 5), 100.0)
    np.fill_diagonal(distances, 0)  # Zero diagonal
    threshold = find_threshold_for_max_cluster_size(distances, max_cluster_size=2)
    assert (
        threshold == 100.0
    ), "Threshold should be the large value for a uniform large distance matrix"

    # Test case 5: Single element distance matrix
    distances = np.array([[0.0]])
    threshold = find_threshold_for_max_cluster_size(distances, max_cluster_size=1)
    assert threshold == 0, "Threshold should be 0 for a single element distance matrix"


def test_create_distinguish_base_layer_topics_prompt():
    # Test case 1: Basic functionality
    topic_indices = [0, 1]
    attempted_topic_names = ["Topic A", "Topic B"]
    representations = {
        "contrastive": [["keyword1", "keyword2"], ["keyword3", "keyword4"]],
        "topical": [["sentence1", "sentence2"], ["sentence3", "sentence4"]],
    }
    document_type = "documents"
    corpus_description = "test corpus"
    prompt = create_distinguish_base_layer_topics_prompt(
        topic_indices,
        attempted_topic_names,
        representations,
        document_type,
        corpus_description,
    )
    assert (
        "Topic A" in prompt and "Topic B" in prompt
    ), "Prompt should contain topic names"
    assert (
        "keyword1" in prompt and "keyword3" in prompt
    ), "Prompt should contain keywords"
    assert (
        "sentence1" in prompt and "sentence3" in prompt
    ), "Prompt should contain sentences"

    # Test case 2: Single topic name
    topic_indices = [0]
    attempted_topic_names = ["Topic A"]
    representations = {
        "contrastive": [["keyword1", "keyword2"]],
        "topical": [["sentence1", "sentence2"]],
    }
    prompt = create_distinguish_base_layer_topics_prompt(
        topic_indices,
        attempted_topic_names,
        representations,
        document_type,
        corpus_description,
    )
    assert "Topic A" in prompt, "Prompt should contain the single topic name"
    assert "keyword1" in prompt, "Prompt should contain keywords"
    assert "sentence1" in prompt, "Prompt should contain sentences"

    # Test case 3: No contrastive representations
    topic_indices = [0, 1]
    attempted_topic_names = ["Topic A", "Topic B"]
    representations = {
        "topical": [["sentence1", "sentence2"], ["sentence3", "sentence4"]]
    }
    prompt = create_distinguish_base_layer_topics_prompt(
        topic_indices,
        attempted_topic_names,
        representations,
        document_type,
        corpus_description,
    )
    assert (
        "Topic A" in prompt and "Topic B" in prompt
    ), "Prompt should contain topic names"
    assert (
        "sentence1" in prompt and "sentence3" in prompt
    ), "Prompt should contain sentences"
    assert "False" in prompt, "Prompt should contain False for missing keywords"

    # Test case 4: No topical representations
    topic_indices = [0, 1]
    attempted_topic_names = ["Topic A", "Topic B"]
    representations = {
        "contrastive": [["keyword1", "keyword2"], ["keyword3", "keyword4"]]
    }
    prompt = create_distinguish_base_layer_topics_prompt(
        topic_indices,
        attempted_topic_names,
        representations,
        document_type,
        corpus_description,
    )
    assert (
        "Topic A" in prompt and "Topic B" in prompt
    ), "Prompt should contain topic names"
    assert (
        "keyword1" in prompt and "keyword3" in prompt
    ), "Prompt should contain keywords"
    assert "False" in prompt, "Prompt should contain False for missing sentences"

    # Test case 5: Different summary kind
    topic_indices = [0, 1]
    attempted_topic_names = ["Topic A", "Topic B"]
    representations = {
        "contrastive": [["keyword1", "keyword2"], ["keyword3", "keyword4"]],
        "topical": [["sentence1", "sentence2"], ["sentence3", "sentence4"]],
    }
    summary_kind_idx = 2
    prompt = create_distinguish_base_layer_topics_prompt(
        topic_indices,
        attempted_topic_names,
        representations,
        document_type,
        corpus_description,
        summary_kind_idx=summary_kind_idx,
    )
    assert (
        _SUMMARY_KINDS[summary_kind_idx] in prompt
    ), "Prompt should contain the correct summary kind"


def test_longest_keyphrases():
    # Test case 1: Basic functionality with overlapping phrases
    candidate_keyphrases = ["data", "big data", "data science", "science"]
    result = longest_keyphrases(candidate_keyphrases)
    expected = ["big data", "data science"]
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 2: No overlapping phrases
    candidate_keyphrases = [
        "machine learning",
        "artificial intelligence",
        "deep learning",
    ]
    result = longest_keyphrases(candidate_keyphrases)
    expected = ["machine learning", "artificial intelligence", "deep learning"]
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 3: All phrases are the same
    candidate_keyphrases = ["AI", "AI", "AI"]
    result = longest_keyphrases(candidate_keyphrases)
    expected = ["AI"]
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 4: Empty list
    candidate_keyphrases = []
    result = longest_keyphrases(candidate_keyphrases)
    expected = []
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 5: Single phrase
    candidate_keyphrases = ["neural networks"]
    result = longest_keyphrases(candidate_keyphrases)
    expected = ["neural networks"]
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 6: Phrases with spaces
    candidate_keyphrases = ["AI", "AI research", "research in AI"]
    result = longest_keyphrases(candidate_keyphrases)
    expected = ["research in AI"]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_topical_sentences_for_cluster():
    # Test case 1: Basic functionality
    docs = np.array(["sentence1", "sentence2", "sentence3", "sentence4"])
    vector_array = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]])
    pointset = [0, 1, 2, 3]
    centroid_vector = np.array([0.25, 0.25])
    n_sentence_examples = 2
    result = topical_sentences_for_cluster(
        docs, vector_array, pointset, centroid_vector, n_sentence_examples
    )
    assert (
        len(result) == n_sentence_examples
    ), f"Expected {n_sentence_examples} sentences, but got {len(result)}"
    assert all(
        sentence in docs for sentence in result
    ), "All returned sentences should be in the original docs"

    # Test case 2: Single sentence in pointset
    docs = np.array(["sentence1"])
    vector_array = np.array([[0.1, 0.2]])
    pointset = [0]
    centroid_vector = np.array([0.1, 0.2])
    n_sentence_examples = 1
    result = topical_sentences_for_cluster(
        docs, vector_array, pointset, centroid_vector, n_sentence_examples
    )
    assert (
        len(result) == n_sentence_examples
    ), f"Expected {n_sentence_examples} sentence, but got {len(result)}"
    assert result[0] == "sentence1", "The only sentence should be 'sentence1'"

    # Test case 3: More examples requested than available
    docs = np.array(["sentence1", "sentence2"])
    vector_array = np.array([[0.1, 0.2], [0.2, 0.1]])
    pointset = [0, 1]
    centroid_vector = np.array([0.15, 0.15])
    n_sentence_examples = 5
    result = topical_sentences_for_cluster(
        docs, vector_array, pointset, centroid_vector, n_sentence_examples
    )
    assert len(result) == len(
        docs
    ), f"Expected {len(docs)} sentences, but got {len(result)}"
    assert all(
        sentence in docs for sentence in result
    ), "All returned sentences should be in the original docs"

    # Test case 4: Empty pointset
    docs = np.array(["sentence1", "sentence2", "sentence3", "sentence4"])
    vector_array = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]])
    pointset = []
    centroid_vector = np.array([0.25, 0.25])
    n_sentence_examples = 2
    result = topical_sentences_for_cluster(
        docs, vector_array, pointset, centroid_vector, n_sentence_examples
    )
    assert result == [], "Expected an empty list for an empty pointset"

    # Test case 5: Identical vectors
    docs = np.array(["sentence1", "sentence2", "sentence3", "sentence4"])
    vector_array = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    pointset = [0, 1, 2, 3]
    centroid_vector = np.array([0.1, 0.2])
    n_sentence_examples = 2
    result = topical_sentences_for_cluster(
        docs, vector_array, pointset, centroid_vector, n_sentence_examples
    )
    assert (
        len(result) == n_sentence_examples
    ), f"Expected {n_sentence_examples} sentences, but got {len(result)}"
    assert all(
        sentence in docs for sentence in result
    ), "All returned sentences should be in the original docs"

def test_diversify():
    # Test case 1: Basic functionality
    query_vector = np.array([0.1, 0.2])
    candidate_neighbor_vectors = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]])
    alpha = 1.0
    max_candidates = 2
    result = diversify(query_vector, candidate_neighbor_vectors, alpha, max_candidates)
    assert len(result) == max_candidates, f"Expected {max_candidates} candidates, but got {len(result)}"
    assert all(isinstance(idx, int) for idx in result), "All indices should be integers"

    # Test case 2: Single candidate
    query_vector = np.array([0.1, 0.2])
    candidate_neighbor_vectors = np.array([[0.1, 0.2]])
    alpha = 1.0
    max_candidates = 1
    result = diversify(query_vector, candidate_neighbor_vectors, alpha, max_candidates)
    assert len(result) == 1, f"Expected 1 candidate, but got {len(result)}"
    assert result[0] == 0, "The only candidate should be the first one"

    # Test case 3: More candidates requested than available
    query_vector = np.array([0.1, 0.2])
    candidate_neighbor_vectors = np.array([[0.1, 0.2], [0.2, 0.1]])
    alpha = 1.0
    max_candidates = 5
    result = diversify(query_vector, candidate_neighbor_vectors, alpha, max_candidates)
    assert len(result) == len(candidate_neighbor_vectors), f"Expected {len(candidate_neighbor_vectors)} candidates, but got {len(result)}"
    assert all(isinstance(idx, int) for idx in result), "All indices should be integers"

    # Test case 4: Identical vectors
    query_vector = np.array([0.1, 0.2])
    candidate_neighbor_vectors = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    alpha = 1.0
    max_candidates = 2
    result = diversify(query_vector, candidate_neighbor_vectors, alpha, max_candidates)
    assert len(result) == max_candidates, f"Expected {max_candidates} candidates, but got {len(result)}"
    assert all(isinstance(idx, int) for idx in result), "All indices should be integers"

    # Test case 5: Different alpha value
    query_vector = np.array([0.1, 0.2])
    candidate_neighbor_vectors = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]])
    alpha = 0.5
    max_candidates = 2
    result = diversify(query_vector, candidate_neighbor_vectors, alpha, max_candidates)
    assert len(result) == max_candidates, f"Expected {max_candidates} candidates, but got {len(result)}"
    assert all(isinstance(idx, int) for idx in result), "All indices should be integers"

    # Test case 6: Empty candidate neighbor vectors
    query_vector = np.array([0.1, 0.2])
    candidate_neighbor_vectors = np.array([])
    alpha = 1.0
    max_candidates = 2
    result = diversify(query_vector, candidate_neighbor_vectors, alpha, max_candidates)
    assert result == [], "Expected an empty list for empty candidate neighbor vectors"

def test_build_cluster_layers():
    # Test case 1: Basic functionality with small dataset
    point_vectors = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]])
    point_locations = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]])
    min_clusters = 2
    min_samples = 2
    base_min_cluster_size = 2
    membership_strength_threshold = 0.2
    next_cluster_size_quantile = 0.8
    verbose = False

    vector_layers, location_layers, pointset_layers, metacluster_layers = build_cluster_layers(
        point_vectors,
        point_locations,
        min_clusters=min_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        membership_strength_threshold=membership_strength_threshold,
        next_cluster_size_quantile=next_cluster_size_quantile,
        verbose=verbose,
    )

    assert len(vector_layers) > 0, "There should be at least one layer of vectors"
    assert len(location_layers) > 0, "There should be at least one layer of locations"
    assert len(pointset_layers) > 0, "There should be at least one layer of pointsets"
    assert len(metacluster_layers) > 0, "There should be at least one layer of metaclusters"

    # Test case 2: Larger dataset with different parameters
    point_vectors = np.random.rand(10, 2)
    point_locations = np.random.rand(10, 2)
    min_clusters = 3
    min_samples = 3
    base_min_cluster_size = 3
    membership_strength_threshold = 0.3
    next_cluster_size_quantile = 0.7
    verbose = False

    vector_layers, location_layers, pointset_layers, metacluster_layers = build_cluster_layers(
        point_vectors,
        point_locations,
        min_clusters=min_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        membership_strength_threshold=membership_strength_threshold,
        next_cluster_size_quantile=next_cluster_size_quantile,
        verbose=verbose,
    )

    assert len(vector_layers) > 0, "There should be at least one layer of vectors"
    assert len(location_layers) > 0, "There should be at least one layer of locations"
    assert len(pointset_layers) > 0, "There should be at least one layer of pointsets"
    assert len(metacluster_layers) > 0, "There should be at least one layer of metaclusters"

    # Test case 3: Edge case with single point
    point_vectors = np.array([[0.1, 0.2]])
    point_locations = np.array([[0.1, 0.2]])
    min_clusters = 1
    min_samples = 1
    base_min_cluster_size = 1
    membership_strength_threshold = 0.1
    next_cluster_size_quantile = 0.5
    verbose = False

    vector_layers, location_layers, pointset_layers, metacluster_layers = build_cluster_layers(
        point_vectors,
        point_locations,
        min_clusters=min_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        membership_strength_threshold=membership_strength_threshold,
        next_cluster_size_quantile=next_cluster_size_quantile,
        verbose=verbose,
    )

    assert len(vector_layers) == 1, "There should be exactly one layer of vectors"
    assert len(location_layers) == 1, "There should be exactly one layer of locations"
    assert len(pointset_layers) == 1, "There should be exactly one layer of pointsets"
    assert len(metacluster_layers) == 1, "There should be exactly one layer of metaclusters"

    # Test case 4: Edge case with empty input
    point_vectors = np.array([])
    point_locations = np.array([])
    min_clusters = 1
    min_samples = 1
    base_min_cluster_size = 1
    membership_strength_threshold = 0.1
    next_cluster_size_quantile = 0.5
    verbose = False

    vector_layers, location_layers, pointset_layers, metacluster_layers = build_cluster_layers(
        point_vectors,
        point_locations,
        min_clusters=min_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        membership_strength_threshold=membership_strength_threshold,
        next_cluster_size_quantile=next_cluster_size_quantile,
        verbose=verbose,
    )

    assert len(vector_layers) == 0, "There should be no layers of vectors"
    assert len(location_layers) == 0, "There should be no layers of locations"
    assert len(pointset_layers) == 0, "There should be no layers of pointsets"
    assert len(metacluster_layers) == 0, "There should be no layers of metaclusters"
    
def test_build_cluster_tree():
    # Test case 1: Basic functionality with simple labels
    labels = [
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
    ]
    result = build_cluster_tree(labels)
    expected = {
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 0), (0, 1)],
        (2, 0): [(1, 0), (1, 1)],
    }
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 2: Single layer of labels
    labels = [
        np.array([0, 1, 2, 3]),
    ]
    result = build_cluster_tree(labels)
    expected = {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (1, 2): [(0, 2)],
        (1, 3): [(0, 3)],
    }
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 3: Multiple layers with more complex structure
    labels = [
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([0, 1, 2, 3, 4, 5]),
    ]
    result = build_cluster_tree(labels)
    expected = {
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 0), (1, 1)],
        (2, 2): [(1, 0), (1, 1)],
        (2, 3): [(1, 0), (1, 1)],
        (2, 4): [(1, 0), (1, 1)],
        (2, 5): [(1, 0), (1, 1)],
        (3, 0): [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
    }
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 4: Edge case with empty labels
    labels = []
    result = build_cluster_tree(labels)
    expected = {}
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 5: Edge case with single element labels
    labels = [
        np.array([0]),
    ]
    result = build_cluster_tree(labels)
    expected = {
        (1, 0): [(0, 0)],
    }
    assert result == expected, f"Expected {expected}, but got {result}"



