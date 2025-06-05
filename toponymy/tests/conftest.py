# conftest.py
from pathlib import Path

import numpy as np
import pandas as pd
import json
import umap
import pytest

from sentence_transformers import SentenceTransformer
from toponymy.llm_wrappers import HuggingFace, AsyncHuggingFace
from toponymy.clustering import centroids_from_labels, ToponymyClusterer

@pytest.fixture(scope="function")
def llm():
    """Fixture for the LLM model."""
    return HuggingFace("Qwen/Qwen2.5-0.5B-Instruct")
    # return HuggingFace("Qwen/Qwen3-0.6B", llm_specific_instructions=" /no_think")

@pytest.fixture(scope="function")
def null_llm():
    """Fixture for a null LLM model."""
    return HuggingFace("hf-internal-testing/tiny-random-gpt2")

@pytest.fixture(scope="function")
def async_llm():
    """Fixture for the async LLM model."""
    return AsyncHuggingFace(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_concurrent_requests=4,
    )
    # return AsyncHuggingFace(
    #     "Qwen/Qwen3-0.6B",
    #     llm_specific_instructions=" /no_think",
    #     max_concurrent_requests=4,
    # )


@pytest.fixture(scope="function")
def embedder():
    """Fixture for the sentence transformer embedder."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def subtopic_objects():
    """Load subtopic objects from JSON file."""
    json_path = Path(__file__).parent / "subtopic_objects.json"
    with open(json_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def all_sentences(subtopic_objects):
    """Extract all sentences from subtopic objects."""
    return sum(
        [x["sentences"] for subtopics in subtopic_objects 
         for x in subtopics["subtopics"]],
        [],
    )

@pytest.fixture(scope="session")
def cluster_label_vector():
    """Generate cluster label vector."""
    return np.arange(5).repeat(25)


@pytest.fixture(scope="session")
def subtopic_label_vector():
    """Generate subtopic label vector."""
    return np.arange(25).repeat(5)


@pytest.fixture(scope="session")
def object_vectors():# embedder, all_sentences):
    """Generate object vectors using the embedder."""
    # return embedder.encode(all_sentences)
    npy_path = Path(__file__).parent / "object_vectors.npy"
    return np.load(npy_path)


@pytest.fixture(scope="session")
def clusterable_vectors(object_vectors):
    """Generate clusterable vectors using UMAP."""
    return umap.UMAP(
        metric="cosine",
        n_neighbors=15,
        n_components=2,
        min_dist=0.001,
        random_state=42
    ).fit_transform(object_vectors)

@pytest.fixture(scope="session")
def cluster_centroid_vectors(cluster_label_vector, object_vectors):
    """Calculate cluster centroid vectors from labels."""
    return centroids_from_labels(cluster_label_vector, object_vectors)

@pytest.fixture(scope="session")
def subtopic_centroid_vectors(subtopic_label_vector, object_vectors):
    """Calculate subtopic centroid vectors from labels."""
    return centroids_from_labels(subtopic_label_vector, object_vectors)

@pytest.fixture(scope="session")
def subtopics(subtopic_objects):
    """Extract subtopics from subtopic objects."""
    return [[x["subtopic"] for x in topic["subtopics"]] for topic in subtopic_objects]

@pytest.fixture(scope="session")
def all_subtopics(subtopics):
    """Flatten the list of subtopics."""
    return sum(subtopics, [])

@pytest.fixture(scope="session")
def subtopic_vectors():# embedder, all_subtopics):
    """Generate subtopic vectors using the embedder."""
    # return embedder.encode(all_subtopics)
    npy_path = Path(__file__).parent / "subtopic_vectors.npy"
    return np.load(npy_path)

@pytest.fixture(scope="session")
def cluster_tree():
    """Create a mock cluster tree structure."""
    return {(1, i): [(0, i * 5 + j) for j in range(5)] for i in range(5)}

@pytest.fixture(scope="session")
def clusterer():
    """Create a mock clusterer."""
    return ToponymyClusterer(
        min_samples=5,
        base_min_cluster_size=4,
        next_cluster_size_quantile=1.0,
        min_clusters=4,
        verbose=True,
    )

@pytest.fixture(scope="session")
def test_objects():
    """Load test objects from JSON file."""
    json_path = Path(__file__).parent / "test_objects.json"
    with open(json_path, "r") as f:
        return json.load(f)
    
@pytest.fixture(scope="session")
def topic_objects():
    """Load topic objects from JSON file."""
    json_path = Path(__file__).parent / "topic_objects.json"
    with open(json_path, "r") as f:
        return json.load(f)
    
@pytest.fixture(scope="session")
def all_topic_objects(topic_objects):
    """Extract all topic objects from the topic objects."""
    return sum([x["paragraphs"] for x in topic_objects], [])

@pytest.fixture(scope="session")
def topic_vectors(): #embedder, all_topic_objects):
    # return embedder.encode(all_topic_objects)
    npy_path = Path(__file__).parent / "topic_vectors.npy"
    return np.load(npy_path)

@pytest.fixture(scope="session")
def test_object_cluster_label_vector():
    return np.concatenate([np.arange(10).repeat(10), np.full(10, -1)])

@pytest.fixture(scope="session")
def test_object_centroid_vectors(test_object_cluster_label_vector, topic_vectors):
    """Calculate centroid vectors for test objects."""
    return centroids_from_labels(test_object_cluster_label_vector, topic_vectors)

@pytest.fixture(scope="session")
def centroid_vectors(cluster_label_vector, topic_vectors):
    """Calculate centroid vectors for topics."""
    return centroids_from_labels(cluster_label_vector, topic_vectors)