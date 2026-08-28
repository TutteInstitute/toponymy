# conftest.py

from pathlib import Path
import httpx
import json
import litellm
import logging
import numpy as np
import os
import pandas as pd
import psutil
import pytest
import requests
import shutil
import subprocess
import time
import umap

from sentence_transformers import SentenceTransformer
from toponymy.llm_wrappers import HuggingFaceNamer, AsyncHuggingFaceNamer
from toponymy.clustering import centroids_from_labels, ToponymyClusterer
from toponymy.tools.notebook_test_helpers import OLLAMA_CI_MODELS
from toponymy.tests.helpers.llm_test_config import make_mock_data

logger = logging.getLogger(__name__)

# fallback to httpx transport for testing to avoid aiohttp issues in test environments
litellm.disable_aiohttp_transport = True


@pytest.fixture(scope="function")
def llm():
    """Fixture for the LLM model."""
    return HuggingFaceNamer("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
    # return HuggingFace("Qwen/Qwen3-0.6B", llm_specific_instructions=" /no_think")


@pytest.fixture(scope="function")
def null_llm():
    """Fixture for a null LLM model."""
    return HuggingFaceNamer("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="function")
def async_llm():
    """Fixture for the async LLM model."""
    return AsyncHuggingFaceNamer(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_concurrent_requests=4,
        device="cpu",
    )
    # return AsyncHuggingFace(
    #     "Qwen/Qwen3-0.6B",
    #     llm_specific_instructions=" /no_think",
    #     max_concurrent_requests=4,
    # )


def ollama_has_model(model_or_family: str, family=False) -> bool:
    """Check if the specified Ollama model or family is available locally."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        response.raise_for_status()

        models = [m["name"] for m in response.json().get("models", [])]

        if family:
            return any(m.startswith(model_or_family + ":") for m in models)
        return any(m == model_or_family for m in models)

    except Exception:
        return False


def _ollama_service_up() -> bool:
    try:
        response = httpx.get("http://localhost:11434/api/version", timeout=2)
        response.raise_for_status()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def _ollama_sufficient_resources():
    """Skip tests if the machine doesn't have enough free memory/disk for Ollama.

    Requirements are relaxed in CI environments.
    """

    is_ci = (
        os.getenv("CI", "").lower() == "true"
        or os.getenv("AZURE_PIPELINES", "").lower() == "true"
    )

    min_memory_gb = 3.0 if is_ci else 4.0
    min_disk_gb = 1.5 if is_ci else 2.0

    available_memory = psutil.virtual_memory().available / (1024**3)
    available_disk = shutil.disk_usage("/tmp").free / (1024**3)

    if available_memory < min_memory_gb:
        pytest.skip(
            f"Insufficient memory for Ollama test: {available_memory:.1f}GB < {min_memory_gb}GB required"
        )
    if available_disk < min_disk_gb:
        pytest.skip(
            f"Insufficient disk space for Ollama test: {available_disk:.1f}GB < {min_disk_gb}GB required"
        )


@pytest.fixture(scope="session")
def ollama_running(_ollama_sufficient_resources) -> bool:
    """Check if Ollama is installed and if the service is available. Try
    to start `ollama serve` if it isn't already running, and tear it
    down afterward -- but only if this fixture is what started it.
    """
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Ollama not installed")
        yield False
        return

    started_process: subprocess.Popen | None = None

    if not _ollama_service_up():
        try:
            started_process = subprocess.Popen(
                ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            time.sleep(3)

            if not _ollama_service_up():
                started_process.terminate()
                started_process.wait(timeout=5)
                logger.warning("Could not start Ollama service for testing")
                yield False
                return
        except Exception:
            logger.warning("Could not start Ollama service for testing")
            yield False
            return
    yield True

    if started_process is not None:
        logger.info("Stopping Ollama service that was started for testing")
        started_process.terminate()
        try:
            started_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            started_process.kill()
            started_process.wait()


@pytest.fixture
def mock_data():
    """Make mock data for testing LLM responses."""
    return make_mock_data()


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
        [
            x["sentences"]
            for subtopics in subtopic_objects
            for x in subtopics["subtopics"]
        ],
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
def object_vectors():  # embedder, all_sentences):
    """Generate object vectors using the embedder."""
    # return embedder.encode(all_sentences)
    npy_path = Path(__file__).parent / "object_vectors.npy"
    return np.load(npy_path)


@pytest.fixture(scope="session")
def clusterable_vectors(object_vectors):
    """Generate clusterable vectors using UMAP."""
    return umap.UMAP(
        metric="cosine", n_neighbors=15, n_components=2, min_dist=0.001, random_state=42
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
def subtopic_vectors():  # embedder, all_subtopics):
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
def topic_vectors():  # embedder, all_topic_objects):
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


@pytest.fixture(scope="session")
def premade_topic_model_path():
    file_path = Path(__file__).parent / "mock-20ng.tm.zip"
    return file_path


@pytest.fixture(scope="session")
def notebook_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("nb_outputs")


@pytest.fixture(scope="function")
def notebook_testing_env(notebook_output_dir):
    """
    Sets/unsets the NOTEBOOK_TESTING environment variable to signal
    notebook_test_replacement decorator, and sets/unsets the OPENAI_API_KEY
    to a non-existing key make sure live calls aren't made by accident.
    """
    old = os.environ.get("NOTEBOOK_TESTING")
    old_openai = os.environ.get("OPENAI_API_KEY")
    old_output_dir = os.environ.get("NB_TEST_OUTPUT_DIR")

    os.environ["NOTEBOOK_TESTING"] = "true"
    os.environ["OPENAI_API_KEY"] = "notarealkey"
    os.environ["NB_TEST_OUTPUT_DIR"] = str(notebook_output_dir)

    try:
        yield
    finally:
        if old is None:
            os.environ.pop("NOTEBOOK_TESTING", None)
        else:
            os.environ["NOTEBOOK_TESTING"] = old
        if old_openai is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_openai
        if old_output_dir is None:
            os.environ.pop("NB_TEST_OUTPUT_DIR", None)
        else:
            os.environ["NB_TEST_OUTPUT_DIR"] = old_output_dir
