"""Retained upstream data fixtures; model checks remain explicit opt-in tests."""

import json
from pathlib import Path

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parent
DATA = DIRECTORY.parent / "data"
MODEL_TESTS = pytest.StashKey[list[tuple[str, str]]]()


def pytest_addoption(parser):
    parser.addoption(
        "--run-local-models",
        action="store_true",
        default=False,
        help="Run retained model tests using locally cached models/services only",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local_model(reason): requires a cached model or local model service"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    model_tests = []
    for item in items:
        if item.path.resolve().parent != DIRECTORY:
            continue
        marker = item.get_closest_marker("local_model")
        if "embedder" in item.fixturenames:
            reason = "Requires cached sentence-transformers/all-MiniLM-L6-v2; model execution is opt-in"
            item.add_marker(pytest.mark.local_model(reason=reason))
        elif marker is not None:
            reason = marker.kwargs.get("reason", "Requires a local model")
        else:
            continue
        model_tests.append((item.nodeid, reason))
        if not config.getoption("--run-local-models"):
            item.add_marker(
                pytest.mark.skip(reason=reason + "; use --run-local-models")
            )
    config.stash[MODEL_TESTS] = model_tests


def pytest_terminal_summary(terminalreporter):
    model_tests = terminalreporter.config.stash.get(MODEL_TESTS, [])
    if model_tests:
        terminalreporter.section("retained local-model tests (collection policy)")
        for nodeid, reason in model_tests:
            terminalreporter.write_line(f"{nodeid}: {reason}")


def _json_data(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture
def embedder(request):
    if not request.config.getoption("--run-local-models"):
        pytest.skip("Cached sentence-transformer execution requires --run-local-models")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


@pytest.fixture(scope="session")
def subtopic_objects():
    return _json_data("subtopic_objects.json")


@pytest.fixture(scope="session")
def all_sentences(subtopic_objects):
    return sum(
        [
            entry["sentences"]
            for topic in subtopic_objects
            for entry in topic["subtopics"]
        ],
        [],
    )


@pytest.fixture(scope="session")
def cluster_label_vector():
    return np.arange(5).repeat(25)


@pytest.fixture(scope="session")
def subtopic_label_vector():
    return np.arange(25).repeat(5)


@pytest.fixture(scope="session")
def object_vectors():
    return np.load(DATA / "object_vectors.npy", allow_pickle=False)


@pytest.fixture(scope="session")
def cluster_centroid_vectors(cluster_label_vector, object_vectors):
    from toponymy.utility_functions import centroids_from_labels

    return centroids_from_labels(cluster_label_vector, object_vectors)


@pytest.fixture(scope="session")
def subtopic_centroid_vectors(subtopic_label_vector, object_vectors):
    from toponymy.utility_functions import centroids_from_labels

    return centroids_from_labels(subtopic_label_vector, object_vectors)


@pytest.fixture(scope="session")
def subtopics(subtopic_objects):
    return [
        [entry["subtopic"] for entry in topic["subtopics"]]
        for topic in subtopic_objects
    ]


@pytest.fixture(scope="session")
def all_subtopics(subtopics):
    return sum(subtopics, [])


@pytest.fixture(scope="session")
def subtopic_vectors():
    return np.load(DATA / "subtopic_vectors.npy", allow_pickle=False)


@pytest.fixture(scope="session")
def test_objects():
    return _json_data("test_objects.json")


@pytest.fixture(scope="session")
def topic_objects():
    return _json_data("topic_objects.json")


@pytest.fixture(scope="session")
def all_topic_objects(topic_objects):
    return sum([topic["paragraphs"] for topic in topic_objects], [])


@pytest.fixture(scope="session")
def topic_vectors():
    return np.load(DATA / "topic_vectors.npy", allow_pickle=False)


@pytest.fixture(scope="session")
def test_object_cluster_label_vector():
    return np.concatenate([np.arange(10).repeat(10), np.full(10, -1)])


@pytest.fixture(scope="session")
def notebook_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("nb_outputs")


@pytest.fixture
def notebook_testing_env(notebook_output_dir, monkeypatch):
    monkeypatch.setenv("NOTEBOOK_TESTING", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "notarealkey")
    monkeypatch.setenv("NB_TEST_OUTPUT_DIR", str(notebook_output_dir))
