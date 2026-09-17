"""Execute reviewed local documentation through the real Jupyter runner.

Jupyter needs local sockets: run this file outside an all-sockets-denied guard.
The temporary kernel uses this interpreter and never registers a user kernel.
The source checks detect accidental reintroduction of live-model workflows;
they are not a sandbox for arbitrary hostile Python.
"""

import ast

import nbformat
from nbformat.v4 import new_code_cell, new_notebook
import pytest

from toponymy.tools.notebook_runner import run_notebook
from toponymy.tools.notebook_test_helpers import doc_dir, get_notebooks

RESULT_CHECKS = {
    "basic_usage.ipynb": """
assert pipeline.topic_names_ == [{10: 'Demo topic 1', 42: 'Demo topic 2'}, {7: 'Demo topic 3'}]
assert pipeline.topic_sizes_ == [{10: 2, 42: 2}, {7: 4}]
assert pipeline.request_counts_['naming'] == 3
assert pipeline.request_counts_['disambiguation'] == 0
assert namer.calls == 3
""",
    "saving_loading.ipynb": """
assert loaded.topic_names == [{10: 'Demo topic 1', 42: 'Demo topic 2'}, {7: 'Demo topic 3'}]
assert loaded.topics[(0, 42)].members.tolist() == [2, 3]
assert loaded.topics[(0, 10)].prompt == pipeline.topics_[(0, 10)].prompt
assert loaded.metadata == pipeline.topic_model_.metadata
""",
    "topic_summaries.ipynb": """
assert set(pipeline.topics_) == {(0, 10), (0, 42), (1, 7)}
for topic in pipeline.topics_.values():
    assert topic.summary == 'A scripted summary for this demonstration.'
    assert topic.explanation == 'A scripted explanation of the supplied examples.'
    assert 'two concise sentences' in topic.prompt.user
assert pipeline.llm_wrapper.calls == 3
""",
    "clusterers.ipynb": """
assert [len(layer) for layer in kmeans] == [4, 1]
assert plscan.cluster_layers_
assert all(len(layer.labels) == 96 for layer in plscan)
assert any(len(layer) >= 2 for layer in plscan)
assert clusterer.cluster_layers_[0].labels.tolist() == [10, 10, 42, 42]
""",
    "clustering_options.ipynb": """
assert 1 <= len(clusterer.cluster_layers_) <= 3
assert tree == {
    (2, 10): [(0, 4), (1, 2), (1, 3)],
    (1, 2): [(0, 19)],
    (3, 0): [(2, 10)],
}
""",
    "exemplar_texts.ipynb": """
assert len(features) == 2
assert set(features[0][0]) == set(objects[:2])
assert set(features[0][1]) == set(objects[2:])
assert set(selected.indices_[0][0]) == {0, 1}
assert set(selected.indices_[0][1]) == {2, 3}
assert len(evidence[1][0]) == 2
""",
    "keyphrases.ipynb": """
assert len(features) == 2
assert set(features[0][0]) == {'apple', 'orchard'}
assert set(features[0][1]) == {'river', 'water'}
assert set(features[1][0]).issubset(set(vocabulary))
assert extractor.feature_key == 'cluster_keywords'
""",
    "how_toponymy_works.ipynb": """
assert pipeline.topics_[(1, 7)].features['cluster_subtopics']['major'] == ['Demo topic 1', 'Demo topic 2']
assert pipeline.request_counts_['naming'] == 3
assert namer.calls == 3
assert not hasattr(pipeline.cluster_layers_[0][0], 'name')
""",
    "debugging_llm_runs.ipynb": """
assert len(create_audit_df(pipeline)) == 3
assert create_layer_summary_df(pipeline)['num_clusters'].tolist() == [2, 1]
assert pipeline.request_counts_ == {'naming': 3, 'disambiguation': 0, 'name_embeddings': 0}
assert namer.calls == 3
""",
}

HISTORICAL_NOTEBOOKS = {
    "test_audit_functionality.ipynb": "Historical v0.5 audit experiment uses external data and live model providers",
    "test_max_layers_newsgroups.ipynb": "Historical v0.5 clustering experiment uses removed APIs and downloaded data",
}

REVIEWED_IMPORTS = {
    "numpy": None,
    "json": None,
    "toponymy": {
        "Toponymy",
        "PrecomputedClusterer",
        "TopicModel",
        "PLSCANClusterer",
        "KMeansClusterer",
    },
    "toponymy.clustering": {
        "validate_cluster_tree",
        "build_cluster_layers",
        "build_cluster_tree",
    },
    "toponymy.feature_extraction": {
        "TextExemplarExtractor",
        "TextKeyphraseExtractor",
        "SubtopicExtractor",
    },
    "toponymy.templates": {"TextTemplate", "SummaryTemplate"},
    "toponymy.audit": {"create_audit_df", "create_layer_summary_df"},
    "scipy.sparse": {"csr_matrix"},
    "pathlib": {"Path"},
    "tempfile": {"TemporaryDirectory"},
}


def _review_local_code(notebook):
    """Require explicit review before adding imports, URLs or dynamic execution."""
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if cell.outputs or cell.execution_count is not None:
            raise ValueError(
                "Current examples must not carry recorded execution outputs"
            )
        if "skip-execution" in cell.metadata.get("tags", []):
            raise ValueError("Every current example cell must be exercised by its test")
        parsed = ast.parse(cell.source, filename=f"cell-{cell_index}")
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                if any(alias.name not in ("numpy", "json") for alias in node.names):
                    raise ValueError("Unreviewed notebook import")
            elif isinstance(node, ast.ImportFrom):
                allowed = REVIEWED_IMPORTS.get(node.module)
                if (
                    node.level
                    or allowed is None
                    or any(alias.name not in allowed for alias in node.names)
                ):
                    raise ValueError("Unreviewed notebook import")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(
                    scheme in node.value.lower()
                    for scheme in ("http://", "https://", "hf://")
                ):
                    raise ValueError("Network resource in a local notebook code cell")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {
                    "eval",
                    "exec",
                    "compile",
                    "__import__",
                    "getattr",
                    "get_ipython",
                }:
                    raise ValueError("Dynamic execution in a local notebook code cell")


def test_notebook_inventory_requires_an_execution_policy():
    actual = {path.name for path in get_notebooks(doc_dir())}
    assert actual == RESULT_CHECKS.keys() | HISTORICAL_NOTEBOOKS.keys()


@pytest.mark.parametrize("name", sorted(RESULT_CHECKS))
def test_current_notebooks_start_with_a_document_title(name):
    notebook = nbformat.read(doc_dir() / name, as_version=4)
    first_markdown = next(
        cell for cell in notebook.cells if cell.cell_type == "markdown"
    )
    title = first_markdown.source.splitlines()[0]
    assert title.startswith("# ")
    assert title[2:].strip()


@pytest.mark.parametrize("name", sorted(HISTORICAL_NOTEBOOKS))
def test_historical_notebooks_are_labelled_and_disabled(name):
    notebook = nbformat.read(doc_dir() / name, as_version=4)
    assert notebook.metadata.nbsphinx.execute == "never"
    assert any(
        "Historical v0.5 experiment" in cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    assert all(
        "skip-execution" in cell.metadata.get("tags", [])
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


@pytest.mark.parametrize(
    "name", sorted(RESULT_CHECKS.keys() | HISTORICAL_NOTEBOOKS.keys())
)
def test_doc_notebook(name, tmp_path, local_notebook_kernel):
    if name in HISTORICAL_NOTEBOOKS:
        pytest.skip(HISTORICAL_NOTEBOOKS[name])
    original_path = doc_dir() / name
    original = original_path.read_bytes()
    notebook = nbformat.reads(original.decode("utf-8"), as_version=4)
    assert notebook.metadata.nbsphinx.execute == "never"
    _review_local_code(notebook)
    marker = "VERIFIED_LOCAL_NOTEBOOK:" + name
    notebook.cells.append(new_code_cell(RESULT_CHECKS[name] + f"\nprint({marker!r})"))
    path = tmp_path / name
    nbformat.write(notebook, path)

    executed = run_notebook(str(path), timeout=180, kernel_name=local_notebook_kernel)

    assert all(
        cell.execution_count is not None
        for cell in executed.cells
        if cell.cell_type == "code"
    )
    streams = [
        output.get("text", "")
        for cell in executed.cells
        for output in cell.get("outputs", [])
    ]
    assert any(marker in output for output in streams)
    assert original_path.read_bytes() == original


@pytest.mark.parametrize(
    "source",
    [
        "import openai",
        "from toponymy.llm_wrappers import OpenAINamer",
        "import numpy as np\nnp.load('https://example.invalid/data.npy')",
        "exec('import requests')",
    ],
)
def test_local_notebook_review_rejects_external_workflows(source):
    with pytest.raises(ValueError):
        _review_local_code(new_notebook(cells=[new_code_cell(source)]))
