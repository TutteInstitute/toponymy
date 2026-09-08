"""Real, small optional SHAP integrations; no provider calls or model downloads."""

from types import SimpleNamespace
import builtins
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from toponymy.feature_extraction import TreeSHAPKeyphraseExtractor


@pytest.mark.skipif(
    importlib.util.find_spec("shap") is None, reason="requires optional treeshap extra"
)
def test_exact_tree_shap_finds_present_contrastive_terms_and_refits():
    matrix = sp.csr_matrix(
        [[4, 0, 1], [3, 0, 1], [2, 0, 1], [0, 4, 1], [0, 3, 1], [0, 2, 1], [99, 99, 0]]
    )
    vocabulary = ["apple", "engine", "shared"]
    clustering = [SimpleNamespace(labels=np.array([17, 17, 17, 205, 205, 205, -1]))]
    extractor = TreeSHAPKeyphraseExtractor(
        n_keyphrases=2, n_estimators=12, max_samples_per_class=3
    )
    result = extractor.fit_predict(
        [""] * 7,
        clustering,
        object_x_keyphrase_matrix=matrix,
        keyphrase_list=vocabulary,
    )
    assert result == [[["apple"], ["engine"]]]
    assert extractor.classifier_count_ == 2
    assert extractor.attribution_scores_[0][0][0] > 0
    # Absence of the competing term may support membership, but is not a phrase
    # describing documents in this cluster and must be excluded from output.
    assert "engine" not in result[0][0]
    assert extractor.attribution_scores_[0][0][2] == 0
    changed = extractor.fit_predict(
        [""] * 7,
        clustering,
        object_x_keyphrase_matrix=matrix,
        keyphrase_list=["pear", "motor", "shared"],
    )
    assert changed == [[["pear"], ["motor"]]]


@pytest.mark.skipif(
    importlib.util.find_spec("shap") is None, reason="requires optional treeshap extra"
)
def test_tree_shap_actual_text_vectorization_and_dense_size_bound():
    objects = [
        "apple pear fruit",
        "apple pear food",
        "engine motor vehicle",
        "engine motor machine",
    ]
    extractor = TreeSHAPKeyphraseExtractor(
        n_keyphrases=2, max_features=4, max_samples_per_class=2, n_estimators=12
    )
    result = extractor.fit_predict(
        objects, [SimpleNamespace(labels=np.array([9, 9, 40, 40]))]
    )
    assert len(extractor.keyphrase_list_) <= 4
    assert len(result[0]) == 2
    assert all(result[0])


@pytest.mark.parametrize("labels", [[], [-1], [19, 19]])
def test_tree_shap_requires_a_real_contrast_and_does_not_import_shap(
    labels, monkeypatch
):
    original_import = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise AssertionError("SHAP should not be loaded without a contrast")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shap)
    clustering = [SimpleNamespace(labels=np.array(labels, dtype=np.int64))]
    extractor = TreeSHAPKeyphraseExtractor()
    expected = [[[]]] if labels == [19, 19] else [[]]
    assert extractor.fit_predict(["x"] * len(labels), clustering) == expected
    assert extractor.classifier_count_ == 0


@pytest.mark.parametrize(
    "configuration",
    [
        {"max_features": 0},
        {"max_samples_per_class": -2},
        {"n_estimators": 0},
        {"n_keyphrases": 0},
    ],
)
def test_tree_shap_validates_resource_bounds(configuration):
    with pytest.raises(ValueError, match="positive"):
        TreeSHAPKeyphraseExtractor(**configuration).fit_predict([], [])
