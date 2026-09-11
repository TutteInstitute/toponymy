"""Regression for a valid partially named snapshot; no provider calls."""

import numpy as np
import pytest
from scipy import sparse
from toponymy.serialization import Topic, TopicModel
from toponymy.templates import Prompt


@pytest.mark.parametrize("format", ["zip", "lance"])
def test_partial_prompt_snapshot_keeps_absent_prompt(format, tmp_path):
    model = TopicModel(
        None,
        {(1, 0): [(0, 0), (0, 1)]},
        [sparse.csr_matrix(np.eye(2, dtype=np.uint8) * 255)],
        np.eye(2),
        topics={
            (0, 0): Topic(0, 0, np.array([0]), prompt=Prompt("S", "U"), name="Ready"),
            (0, 1): Topic(0, 1, np.array([1])),
        },
    )
    path = tmp_path / format
    if format == "zip":
        model.to_file(path)
        loaded = TopicModel.from_file(path)
    else:
        model.to_lance(path)
        loaded = TopicModel.from_lance(path)
    assert loaded.topics[(0, 1)].prompt is None
    assert loaded.topics[(0, 0)].prompt == Prompt("S", "U")
    assert loaded.topics[(0, 0)].name == "Ready"
    assert loaded.topics[(0, 1)].name is None
    assert all(topic.name_embedding is None for topic in loaded.topics.values())
