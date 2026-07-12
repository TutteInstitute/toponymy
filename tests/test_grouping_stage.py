from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from toponymy.types import Cluster, ClusterLayer
from toponymy.new_clustering import build_cluster_layers


@pytest.mark.parametrize("labels", [[], [np.array([], dtype=int)], [[-1, -1]]])
def test_empty_partitions(labels):
    layers = build_cluster_layers(labels)
    assert all(len(layer) == 0 for layer in layers)


@pytest.mark.parametrize(
    "labels",
    [
        [np.array([0, -2])],
        [np.array([True, False])],
        [np.array([0.0, 1.0])],
        [np.array([0.5])],
        [np.array([np.nan])],
        [np.array([np.inf])],
        [np.array([1 + 0j])],
        [np.array(["1"])],
        [np.array([0], dtype=object)],
        [np.array([2**64 - 1], dtype=np.uint64)],
        [np.zeros((2, 1), dtype=int)],
        [np.array([0]), np.array([0, 1])],
        np.array([0, 1]),
    ],
)
def test_label_validation(labels):
    for builder in (build_cluster_layers,):
        with pytest.raises(ValueError):
            builder(labels)


def test_sparse_noncontiguous_and_maximum_ids():
    maximum = np.iinfo(np.int64).max
    source = np.array([maximum, -1, 8, -1, maximum, -1], dtype=np.int64)[::2]
    layer = build_cluster_layers([source])[0]
    assert [cluster.label for cluster in layer] == [8, maximum]
    assert not np.shares_memory(layer.labels, source)
    np.testing.assert_array_equal(layer[1].members, [0, 2])
