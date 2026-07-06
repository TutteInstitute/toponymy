"""Stable clustering output properties retained from the modular refactor."""

import numpy as np

from toponymy.new_clustering import build_cluster_layers, build_cluster_tree


def test_grouping_preserves_original_ids_and_member_order():
    labels = [np.array([19, -1, 4, 19, 4, -1])]
    layer = build_cluster_layers(labels)[0]
    assert layer.layer_index == 0
    assert [cluster.label for cluster in layer] == [4, 19]
    np.testing.assert_array_equal(layer[0].members, [2, 4])
    np.testing.assert_array_equal(layer[1].members, [0, 3])


def test_nested_tree_attaches_clusters_to_nearest_parent():
    labels = [
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        np.zeros(8, dtype=int),
    ]
    tree = build_cluster_tree(labels)
    assert set(tree[(1, 0)]) == {(0, 0), (0, 1)}
    assert set(tree[(1, 1)]) == {(0, 2), (0, 3)}
    assert set(tree[(2, 0)]) == {(1, 0), (1, 1)}
    assert tree[(3, 0)] == [(2, 0)]


def test_separate_builds_do_not_share_member_arrays():
    labels = [np.array([1, 0, 1, 0])]
    first = build_cluster_layers(labels)
    second = build_cluster_layers(labels)
    assert not np.shares_memory(first[0][0].members, second[0][0].members)
    assert not np.shares_memory(first[0][0].members, labels[0])
