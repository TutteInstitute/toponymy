from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from toponymy.types import Cluster, ClusterLayer


def test_cluster_owns_members_and_has_no_naming_state():
    source = np.array([4, 2, 0])
    cluster = Cluster(51, source)
    source[:] = 19
    np.testing.assert_array_equal(cluster.members, [0, 2, 4])
    assert not cluster.members.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cluster.members[0] = 7
    with pytest.raises(FrozenInstanceError):
        cluster.label = 1
    assert not hasattr(cluster, "name")
    assert not hasattr(cluster, "features")


def test_layer_owns_labels_and_orders_clusters():
    source = np.array([19, -1, 4, 19])
    clusters = [Cluster(19, [0, 3]), Cluster(4, [2])]
    layer = ClusterLayer(clusters, 0, source)
    source[:] = -1
    clusters.clear()
    assert isinstance(layer.clusters, tuple)
    assert [cluster.label for cluster in layer] == [4, 19]
    assert layer.cluster_labels is layer.labels
    np.testing.assert_array_equal(layer.labels, [19, -1, 4, 19])
    assert not layer.labels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        layer.labels[:] = 0
    with pytest.raises(FrozenInstanceError):
        layer.labels = np.zeros(4, dtype=int)


@pytest.mark.parametrize(
    "label,members",
    [
        (-1, [0]),
        (True, [0]),
        (1.0, [0]),
        (2**64, [0]),
        (0, []),
        (0, [-1]),
        (0, [0, 0]),
        (0, [0.5]),
        (0, [[0]]),
    ],
)
def test_invalid_cluster(label, members):
    with pytest.raises(ValueError):
        Cluster(label, members)


@pytest.mark.parametrize(
    "clusters,index,labels,error",
    [
        ([object()], 0, [0], TypeError),
        ([Cluster(0, [0]), Cluster(0, [1])], 0, [0, 0], ValueError),
        ([Cluster(0, [0]), Cluster(1, [0])], 0, [0], ValueError),
        ([Cluster(0, [1])], 0, [0], ValueError),
        ([Cluster(0, [0])], 0, [-1], ValueError),
        ([], -1, [], ValueError),
        ([], False, [], ValueError),
    ],
)
def test_invalid_layer(clusters, index, labels, error):
    with pytest.raises(error):
        ClusterLayer(clusters, index, labels)
