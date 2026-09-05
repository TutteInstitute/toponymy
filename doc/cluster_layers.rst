Cluster layers and topic state
==============================

``toponymy.types.Cluster`` stores an original nonnegative label and its sorted,
unique observation indices. ``ClusterLayer`` stores a layer index, a tuple of
clusters ordered by ID, and a complete label array. ``layer.cluster_labels``
is an alias for ``layer.labels``. Arrays are owned and read-only; naming state
is stored separately on ``serialization.Topic``.

.. code-block:: python

   from toponymy.clustering import build_cluster_layers, build_cluster_tree

   labels = [[10, -1, 42, 10], [7, -1, 7, 7]]
   layers = build_cluster_layers(labels)
   tree = build_cluster_tree(labels)
   assert layers[0][0].label == 10
   assert layers[0][0].members.tolist() == [0, 3]
   assert tree[(1, 7)] == [(0, 10), (0, 42)]

The ``[0]`` in ``layers[0][0]`` is the ordinal of a sorted cluster, not its ID.
Use original IDs in topic keys and tree edges. Gaps in IDs do not create empty
clusters; noise never creates a cluster.

Trees map a parent ``(layer, ID)`` to child keys. A parent must be in a higher
layer and fully contain the child's members. The builder finds the nearest
such layer, skipping crossing clusters and partial noise. Nodes without an
ancestor attach to the synthetic root ``(number_of_layers, 0)``.
``validate_cluster_tree(tree, layers)`` rejects false containment, unknown or
missing nodes, repeated parents and cycles. An empty hierarchy has ``{}``.

After preparation, inspect mutable topic state through
``pipeline.topics_[(layer, ID)]``. A topic contains ``members``, ``features``,
``prompt``, ``name``, ``summary`` and ``explanation``. Feature extraction and
naming do not add those fields to cluster layers.
