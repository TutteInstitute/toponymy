================
PLSCANClusterer
================

``PLSCANClusterer`` is a Toponymy ``Clusterer`` wrapper around
``fast_plscan.PLSCAN``. It is useful when you want PLSCAN-based layered
clustering inside Toponymy while still using the standard Toponymy clusterer
interface. It does not replace ``ToponymyClusterer``; it is an alternative
clusterer for users who specifically want PLSCAN's persistence-based cluster
layers.

When to use it
--------------

Use ``PLSCANClusterer`` when you want to:

* cluster on ``clusterable_vectors``, often a low-dimensional map or another
  clusterable representation;
* compute Toponymy centroids from ``embedding_vectors``;
* inspect PLSCAN layers through the same ``cluster_layers_`` and
  ``cluster_tree_`` interface used by other Toponymy clusterers.

Basic usage
-----------

.. code-block:: python

   from toponymy import ClusterLayerText, PLSCANClusterer

   clusterer = PLSCANClusterer(
       min_clusters=6,
       min_samples=5,
       base_min_cluster_size=10,
       max_layers=4,
   )

   cluster_layers, cluster_tree = clusterer.fit_predict(
       clusterable_vectors=clusterable_vectors,
       embedding_vectors=embedding_vectors,
       layer_class=ClusterLayerText,
   )

   # Toponymy layer objects and the parent/child cluster tree
   clusterer.cluster_layers_
   clusterer.cluster_tree_

   # PLSCAN metadata retained for the returned layers
   clusterer.cluster_probabilities_
   clusterer.cluster_cut_sizes_

``cluster_layers`` is the list of Toponymy cluster layer objects. ``cluster_tree``
maps clusters between neighboring layers.

Notes
-----

* ``clusterable_vectors`` are passed to ``fast_plscan.PLSCAN.fit(...)``.
* ``embedding_vectors`` are used for Toponymy centroid construction.
* ``-1`` labels are preserved as noise or unlabelled points.
* ``cluster_probabilities_`` and ``cluster_cut_sizes_`` are stored on the
  clusterer object for the returned layers.
* ``n_threads=-1`` uses the upstream PLSCAN default thread behavior.
