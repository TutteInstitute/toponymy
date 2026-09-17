PLSCANClusterer
===============

``PLSCANClusterer`` is the default adapter around ``fast_hdbscan.PLSCAN``.
It delegates clustering to fast-hdbscan and converts its labels into the same
owned layer and containment-tree contract used by the other clusterers.

.. code-block:: python

   from toponymy import PLSCANClusterer

   clusterer = PLSCANClusterer(
       min_samples=5,
       base_min_cluster_size=10,
       max_layers=4,
       reproducible=True,
   )
   layers, tree = clusterer.fit_predict(clusterable_vectors)

   for layer in layers:
       print(layer.layer_index, [cluster.label for cluster in layer])

Pass only the clustering representation to the clusterer. Semantic embeddings,
feature extractors, templates and naming providers belong to ``Toponymy``.

``base_min_cluster_size`` controls the finest clustering resolution.
``base_n_clusters`` requests an approximate finest-layer count instead.
``max_layers`` limits the number of selected layers; it does not promise that
every input yields that many nonempty layers. ``reproducible=True`` selects
the maintained estimator's deterministic computation path.

Algorithm-specific diagnostics remain on ``clusterer.plscan_``, including
``membership_strength_layers_``, ``layer_persistence_scores_`` and
``min_cluster_sizes_``. For an empty or undersized fit that bypassed the
estimator, ``plscan_`` is ``None``.

The adapter requires fast-hdbscan >=0.3.2. Dense finite real vectors are the
ordinary input. ``metric="precomputed"`` instead accepts a scipy sparse square
distance graph with finite, nonnegative stored distances; an explicit zero
edge differs from a missing edge. Other metrics may need fast-hdbscan's
optional nearest-neighbor dependency. ``cannot_link`` constraints require
``algorithm="kruskal"``.

No naming configuration is stored on cluster layers. Noise stays ``-1``;
empty and all-noise results are valid. See :doc:`cluster_layers` for the full
membership and tree contract.
