Pipeline and results
====================

.. autoclass:: toponymy.Toponymy
   :members:

.. autoclass:: toponymy.serialization.Topic
   :members:

.. autoclass:: toponymy.TopicModel
   :members:

Clustering
----------

.. automodule:: toponymy.clustering
   :members: Clusterer, PLSCANClusterer, EVoCClusterer, KMeansClusterer, PrecomputedClusterer, build_cluster_layers, build_cluster_tree, validate_cluster_tree

.. automodule:: toponymy.types
   :members: Cluster, ClusterLayer

Feature extraction
------------------

.. automodule:: toponymy.feature_extraction
   :members: FeatureExtractorBase, TextExemplarExtractor, TextKeyphraseExtractor, SubtopicExtractor, TreeSHAPKeyphraseExtractor

Prompts and templates
---------------------

.. automodule:: toponymy.templates
   :members: Prompt, Template, TextTemplate, SummaryTemplate, MultilingualENFRTemplate
