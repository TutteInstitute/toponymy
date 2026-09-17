.. image:: toponymy_text_horizontal.png
   :width: 600
   :alt: Toponymy logo
   :align: center

Toponymy: topics in embedding space
===================================

Toponymy combines clustering, representative evidence, and language models to
name groups of objects at several resolutions. This guide describes v0.6.
The default pipeline uses PLSCAN and exemplar text; optional extractors add
keyphrases, named child topics, or contrastive TreeSHAP evidence.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   intro
   installation
   basic_usage
   migration
   benchmark_results
   params_and_options
   saving_loading
   debugging_llm_runs

.. toctree::
   :maxdepth: 1
   :caption: Components

   how_toponymy_works
   clusterers
   plscan_clusterer
   clustering_options
   cluster_layers
   exemplar_texts
   keyphrases
   topic_summaries
   llm_wrappers
   embedding_wrappers

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   historical_examples

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
