Toponymy
========

.. image:: doc/toponymy_text_horizontal.png
   :width: 600
   :align: center
   :alt: Toponymy

Toponymy names groups of documents or other objects in embedding space, at
multiple resolutions. It combines maintained clustering algorithms, selected
evidence from each cluster, and a language model. The name comes from the
Greek *topos* (place) and *onuma* (name).

`Documentation <https://toponymy.readthedocs.io/>`_ describes the public API.
The `architecture wiki <https://github.com/TutteInstitute/toponymy/wiki/Toponymy-Architecture>`_
provides additional design context. This source tree documents the v0.6 API;
see ``doc/migration.rst`` when moving from v0.5.

Installation
------------

Install a released package with ``pip install toponymy``, or install this source
tree with ``pip install .``. Python 3.10 or newer is required. PLSCAN is the
default clusterer. EVoC and TreeSHAP are optional extras:

.. code-block:: shell

   pip install 'toponymy[evoc]'
   pip install 'toponymy[treeshap]'

A local example
---------------

In a source checkout, ``examples/local_pipeline.py`` uses small arrays,
precomputed clusters with sparse IDs, and a deterministic naming provider.
It needs no model downloads, credentials, or service calls:

.. code-block:: shell

   python examples/local_pipeline.py
   python examples/local_pipeline.py --output topics.toponymy

For your own objects and vectors, the main workflow is:

.. code-block:: python

   from toponymy import Toponymy

   pipeline = Toponymy(llm_wrapper=your_naming_provider)
   pipeline.prepare(objects, embedding_vectors, clusterable_vectors)

   for key, topic in pipeline.topics_.items():
       print(key, topic.features, topic.prompt.system, topic.prompt.user)

   pipeline.name_topics()
   print(pipeline.topic_names_)

``prepare`` builds clusters, extracts evidence and renders initial prompts
without calling the naming provider. ``name_topics`` performs naming.
``fit(objects, embedding_vectors, clusterable_vectors)`` combines both stages.
With an asynchronous wrapper, use ``await pipeline.fit_async(...)`` or
``await pipeline.name_topics_async()`` after preparation.

``embedding_vectors`` is a finite numeric matrix with one row per object.
Feature extraction uses these semantic vectors. Optional ``clusterable_vectors``
can have a different dimension, for example a two-dimensional map. If omitted,
clustering uses the semantic vectors. Do not mutate input matrices while a fit
or its result is using them.

Choosing the evidence
---------------------

The default uses four representative texts per cluster through
``TextExemplarExtractor``. It requires no additional text embedding provider.
To include keyphrases or already named child topics, configure extractors:

.. code-block:: python

   from toponymy.feature_extraction import (
       TextExemplarExtractor, TextKeyphraseExtractor, SubtopicExtractor,
   )

   pipeline = Toponymy(
       llm_wrapper=your_naming_provider,
       text_embedding_model=your_text_embedder,
       feature_extractors=[
           TextExemplarExtractor(n_exemplars=6),
           TextKeyphraseExtractor(n_keyphrases=12),
           SubtopicExtractor(n_subtopics=16),
       ],
   )

Keyphrases need a text embedder only when their embeddings must be generated.
Supplying a count matrix, vocabulary and keyphrase vectors through
``feature_options`` requires no text embedder. Child-topic
features are filled after their lower layers have names, so upper-layer prompts
are refreshed during naming. ``TreeSHAPKeyphraseExtractor`` offers an opt-in
contrastive alternative with its own optional dependency.

Results and storage
-------------------

``pipeline.topics_`` maps ``(layer_index, original_cluster_id)`` to topic state:
members, features, prompt, name, optional summary and explanation. Each entry
in ``topic_names_`` and ``topic_sizes_`` is a dictionary keyed by the original
cluster ID. IDs may be sparse; they are not list offsets. Noise has label ``-1``
and does not create a topic. ``topic_name_vectors_`` gives names aligned to
objects in each layer, with ``"Unlabelled"`` for noise.

.. code-block:: python

   from toponymy import TopicModel

   pipeline.topic_model_.to_file("topics.toponymy")
   saved = TopicModel.from_file("topics.toponymy")
   print(saved.topic_names)

The existing ``TopicModel`` is the results and persistence interface. Version
0.2 files retain topic features and prompts; version 0.1 files remain readable.
For summaries, configure ``SummaryTemplate`` rather than a cluster layer class.

If you have a two-dimensional document map, the optional DataMapPlot library
can display the object-aligned names at each resolution:

.. code-block:: python

   import datamapplot

   plot = datamapplot.create_interactive_plot(
       document_map, *pipeline.topic_name_vectors_
   )

Similar names are disambiguated by default. Exact duplicates need no embedder;
an available text embedder also enables similarity comparisons. Disambiguation
can add provider calls, which are visible in ``request_counts_``. Scripted
responses in the local example demonstrate the pipeline, not topic quality.

The documentation notebooks and example datasets are source-checkout assets;
they are not installed with the package. Install ``toponymy[example-notebooks]``
to run your own local notebooks through ``toponymy.tools.notebook_runner``.
Pass explicit notebook paths. For the historical bundled-data loaders, set
``TOPONYMY_EXAMPLES_DIR`` to an existing local examples directory; this setting
does not download data.

Contributing and license
------------------------

See ``CONTRIBUTING.md`` for the formatter, tests, and contribution workflow.
Toponymy is MIT licensed; see ``LICENSE``.
