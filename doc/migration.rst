Migrating from v0.5 to v0.6
===========================

The public ``Toponymy`` entry point now runs the modular pipeline. Its default
clusterer is PLSCAN, and its default evidence is exemplar text. Adding
keyphrases is an explicit choice. A text embedder is needed only when their
embeddings must be generated; supplied count matrices, vocabularies and
keyphrase vectors require no text embedder.

Installed packages and source examples
--------------------------------------

The documentation notebooks, ``examples/local_pipeline.py`` and historical
example datasets remain in the source checkout. They are not installed with
the package. ``toponymy[example-notebooks]`` installs the local notebook runner
dependencies; pass the paths of notebooks you want to execute. Historical
bundled-data loaders use a checkout's ``examples`` directory, or the existing local
directory selected by ``TOPONYMY_EXAMPLES_DIR``. Missing data raises an error
without downloading a replacement.

.. list-table:: Configuration changes
   :header-rows: 1
   :widths: 38 62

   * - Earlier configuration
     - v0.6 configuration
   * - ``ToponymyClusterer`` and legacy kernel options
     - ``PLSCANClusterer``; the old class name is a deprecated alias using PLSCAN
   * - ``layer_class=ClusterLayerText``
     - ``feature_extractors=[TextExemplarExtractor(...), ...]``
   * - ``keyphrase_builder=...`` on ``Toponymy``
     - ``TextKeyphraseExtractor(keyphrase_builder=...)`` in ``feature_extractors``
   * - ``ClusterLayerSummaryText`` and layer-level summary settings
     - ``prompt_template=SummaryTemplate(..., summary_kind=...)``
   * - Names, features and prompts stored on cluster layers
     - ``pipeline.topics_[(layer_index, original_cluster_id)]``
   * - Names indexed as a dense list of cluster IDs
     - Per-layer dictionaries in ``topic_names_`` and ``topic_sizes_``
   * - ``new_*`` implementation imports
     - ``clustering``, ``types``, ``feature_extraction``, ``templates`` and ``toponymy``

Inputs and ownership
--------------------

Call ``fit(objects, embedding_vectors, clusterable_vectors=None)``. Semantic
vectors and optional clustering vectors each have one row per object, but may
have different dimensions. ``object_vectors=`` is a compatibility spelling
for ``embedding_vectors=``; supply one spelling. Vectors must contain finite
real numbers. They are borrowed through read-only views; callers must not
mutate them while the pipeline or its results use them.

Cluster labels require integer dtype. ``-1`` is noise; other negative IDs are
invalid. Original nonnegative IDs survive grouping and persistence. A cluster
layer owns its read-only labels and member indices. Create another layer to
change membership. Empty input and all-noise partitions are valid results.

``PrecomputedClusterer(labels)`` accepts a sequence of label vectors, one per
layer. Its ``fit(vectors)`` validates the observation count. When labels were
not supplied to the constructor, ``fit(label_layers)`` remains supported.

EVoC fitting
------------

``EVoCClusterer`` fits in a fresh Python process by default. EVoC and
fast_hdbscan currently define distinct internal namedtuple classes with the
same names and fields. Numba's process-wide runtime type cache can confuse
these classes when both libraries fit in one process. Isolation keeps both
algorithms usable without changing their kernels or suppressing failures.

The adapter preserves EVoC's naturally selected layers and returns its actual
fitted model as ``evoc_``. Input passes through a temporary memory-mapped file;
the process and temporary files are cleaned up after completion or interruption.
The additional interpreter startup, compilation, disk space and model handoff
costs apply to every fit. Child failures propagate with their error output.
The caller's environment and thread settings are inherited unchanged.

``EVoCClusterer(isolated=False)`` opts into direct fitting for a process where
only EVoC executes clustering kernels. Accessing fitted attributes on ``evoc_``
is supported; directly calling its kernel-executing methods bypasses isolation.
Use the adapter's ``fit`` method when refitting alongside PLSCAN.

Staged naming
-------------

``prepare`` creates topic state, features and initial prompts without calling
the naming provider. ``name_topics`` or ``await name_topics_async()`` completes
naming. Upper-layer prompts are refreshed after layer-dependent extractors
receive lower-layer names. Every new ``fit`` recomputes data-dependent stages.

Disambiguation remains enabled. It detects duplicate names, and uses semantic
name similarity when a text embedder is present. Use ``disambiguate=False``
only when that functional change is intended; it can change request counts.
Similarity groups use complete linkage and each rename request contains at most
``max_disambiguation_group_size`` topics (four by default).

The old ``create_cluster_layers`` and ``build_raw_cluster_layers`` engine helpers
are removed. Fit a clusterer and read its ``cluster_layers_`` and ``cluster_tree_``.
Clusterer fits no longer accept runtime layer classes or naming configuration;
configure those on ``Toponymy``. The unused ``next_cluster_size_quantile`` and
``show_progress_bar`` adapter options are removed; use ``verbose`` for progress.

Prompts and persistence
-----------------------

``Prompt(system, user, json_schema=None)`` separates provider-independent
messages. A template renders prompts and parses responses; it does not mutate
the feature dictionaries. ``SummaryTemplate`` returns name, summary and
explanation, stored on each topic.

Child summary evidence
~~~~~~~~~~~~~~~~~~~~~~

``SubtopicExtractor()`` supplies lower topics' names to each parent's prompt.
Choose ``source="summary"`` or ``source="explanation"`` to use the corresponding
fields produced by ``SummaryTemplate`` instead::

    from toponymy.feature_extraction import SubtopicExtractor, TextExemplarExtractor
    from toponymy.templates import SummaryTemplate

    model = Toponymy(
        llm_wrapper,
        clusterer=clusterer,
        feature_extractors=[
            TextExemplarExtractor(),
            SubtopicExtractor(source="summary"),
        ],
        prompt_template=SummaryTemplate("documents", "my corpus"),
    )

Lower layers finish naming before this evidence is extracted. Direct children
are ordered by membership size and original topic key; duplicate evidence is
included once. A missing or empty selected field raises ``ValueError`` before
the parent naming request. It does not fall back to the child's name.
The selectable source replaces the former summary-layer ownership of child
summaries and explanations; it selects one field for each extractor.

Stored results
~~~~~~~~~~~~~~

``pipeline.topic_model_`` is the existing ``serialization.TopicModel``.
``TopicModel.to_file`` writes format 0.2, and ``from_file`` reads 0.1 and 0.2.
Earlier files retain the information they originally stored; loading does not
invent missing prompt or summary state. ``topic_df`` is a materialized view:
edit a topic's state explicitly instead of expecting DataFrame edits to update
the model.
