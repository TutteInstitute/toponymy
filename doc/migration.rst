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

Set ``Toponymy(..., reuse_clusterer=True)`` to consume an already fitted
clusterer without refitting it. This keeps separately fitted metadata extractors
aligned with that hierarchy. Reuse validates the layer row counts and tree, and
captures the fitted structure for this pipeline. The caller must supply the same
observations in the same order; matching row counts cannot establish identity.
An unfitted clusterer is an error. The default ``False`` still fits on every
``prepare``/``fit``.

EVoC fitting
------------

``EVoCClusterer`` fits in a fresh Python process by default. EVoC and
fast_hdbscan currently define distinct internal namedtuple classes with the
same names and fields. Numba's process-wide runtime type cache can confuse
these classes when both libraries fit in one process. Isolation keeps both
algorithms usable without changing their kernels or suppressing failures.

The adapter preserves EVoC's naturally selected layers. ``evoc_`` contains its
fitted data and constructor values in a ``types.SimpleNamespace``, including
the native tree materialized inside the fit process. It is no longer an EVoC
estimator: use the adapter's ``fit``/``fit_predict`` for refits and
``get_params``/``set_params`` for configuration. This prevents copied or
unpickled fitted state from accidentally executing native kernels in the parent.
The adapter's ``fit_predict`` returns ``(layers, tree)``. For the selected label
vector previously returned by native ``evoc_.fit_predict(X)``, use
``adapter.fit(X).evoc_.labels_``.
Input passes through a temporary memory-mapped file;
the process and temporary files are cleaned up after completion or interruption.
On Windows, interruption uses ``taskkill /T /F`` to stop the interpreter tree
before removing the mapped input and inherited log. Waiting for a virtual
environment's launcher alone can race those file handles. The adapter does not
provide a total fit deadline or a guarantee about delivery of OS console signals.
If termination or file cleanup fails, the initiating exception remains primary
and the cleanup error is chained to it. Cleanup cannot be guaranteed when the
operating system refuses termination or file removal.
The additional interpreter startup, compilation, disk space and model handoff
costs apply to every fit. Child failures propagate with their error output.
The caller's environment and thread settings are inherited unchanged.

``EVoCClusterer(isolated=False)`` opts into direct fitting for a process where
only EVoC executes clustering kernels. It returns the same data-only ``evoc_``
contract. Use the default isolated adapter when refitting alongside PLSCAN.

Staged naming
-------------

``prepare`` creates topic state, features and initial prompts without calling
the naming provider. ``name_topics`` or ``await name_topics_async()`` completes
naming. Upper-layer prompts are refreshed after layer-dependent extractors
receive lower-layer names. Every new ``fit`` recomputes data-dependent stages.
Explicit ``reuse_clusterer=True`` reuses clustering only; feature and naming
state are fresh for each prepared fit.

Disambiguation remains enabled. It detects duplicate names, and uses semantic
name similarity when a text embedder is present. Use ``disambiguate=False``
only when that functional change is intended; it can change request counts.
Similarity groups use complete linkage and each rename request contains at most
``max_disambiguation_group_size`` topics (four by default).

Each topic retains its current numerical ``name_embedding`` and the exact
``embedded_name`` text when a naming stage needs them. These small vectors are
owned and read-only; changing ``topic.name`` clears both fields. Exact current
names reuse vectors within a prepared fit, including across layers. A changed
name is encoded when next needed, and a new ``prepare`` starts fresh state.
Keep the text embedder and its configuration fixed during a prepared fit;
replacing it requires preparing again. ``name_embedding_context`` records the
embedder class and descriptive model settings, not a portable model fingerprint.
The existing ``topic_name_vectors_`` remains object-aligned text with
``Unlabelled`` for noise; it is separate from numerical embeddings.

``topic_model_.disambiguation_history`` retains each group's original names,
input name vectors, prompt, outputs, status, attempts and errors. A failed or
cancelled naming call leaves this evidence inspectable. Retrying on the same
prepared pipeline resumes unfinished groups using their saved prompts and does
not submit successful groups again. Original topic prompts also survive retries.
Changing a pending group's names requires preparing again. Loaded archives are
result snapshots; they do not reconstruct a provider or a resumable pipeline.

The old ``create_cluster_layers`` and ``build_raw_cluster_layers`` engine helpers
are removed. Fit a clusterer and read its ``cluster_layers_`` and ``cluster_tree_``.
Clusterer fits no longer accept runtime layer classes or naming configuration;
configure those on ``Toponymy``. The unused ``next_cluster_size_quantile`` and
``show_progress_bar`` adapter options are removed; use ``verbose`` for progress.

Prompts and persistence
-----------------------

``Prompt(system, user, json_schema=None, *, combined=None)`` separates provider-independent
messages. A template renders prompts and parses responses; it does not mutate
the feature dictionaries. ``SummaryTemplate`` returns name, summary and
explanation, stored on each topic.

``combined=`` carries a separately authored rendering for providers without
system messages. When absent, wrappers join system and user with two newlines;
an explicitly empty combined string stays empty. The third positional argument
remains the schema. All renderings stay on this one canonical Prompt and survive
ZIP/Lance persistence.

Embedding adapters
~~~~~~~~~~~~~~~~~~

``AnthropicEmbedder`` now raises ``NotImplementedError`` at construction:
the earlier implementation called an embeddings endpoint absent from the
Anthropic SDK. Use an embedding provider's adapter or a local ``encode``
implementation; Anthropic topic naming is still supported.

Indexed embedding responses are aligned to input order and rejected if an
index is duplicated or missing, or a vector is nonfinite or malformed.
Empty input to the HTTP embedding adapters returns a ``(0, 0)`` float array
without a request; the dimension is unknown until a nonempty response.
Azure's SDK owns transport retries; the wrapper no longer retries authentication
errors or malformed results. Voyage requests have bounded connect/read waits.

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
ZIP and Lance writers use format 0.3; readers accept 0.1, 0.2 and 0.3.
Earlier files retain the information they originally stored; loading does not
invent missing prompt or summary state. ``topic_df`` is a materialized view:
edit a topic's state explicitly instead of expecting DataFrame edits to update
the model.

Format 0.3 preserves current name vectors with their exact dtype and text,
embedding context and disambiguation history. Legacy results have no name vectors
and an empty history. Writers validate these fields before replacing an existing
artifact. Historical embedding dtypes must represent their recorded values
exactly; use an array's ``tolist()`` when constructing a record manually.

Both readers validate topic identities, membership alignment and topology before
returning. Repeated JSON keys, colliding decoded IDs, duplicate topic rows and
contradictory inventories are errors. Tree edges may skip layers, but must
descend and preserve containment. An explicit synthetic root must reach every
topic; rootless forests remain readable. Noise has no topic row.

Legacy label-indexed membership matrices remain supported when unrepresented
columns are empty. Lance membership values must be lossless integers in 0..255;
fractional weights require ZIP storage. Lance dtype declarations must match the
stored vector and graph types. Materializing a topic table copies exposed
keyword lists and preserves borrowed sparse matrices.
