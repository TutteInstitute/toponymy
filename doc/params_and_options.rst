Parameters and options
======================

``Toponymy(llm_wrapper, text_embedding_model=None, clusterer=None, *,
feature_extractors=None, prompt_template=None, disambiguate=True, ...)``
configures the stages. Constructor defaults create fresh component instances.

.. code-block:: python

   from toponymy import Toponymy, PLSCANClusterer
   from toponymy.feature_extraction import (
       TextExemplarExtractor, TextKeyphraseExtractor, SubtopicExtractor,
   )
   from toponymy.templates import TextTemplate

   pipeline = Toponymy(
       llm_wrapper=namer,
       text_embedding_model=embedder,
       clusterer=PLSCANClusterer(base_min_cluster_size=10, max_layers=4),
       feature_extractors=[
           TextExemplarExtractor(n_exemplars=6),
           TextKeyphraseExtractor(selection_method="information_weighted", n_keyphrases=12),
           SubtopicExtractor(n_subtopics=16),
       ],
       prompt_template=TextTemplate("research abstracts", "a collection of research abstracts"),
   )

Each extractor fills one distinct template feature key. Exemplars fill
``cluster_sentences``, keyphrases fill ``cluster_keywords``, and child topics
fill ``cluster_subtopics``. Choose either ``TextKeyphraseExtractor`` or
``TreeSHAPKeyphraseExtractor`` for the keyword slot.

The default ``TextExemplarExtractor`` uses central exemplars. Its other methods
are ``random``, ``facility_location`` and ``saturated_coverage``. See
:doc:`exemplar_texts` and :doc:`keyphrases` for numerical input examples.

``lowest_detail_level`` and ``highest_detail_level`` are ordered values between
zero and one that select how specific the names should be across layers.
``disambiguation_threshold`` is the cosine similarity threshold for name
embeddings when an embedder is supplied. Exact duplicate detection also works
without an embedder.

Set ``reuse_clusterer=True`` to use an already fitted hierarchy for the same
observations in the same order. ``prepare`` still creates fresh topic and feature
state; it validates alignment without refitting that clusterer.

``SubtopicExtractor`` defaults to size-ordered child names. Its
``selection_method`` can instead be ``central``, ``information_weighted``,
``facility_location`` or ``saturated_coverage`` to select base-topic evidence
using retained name embeddings. ``source="summary"`` or ``"explanation"`` chooses
the selected text field. See :doc:`migration` for prerequisites and ID mapping.

``stage_timings_`` exposes local stage durations; ``request_counts_`` separates
naming, disambiguation and name-embedding requests. They describe the current
run, rather than estimates of provider billing or semantic quality.
