Embedding providers
===================

Supply semantic document vectors to ``Toponymy.prepare`` or ``fit``. The default
exemplar extractor uses those vectors and needs no additional text embedder.
Optional ``clusterable_vectors`` are used for clustering alone and may have a
different dimension.

``text_embedding_model`` is required when the pipeline includes
``TextKeyphraseExtractor``. It also enables semantic comparisons of generated
names during disambiguation. Exact duplicate detection works without it.
A compatible provider exposes ``encode(texts, ...)`` and returns a finite
numeric matrix with one row per supplied text.

.. code-block:: python

   from toponymy import Toponymy
   from toponymy.feature_extraction import TextExemplarExtractor, TextKeyphraseExtractor

   pipeline = Toponymy(
       llm_wrapper=namer,
       text_embedding_model=embedder,
       feature_extractors=[TextExemplarExtractor(), TextKeyphraseExtractor()],
   )

Use your existing text embedder or an installed SDK wrapper such as
``OpenAIEmbedder`` or ``CohereEmbedder``. SDK-specific names are available when
their optional dependencies are installed. Model downloads and remote
``encode`` calls are separate operations that the :doc:`basic_usage` example
does not require.

When calling a keyphrase extractor directly, a count matrix, vocabulary and
precomputed keyphrase vectors can replace text encoding. These vectors must
have compatible dimensions for the selected numerical method. See
:doc:`keyphrases` for a small local example.

Embeddings and their preprocessing define the geometry of similarity
comparisons. Keep their configuration consistent within an extraction or
comparison, and assess topic quality separately from structural tests.
