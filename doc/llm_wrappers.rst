Naming providers
================

The pipeline passes ``Prompt(system, user, json_schema=None)`` to the naming
wrapper, along with the selected template's response parser. Inspect the
messages through ``pipeline.topics_`` after ``prepare``. Features that depend
on lower-layer names are added before naming the corresponding upper layer.

The :doc:`basic_usage` example uses a deterministic local fake. For a real
provider, configure a supported model and credentials explicitly. The following
setup constructs a wrapper; calling ``name_topics`` or ``fit`` then sends
requests to that provider.

.. code-block:: python

   import os
   from toponymy import Toponymy
   from toponymy.llm_wrappers import LiteLLMNamer

   namer = LiteLLMNamer(
       model=os.environ["TOPONYMY_MODEL"],
       api_key=os.environ["TOPONYMY_API_KEY"],
       use_json_schema=True,
   )
   pipeline = Toponymy(namer)
   pipeline.prepare(objects, embedding_vectors)

Schema policy
-------------

``use_json_schema=True`` requires a prompt schema and provider support. An
unsupported required schema raises ``InvalidLLMInputError`` rather than silently
falling back. ``False`` disables schema requests; ``None`` allows automatic
capability selection. ``use_json_object`` controls JSON-object output separately.
Do not force both modes or combine an explicit mode with a conflicting
``provider_kwargs['response_format']``.

Templates parse JSON structurally and validate fields by meaning, independently
of key order. Invalid responses are errors; they are not successful empty topic
names. Unexpected exceptions and cancellation propagate. Retry policy is bounded
and distinguishes transient transport failures from invalid requests and
authentication or permission failures.

Sync, async and batch
---------------------

``LLMWrapper.generate_topic_name`` handles one synchronous prompt.
``AsyncLLMWrapper.generate_topic_names`` returns results aligned with the input
prompts and bounds concurrent requests. With ``AsyncLiteLLMNamer``:

.. code-block:: python

   from toponymy.llm_wrappers import AsyncLiteLLMNamer

   async_namer = AsyncLiteLLMNamer(
       model=os.environ["TOPONYMY_MODEL"],
       api_key=os.environ["TOPONYMY_API_KEY"],
       max_concurrent_requests=4,
   )
   pipeline = Toponymy(async_namer)
   pipeline.prepare(objects, embedding_vectors)
   await pipeline.name_topics_async()

Await async operations in the caller's event loop. Sync methods do not create
hidden loops. Provider batch transports keep their own submission and result
retrieval behavior; a batch can report per-item failure.

Convenience factories such as ``OpenAINamer``, ``AnthropicNamer`` and
``OllamaNamer`` configure the same wrapper interfaces. Local model integrations
may require optional packages and separately provisioned model files. Model
availability, service limits and prices belong to the provider's current
documentation; no fixed model-quality or pricing claim is made here.

Callbacks and ``request_counts_`` help inspect actual request behavior. A
connectivity check itself can make a provider call; it is not needed to inspect
prepared prompts. See :doc:`debugging_llm_runs` and :doc:`api_llm_wrappers`.
