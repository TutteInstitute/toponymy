LLM Wrappers
------------

Wrappers accept the canonical ``Prompt`` and the selected template's response
parser. See :doc:`llm_wrappers` for system/user separation, schema policy,
bounded retries and the distinction between sync, async and batch transports.

.. currentmodule:: toponymy.llm_wrappers

.. autoclass:: toponymy.llm_wrappers.LLMWrapper
   :members: generate_topic_name, generate_topic_cluster_names, supports_json_schema

.. autoclass:: toponymy.llm_wrappers.AsyncLLMWrapper
   :members: generate_topic_names, generate_topic_cluster_names, supports_json_schema

``LiteLLMNamer`` and ``AsyncLiteLLMNamer`` adapt configured provider models.
An explicitly required structured-output capability must be supported by that
provider; unsupported requirements raise ``InvalidLLMInputError``.

.. autoclass:: toponymy.llm_wrappers.LiteLLMNamer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.AsyncLiteLLMNamer
   :members:
   :undoc-members:
   :show-inheritance:

LLM Convenience Wrappers
^^^^^^^^^^^^^^^^^^^^^^^^

The following wrappers include helpful presets for convenience purposes (and their Async equivalents). 

.. autofunction:: toponymy.llm_wrappers.OpenAINamer

.. autofunction:: toponymy.llm_wrappers.AnthropicNamer

.. autofunction:: toponymy.llm_wrappers.CohereNamer

.. autofunction:: toponymy.llm_wrappers.AzureAINamer


Local LLM Wrappers
^^^^^^^^^^^^^^^^^^

These wrappers are designed to work with locally running LLMs.

.. autofunction:: toponymy.llm_wrappers.OllamaNamer


.. autoclass:: toponymy.llm_wrappers.HuggingFaceNamer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.LlamaCppNamer
   :members:
   :undoc-members:
   :show-inheritance:
