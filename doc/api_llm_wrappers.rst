LLM Wrappers
------------

The LLM Wrappers are a set of classes that provide a unified interface for different LLMs.

They are designed to be used with the Toponymy API and provide a consistent way to interact with different LLMs.

.. currentmodule:: toponymy.llm_wrappers

Local LLM Wrappers
^^^^^^^^^^^^^^^^^^

These wrappers are designed to work with locally running LLMs.

.. autoclass:: toponymy.llm_wrappers.HuggingFaceNamer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.LlamaCppNamer
   :members:
   :undoc-members:
   :show-inheritance:

LLM Service Wrappers
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toponymy.llm_wrappers.OpenAINamer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.CohereNamer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.AnthropicNamer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.AzureAINamer
    :members:
    :undoc-members:
    :show-inheritance: