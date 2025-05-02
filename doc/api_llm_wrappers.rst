LLM Wrappers
------------

The LLM Wrappers are a set of classes that provide a unified interface for different LLMs.

They are designed to be used with the Toponymy API and provide a consistent way to interact with different LLMs.

.. currentmodule:: toponymy.llm_wrappers

Local LLM Wrappers
^^^^^^^^^^^^^^^^^^

These wrappers are designed to work with locally running LLMs.

.. autoclass:: toponymy.llm_wrappers.HuggingFace
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.LlamaCpp
   :members:
   :undoc-members:
   :show-inheritance:

LLM Service Wrappers
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toponymy.llm_wrappers.OpenAI
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.Cohere
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.Anthropic
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: toponymy.llm_wrappers.AzureAI
    :members:
    :undoc-members:
    :show-inheritance: