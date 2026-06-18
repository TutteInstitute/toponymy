LLM Wrappers
------------

The LLM wrappers provide a unified interface for working with different language models.

They are designed for use with the Toponymy API and offer a consistent way to configure and interact with LLM providers.

.. currentmodule:: toponymy.llm_wrappers

Most wrappers are convenience layers around `(Async)LiteLLMNamer` with sensible defaults. Any model supported by LiteLLM can also be used directly via `LiteLLMNamer` by specifying the appropriate parameters.

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