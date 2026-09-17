Embedding API
=============

.. autoclass:: toponymy.embedding_wrappers.TextEmbedderProtocol
   :members:

``encode`` returns a numeric matrix in the order of the supplied texts.
See :doc:`embedding_wrappers` for how the pipeline uses it. Provider-specific
classes are conditional on their installed SDKs; consult the installed module's
class signatures for provider and model configuration.
