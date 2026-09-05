Introduction
============

Embeddings place related documents or other objects near each other. Clustering
can reveal groups in that space, but cluster IDs alone do not explain them.
Toponymy selects evidence from each group and asks a language model for a name.
Multiple resolutions can describe fine topics and broader themes.

The stages are explicit:

1. Supply objects and their semantic embedding vectors. A separate clustering
   representation is optional; dimensionality reduction is not a prerequisite.
2. Cluster with PLSCAN, EVoC, KMeans, or precomputed labels. Only actual
   containment creates a parent-child edge, since resolutions need not nest.
3. Extract exemplar texts by default. Select additional extractors when their
   evidence is useful for the corpus.
4. Inspect features and initial system/user prompts with ``prepare``.
5. Name topics, then populate child-topic features for higher layers and
   disambiguate similar names. These stages can make further provider requests.
6. Explore and save the existing ``TopicModel`` results object.

Clusters store membership. Topics store features and generated names. This
separation lets you inspect each stage without storing provider settings or
mutable naming state in a clusterer.

Start with :doc:`installation` and the local :doc:`basic_usage` example.
Moving from v0.5? Read :doc:`migration` first.
