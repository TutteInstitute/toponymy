# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Prior Topics for Incremental Updates
#
# Using the 20-newsgroups dataset, we show how providing known labels for clusters can
# reduce LLM calls for incremental updates. We use 99% of the dataset for the initial
# Toponymy labeling, and then incrementally add the remaining 1% of the dataset on a
# subsequent run. This subtle update should not cause major changes to topic names, so
# many clusters can reuse the topic names from the initial run.
#
# This example starts the same way as the basic usage example: load the embedded
# 20-newsgroups data, extract the vectors, configure an embedding model, configure an
# LLM wrapper, and fit a `Toponymy` model.

# %%
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# %% [markdown]
# For the dataset we'll use the same embedded 20-newsgroups parquet file from the
# basic usage example. It includes cleaned post text, high-dimensional embedding
# vectors, and a 2D map representation that works well for clustering and plotting.

# %%
newsgroups_df = pd.read_parquet(
    "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
)

# %%
newsgroups_df.head()

# %% [markdown]
# Extract the text, embedding vectors, and clusterable 2D vectors in the format
# Toponymy expects.

# %%
text = newsgroups_df["post"].str.strip().values
embedding_vectors = np.stack(newsgroups_df["embedding"].values)
clusterable_vectors = np.stack(newsgroups_df["map"].values)

# %% [markdown]
# We will simulate an incremental update by using the first 99% of the data for the
# first fit. The second fit uses the full dataset, which is equivalent to appending
# the remaining documents to a previously labeled corpus.

# %%
initial_size = int(0.99 * len(text))
remaining_indices = np.arange(initial_size, len(text))

initial_text = text[:initial_size]
initial_embedding_vectors = embedding_vectors[:initial_size]
initial_clusterable_vectors = clusterable_vectors[:initial_size]

initial_size, len(remaining_indices)

# %% [markdown]
# As in the basic usage example, we need a text embedding model for semantic
# similarity work inside Toponymy.

# %%
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# %% [markdown]
# Toponymy supports several LLM providers. This example follows the basic usage
# notebook and uses an Azure AI Foundry Cohere model, but you can substitute any
# Toponymy LLM wrapper.

# %%
from toponymy import KeyphraseBuilder, Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import AzureAINamer

azure_api_key = open("../azure_cohere_api_key.txt").read().strip()

llm = AzureAINamer(
    azure_api_key,
    endpoint="https://azureaitimcuse5821437469.services.ai.azure.com/models",
    model="Cohere-command-r-08-2024",
)

# %% [markdown]
# We'll use the same model configuration for both runs. The factory below creates a
# fresh `ToponymyClusterer` for each fit so that the second run builds cluster layers
# for the full, updated dataset.

# %%
def make_topic_model(previous_cluster_layers=None):
    return Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedding_model,
        clusterer=ToponymyClusterer(min_clusters=4, verbose=True),
        keyphrase_builder=KeyphraseBuilder(
            ngram_range=(1, 6),
            max_features=15_000,
            verbose=True,
        ),
        object_description="newsgroup posts",
        corpus_description="20-newsgroups dataset",
        exemplar_delimiters=["<EXAMPLE_POST>\n", "\n</EXAMPLE_POST>\n\n"],
        previous_cluster_layers=previous_cluster_layers,
    )


# %% [markdown]
# First fit the topic model on the initial 99% of the corpus. This is the same fit
# workflow as the basic usage example, but using the subset of data we selected above.

# %%
initial_topic_model = make_topic_model()

# %%
# %%time
initial_topic_model.fit(
    initial_text,
    embedding_vectors=initial_embedding_vectors,
    clusterable_vectors=initial_clusterable_vectors,
)

# %%
initial_topic_model.topic_names_[-1]

# %% [markdown]
# Now fit a new topic model on the complete dataset. Toponymy will compare clusters
# in each new layer to the prior layers from the initial run. The prior label arrays
# are shorter than the current dataset, which is fine for append-only updates: Toponymy
# treats them as labels for the prefix of the current vectors. When a new cluster is
# sufficiently similar to a prior cluster, it reuses the previous topic name instead
# of sending that prompt to the LLM.

# %%
updated_topic_model = make_topic_model(
    previous_cluster_layers=initial_topic_model.cluster_layers_
)

# %%
# %%time
updated_topic_model.fit(
    text,
    embedding_vectors=embedding_vectors,
    clusterable_vectors=clusterable_vectors,
)

# %% [markdown]
# The updated model contains topic labels for all documents, including the 1% that
# was held back from the initial fit.

# %%
topics_per_document = [
    cluster_layer.topic_name_vector
    for cluster_layer in updated_topic_model.cluster_layers_
]

topics_per_document[-1][remaining_indices]

# %% [markdown]
# We can also inspect how many topic names were reused in each layer during the
# incremental update.

# %%
reused_topic_counts = [
    len(cluster_layer._prior_topic_reuse_indices)
    for cluster_layer in updated_topic_model.cluster_layers_
]

total_topic_counts = [
    len(cluster_layer.topic_names)
    for cluster_layer in updated_topic_model.cluster_layers_
]

pd.DataFrame(
    {
        "layer": np.arange(len(updated_topic_model.cluster_layers_)),
        "reused_topics": reused_topic_counts,
        "total_topics": total_topic_counts,
        "llm_named_topics": (
            np.array(total_topic_counts) - np.array(reused_topic_counts)
        ),
    }
)

# %% [markdown]
# Finally, compare the highest-level topic names from the initial and updated runs.
# For a small incremental update, most of these names should remain stable.

# %%
pd.DataFrame(
    {
        "initial_top_level_topics": pd.Series(initial_topic_model.topic_names_[-1]),
        "updated_top_level_topics": pd.Series(updated_topic_model.topic_names_[-1]),
    }
)
