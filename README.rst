===========
Toponymy
===========

.. image:: doc/toponymy_text_horizontal.png
  :width: 600
  :align: center
  :alt: Toponymy

The package name Toponymy is derived from the Greek topos ‘place’ + onuma ‘name’.  Thus, the naming of places.  
The goal of Toponymy is to put names to places in the space of information. This could be a corpus of documents,
in which case Toponymy can be viewed as a topic naming library.  It could also be a collection of images, in which case
Toponymy could be used to name the themes of the images.  The goal is to provide a names that can allow a user to
navigate through the space of information in a meaningful way.

Toponymy is designed to scale to very large corpora and collections, providing meaningful names on multiple scales,
from broad themes to fine-grained topics.  We make use a custom clustering methods, information extraction, 
and large language models to power this. The library is designed to be flexible and easy to use.

As of now this is an beta version of the library. Things can and will break right now.
We welcome feedback, use cases and feature suggestions.

------------------
Basic Installation
------------------

You can install Toponymy using

.. code-block:: bash

    pip install toponymy


To install the latest version of Toponymy from source you can do so by cloning the repository and running:

.. code-block:: bash

    git clone https://github.com/TutteInstitute/toponymy
    cd toponymy
    pip install .

-----------
Basic Usage
-----------

We will need documents, document vectors and a low dimensional representation of these document vector to construct
a representation.  This can be very expensive without a GPU so we recommend storing and reloading these vectors as 
needed. For ease of experimentation we have precomputed and stored such vectors for the `20-Newsgroups dataset <http://qwone.com/~jason/20Newsgroups/>`_  
on hugging face.  Code to retrieve these vectors is below.

.. code-block:: python

    pip install pandas

    import numpy as np
    import pandas as pd
    newsgroups_df = pd.read_parquet("hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet")
    text = newsgroups_df["post"].str.strip().values
    document_vectors = np.stack(newsgroups_df["embedding"].values)
    document_map = np.stack(newsgroups_df["map"].values)

Toponymy also requires an embedding model for determining which of the documents will be most relevant to each
of our clusters.  This doesn't have to be the embedding model that our documents were embedded with but it 
should be similar.

.. code-block:: python

    pip install sentence_transformers

    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


Once the low-dimensional representation is available (``document_map`` in this case), we can do the topic naming. 
Toponymy will make use of a clusterer (such as ``ToponymyClusterer``) to create a balanced hierarchical layered 
clustering of our documents. It will then use a variety of sampling and summarization techniques to construct prompts 
describing each cluster to pass to a large language model (LLM).  If you would like to experiment with testing 
various cluster parameters in order construct cluster layers appropriate to your data feel free to cluster 
your data ahead of time via:

.. code-block:: python

    from toponymy import ToponymyClusterer
    clusterer = ToponymyClusterer(min_clusters=4)
    clusterer.fit(clusterable_vectors=document_map, embedding_vectors=document_vectors)
    for i, layer in enumerate(clusterer.cluster_layers_):
        print(f'{len(np.unique(layer.cluster_labels))-1} clusters in layer {i}')

    428 clusters in layer 0
    136 clusters in layer 1
    42 clusters in layer 2
    14 clusters in layer 3
    5 clusters in layer 4

Toponymy supports multiple LLMs, including Cohere, OpenAI, and Anthropic via service calls, and local models via
Huggingface and LlamaCpp. Here we show an example using OpenAI. The following code will generate a topic naming
for the documents in the data set using an ``embedding_model``, ``document_vectors`` and ``document_map`` created above.

.. code-block:: python

    from toponymy import Toponymy, KeyphraseBuilder
    from toponymy.llm_wrappers import OpenAI

    openai_api_key = open("openai_key.txt").read().strip()
    llm = OpenAI('openai_api_key')

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedding_model,
        clusterer=clusterer,
        object_description="newsgroup posts",
        corpus_description="20-newsgroups dataset",
        exemplar_delimiters=["<EXAMPLE_POST>\n","\n</EXAMPLE_POST>\n\n"],
    )
    topic_model.fit(text, document_vectors, document_map)

    topic_names = topic_model.topic_names_
    topics_per_document = [cluster_layer.topic_name_vector for cluster_layer in topic_model.cluster_layers_]
    
``topic_names`` is a list of lists which can be used to explore the unique topic names in each layer or resolution.
Let's examine the last two layers of topics.

.. code-block:: python

    topic_names[-2:]

    [['NHL Playoffs and Player Analysis',
    'Major League Baseball Analysis',
    'Space Exploration and Technology Innovations',
    'Encryption Policy and Government Surveillance',
    'Health and Alternative Treatments',
    'Israeli-Palestinian and Lebanese Conflicts',
    'Automotive Performance and Safety',
    'Christian Theology and Debates',
    'Waco Siege and Government Accountability',
    'Debates on Morality and Free Speech',
    'Gun Rights and Legislation',
    'X Window System and Graphics Software',
    'Hard Drive Technologies and Troubleshooting',
    'Vintage Computer Hardware and Upgrades'],
    ['Sports Analysis',
    'Religion and Government Accountability',
    'Automotive Performance and Safety',
    'X Window System and Graphics Software',
    'Computer Hardware']]


``topics_per_document`` contains topic labels for each document, with one list for each level of resultion in our 
cluster layers.  In our above case this will be a list of 5 layers each containing a list of 18,170 topic names.  
Documents that aren't contained within a cluster at a given layer are given the topic ``Unlabelled``.

.. code-block:: python
    
    topics_per_document
    

    [array(['Unlabelled',
            'Discussion on VESA Local Bus Video Cards and Performance',
            'Unlabelled', ...,
            'Cooling Solutions and Components for CPUs and Power Supplies',
            'Algorithms for Finding Sphere from Four Points in 3D',
            'Automotive Discussions on Performance Cars and Specifications'], dtype=object),
    array(['NHL Playoff Analysis and Predictions',
            'Graphics Card Performance and Benchmark Discussions',
            'Armenian Genocide and Turkish Atrocities Discourse', ...,
            'Cooling Solutions and Components for CPUs and Power Supplies',
            'Algorithms for 3D Polygon Processing and Geometry',
            'Discussions on SUVs and Performance Cars'], dtype=object),
    array(['NHL Playoff Analysis and Predictions',
            'Video Card Drivers and Performance',
            'Armenian Genocide and Turkish Atrocities', ..., 'Unlabelled',
            'Unlabelled', 'Automotive Performance and Used Cars'], dtype=object),
    array(['NHL Playoffs and Player Analysis',
            'Vintage Computer Hardware and Upgrades', 'Unlabelled', ...,
            'Unlabelled', 'X Window System and Graphics Software',
            'Automotive Performance and Safety'], dtype=object),
    array(['Sports Analysis', 'Computer Hardware', 'Unlabelled', ...,
            'Unlabelled', 'X Window System and Graphics Software',
            'Automotive Performance and Safety'], dtype=object)]

At this point we recommend that you explore your data and topic names with an interactive visualization library.  
Our `DataMapPlot <https://github.com/TutteInstitute/datamapplot>`_ library is particularly well suited to exploring 
data maps along with layers of topic names.  It takes requires our ``document_map``, ``document_vectors`` and newly created ``topics_per_document``.

-------------------
Vector Construction
-------------------

If you do not have ready made document vectors and low dimensional representations of your data you will need to compute 
your own. For faster encoding change device to: "cuda", "mps", "npu" or "cpu" depending on hardware availability. Alternatively,
one could make use of an API call to embedding service.  Embedding wrappers can be found in:

.. code-block:: python

    from toponymy.embedding_wrappers import OpenAIEmbedder

or the embedding wrapper of your choice. Once we generate document vectors we will need to construct a low dimensional representation.  
Here we do that via our UMAP library.  

.. code-block:: python

    pip install umap-learn
    pip install pandas
    pip install sentence_transformers

    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import umap

    newsgroups_df = pd.read_parquet("hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet")
    text = newsgroups_df["post"].str.strip().values
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    document_vectors = embedding_model.encode(text, show_progress_bar=True)
    document_map = umap.UMAP(metric='cosine').fit_transform(document_vectors)

-------
License
-------

Toponymy is MIT licensed. See the LICENSE file for details.

------------
Contributing
------------

Contributions are more than welcome! If you have ideas for features of projects please get in touch. Everything from
code to notebooks to examples and documentation are all *equally valuable* so please don't feel you can't contribute.
To contribute please `fork the project <https://github.com/TutteInstitute/toponymy/fork>`_ make your
changes and submit a pull request. We will do our best to work through any issues with you and get your code merged in.
