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

As of now this is an early beta version of the library. Things can and will break right now.
We welcome feedback, use cases and feature suggestions.

------------------
Basic Installation
------------------

For now install the latest version of Toponymy from source you can do so by cloning the repository and running:

.. code-block:: bash

    git clone https://github.com/TutteInstitute/toponymy
    cd toponymy
    pip install .

-----------
Basic Usage
-----------

We will need documents, document vectors and a low dimensional representation of these document vector to construct
a representation.  This can be very expensive without a GPU so we recommend storing and reloading these vectors as 
needed.  For faster encoding change device to: "cuda", "mps", "npu" or "cpu" depending on hardware availability.  Once we 
generate document vectors we will need to construct a low dimensional representation.  Here we do that via our UMAP library.

.. code-block:: python

    data = pd.read_csv("hf://datasets/CShorten/ML-ArXiv-Papers/ML-Arxiv-Papers.csv")
    text =data.title+" "+data.abstract
    embedding_model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2", device="cpu") 
    document_vectors = embedding_model.encode(text, show_progress_bar=True)
    document_map = umap.UMAP(metric='cosine').fit_transform(document_vectors)

Once the low-dimensional representation is available (``document_map`` in this case), we can do the topic naming. 
Toponymy supports multiple LLMs, including Cohere, OpenAI, and Anthropic via service calls, and local models via
Huggingface and LlamaCpp. Here we show an example using Cohere.  The following code will generate a topic naming
for the documents in the data set using the embedding_model, document_vectors and document_map created above.

.. code-block:: python

    from toponymy import Toponymy, ToponymyClusterer, ClusterLayerText, KeyphraseBuilder
    from toponymy.llm_wrappers import Cohere

    llm = Cohere(api_key='your_api_key')

    topic_model = Toponymy(
        llm=llm,
        embedding_model=embedding_model,
        layer_class=ClusterLayerText,
        clusterer=ToponymyClusterer(),
        keyphrase_builder=KeyphraseBuilder(),
        object_description="paper titles and abstracts",
        corpus_description="AI papers",
    )
    topic_model.fit(text, document_vectors, document_map)

    topic_names = topic_model.topic_names_
    topics_per_document = topic_model.topic_name_vectors_

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
