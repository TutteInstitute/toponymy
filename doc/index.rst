.. Toponymy documentation master file, created by
   sphinx-quickstart on Wed Apr 30 14:55:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: toponymy_text_horizontal.png
  :width: 600
  :alt: Toponymy logo
  :align: center

Toponymy: Topic Modelling in Embedding Space
============================================

The package name Toponymy is derived from the Greek topos ‘place’ + onuma ‘name’.  Thus, the naming of places.  
The goal of Toponymy is to put names to places in the space of information. This could be a corpus of documents,
in which case Toponymy can be viewed as a topic naming library.  It could also be a collection of images, in which case
Toponymy could be used to name the themes of the images.  The goal is to provide a names that can allow a user to
navigate through the space of information in a meaningful way.

Toponymy is designed to scale to very large corpora and collections, providing meaningful names on multiple scales,
from broad themes to fine-grained topics.  We make use a custom clustering methods, information extraction, 
and large language models to power this. The library is designed to be flexible and easy to use.

**Quick Install**

.. code-block:: bash

   pip install toponymy

**Beta Status**

   Please note: Toponymy is currently in beta. APIs may evolve. We welcome
   `feedback and contributions <Your Contribution/Issues Link>`_!

User Guide
----------

Toponymy is designed to be easy to use.  The user guide provides a quick start to the library,
and a tour of some of the richer functionality and uses cases.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   intro
   installation
   basic_usage

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage:

   clusterers
   clustering_options
   keyphrases
   llm_wrappers
   embedding_wrappers
   cluster_layers

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   ml_papers
   images
   long_documents
   audio_samples

.. toctree::
   :maxdepth: 1
   :caption: Sundries:

   api
   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`