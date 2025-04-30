Installation
============

To get started using Toponymy you first have to get Toponymy installed. Fortunately, this is easy to do.
Toponymy is available on PyPI, so you can install it using pip:

.. code-block:: shell

    pip install toponymy

If you are using conda, you can also install Toponymy using conda-forge (comming soon):
.. code-block:: shell

    conda install -c conda-forge toponymy

To get the latest and greatest version of Toponymy, you can also install it from source.
.. code-block:: shell

    git clone https://github.com/TutteInstitute/toponymy.git
    cd toponymy
    pip install -r requirements.txt
    pip install -e .

This provides the basics to get started, but Toponymy makes use of LLMs and embedding models as well. Instead 
of havign a very large dependency list, we allow you to install what you need, and different options 
will become available within Toponymy based on what you have installed.

If you are interested in working with local models, you can install ``llama-cpp-python`` or huggingface's ``transformers``.
This will enable you to use the LlamaCpp and HuggingFace models, respectively. For LlamaCpp you will also need to download
the models you want to use. You can find the models on HuggingFace or other model repositories. For the HuggingFace models, 
you can specify the model name and the HuggingFace class will handle the downloading for you.

.. code-block:: shell

    pip install llama-cpp-python
    pip install transformers

If you want to use LLMs from OpenAI, Cohere, or Anthropic, you will need to install the ``openai``, ``cohere`` or ``anthropic`` 
packages respectively.

.. code-block:: shell

    pip install openai
    pip install cohere
    pip install anthropic

If you are using models on Azure's AI Foundry you will need to install the azure-ai-inference package.
.. code-block:: shell

    pip install azure-ai-inference

You can install some or all of these packages, but you will need to install at least one of them to use Toponymy to its fullest.