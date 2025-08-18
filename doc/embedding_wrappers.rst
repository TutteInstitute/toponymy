==================
Embedding Wrappers
==================

This document provides an overview of the embedding wrappers available in the `embedding_wrappers` module. These wrappers allow various text embedding services and APIs to be used seamlessly within Toponymy for generating vector representations of keyphrases and topic names.

Embedding models play a crucial role in Toponymy's topic naming process. While your documents may already have embeddings from any model, Toponymy uses a separate embedding model internally to encode and compare keyphrases and topic names. This allows for semantic similarity calculations that ensure diversity among selected keyphrases and enable effective topic name disambiguation.

-----------------------------
Installing required libraries
-----------------------------

Each wrapper may require specific libraries to be installed. You can install them using pip or uv.
For example, to enable the use of the OpenAI embedding wrapper you would need to install the `openai` library:

.. code-block:: bash

    pip install openai

The following wrappers require the following libraries:

- `openai`: For the OpenAI embedding wrapper.
- `anthropic`: For the Anthropic embedding wrapper.
- `cohere`: For the Cohere embedding wrapper.
- `azure-ai-inference`: For the Azure AI embedding wrapper.
- `mistralai`: For the Mistral embedding wrapper.
- `requests`: For the Voyage AI embedding wrapper.
- `vllm`: For the vLLM embedding wrapper.

--------------------
Role in Topic Naming
--------------------

Understanding the role of embeddings in Toponymy's workflow is essential for choosing the right embedding model. Unlike document embeddings, which need to capture the full semantic content of potentially long texts, the embedding models used by Toponymy focus specifically on short keyphrases and topic names. This creates different requirements and opens up different optimization opportunities.

The primary use cases for embeddings in Toponymy include **keyphrase selection diversity**, where embeddings ensure that selected keyphrases for each cluster represent diverse aspects of the topic rather than near-duplicates; **topic name disambiguation**, where semantically similar topic names are identified and re-prompted to create more distinctive labels; and **subtopic selection**, where embeddings help select representative subtopic names from lower clustering layers to inform higher-level topic naming.

Since these embeddings are used for comparison rather than absolute representation, the choice of embedding model is somewhat flexible. The key requirements are reasonable semantic understanding of domain-specific terminology, consistency in representation, and computational efficiency for processing potentially thousands of keyphrases. You don't necessarily need the most powerful or expensive embedding model—a good balance of quality and speed is often optimal.

Most embedding wrappers in Toponymy process texts in batches of 96 items to balance API efficiency with memory usage. They include progress bars for long-running operations and handle API rate limiting and retry logic automatically. All wrappers return standardized numpy arrays, ensuring consistent interfaces regardless of the underlying embedding service.

-----------------
Available Wrappers
-----------------

API-Based Embedding Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**OpenAIEmbedder**

The OpenAIEmbedder provides access to OpenAI's text embedding models through their API. OpenAI's embedding models are known for their strong performance across diverse domains and languages.

.. code-block:: python

    from toponymy.embedding_wrappers import OpenAIEmbedder
    
    # Initialize with OpenAI API
    embedder = OpenAIEmbedder(
        api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
        model="text-embedding-3-small",  # Cost-effective and performant
        base_url="https://api.openai.com/v1"  # Optional custom endpoint
    )
    
    # Generate embeddings for keyphrases
    keyphrases = ["machine learning", "neural networks", "deep learning"]
    embeddings = embedder.encode(keyphrases, show_progress_bar=True)

**Available Models:**

- `text-embedding-3-small`: 1536 dimensions, $0.02/1M tokens (recommended)
- `text-embedding-3-large`: 3072 dimensions, $0.13/1M tokens (higher quality)
- `text-embedding-ada-002`: 1536 dimensions, $0.10/1M tokens (legacy)

**CohereEmbedder**

The CohereEmbedder provides access to Cohere's embedding models, which are optimized for search and retrieval tasks. This makes them particularly well-suited for Toponymy's keyphrase comparison needs.

.. code-block:: python

    from toponymy.embedding_wrappers import CohereEmbedder
    
    # Initialize with Cohere API
    embedder = CohereEmbedder(
        api_key="your-cohere-api-key",  # Or set CO_API_KEY env var
        model="embed-multilingual-v3.0",  # Supports multiple languages
        base_url=None,  # Optional custom endpoint
        httpx_client=None  # Optional custom HTTP client
    )
    
    # Generate embeddings
    embeddings = embedder.encode(
        texts=["category theory", "topology", "algebra"],
        show_progress_bar=True
    )

The Cohere embedder uses `input_type="search_query"` by default, which is optimized for comparing keyphrases and topic names against document content.

**AnthropicEmbedder**

The AnthropicEmbedder provides access to embedding capabilities through Anthropic's API. While primarily known for their language models, Anthropic also offers embedding services.

.. code-block:: python

    from toponymy.embedding_wrappers import AnthropicEmbedder
    
    # Initialize with Anthropic API
    embedder = AnthropicEmbedder(
        api_key="your-anthropic-api-key",  # Or set ANTHROPIC_API_KEY env var
        model="claude-3-haiku-20240307",  # Model for embedding generation
        base_url=None,  # Optional custom endpoint
        httpx_client=None  # Optional custom HTTP client
    )

**Note**: The Anthropic embedder processes texts individually rather than in batches, which may result in slower processing for large keyphrase lists.

**AzureAIEmbedder**

The AzureAIEmbedder provides access to embedding models deployed through Azure AI services, offering enterprise-grade infrastructure with comprehensive compliance and security features.

.. code-block:: python

    from toponymy.embedding_wrappers import AzureAIEmbedder
    
    # Initialize with Azure AI
    embedder = AzureAIEmbedder(
        api_key="your-azure-api-key",
        endpoint="https://your-endpoint.inference.ai.azure.com",
        model="your-deployed-embedding-model"
    )
    
    # Generate embeddings with automatic retry logic
    embeddings = embedder.encode(
        texts=["machine learning", "data science", "artificial intelligence"],
        show_progress_bar=True
    )

The Azure AI embedder includes built-in retry logic with exponential backoff to handle transient API failures gracefully.

**MistralEmbedder**

The MistralEmbedder provides access to Mistral's embedding models through their API, offering competitive performance and pricing for text embedding tasks.

.. code-block:: python

    from toponymy.embedding_wrappers import MistralEmbedder
    
    # Initialize with Mistral API
    embedder = MistralEmbedder(
        api_key="your-mistral-api-key",
        model="mistral-embed"  # Mistral's embedding model
    )
    
    # Generate embeddings
    embeddings = embedder.encode(
        texts=["natural language processing", "text mining", "information retrieval"],
        show_progress_bar=True
    )

**VoyageAIEmbedder**

The VoyageAIEmbedder provides access to Voyage AI's embedding models, which are specifically optimized for retrieval and search applications, making them well-suited for Toponymy's needs.

.. code-block:: python

    from toponymy.embedding_wrappers import VoyageAIEmbedder
    
    # Initialize with Voyage AI API
    embedder = VoyageAIEmbedder(
        api_key="your-voyage-api-key",
        model="voyage-2"  # High-performance embedding model
    )
    
    # Generate embeddings
    embeddings = embedder.encode(
        texts=["computer vision", "image processing", "pattern recognition"],
        show_progress_bar=True
    )

Local Embedding Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~

**VLLMEmbedder**

The VLLMEmbedder provides high-performance local embedding generation using the vLLM library. This wrapper is ideal for scenarios requiring data privacy, high throughput, or freedom from API costs.

.. code-block:: python

    from toponymy.embedding_wrappers import VLLMEmbedder
    
    # Initialize with a local embedding model
    embedder = VLLMEmbedder(
        model="all-MiniLM-L6-v2",  # Popular and efficient embedding model
        kwargs={
            "tensor_parallel_size": 1,  # Number of GPUs for tensor parallelism
            "gpu_memory_utilization": 0.8,  # Fraction of GPU memory to use
            "max_model_len": 512  # Maximum sequence length
        }
    )
    
    # Generate embeddings locally
    embeddings = embedder.encode(
        texts=["distributed systems", "microservices", "containerization"],
        show_progress_bar=True
    )

**Supported Models:**

Popular embedding models that work well with vLLM include:

- `all-MiniLM-L6-v2`: Fast and efficient, good for most use cases
- `all-mpnet-base-v2`: Higher quality, more resource intensive
- `sentence-transformers/all-MiniLM-L6-v2`: Explicit sentence-transformers model
- `intfloat/e5-base-v2`: Strong performance on various tasks

Using Local Embedding Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While not technically a wrapper, many users find that using SentenceTransformers directly provides an excellent balance of simplicity, performance, and model selection for Toponymy's embedding needs:

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    
    # Initialize a local embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Use directly with Toponymy
    from toponymy.toponymy import Toponymy
    
    topic_model = Toponymy(
        # ... other parameters ...
        text_embedding_model=embedding_model,  # Pass the model directly
    )

This approach provides direct access to the extensive SentenceTransformers model library and avoids the overhead of wrapper layers for local processing.

------------------------------------
Choosing the Right Embedding Wrapper
------------------------------------

Selecting the appropriate embedding wrapper depends on several key factors that mirror but differ from the considerations for LLM wrappers. The primary dimensions include cost efficiency, processing speed, data privacy requirements, model quality, and integration complexity. However, because embedding models are used for comparison tasks rather than generation, and because keyphrases are typically much shorter than full documents, the requirements are often less stringent than for document embeddings.

**Cost Considerations**

For API-based embedding services, costs are typically much lower than LLM costs because embeddings are smaller and require less computation than text generation. However, for projects processing large numbers of keyphrases, costs can still accumulate:

.. list-table:: Embedding Cost Comparison (approximate)
   :header-rows: 1
   :widths: 30 25 25 20

   * - Provider
     - Model
     - Cost per 1M tokens
     - Notes
   * - OpenAI
     - text-embedding-3-small
     - $0.02
     - **Recommended**
   * - Cohere
     - embed-multilingual-v3.0
     - $0.10
     - Multilingual support
   * - OpenAI
     - text-embedding-3-large
     - $0.13
     - Higher dimensionality
   * - Voyage AI
     - voyage-2
     - $0.10
     - Optimized for retrieval
   * - Local (vLLM)
     - all-MiniLM-L6-v2
     - Hardware only
     - One-time setup cost

**Quality vs. Speed Trade-offs**

For Toponymy's specific use cases, embedding quality requirements are moderate because the models are used for relative comparisons rather than absolute semantic understanding. This means that many embedding models will perform adequately:

**High Performance (Recommended):**

.. code-block:: python

    # Best balance of cost, speed, and quality
    from toponymy.embedding_wrappers import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    
    # Good multilingual support
    from toponymy.embedding_wrappers import CohereEmbedder
    embedder = CohereEmbedder(model="embed-multilingual-v3.0")
    
    # Local processing with good performance
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

**Budget-Conscious:**

.. code-block:: python

    # Most cost-effective API option
    from toponymy.embedding_wrappers import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    
    # Free local processing (after hardware costs)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

**High Quality:**

.. code-block:: python

    # Higher dimensional embeddings for better precision
    from toponymy.embedding_wrappers import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-large")
    
    # Specialized for retrieval tasks
    from toponymy.embedding_wrappers import VoyageAIEmbedder
    embedder = VoyageAIEmbedder(model="voyage-2")

**Privacy and Security**

For organizations with strict data privacy requirements, local embedding models are essential:

.. code-block:: python

    # Complete data privacy with local processing
    from toponymy.embedding_wrappers import VLLMEmbedder
    embedder = VLLMEmbedder(model="all-MiniLM-L6-v2")
    
    # Alternative local approach
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

**Enterprise Integration**

For enterprise environments, Azure AI integration often provides the best fit with existing infrastructure:

.. code-block:: python

    # Enterprise-grade with compliance features
    from toponymy.embedding_wrappers import AzureAIEmbedder
    embedder = AzureAIEmbedder(
        api_key="your-azure-api-key",
        endpoint="https://your-endpoint.inference.ai.azure.com",
        model="your-deployed-embedding-model"
    )

---------------------
Performance Guidance
---------------------

Embedding performance in Toponymy is generally not a bottleneck compared to LLM processing, but understanding performance characteristics can help optimize your workflows, especially when processing large numbers of keyphrases.

**Batch Size Optimization**

Most embedding wrappers use a default batch size of 96 items, which provides a good balance between API efficiency and memory usage. For local models, you may be able to increase batch sizes:

.. code-block:: python

    # For local models, larger batches may be more efficient
    from toponymy.embedding_wrappers import VLLMEmbedder
    
    embedder = VLLMEmbedder(
        model="all-MiniLM-L6-v2",
        kwargs={
            "max_num_seqs": 256,  # Process more sequences in parallel
            "gpu_memory_utilization": 0.9
        }
    )

**Processing Large Keyphrase Lists**

When working with very large keyphrase lists (>10,000 items), consider the following optimizations:

.. code-block:: python

    # Enable progress bars for long-running operations
    embeddings = embedder.encode(
        texts=large_keyphrase_list,
        show_progress_bar=True,  # Monitor progress
        verbose=True  # Additional logging
    )
    
    # For very large lists, consider processing in chunks
    import numpy as np
    
    chunk_size = 5000
    all_embeddings = []
    
    for i in range(0, len(large_keyphrase_list), chunk_size):
        chunk = large_keyphrase_list[i:i+chunk_size]
        chunk_embeddings = embedder.encode(chunk, show_progress_bar=True)
        all_embeddings.append(chunk_embeddings)
    
    final_embeddings = np.vstack(all_embeddings)

**Memory Management**

For memory-constrained environments, consider using smaller embedding models or processing in smaller batches:

.. code-block:: python

    # Smaller model for memory-constrained environments
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~90MB model
    
    # Alternative: Even smaller model
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ~60MB model

-----------------------
Integration with Toponymy
-----------------------

Embedding wrappers integrate seamlessly with Toponymy's main workflow. Here's how to use them effectively in different scenarios:

**Basic Usage**

.. code-block:: python

    from toponymy.toponymy import Toponymy
    from toponymy.embedding_wrappers import OpenAIEmbedder
    from toponymy.llm_wrappers import OpenAI
    
    # Initialize embedding and LLM models
    embedding_model = OpenAIEmbedder(
        api_key="your-openai-api-key",
        model="text-embedding-3-small"
    )
    
    llm_model = OpenAI(
        api_key="your-openai-api-key",
        model="gpt-4o-mini"
    )
    
    # Create Toponymy instance
    topic_model = Toponymy(
        llm_wrapper=llm_model,
        text_embedding_model=embedding_model,
        # ... other parameters ...
    )

**Mixed Local and API Approach**

You can use local embeddings with API-based LLMs, or vice versa, depending on your specific requirements:

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    from toponymy.llm_wrappers import OpenAI
    
    # Local embeddings for privacy, API LLM for quality
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm_model = OpenAI(api_key="your-api-key", model="gpt-4o-mini")
    
    topic_model = Toponymy(
        llm_wrapper=llm_model,
        text_embedding_model=embedding_model,
    )

**Enterprise Configuration**

For enterprise environments, Azure AI services can provide a unified platform:

.. code-block:: python

    from toponymy.embedding_wrappers import AzureAIEmbedder
    from toponymy.llm_wrappers import AzureAI
    
    # Unified Azure AI configuration
    embedding_model = AzureAIEmbedder(
        api_key="your-azure-api-key",
        endpoint="https://your-embedding-endpoint.inference.ai.azure.com",
        model="your-embedding-model"
    )
    
    llm_model = AzureAI(
        api_key="your-azure-api-key",
        endpoint="https://your-llm-endpoint.inference.ai.azure.com",
        model="your-llm-model"
    )
    
    topic_model = Toponymy(
        llm_wrapper=llm_model,
        text_embedding_model=embedding_model,
    )

**Troubleshooting Common Issues**

**Authentication Errors:**

Ensure API keys are correctly set either as parameters or environment variables:

.. code-block:: bash

    export OPENAI_API_KEY="your-openai-api-key"
    export CO_API_KEY="your-cohere-api-key"
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

**Memory Issues with Local Models:**

Reduce batch sizes or use smaller models:

.. code-block:: python

    # Reduce memory usage
    from toponymy.embedding_wrappers import VLLMEmbedder
    
    embedder = VLLMEmbedder(
        model="all-MiniLM-L6-v2",
        kwargs={
            "gpu_memory_utilization": 0.5,  # Use less GPU memory
            "max_model_len": 256  # Shorter sequences
        }
    )

**Rate Limiting Issues:**

Most wrappers include automatic retry logic, but you can adjust batch sizes if needed:

.. code-block:: python

    # Process in smaller batches to avoid rate limits
    import time
    
    batch_size = 48  # Smaller than default 96
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embedder.encode(batch)
        all_embeddings.append(embeddings)
        time.sleep(1)  # Brief pause between batches
    
    final_embeddings = np.vstack(all_embeddings)

-----------------------
Recommended Configurations
-----------------------

Based on common use cases and requirements, here are recommended embedding configurations:

**For Getting Started:**

.. code-block:: python

    # Simple, reliable, cost-effective
    from toponymy.embedding_wrappers import OpenAIEmbedder
    
    embedder = OpenAIEmbedder(
        api_key="your-openai-api-key",
        model="text-embedding-3-small"
    )

**For Budget-Conscious Projects:**

.. code-block:: python

    # Free after initial setup, good performance
    from sentence_transformers import SentenceTransformer
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

**For Maximum Privacy:**

.. code-block:: python

    # Complete local processing
    from toponymy.embedding_wrappers import VLLMEmbedder
    
    embedder = VLLMEmbedder(model="all-MiniLM-L6-v2")

**For Multilingual Content:**

.. code-block:: python

    # Strong multilingual support
    from toponymy.embedding_wrappers import CohereEmbedder
    
    embedder = CohereEmbedder(
        api_key="your-cohere-api-key",
        model="embed-multilingual-v3.0"
    )

**For Enterprise Environments:**

.. code-block:: python

    # Enterprise-grade with compliance
    from toponymy.embedding_wrappers import AzureAIEmbedder
    
    embedder = AzureAIEmbedder(
        api_key="your-azure-api-key",
        endpoint="https://your-endpoint.inference.ai.azure.com",
        model="your-deployed-model"
    )

**For High-Performance Requirements:**

.. code-block:: python

    # High-quality embeddings for demanding applications
    from toponymy.embedding_wrappers import OpenAIEmbedder
    
    embedder = OpenAIEmbedder(
        api_key="your-openai-api-key",
        model="text-embedding-3-large"
    )

The choice of embedding wrapper should align with your overall project architecture, security requirements, and performance needs. Since embedding costs are typically much lower than LLM costs, it's often worth choosing a slightly higher-quality option for better topic naming results, especially for production applications where topic quality is important.