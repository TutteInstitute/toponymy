============
LLM Wrappers
============

This document provides an overview of the LLM wrappers available in the `llm_wrappers` module. These wrappers
allow various LLM services and APIs to be used seamlessly within Toponymy.

-----------------------------
Installing required libraries
-----------------------------

Each wrapper may require specific libraries to be installed. You can install them using pip or uv.
For example, to enable the use of the OpenAI LLM wrapper you would need to install the `openai` library:
```bash
pip install openai
```
The following wrappers require the following libraries:

- `openai`: For the OpenAI LLM wrapper.
- `anthropic`: For the Anthropic LLM wrapper.
- `google-generativeai`: For the Google Generative AI wrapper.
- `azure-ai-inference`: For the Azure OpenAI wrapper.
- `cohere`: For the Cohere LLM wrapper.
- `ollama`: For the Ollama LLM wrapper (You will also have to install ollama itself).
- `llama-cpp-python`: For the Llama.cpp wrapper (this may require a more complex installation process depending on your system).
- `transformers` and `huggingface-hub`: For the Hugging Face LLM wrapper.
- `vllm`: For the vLLM wrapper.

--------------
Basic Wrappers
--------------

Basic wrappers provide synchronous access to various LLM services and local models, processing requests sequentially one at a time. This approach makes them particularly well-suited for simple use cases, debugging workflows, or situations where you need to carefully control the rate of API calls to avoid hitting rate limits. Each basic wrapper implements the `LLMWrapper` interface, ensuring a consistent API across different model providers.

These wrappers are designed to handle the core functionality needed for topic modeling workflows. They can generate descriptive names for individual topics based on the most representative documents or keywords, as well as create names for groups of related topics in hierarchical clustering scenarios. The wrappers include built-in temperature control to adjust the creativity and randomness of generated text, automatic retry logic with exponential backoff to handle transient API failures gracefully, and support for system prompts where the underlying model allows it, enabling better instruction following and more consistent outputs.

Local Model Wrappers
~~~~~~~~~~~~~~~~~~~~

**LlamaCpp**

The LlamaCpp wrapper provides access to local GGUF format models through the llama-cpp-python library. This wrapper is ideal for running models locally without requiring API keys or internet connectivity.

.. code-block:: python

    from toponymy.llm_wrappers import LlamaCpp
    
    # Initialize with a local model file
    llm = LlamaCpp(
        model_path="/path/to/your/model.gguf",
        llm_specific_instructions="Be concise and descriptive",
        n_ctx=4096,  # Context window size
        n_gpu_layers=35  # Number of layers to offload to GPU
    )

**Note**: LlamaCpp does not support system prompts, so all instructions must be included in the main prompt.

**HuggingFace**

The HuggingFace wrapper provides access to models hosted on Hugging Face Hub. This wrapper downloads and runs models locally using the transformers library.

.. code-block:: python

    from toponymy.llm_wrappers import HuggingFace
    
    # Initialize with a Hugging Face model
    llm = HuggingFace(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        llm_specific_instructions="Generate clear, descriptive topic names",
        device_map="auto",  # Automatically map to available devices
        torch_dtype="float16"  # Use half precision for efficiency
    )

**VLLM**

The VLLM wrapper provides high-performance inference for Hugging Face models using the vLLM library. It offers better throughput than the standard HuggingFace wrapper, especially for longer sequences.

.. code-block:: python

    from toponymy.llm_wrappers import VLLM
    
    # Initialize with vLLM for better performance
    llm = VLLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        llm_specific_instructions="Focus on clarity and accuracy",
        tensor_parallel_size=1,  # Number of GPUs for tensor parallelism
        max_model_len=4096  # Maximum sequence length
    )

API-Based Wrappers
~~~~~~~~~~~~~~~~~~

**OpenAI**

The OpenAI wrapper provides access to OpenAI's GPT models through their API. It supports the latest GPT models and includes full system prompt support.

.. code-block:: python

    from toponymy.llm_wrappers import OpenAI
    
    # Initialize with OpenAI API
    llm = OpenAI(
        api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
        model="gpt-4o-mini",  # Cost-effective model for topic naming
        llm_specific_instructions="Be precise and domain-appropriate",
        base_url="https://api.openai.com/v1"  # Optional custom endpoint
    )

**Anthropic**

The Anthropic wrapper provides access to Claude models through Anthropic's API. Claude models are particularly good at following complex instructions and maintaining consistency.

.. code-block:: python

    from toponymy.llm_wrappers import Anthropic
    
    # Initialize with Anthropic API
    llm = Anthropic(
        api_key="your-anthropic-api-key",  # Or set ANTHROPIC_API_KEY env var
        model="claude-3-haiku-20240307",  # Fast and cost-effective
        llm_specific_instructions="Generate coherent, descriptive names"
    )

**Cohere**

The Cohere wrapper provides access to Cohere's Command models, which offer good performance for text generation tasks at competitive pricing.

.. code-block:: python

    from toponymy.llm_wrappers import Cohere
    
    # Initialize with Cohere API
    llm = Cohere(
        api_key="your-cohere-api-key",  # Or set CO_API_KEY env var
        model="command-r-08-2024",  # Balanced performance and cost
        llm_specific_instructions="Keep names concise but informative",
        base_url="https://api.cohere.com"  # Optional custom endpoint
    )

**AzureAI**

The AzureAI wrapper provides access to models through Azure AI services, supporting various models deployed on Azure infrastructure.

.. code-block:: python

    from toponymy.llm_wrappers import AzureAI
    
    # Initialize with Azure AI
    llm = AzureAI(
        api_key="your-azure-api-key",
        endpoint="https://your-endpoint.inference.ai.azure.com",
        model="your-deployed-model-name",
        llm_specific_instructions="Generate professional topic names"
    )

---------------------
Asynchronous Wrappers
---------------------

Asynchronous wrappers represent a significant step up in capability, enabling concurrent processing of multiple prompts simultaneously rather than handling them one by one. This concurrent approach can dramatically improve throughput when working with large numbers of topics, making them particularly valuable for production workflows or research projects involving substantial datasets. These wrappers implement the `AsyncLLMWrapper` interface and are especially useful when you need to process many topics at once while still respecting API rate limits and managing resources effectively.

The primary advantage of asynchronous processing lies in its ability to maximize the utilization of both network resources and API quotas. Instead of waiting for each individual request to complete before starting the next one, async wrappers can maintain multiple requests in flight simultaneously, leading to much better overall throughput. They include sophisticated rate limit management with configurable concurrency controls, allowing you to tune the number of simultaneous requests based on your API provider's limits and your specific needs. This approach also makes more efficient use of network and compute resources, as the system can continue processing other requests while waiting for responses from the API.

**Usage Pattern:**

.. code-block:: python

    import asyncio
    from toponymy.llm_wrappers import AsyncAnthropic
    
    async def process_topics():
        llm = AsyncAnthropic(
            api_key="your-api-key",
            max_concurrent_requests=5  # Control concurrency
        )
        
        # Process multiple prompts concurrently
        prompts = [prompt1, prompt2, prompt3, ...]
        results = await llm.generate_topic_names(prompts)
        
        await llm.close()  # Clean up resources
        return results
    
    # Run the async function
    results = asyncio.run(process_topics())

**Available Async Wrappers:**

**AsyncHuggingFace**

Provides asynchronous access to Hugging Face models. Primarily useful for testing async workflows with local models.

.. code-block:: python

    from toponymy.llm_wrappers import AsyncHuggingFace
    
    llm = AsyncHuggingFace(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_concurrent_requests=3  # Limited by local hardware
    )

**AsyncVLLM**

Asynchronous wrapper for vLLM, offering high-performance batch processing for local model inference.

.. code-block:: python

    from toponymy.llm_wrappers import AsyncVLLM
    
    llm = AsyncVLLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_concurrent_requests=10,
        tensor_parallel_size=2
    )

**AsyncCohere**

Provides concurrent access to Cohere's API, allowing efficient processing of multiple topic naming requests.

.. code-block:: python

    from toponymy.llm_wrappers import AsyncCohere
    
    llm = AsyncCohere(
        api_key="your-cohere-api-key",
        model="command-r-08-2024",
        max_concurrent_requests=10  # Adjust based on rate limits
    )

**AsyncAnthropic**

Enables concurrent processing with Anthropic's Claude models, ideal for large-scale topic naming tasks.

.. code-block:: python

    from toponymy.llm_wrappers import AsyncAnthropic
    
    llm = AsyncAnthropic(
        api_key="your-anthropic-api-key",
        model="claude-3-haiku-20240307",
        max_concurrent_requests=5  # Conservative rate limiting
    )

**AsyncOpenAI**

Provides asynchronous access to OpenAI's GPT models with configurable concurrency controls.

.. code-block:: python

    from toponymy.llm_wrappers import AsyncOpenAI
    
    llm = AsyncOpenAI(
        api_key="your-openai-api-key",
        model="gpt-4o-mini",
        max_concurrent_requests=8
    )

**AsyncAzureAI**

Enables concurrent processing with Azure AI services, supporting various deployed models.

.. code-block:: python

    from toponymy.llm_wrappers import AsyncAzureAI
    
    llm = AsyncAzureAI(
        api_key="your-azure-api-key",
        endpoint="https://your-endpoint.inference.ai.azure.com",
        max_concurrent_requests=6
    )

--------------
Batch Wrappers
--------------

Batch wrappers provide specialized processing modes that optimize for specific use cases where the standard synchronous or asynchronous approaches may not be ideal. These wrappers are designed to handle very large workloads efficiently, often implementing different trade-offs between processing speed, cost efficiency, and operational complexity. They are particularly valuable for research projects, large-scale data processing tasks, or situations where cost optimization is more important than immediate results.

The most significant advantage of batch processing is cost efficiency. Many API providers offer substantial discounts for batch processing in exchange for longer processing times and delayed results. This makes batch wrappers ideal for scenarios where you have large datasets to process but don't need immediate results, such as periodic analysis of accumulated documents, research projects with budget constraints, or background processing of historical data. The trade-off is that results are not available immediately, and the processing pipeline needs to be designed to handle asynchronous result retrieval and potential batch failures gracefully.

**BatchAnthropic**

The BatchAnthropic wrapper uses Anthropic's Batch API, which provides significant cost savings (50% discount) in exchange for longer processing times. This wrapper is ideal for large-scale, non-time-sensitive topic naming tasks.

**Key Features:**

- **Cost Efficiency**: 50% discount on API costs compared to standard API
- **Large Scale Processing**: Designed for processing thousands of prompts
- **Delayed Results**: Processing takes up to 24 hours to complete
- **Automatic Result Retrieval**: Handles batch submission, monitoring, and result collection

**Usage:**

.. code-block:: python

    from toponymy.llm_wrappers import BatchAnthropic
    import asyncio
    
    async def batch_process_topics():
        llm = BatchAnthropic(
            api_key="your-anthropic-api-key",
            model="claude-3-haiku-20240307"
        )
        
        # Submit a large batch of prompts
        prompts = [...]  # List of hundreds or thousands of prompts
        
        # This will submit the batch and wait for completion
        # (up to 24 hours)
        results = await llm.generate_topic_names(prompts)
        
        return results
    
    # For very large batches, consider running this as a background task
    results = asyncio.run(batch_process_topics())

**When to Use Batch Wrappers:**

- Processing large datasets (1000+ topics) where cost is a primary concern
- Non-interactive workflows where delay is acceptable
- Research projects with budget constraints
- Periodic batch processing of accumulated data

**Considerations:**

- Results are not available immediately (up to 24 hours)
- Less suitable for interactive applications
- Requires careful error handling for failed batches
- Best for homogeneous workloads (similar prompt types)

-------------------------
Choosing the Right Wrapper
-------------------------

Selecting the appropriate wrapper depends on understanding your specific requirements across several key dimensions: cost constraints, processing speed needs, data privacy requirements, and the scale of your topic modeling project. Each type of wrapper represents different trade-offs, and the optimal choice often depends on the specific context of your use case rather than a one-size-fits-all recommendation.

For users just getting started with topic modeling or working with smaller datasets, the decision process is relatively straightforward. Basic synchronous wrappers provide the simplest development experience and are easier to debug when things go wrong. If you're working with fewer than 100 topics or doing exploratory analysis where you need to iterate quickly on prompts and settings, the sequential processing approach of basic wrappers is often preferable to the additional complexity of async implementations.

Privacy considerations play an increasingly important role in wrapper selection, particularly for organizations handling sensitive data or operating in regulated industries. Local model wrappers like LlamaCpp, HuggingFace, and VLLM ensure that your data never leaves your infrastructure, providing complete control over data processing and compliance. However, this privacy comes with the trade-off of requiring suitable hardware resources and the technical expertise to manage model deployment and maintenance.

For production environments and larger-scale deployments, the choice becomes more nuanced. High throughput requirements typically favor asynchronous wrappers, which can process multiple topics concurrently and make much more efficient use of API quotas and network resources. Real-time applications benefit from async wrappers with carefully tuned concurrency limits that balance speed with API rate limit compliance. Enterprise environments often gravitate toward solutions like AzureAI that integrate well with existing infrastructure and provide the compliance and security features required for corporate deployments.

Understanding the fundamental differences between local and API-based models is crucial for making informed decisions about your topic modeling infrastructure. Local models, accessed through wrappers like LlamaCpp, HuggingFace, and VLLM, eliminate ongoing API costs entirely and provide complete data privacy since all processing happens on your own hardware. This approach is particularly attractive for organizations with strict data governance requirements or projects with long-term, high-volume processing needs where API costs would accumulate significantly over time. However, local deployment requires substantial hardware investments, particularly GPU resources for reasonable performance, along with the technical expertise to manage model deployment, updates, and maintenance.

API-based models represent the opposite trade-off, offering a pay-per-use model that eliminates hardware requirements and provides access to cutting-edge models without the need for local infrastructure management. Services like OpenAI, Anthropic, and Cohere handle all the complexities of model hosting, scaling, and maintenance, allowing you to focus on your core application logic. The downside is the ongoing cost per request and the requirement for internet connectivity, along with the need to trust third-party services with your data processing.

The choice between local and API models often comes down to volume and usage patterns. For occasional use, small projects, or experimentation, API models typically provide better value and lower barrier to entry. For high-volume, production deployments, or scenarios with strict privacy requirements, the upfront investment in local model infrastructure often pays dividends in the long term through eliminated API costs and enhanced data control.

**Model Selection:**

- **Quality priority**: GPT-4, Claude-3-Opus (higher cost, limited quality gain for topic naming)
- **Balanced**: GPT-4o-mini, Claude-3-Haiku, Command-R (recommended)
- **Cost priority**: Smaller local models, though may require more prompt engineering

**Recommended Models by Provider**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Topic naming is generally a simpler task than complex reasoning or code generation, which means that the cheapest and fastest models from each provider are typically sufficient and offer the best value. More expensive, advanced models provide limited benefit for this specific use case.

**OpenAI Models:**

.. code-block:: python

    # Recommended: Cost-effective and sufficient for topic naming
    llm = OpenAI(model="gpt-4o-mini")  # ~$0.15/1M input tokens
    
    # Alternative: Slightly better quality, higher cost
    llm = OpenAI(model="gpt-4")       # ~$2.50/1M input tokens
    
    # Not recommended for topic naming: Expensive with minimal benefit
    llm = OpenAI(model="o1-preview")   # ~$15/1M input tokens
    llm = OpenAI(model="gpt-4")        # ~$30/1M input tokens

**Anthropic Models:**

.. code-block:: python

    # Recommended: Fast, cost-effective, excellent for topic naming
    llm = Anthropic(model="claude-3-haiku-20240307")    # ~$0.25/1M input tokens
    
    # Alternative: Better instruction following, moderate cost
    llm = Anthropic(model="claude-3-5-sonnet-20241022") # ~$3/1M input tokens
    
    # Not recommended for topic naming: Expensive with minimal benefit
    llm = Anthropic(model="claude-3-opus-20240229")     # ~$15/1M input tokens

**Cohere Models:**

.. code-block:: python

    # Recommended: Excellent value for topic naming tasks
    llm = Cohere(model="command-r-08-2024")    # ~$0.15/1M input tokens
    
    # Alternative: Slightly better performance
    llm = Cohere(model="command-r-plus-08-2024") # ~$2.50/1M input tokens

**Local Model Recommendations:**

For local models, smaller instruction-tuned models typically work well for topic naming:

.. code-block:: python

    # Recommended local models (in order of preference)
    
    # 7B models - good balance of quality and resource requirements
    llm = HuggingFace(model="mistralai/Mistral-7B-Instruct-v0.3")
    llm = VLLM(model="microsoft/DialoGPT-medium")
    
    # 13B models - better quality, higher resource requirements
    llm = HuggingFace(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    # Smaller models - for resource-constrained environments
    llm = HuggingFace(model="microsoft/DialoGPT-small")

Topic naming is fundamentally different from many other natural language processing tasks that typically drive the development of large language models. While tasks like creative writing, complex reasoning, code generation, or multi-step problem solving benefit significantly from the most advanced and expensive models, topic naming has several characteristics that make it well-suited to simpler, more cost-effective models.

The core challenge in topic naming is pattern recognition and summarization rather than complex reasoning or creativity. When generating a topic name, the model needs to identify the common themes and concepts present in a collection of documents or keywords, then produce a concise, descriptive label that captures the essence of that topic. This process primarily involves recognizing patterns in text and applying learned associations between concepts and their typical names or descriptions. The task has clear, well-defined instructions with relatively straightforward prompts, making it easier for smaller models to understand and execute successfully.

Additionally, topic names are typically short responses of just 2-5 words, which means models don't need to maintain long-form coherence or manage complex narrative structures. The output should be deterministic and descriptive rather than creative or novel, focusing on clarity and accuracy rather than originality. These characteristics align well with the strengths of smaller, faster models, which excel at pattern recognition and can produce consistent, high-quality results for well-defined tasks without the computational overhead and cost of larger models.

**Cost Impact Example:**

For processing 1,000 topics with typical prompt sizes (~500 tokens each):

.. list-table:: Model Cost Comparison
   :header-rows: 1
   :widths: 30 20 20 30

   * - Model
     - Cost per 1K topics
     - Quality for topic naming
     - Recommendation
   * - GPT-4o-mini
     - ~$0.25
     - Excellent
     - **Recommended**
   * - Claude-3-Haiku
     - ~$0.35
     - Excellent
     - **Recommended**
   * - Command-R
     - ~$0.25
     - Very Good
     - **Recommended**
   * - GPT-4o
     - ~$4.00
     - Excellent+
     - Unnecessary expense
   * - Claude-3.5-Sonnet
     - ~$5.00
     - Excellent+
     - Unnecessary expense
   * - Claude-3-Opus
     - ~$25.00
     - Excellent++
     - **Not recommended**

The quality difference between recommended models and premium models for topic naming is typically negligible, while the cost difference can be 10-100x higher.