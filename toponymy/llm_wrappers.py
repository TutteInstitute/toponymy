import string
import re
from warnings import warn
import json

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from tqdm.auto import tqdm


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 400  # Default Cohere rate limit
    concurrent_requests: int = 50  # Default concurrent requests


import tokenizers
import transformers

_GET_TOPIC_NAME_REGEX = (
    r'\{\s*"topic_name":\s*.*?,\s*"topic_specificity":\s*\d+\.\d+\s*\}'
)
_GET_TOPIC_CLUSTER_NAMES_REGEX = (
    r'\{\s*"new_topic_name_mapping":\s*.*?,\s*"topic_specificities": .*?\}'
)

try:

    import llama_cpp

    class LlamaCppWrapper:

        def __init__(self, model_path, **kwargs):
            self.model_path = model_path
            for arg, val in kwargs.items():
                if arg == "n_ctx":
                    continue
                setattr(self, arg, val)
            self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)

        def generate_topic_name(self, prompt, temperature=0.8):
            try:
                topic_name_info = self.llm(
                    prompt, temperature=temperature, max_tokens=256
                )["choices"][0]["text"]
                topic_name_info = re.findall(_GET_TOPIC_NAME_REGEX, topic_name_info)[0]
                topic_name_info = json.loads(topic_name_info)
                topic_name = topic_name_info["topic_name"]
                return topic_name
            except Exception as e:
                warn(f"Failed to generate topic name with LlamaCpp: {e}")
                return ""

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm(
                    prompt, temperature=temperature, max_tokens=1024
                )
                topic_name_info_text = topic_name_info_raw["choices"][0]["text"]
                topic_name_info = re.findall(
                    _GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text
                )[0]
                topic_name_info = json.loads(topic_name_info)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names)
                ]
                return result
            except:
                return old_names

except ImportError:
    pass

try:
    import huggingface_hub
    import transformers

    class HuggingFaceWrapper:

        def __init__(self, model, **kwargs):
            self.model = model
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)

        def generate_topic_name(self, prompt, temperature=0.8):
            try:
                topic_name_info_raw = self.llm(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=64,
                    temperature=temperature,
                    do_sample=True,
                )
                topic_name_info_text = topic_name_info_raw[0]["generated_text"][-1][
                    "content"
                ]
                topic_name_info = re.findall(
                    _GET_TOPIC_NAME_REGEX, topic_name_info_text
                )[0]
                topic_name_info = json.loads(topic_name_info)
                topic_name = topic_name_info["topic_name"]
            except Exception as e:
                warn(f"Failed to generate topic name with HuggingFace: {e}")
                topic_name = ""

            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=1024,
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw[0]["generated_text"][-1][
                    "content"
                ]
                topic_name_info = re.findall(
                    _GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text
                )[0]
                topic_name_info = json.loads(topic_name_info)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names)
                ]
                return result
            except:
                return old_names

except ImportError:
    pass

try:
    import cohere

    class CohereWrapper:

        def __init__(
            self,
            API_KEY,
            model="command-r-08-2024",
            local_tokenizer=None,
            requests_per_minute=300,
            concurrent_requests=20,
        ):
            # Sync client for regular operations
            self.llm = cohere.Client(api_key=API_KEY)
            # Async client for batch operations
            self.async_llm = cohere.AsyncClient(api_key=API_KEY)

            try:
                self.llm.models.get(model)
            except cohere.errors.not_found_error.NotFoundError:
                models = [x.name for x in self.llm.models.list().models]
                msg = f"Model '{model}' not found, try one of {models}"
                raise ValueError(msg)
            self.model = model

            # Initialize rate limiting components
            self.rate_limit = RateLimitConfig(requests_per_minute, concurrent_requests)
            self.request_timestamps = []

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                ).text
                topic_name_info = json.loads(topic_name_info_raw)
                topic_name = topic_name_info["topic_name"]
            except:
                topic_name = ""
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.8):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                )
                topic_name_info_text = topic_name_info_raw.text
                topic_name_info = json.loads(topic_name_info_text)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names)
                ]
                return result
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Cohere: {e}")
                print(topic_name_info_text)
                return old_names

        async def generate_topic_names_batch(self, prompts, temperature=0.5):

            async def _generate_single_topic_name_async(prompt):
                try:
                    response = await self.async_llm.chat(
                        message=prompt,
                        model=self.model,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    topic_name_info = json.loads(response.text)
                    return topic_name_info["topic_name"]
                except Exception as e:
                    warn(f"Failed to generate topic name: {e}")
                    return ""

            results = []
            for i in tqdm(
                range(0, len(prompts), self.rate_limit.concurrent_requests),
                desc="Generating topic names in batches",
            ):
                # Limit the number of concurrent requests
                tasks = [
                    _generate_single_topic_name_async(prompt)
                    for prompt in prompts[i : i + self.rate_limit.concurrent_requests]
                ]

                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    self.request_timestamps.extend(
                        [datetime.now()] * self.rate_limit.concurrent_requests
                    )
                    # Handle any exceptions in results
                    results.extend(
                        [
                            result if not isinstance(result, Exception) else ""
                            for result in batch_results
                        ]
                    )
                except Exception as e:
                    warn(f"Batch processing failed: {e}")
                    results.extend([""] * self.rate_limit.concurrent_requests)

                if len(self.request_timestamps) >= self.rate_limit.requests_per_minute:
                    # Wait for the next minute to make more requests
                    next_request_time = self.request_timestamps[0] + timedelta(
                        minutes=1
                    )
                    wait_time = max(
                        2, (next_request_time - datetime.now()).total_seconds()
                    )
                    await asyncio.sleep(wait_time)
                    self.request_timestamps = self.request_timestamps[
                        self.rate_limit.requests_per_minute :
                    ]

            return results

        async def generate_topic_cluster_names_batch(
            self, prompts, old_names_per_cluster, temperature=0.8
        ):

            async def _generate_cluster_topic_names_async(
                prompt, old_names, temperature=0.8
            ):
                try:
                    topic_name_info_raw = await self.async_llm.chat(
                        message=prompt,
                        model=self.model,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        max_tokens=4096,
                    )
                    topic_name_info_text = topic_name_info_raw.text
                    topic_name_info = json.loads(topic_name_info_text)
                    mapping = topic_name_info["new_topic_name_mapping"]
                    result = [
                        mapping.get(f"{n}. {name}", name)
                        for n, name in enumerate(old_names)
                    ]
                    return result
                except Exception as e:
                    warn(f"Failed to generate topic cluster names with Cohere: {e}")
                    return old_names

            # Wait for the next minute to make more requests
            if self.request_timestamps:
                next_request_time = self.request_timestamps[0] + timedelta(minutes=1)
                wait_time = max(0, (next_request_time - datetime.now()).total_seconds())
                await asyncio.sleep(wait_time)

            results = []
            for i in tqdm(
                range(0, len(prompts), self.rate_limit.concurrent_requests),
                desc="Generating topic cluster names in batches",
            ):
                # Limit the number of concurrent requests
                tasks = [
                    _generate_cluster_topic_names_async(prompt, old_names)
                    for prompt, old_names in zip(
                        prompts[i : i + self.rate_limit.concurrent_requests],
                        old_names_per_cluster[
                            i : i + self.rate_limit.concurrent_requests
                        ],
                    )
                ]

                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    self.request_timestamps.extend(
                        [datetime.now()] * self.rate_limit.concurrent_requests
                    )
                    # Handle any exceptions in results
                    results.extend(batch_results)
                except Exception as e:
                    warn(f"Batch processing failed: {e}")
                    results.extend(
                        old_names_per_cluster[
                            i : i + self.rate_limit.concurrent_requests
                        ]
                    )

                if len(self.request_timestamps) >= self.rate_limit.requests_per_minute:
                    # Wait for the next minute to make more requests
                    next_request_time = self.request_timestamps[0] + timedelta(
                        minutes=1
                    )
                    wait_time = max(
                        2, (next_request_time - datetime.now()).total_seconds()
                    )
                    await asyncio.sleep(wait_time)
                    self.request_timestamps = self.request_timestamps[
                        self.rate_limit.requests_per_minute :
                    ]

            return results

except:
    pass

try:
    import anthropic

    class AnthropicWrapper:

        def __init__(
            self, API_KEY, model="claude-3-haiku-20240307", requests_per_minute=50, concurrent_requests=2
        ):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.async_llm = anthropic.AsyncAnthropic(api_key=API_KEY)
            self.model = model

            # Initialize rate limiting components
            self.rate_limit = RateLimitConfig(requests_per_minute, concurrent_requests)
            self.request_timestamps = []

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = json.loads(topic_name_info_text)
                topic_name = topic_name_info["topic_name"]
            except:
                topic_name = ""
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = json.loads(topic_name_info_text)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names)
                ]
                return result
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Anthropic: {e}")
                return old_names

        async def generate_topic_names_batch(self, prompts, temperature=0.5):

            async def _generate_single_topic_name_async(prompt):
                try:
                    topic_name_info_raw = self.llm.messages.create(
                        model=self.model,
                        max_tokens=256,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                    topic_name_info_text = topic_name_info_raw.content[0].text
                    topic_name_info = json.loads(topic_name_info_text)
                    topic_name = topic_name_info["topic_name"]
                    return topic_name
                except Exception as e:
                    warn(f"Failed to generate topic name due to: {e}")
                    return ""

            results = []
            for i in tqdm(
                range(0, len(prompts), self.rate_limit.concurrent_requests),
                desc="Generating topic names in batches",
            ):
                # Limit the number of concurrent requests
                tasks = [
                    _generate_single_topic_name_async(prompt)
                    for prompt in prompts[i : i + self.rate_limit.concurrent_requests]
                ]

                try:
                    self.request_timestamps.extend(
                        [datetime.now()] * self.rate_limit.concurrent_requests
                    )
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Handle any exceptions in results
                    results.extend(
                        [
                            result if not isinstance(result, Exception) else ""
                            for result in batch_results
                        ]
                    )
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    warn(f"Batch processing failed: {e}")
                    results.extend([""] * self.rate_limit.concurrent_requests)

                if len(self.request_timestamps) >= self.rate_limit.requests_per_minute:
                    # Wait for the next minute to make more requests
                    next_request_time = self.request_timestamps[0] + timedelta(
                        minutes=1
                    )
                    wait_time = max(
                        2, (next_request_time - datetime.now()).total_seconds()
                    )
                    await asyncio.sleep(wait_time)
                    self.request_timestamps = self.request_timestamps[
                        self.rate_limit.requests_per_minute :
                    ]

            return results

        async def generate_topic_cluster_names_batch(
            self, prompts, old_names_per_cluster, temperature=0.8
        ):

            async def _generate_cluster_topic_names_async(
                prompt, old_names, temperature=0.8
            ):
                try:
                    topic_name_info_raw = self.llm.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                    topic_name_info_text = topic_name_info_raw.content[0].text
                    topic_name_info = json.loads(topic_name_info_text)
                    mapping = topic_name_info["new_topic_name_mapping"]
                    result = [
                        mapping.get(f"{n}. {name}", name)
                        for n, name in enumerate(old_names)
                    ]
                    return result
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    warn(f"Failed to generate topic cluster names with Cohere: {e}")
                    return old_names

            # Wait for the next minute to make more requests
            if self.request_timestamps:
                next_request_time = self.request_timestamps[0] + timedelta(minutes=1)
                wait_time = max(0, (next_request_time - datetime.now()).total_seconds())
                await asyncio.sleep(wait_time)

            results = []
            for i in tqdm(
                range(0, len(prompts), self.rate_limit.concurrent_requests),
                desc="Generating topic cluster names in batches",
            ):
                # Limit the number of concurrent requests
                tasks = [
                    _generate_cluster_topic_names_async(prompt, old_names)
                    for prompt, old_names in zip(
                        prompts[i : i + self.rate_limit.concurrent_requests],
                        old_names_per_cluster[
                            i : i + self.rate_limit.concurrent_requests
                        ],
                    )
                ]

                try:
                    self.request_timestamps.extend(
                        datetime.now() for n in range(self.rate_limit.concurrent_requests)
                    )
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Handle any exceptions in results
                    results.extend(batch_results)
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    warn(f"Batch processing failed: {e}")
                    results.extend(
                        old_names_per_cluster[
                            i : i + self.rate_limit.concurrent_requests
                        ]
                    )

                if len(self.request_timestamps) >= self.rate_limit.requests_per_minute:
                    # Wait for the next minute to make more requests
                    next_request_time = self.request_timestamps[0] + timedelta(
                        minutes=1
                    )
                    wait_time = max(
                        2, (next_request_time - datetime.now()).total_seconds()
                    )
                    await asyncio.sleep(wait_time)
                    self.request_timestamps = self.request_timestamps[
                        self.rate_limit.requests_per_minute :
                    ]

            return results

except:
    pass

try:
    import openai

    class OpenAIWrapper:

        def __init__(self, API_KEY, model="gpt-4o-mini", verbose=False):
            self.llm = openai.OpenAI(api_key=API_KEY)
            self.model = model
            self.verbose = verbose

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                topic_name_info_text = topic_name_info_raw.choices[0].message.content

                topic_name_info = json.loads(topic_name_info_text)
                topic_name = topic_name_info["topic_name"]
                if self.verbose:
                    print(topic_name_info)
            except Exception as e:
                topic_name = ""
                warn(f"{e}\n{prompt}\n{topic_name_info_text}")
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                topic_name_info_text = topic_name_info_raw.choices[0].message.content
                topic_name_info = json.loads(topic_name_info_text)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names)
                ]
                return result
            except:
                return old_names

except:
    pass
