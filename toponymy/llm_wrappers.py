import string
import re
from warnings import warn
import json

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 400   # Default Cohere rate limit
    concurrent_requests: int = 50    # Default concurrent requests

import tokenizers
import transformers

_GET_TOPIC_NAME_REGEX = r'\{\s*"topic_name":\s*.*?, "topic_specificity":\s*\d+\.\d+\s*\}'
_GET_TOPIC_CLUSTER_NAMES_REGEX = r'\{\s*"new_topic_name_mapping":\s*.*?, "topic_specificities": .*?\}'

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
                topic_name_info = self.llm(prompt, temperature=temperature, max_tokens=256)["choices"][0]["text"]
                topic_name_info = re.findall(_GET_TOPIC_NAME_REGEX, topic_name_info)[0]
                topic_name_info = json.loads(topic_name_info)
                topic_name = topic_name_info["topic_name"]
                return topic_name
            except Exception as e:
                warn(f"Failed to generate topic name with LlamaCpp: {e}")
                return ""

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm(prompt, temperature=temperature, max_tokens=1024)
                topic_name_info_text = topic_name_info_raw["choices"][0]["text"]
                topic_name_info = re.findall(_GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text)[0]
                topic_name_info = json.loads(topic_name_info)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [mapping.get(f"{n}. {name}", name) for n, name in enumerate(old_names)]
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
                topic_name_info_raw = self.llm([{"role":"user", "content": prompt}], max_new_tokens=64, temperature=temperature, do_sample=True)
                topic_name_info_text = topic_name_info_raw[0]["generated_text"][-1]['content']
                topic_name_info = re.findall(_GET_TOPIC_NAME_REGEX, topic_name_info_text)[0]
                topic_name_info = json.loads(topic_name_info)
                topic_name = topic_name_info["topic_name"]
            except Exception as e:
                warn(f"Failed to generate topic name with HuggingFace: {e}")
                topic_name = ""

            return topic_name
        
        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm([{"role":"user", "content": prompt}], max_new_tokens=1024, temperature=temperature)
                topic_name_info_text = topic_name_info_raw[0]["generated_text"][-1]['content']
                topic_name_info = re.findall(_GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text)[0]
                topic_name_info = json.loads(topic_name_info)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [mapping.get(f"{n}. {name}", name) for n, name in enumerate(old_names)]
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
            requests_per_minute=500, 
            concurrent_requests=50
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
            self.rate_limit = RateLimitConfig()
            self.semaphore = asyncio.Semaphore(self.rate_limit.concurrent_requests)
            self.request_timestamps = []

        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                    response_format={ "type": "json_object" },
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
                    response_format={ "type": "json_object" },
                    max_tokens=2048,
                )
                topic_name_info_text = topic_name_info_raw.text
                topic_name_info = json.loads(topic_name_info_text)
                mapping = topic_name_info["new_topic_name_mapping"]
                result = [mapping.get(f"{n}. {name}", name) for n, name in enumerate(old_names)]
                return result
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Cohere: {e}")
                return old_names

        async def generate_topic_names_batch(self, prompts, temperature=0.5):

            async def _generate_single_topic_name_async(prompt):
                try:
                    response = await self.async_llm.chat(
                        message=prompt,
                        model=self.model,
                        temperature=temperature,
                        response_format={ "type": "json_object" },
                    )
                    topic_name_info = json.loads(response.text)
                    return topic_name_info["topic_name"]
                except Exception as e:
                    warn(f"Failed to generate topic name: {e}")
                    return ""

            tasks = [
                _generate_single_topic_name_async(prompt)
                for prompt in prompts
            ]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Handle any exceptions in results
                return [
                    result if not isinstance(result, Exception) else ""
                    for result in results
                ]
            except Exception as e:
                warn(f"Batch processing failed: {e}")
                return [""] * len(prompts)
            
except:
    pass

try:
    import anthropic

    class AnthropicWrapper:

        def __init__(
            self, API_KEY, model="claude-3-haiku-20240307", local_tokenizer=None
        ):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.model = model

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
                result = [mapping.get(f"{n}. {name}", name) for n, name in enumerate(old_names)]
                return result
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Anthropic: {e}")
                return old_names

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
                result = [mapping.get(f"{n}. {name}", name) for n, name in enumerate(old_names)]
                return result
            except:
                return old_names

except:
    pass
