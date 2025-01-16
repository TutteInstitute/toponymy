import string
from warnings import warn

import tokenizers
import transformers

from toponymy.templates import GET_TOPIC_CLUSTER_NAMES_REGEX, GET_TOPIC_NAME_REGEX
from abc import ABC, abstractmethod
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

import re
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMWrapper(ABC):

    @abstractmethod
    def generate_topic_name(self, prompt: str, temperature: float) -> str:
        pass

    @abstractmethod
    def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float) -> List[str]:
        pass

try:

    import llama_cpp

    class LlamaCpp(LLMWrapper):

        def __init__(self, model_path: str, **kwargs):
            self.model_path = model_path
            for arg, val in kwargs.items():
                if arg == "n_ctx":
                    continue
                setattr(self, arg, val)
            self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=lambda x: "")
        def generate_topic_name(self, prompt: str, temperature: float = 0.8) -> str:
            try:
                topic_name_info = self.llm(
                    prompt, temperature=temperature, max_tokens=256
                )["choices"][0]["text"]
                topic_name_info = re.findall(
                    GET_TOPIC_NAME_REGEX, topic_name_info, re.DOTALL
                )[0]
                topic_name_info = json.loads(topic_name_info)
                topic_name = topic_name_info["topic_name"]
            except Exception as e:
                raise ValueError(f"Failed to generate topic name with LlamaCpp: {e}")

            return topic_name

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
            try:
                topic_name_info_raw = self.llm(
                    prompt, temperature=temperature, max_tokens=1024
                )
                topic_name_info_text = topic_name_info_raw["choices"][0]["text"]
                topic_name_info = re.findall(
                    GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text
                )[0]
                topic_name_info = json.loads(topic_name_info)
                mapping = topic_name_info["new_topic_name_mapping"]
                if len(mapping) == len(old_names):
                    result = [
                        mapping.get(f"{n}. {name}", name)
                        for n, name in enumerate(old_names, start=1)
                    ]
                    return result
                else:
                    mapping = re.findall(r'"new_topic_name_mapping":\s*\{(.*?)\}', topic_name_info_text, re.DOTALL)[0]
                    new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                    if len(new_names) == len(old_names):
                        return new_names
                    else:
                        # warn(f"Failed to generate enough names when fixing {old_names}; got {mapping}")
                        return old_names
            except:
                return old_names

except ImportError:
    pass

try:
    import huggingface_hub
    import transformers

    class HuggingFace(LLMWrapper):

        def __init__(self, model: str, **kwargs):
            self.model = model
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=lambda x: "")
        def generate_topic_name(self, prompt: str, temperature: float = 0.8) -> str:
            try:
                topic_name_info_raw = self.llm(
                    [{"role": "user", "content": prompt}],
                    return_full_text=False,
                    max_new_tokens=64,
                    temperature=temperature,
                    do_sample=True,
                )
                topic_name_info_text = topic_name_info_raw[0]["generated_text"]
                topic_name_info = re.findall(
                    GET_TOPIC_NAME_REGEX, topic_name_info_text, re.DOTALL
                )[0]
                topic_name_info = json.loads(topic_name_info)
                topic_name = topic_name_info["topic_name"]
            except Exception as e:
                raise ValueError(f"Failed to generate topic name with HuggingFace: {e}")
            
            return topic_name

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
            try:
                topic_name_info_raw = self.llm(
                    [{"role": "user", "content": prompt}],
                    return_full_text=False,
                    max_new_tokens=1024,
                    temperature=temperature,
                    do_sample=True,
                )
                topic_name_info_text = topic_name_info_raw[0]["generated_text"]
                topic_name_info = re.findall(
                    GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text, re.DOTALL
                )[0]
                topic_name_info = json.loads(topic_name_info)
                mapping = topic_name_info["new_topic_name_mapping"]
                if len(mapping) == len(old_names):
                    result = [
                        mapping.get(f"{n}. {name}", name)
                        for n, name in enumerate(old_names, start=1)
                    ]
                    return result
                else:
                    mapping = re.findall(r'"new_topic_name_mapping":\s*\{(.*?)\}', topic_name_info_text, re.DOTALL)[0]
                    new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                    if len(new_names) == len(old_names):
                        return new_names
                    else:
                        # warn(f"Failed to generate enough names when fixing {old_names}; got {mapping}")
                        return old_names
            except:
                return old_names

except ImportError:
    pass

try:

    import json

    import cohere

    class Cohere(LLMWrapper):

        def __init__(self, API_KEY: str, model: str = "command-r-08-2024"):
            self.llm = cohere.Client(api_key=API_KEY)

            try:
                self.llm.models.get(model)
            except cohere.errors.not_found_error.NotFoundError:
                models = [x.name for x in self.llm.models.list().models]
                msg = f"Model '{model}' not found, try one of {models}"
                raise ValueError(msg)
            self.model = model

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=lambda x: "")
        def generate_topic_name(self, prompt: str, temperature: float = 0.5) -> str:
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
                raise ValueError(f"Failed to generate topic name with Cohere")
            return topic_name

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
            topic_name_info_raw = self.llm.chat(
                message=prompt,
                model=self.model,
                temperature=temperature,
            )
            topic_name_info_text = topic_name_info_raw.text
            topic_name_info = re.findall(
                GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text, re.DOTALL
            )[0]
            topic_name_info = json.loads(topic_name_info)

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(r'"new_topic_name_mapping":\s*\{(.*?)\}', topic_name_info_text, re.DOTALL)[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")

except:
    pass

try:

    import json

    import anthropic

    class Anthropic(LLMWrapper):

        def __init__(
            self, API_KEY: str, model: str = "claude-3-haiku-20240307"
        ):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.model = model

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=lambda x: "")
        def generate_topic_name(self, prompt: str, temperature: float = 0.5) -> str:
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
                raise ValueError(f"Failed to generate topic name with Anthropic")
                
            return topic_name

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
            try:
                topic_name_info_raw = self.llm.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = json.loads(topic_name_info_text)
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Anthropic: {e}")
                return old_names

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(r'"new_topic_name_mapping":\s*\{(.*?)\}', topic_name_info_text, re.DOTALL)[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    # warn(f"Failed to generate enough names when fixing {old_names}; got {mapping}")
                    return old_names

except:
    pass

try:
    import json

    import openai

    class OpenAI(LLMWrapper):

        def __init__(self, API_KEY: str, model: str = "gpt-4o-mini", base_url: str = None, verbose: bool = False):
            self.llm = openai.OpenAI(api_key=API_KEY, base_url=base_url)
            self.model = model
            self.verbose = verbose

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=lambda x: "")
        def generate_topic_name(self, prompt: str, temperature: float = 0.5) -> str:
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
                raise ValueError(f"{e}\n{prompt}\n{topic_name_info_text}")
            
            return topic_name

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
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
            except Exception as e:
                warn(f"Failed to generate topic cluster names with OpenAI: {e}")
                return old_names

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    mapping.get(f"{n}. {name}", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(r'"new_topic_name_mapping":\s*\{(.*?)\}', topic_name_info_text, re.DOTALL)[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    # warn(f"Failed to generate enough names when fixing {old_names}; got {mapping}")
                    return old_names

except:
    pass


############################ UNTESTED WRAPPERS ############################

# # MistralAI wrapper
# try:
#     import mistralai
#     from mistralai.client import MistralClient
#     from mistralai.models.chat_completion import ChatMessage
    
#     class Mistral(LLMWrapper):
#         """Wrapper for MistralAI's API"""
        
#         def __init__(self, API_KEY: str, model: str = "mistral-tiny"):
#             """
#             Initialize MistralAI wrapper
            
#             Args:
#                 API_KEY: MistralAI API key
#                 model: Model name (tiny, small, medium, large)
#             """
#             self.llm = MistralClient(api_key=API_KEY)
#             self.model = model

#         @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#         def generate_topic_name(self, prompt: str, temperature: float = 0.5) -> str:
#             try:
#                 messages = [ChatMessage(role="user", content=prompt)]
#                 response = self.llm.chat(
#                     model=self.model,
#                     messages=messages,
#                     temperature=temperature
#                 )
#                 topic_name_info = json.loads(response.messages[-1].content)
#                 topic_name = topic_name_info["topic_name"]
#             except Exception as e:
#                 warn(f"Failed to generate topic name with Mistral: {e}")
#                 topic_name = ""
#             return topic_name

#         @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#         def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
#             try:
#                 messages = [ChatMessage(role="user", content=prompt)]
#                 response = self.llm.chat(
#                     model=self.model,
#                     messages=messages,
#                     temperature=temperature
#                 )
#                 topic_name_info = json.loads(response.messages[-1].content)
#                 mapping = topic_name_info["new_topic_name_mapping"]
#                 if len(mapping) == len(old_names):
#                     return [mapping.get(f"{n}. {name}", name) 
#                            for n, name in enumerate(old_names, start=1)]
#             except Exception as e:
#                 warn(f"Failed to generate topic cluster names with Mistral: {e}")
#             return old_names
# except ImportError:
#     pass

# # Together AI wrapper
# try:
#     import together
    
#     class Together(LLMWrapper):
#         """Wrapper for Together.ai's API"""
        
#         def __init__(self, API_KEY: str, model: str = "togethercomputer/llama-2-7b"):
#             """
#             Initialize Together.ai wrapper
            
#             Args:
#                 API_KEY: Together.ai API key
#                 model: Model name
#             """
#             together.api_key = API_KEY
#             self.model = model

#         @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#         def generate_topic_name(self, prompt: str, temperature: float = 0.5) -> str:
#             try:
#                 response = together.Complete.create(
#                     prompt=prompt,
#                     model=self.model,
#                     temperature=temperature,
#                     max_tokens=256
#                 )
#                 topic_name_info = json.loads(response['output']['choices'][0]['text'])
#                 topic_name = topic_name_info["topic_name"]
#             except Exception as e:
#                 warn(f"Failed to generate topic name with Together: {e}")
#                 topic_name = ""
#             return topic_name

#         @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#         def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
#             try:
#                 response = together.Complete.create(
#                     prompt=prompt,
#                     model=self.model,
#                     temperature=temperature,
#                     max_tokens=1024
#                 )
#                 topic_name_info = json.loads(response['output']['choices'][0]['text'])
#                 mapping = topic_name_info["new_topic_name_mapping"]
#                 if len(mapping) == len(old_names):
#                     return [mapping.get(f"{n}. {name}", name) 
#                            for n, name in enumerate(old_names, start=1)]
#             except Exception as e:
#                 warn(f"Failed to generate topic cluster names with Together: {e}")
#             return old_names
# except ImportError:
#     pass

# # Google Vertex AI wrapper
# try:
#     from google.cloud import aiplatform
#     from vertexai.language_models import TextGenerationModel
    
#     class VertexAI(LLMWrapper):
#         """Wrapper for Google Cloud's Vertex AI"""
        
#         def __init__(self, project_id: str, location: str = "us-central1", 
#                      model: str = "text-bison@002"):
#             """
#             Initialize Vertex AI wrapper
            
#             Args:
#                 project_id: Google Cloud project ID
#                 location: Google Cloud region
#                 model: Model name
#             """
#             aiplatform.init(project=project_id, location=location)
#             self.model = TextGenerationModel.from_pretrained(model)

#         @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#         def generate_topic_name(self, prompt: str, temperature: float = 0.5) -> str:
#             try:
#                 response = self.model.predict(
#                     prompt,
#                     temperature=temperature,
#                     max_output_tokens=256
#                 )
#                 topic_name_info = json.loads(response.text)
#                 topic_name = topic_name_info["topic_name"]
#             except Exception as e:
#                 warn(f"Failed to generate topic name with Vertex AI: {e}")
#                 topic_name = ""
#             return topic_name

#         @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#         def generate_topic_cluster_names(self, prompt: str, old_names: List[str], temperature: float = 0.5) -> List[str]:
#             try:
#                 response = self.model.predict(
#                     prompt,
#                     temperature=temperature,
#                     max_output_tokens=1024
#                 )
#                 topic_name_info = json.loads(response.text)
#                 mapping = topic_name_info["new_topic_name_mapping"]
#                 if len(mapping) == len(old_names):
#                     return [mapping.get(f"{n}. {name}", name) 
#                            for n, name in enumerate(old_names, start=1)]
#             except Exception as e:
#                 warn(f"Failed to generate topic cluster names with Vertex AI: {e}")
#             return old_names
# except ImportError:
#     pass