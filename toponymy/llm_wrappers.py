import string
from warnings import warn

import tokenizers
import transformers

from toponymy.templates import GET_TOPIC_CLUSTER_NAMES_REGEX, GET_TOPIC_NAME_REGEX
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

import re
import os
import httpx

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def repair_json_string_backslashes(s: str) -> str:
    """
    Attempts to repair a string that should be JSON by escaping unescaped backslashes.
    This focuses on the common issue of literal backslashes not being escaped.
    """
    # Define placeholders for known valid JSON escape sequences
    # This helps prevent double-escaping or breaking already correct sequences.
    placeholders = {
        "\\\\": "__DOUBLE_BACKSLASH_PLACEHOLDER__",
        "\\\"": "__ESCAPED_QUOTE_PLACEHOLDER__",
        "\\n": "__NEWLINE_PLACEHOLDER__",
        "\\r": "__CARRIAGE_RETURN_PLACEHOLDER__",
        "\\t": "__TAB_PLACEHOLDER__",
        "\\b": "__BACKSPACE_PLACEHOLDER__",
        "\\f": "__FORMFEED_PLACEHOLDER__",
        "\\/": "__SOLIDUS_PLACEHOLDER__" # Though '/' doesn't always need escaping
    }

    # Step 1: Protect existing valid escape sequences
    temp_s = s
    for original, placeholder in placeholders.items():
        temp_s = temp_s.replace(original, placeholder)

    # Step 2: Escape remaining single backslashes
    # These are likely the problematic ones intended to be literal backslashes.
    temp_s = temp_s.replace("\\", "\\\\")

    # Step 3: Restore the original valid escape sequences
    for original, placeholder in placeholders.items():
        temp_s = temp_s.replace(placeholder, original)

    return temp_s

def llm_output_to_result(llm_output: str, regex: str) -> dict:
    json_portion = re.findall(
        regex, llm_output, re.DOTALL
    )[0]
    try:
        result = json.loads(json_portion)
    except json.JSONDecodeError:
        # Attempt to repair the JSON string
        repaired_json = repair_json_string_backslashes(json_portion)
        result = json.loads(repaired_json)

    return result


class LLMWrapper(ABC):

    @abstractmethod
    def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float) -> str:
        pass

    @abstractmethod
    def generate_topic_cluster_names(
        self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float
    ) -> List[str]:
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

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda x: "",
        )
        def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.8) -> str:
            try:
                if isinstance(prompt, str):
                    topic_name_info = self.llm(
                        prompt, temperature=temperature, max_tokens=256
                    )["choices"][0]["text"]
                elif isinstance(prompt, dict):
                    raise ValueError(
                        f"System/User prompts are not supported for LlamaCpp wrapper"
                    )
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info = self.llm(
                    prompt, temperature=temperature, max_tokens=256
                )["choices"][0]["text"]
                topic_name_info = llm_output_to_result(topic_name_info, GET_TOPIC_NAME_REGEX)
                topic_name = topic_name_info["topic_name"]
            except Exception as e:
                raise ValueError(f"Failed to generate topic name with LlamaCpp: {e}")

            return topic_name

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda retry_state: retry_state.args[2],
        )
        def generate_topic_cluster_names(
            self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.5
        ) -> List[str]:
            if isinstance(prompt, str):
                topic_name_info_raw = self.llm(
                    prompt, temperature=temperature, max_tokens=1024
                )
            elif isinstance(prompt, dict):
                raise ValueError(
                    f"System/User prompts are not supported for LlamaCpp wrapper"
                )
            else:
                raise ValueError(
                    f"Prompt must be a string or a dictionary, got {type(prompt)}"
                )
            
            topic_name_info_text = topic_name_info_raw["choices"][0]["text"]
            topic_name_info = llm_output_to_result(
                topic_name_info_text, GET_TOPIC_CLUSTER_NAMES_REGEX
            )
            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    # Empty names can cause issues, so cater to that is possible
                    mapping.get(f"{n}. {name}") if f"{n}. {name}" in mapping else mapping.get(f"{n}.", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    topic_name_info_text,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")

except ImportError:
    pass

try:
    import huggingface_hub
    import transformers

    class HuggingFace(LLMWrapper):

        def __init__(self, model: str, **kwargs):
            self.model = model
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda x: "",
        )
        def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.8) -> str:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm(
                        [{"role": "user", "content": prompt}],
                        return_full_text=False,
                        max_new_tokens=64,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.llm.tokenizer.eos_token_id,
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm(
                        [{"role": "system", "content": prompt["system"]},
                         {"role": "user", "content": prompt["user"]}],
                        return_full_text=False,
                        max_new_tokens=64,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.llm.tokenizer.eos_token_id,
                    )
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info_text = topic_name_info_raw[0]["generated_text"]
                topic_name_info = llm_output_to_result(topic_name_info_text, GET_TOPIC_NAME_REGEX)
                topic_name = topic_name_info["topic_name"]
            except Exception as e:
                raise ValueError(f"Failed to generate topic name with HuggingFace: {e}")

            return topic_name

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda retry_state: retry_state.args[2],
        )
        def generate_topic_cluster_names(
            self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.5
        ) -> List[str]:
            if isinstance(prompt, str):
                topic_name_info_raw = self.llm(
                    [{"role": "user", "content": prompt}],
                    return_full_text=False,
                    max_new_tokens=1024,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.eos_token_id,
                )
            elif isinstance(prompt, dict):
                topic_name_info_raw = self.llm(
                    [{"role": "system", "content": prompt["system"]},
                     {"role": "user", "content": prompt["user"]}],
                    return_full_text=False,
                    max_new_tokens=1024,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.eos_token_id,
                )
            else:   
                raise ValueError(
                    f"Prompt must be a string or a dictionary, got {type(prompt)}"
                )
            
            topic_name_info_text = topic_name_info_raw[0]["generated_text"]
            topic_name_info = llm_output_to_result(
                topic_name_info_text, GET_TOPIC_CLUSTER_NAMES_REGEX
            )
            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    # Empty names can cause issues, so cater to that is possible
                    mapping.get(f"{n}. {name}") if f"{n}. {name}" in mapping else mapping.get(f"{n}.", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    topic_name_info_text,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")

except ImportError:
    pass

try:

    import json

    import cohere

    class Cohere(LLMWrapper):

        def __init__(self, API_KEY: str, model: str = "command-r-08-2024", base_url: str = None, httpx_client: Optional[httpx.Client] = None):
            if base_url is None:
                base_url = os.getenv("CO_API_URL", "https://api.cohere.com")
            self.llm = cohere.ClientV2(api_key=API_KEY, base_url=base_url, httpx_client=httpx_client)

            try:
                self.llm.models.get(model)
            except cohere.errors.not_found_error.NotFoundError:
                models = [x.name for x in self.llm.models.list().models]
                msg = f"Model '{model}' not found, try one of {models}"
                raise ValueError(msg)
            self.model = model

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda x: "",
        )
        def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.5) -> str:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.chat(
                        messages=[{"role":"user", "content": prompt}],
                        model=self.model,
                        temperature=temperature,
                        # This results in failures more often than useful output
                        # response_format={"type": "json_object"},
                    ).message.content[0].text
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.chat(
                        messages=[
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user"]},
                        ],
                        model=self.model,
                        temperature=temperature,
                        # This results in failures more often than useful output
                        # response_format={"type": "json_object"},
                    ).message.content[0].text
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info = llm_output_to_result(topic_name_info_raw, GET_TOPIC_NAME_REGEX)
                topic_name = topic_name_info["topic_name"]
            except:
                raise ValueError(f"Failed to generate topic name with Cohere")
            return topic_name

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda retry_state: retry_state.args[2],
        )
        def generate_topic_cluster_names(
            self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.5
        ) -> List[str]:
            if isinstance(prompt, str):
                topic_name_info_raw = self.llm.chat(
                    messages=[{"role":"user", "content": prompt}],
                    model=self.model,
                    temperature=temperature,
                ).message.content[0].text
            elif isinstance(prompt, dict):
                topic_name_info_raw = self.llm.chat(
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    model=self.model,
                    temperature=temperature,
                ).message.content[0].text
            else:
                raise ValueError(
                    f"Prompt must be a string or a dictionary, got {type(prompt)}"
                )

            topic_name_info = llm_output_to_result(
                topic_name_info_raw, GET_TOPIC_CLUSTER_NAMES_REGEX
            )

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    # Empty names can cause issues, so cater to that is possible
                    mapping.get(f"{n}. {name}") if f"{n}. {name}" in mapping else mapping.get(f"{n}.", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    topic_name_info_raw,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(
                        f"Failed to generate enough names when fixing {old_names}; got {mapping}"
                    )

except:
    pass

try:

    import json

    import anthropic

    class Anthropic(LLMWrapper):

        def __init__(self, API_KEY: str, model: str = "claude-3-haiku-20240307"):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.model = model

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda x: "",
        )
        def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.5) -> str:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.messages.create(
                        model=self.model,
                        max_tokens=256,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.messages.create(
                        model=self.model,
                        max_tokens=256,
                        messages=[
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user"]},
                        ],
                        temperature=temperature,
                    )
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = llm_output_to_result(
                    topic_name_info_text, GET_TOPIC_NAME_REGEX
                )
                topic_name = topic_name_info["topic_name"]
            except:
                raise ValueError(f"Failed to generate topic name with Anthropic")

            return topic_name

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda retry_state: retry_state.args[2],
        )
        def generate_topic_cluster_names(
            self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.5
        ) -> List[str]:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user"]},
                        ],
                        temperature=temperature,
                    )
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info_text = topic_name_info_raw.content[0].text
                topic_name_info = llm_output_to_result(
                    topic_name_info_text, GET_TOPIC_CLUSTER_NAMES_REGEX
                )
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Anthropic: {e}")
                return old_names

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    # Empty names can cause issues, so cater to that is possible
                    mapping.get(f"{n}. {name}") if f"{n}. {name}" in mapping else mapping.get(f"{n}.", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    topic_name_info_text,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")

except:
    pass

try:
    import json

    import openai

    class OpenAI(LLMWrapper):

        def __init__(
            self,
            API_KEY: str,
            model: str = "gpt-4o-mini",
            base_url: str = None,
            verbose: bool = False,
        ):
            self.llm = openai.OpenAI(api_key=API_KEY, base_url=base_url)
            self.model = model
            self.verbose = verbose

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda x: "",
        )
        def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.5) -> str:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.chat.completions.create(
                        model=self.model,
                        max_tokens=256,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.chat.completions.create(
                        model=self.model,
                        max_tokens=256,
                        messages=[
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user"]},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info_text = topic_name_info_raw.choices[0].message.content

                topic_name_info = llm_output_to_result(
                    topic_name_info_text, GET_TOPIC_NAME_REGEX
                )
                topic_name = topic_name_info["topic_name"]
                if self.verbose:
                    print(topic_name_info)
            except Exception as e:
                raise ValueError(f"{e}\n{prompt}\n{topic_name_info_text}")

            return topic_name

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda retry_state: retry_state.args[2],
        )
        def generate_topic_cluster_names(
            self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.5
        ) -> List[str]:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.chat.completions.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.chat.completions.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user"]},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                else:
                    raise ValueError(
                        f"Prompt must be a string or a dictionary, got {type(prompt)}"
                    )
                
                topic_name_info_text = topic_name_info_raw.choices[0].message.content
                topic_name_info = llm_output_to_result(
                    topic_name_info_text, GET_TOPIC_CLUSTER_NAMES_REGEX
                )
            except Exception as e:
                warn(f"Failed to generate topic cluster names with OpenAI: {e}")
                return old_names

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    # Empty names can cause issues, so cater to that is possible
                    mapping.get(f"{n}. {name}") if f"{n}. {name}" in mapping else mapping.get(f"{n}.", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    topic_name_info_text,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")

except:
    pass


try:
    import azure.ai.inference
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    class AzureAI(LLMWrapper):

        def __init__(self, API_KEY: str, endpoint: str, model: str):
            self.endpoint = endpoint
            self.model = model
            self.llm = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(API_KEY),
            )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda x: "",
        )
        def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.5) -> str:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.complete(
                        model=self.model,
                        max_tokens=256,
                        messages=[UserMessage(prompt)],
                        temperature=temperature,
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.complete(
                        model=self.model,
                        max_tokens=256,
                        messages=[SystemMessage(prompt["system"]), UserMessage(prompt["user"])],
                        temperature=temperature,
                    )
                else:
                    raise ValueError(f"Prompt must be a string or a dictionary, got {type(prompt)}")
                
                topic_name_info_text = topic_name_info_raw.choices[0].message.content
                topic_name_info = llm_output_to_result(topic_name_info_text, GET_TOPIC_NAME_REGEX)
                topic_name = topic_name_info["topic_name"]
            except:
                raise ValueError(f"Failed to generate topic name with AzureAI")

            return topic_name

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry_error_callback=lambda retry_state: retry_state.args[2],
        )
        def generate_topic_cluster_names(
            self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.5
        ) -> List[str]:
            try:
                if isinstance(prompt, str):
                    topic_name_info_raw = self.llm.complete(
                        model=self.model,
                        max_tokens=1024,
                        messages=[UserMessage(prompt)],
                        temperature=temperature,
                    )
                elif isinstance(prompt, dict):
                    topic_name_info_raw = self.llm.complete(
                        model=self.model,
                        max_tokens=1024,
                        messages=[SystemMessage(prompt["system"]), UserMessage(prompt["user"])],
                        temperature=temperature,
                    )
                else:
                    raise ValueError(f"Prompt must be a string or a dictionary, got {type(prompt)}")
                
                topic_name_info_text = topic_name_info_raw.choices[0].message.content
                topic_name_info = llm_output_to_result(
                    topic_name_info_text, GET_TOPIC_CLUSTER_NAMES_REGEX
                )
            except Exception as e:
                warn(f"Failed to generate topic cluster names with AzureAI: {e}")
                return old_names

            mapping = topic_name_info["new_topic_name_mapping"]
            if len(mapping) == len(old_names):
                result = [
                    # Empty names can cause issues, so cater to that is possible
                    mapping.get(f"{n}. {name}") if f"{n}. {name}" in mapping else mapping.get(f"{n}.", name)
                    for n, name in enumerate(old_names, start=1)
                ]
                return result
            else:
                mapping = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    topic_name_info_text,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")
                
except ImportError:
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
