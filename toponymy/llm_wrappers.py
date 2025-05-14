import string
from warnings import warn

import tokenizers
import transformers

from toponymy.templates import GET_TOPIC_CLUSTER_NAMES_REGEX, GET_TOPIC_NAME_REGEX
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

import re
import os
import httpx
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class InvalidLLMInputError(ValueError):
    """A custom exception for invalid LLM input. In these cases we do not want to retry, as the input will not change."""
    pass

def _should_retry(e: Exception) -> bool:
    if isinstance(e, InvalidLLMInputError):
        return False
    return True

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
    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Call the LLM with the given prompt and temperature.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Call the LLM with a system prompt and user prompt.
        This method should be implemented by subclasses.
        """
        pass

    # @abstractmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda x: "",
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_name(self, prompt: Union[str, Dict[str, str]], temperature: float = 0.4) -> str:
        try:
            if isinstance(prompt, str):
                topic_name_info_raw = self._call_llm(prompt, temperature, max_tokens=128)
            elif isinstance(prompt, dict) and self.supports_system_prompts:
                topic_name_info_raw = self._call_llm_with_system_prompt(
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                    temperature=temperature,
                    max_tokens=128,
                )
            else:
                raise InvalidLLMInputError(
                    f"Prompt must be a string or a dictionary, got {type(prompt)}"
                )
            
            topic_name_info = llm_output_to_result(topic_name_info_raw, GET_TOPIC_NAME_REGEX)
            topic_name = topic_name_info["topic_name"]
        except Exception as e:
            raise ValueError(f"Failed to generate topic name with {self.__class__.__name__}")
        return topic_name

    # @abstractmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: retry_state.args[1] if len(retry_state.args) > 1 and isinstance(retry_state.args[1], list) else [],
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_cluster_names(
        self, prompt: Union[str, Dict[str, str]], old_names: List[str], temperature: float = 0.4,
    ) -> List[str]:
        try:
            if isinstance(prompt, str):
                topic_name_info_raw = self._call_llm(prompt, temperature, max_tokens=1024)
            elif isinstance(prompt, dict) and self.supports_system_prompts:
                topic_name_info_raw = self._call_llm_with_system_prompt(
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                    temperature=temperature,
                    max_tokens=1024,
                )
            else:
                raise InvalidLLMInputError(f"Prompt must be a string or a dictionary, got {type(prompt)}")
            
            topic_name_info = llm_output_to_result(
                topic_name_info_raw, GET_TOPIC_CLUSTER_NAMES_REGEX
            )
        except Exception as e:
            warn(f"Failed to generate topic cluster names with {self.__class__.__name__}: {e}")
            return old_names

        mapping = topic_name_info["new_topic_name_mapping"]
        if len(mapping) == len(old_names):
            result = []
            for i, old_name_val in enumerate(old_names, start=1):
                key_with_val = f"{i}. {old_name_val}"
                key_just_index = f"{i}."
                if key_with_val in mapping:
                    result.append(mapping[key_with_val])
                elif key_just_index in mapping: # This was `mapping.get(f"{n}.", name)` which is ambiguous
                    result.append(mapping[key_just_index])
                else:
                    result.append(old_name_val) # Fallback to old name to maintain length
            return result
        else:
            # Fallback to just parsing the string as best we can
            mapping = re.findall(
                r'"new_topic_name_mapping":\s*\{(.*?)\}',
                topic_name_info_raw,
                re.DOTALL,
            )[0]
            new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping, re.DOTALL)
            if len(new_names) == len(old_names):
                return new_names
            else:
                raise ValueError(f"Failed to generate enough names when fixing {old_names}; got {mapping}")

    @property
    def supports_system_prompts(self) -> bool:
        """
        Check if the LLM wrapper supports system prompts.
        By default, it does. Override in subclasses if not supported.
        """
        return True


try:
    import llama_cpp

    class LlamaCpp(LLMWrapper):

        def __init__(self, model_path: str, llm_specific_instructions=None, **kwargs):
            self.model_path = model_path
            for arg, val in kwargs.items():
                if arg == "n_ctx":
                    continue
                setattr(self, arg, val)
            self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm(
                prompt + self.extra_prompting,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            result = response["choices"][0]["text"]
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            raise InvalidLLMInputError(
                "System prompts are not supported for LlamaCpp wrapper"
            )

        @property
        def supports_system_prompts(self) -> bool:
            return False

except ImportError:
    pass

try:
    import huggingface_hub
    import transformers

    class HuggingFace(LLMWrapper):

        def __init__(self, model: str, llm_specific_instructions=None, **kwargs):
            self.model = model
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm(
                [{"role": "user", "content": prompt + self.extra_prompting}],
                return_full_text=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            result = response[0]["generated_text"]
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt + self.extra_prompting}],
                return_full_text=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            result = response[0]["generated_text"]
            print(result)
            return result

except ImportError:
    pass

try:
    import cohere

    class Cohere(LLMWrapper):

        def __init__(self, API_KEY: str, model: str = "command-r-08-2024", base_url: str = None, httpx_client: Optional[httpx.Client] = None, llm_specific_instructions=None):
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
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.chat(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
                # This results in failures more often than useful output
                # response_format={"type": "json_object"},
            )
            result = response.message.content[0].text
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.chat(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                temperature=temperature,
                # This results in failures more often than useful output
                # response_format={"type": "json_object"},
            )
            result = response.message.content[0].text
            return result

except:
    pass

try:
    import anthropic

    class Anthropic(LLMWrapper):

        def __init__(self, API_KEY: str, model: str = "claude-3-haiku-20240307", llm_specific_instructions=None):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.model = model
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
            )
            result = response.content[0].text
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                temperature=temperature,
            )
            result = response.content[0].text
            return result
except:
    pass

try:
    import openai

    class OpenAI(LLMWrapper):

        def __init__(
            self,
            API_KEY: str,
            model: str = "gpt-4o-mini",
            base_url: str = None,
            llm_specific_instructions=None,
            verbose: bool = False,
        ):
            self.llm = openai.OpenAI(api_key=API_KEY, base_url=base_url)
            self.model = model
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.verbose = verbose

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            return result

except:
    pass


try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    class AzureAI(LLMWrapper):

        def __init__(self, API_KEY: str, endpoint: str, model: str, llm_specific_instructions=None):
            self.endpoint = endpoint
            self.model = model
            self.llm = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(API_KEY),
            )
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.complete(
                model=self.model,
                max_tokens=max_tokens,
                messages=[UserMessage(prompt + self.extra_prompting)],
                temperature=temperature,
            )
            result = response.choices[0].message.content
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            response = self.llm.complete(
                model=self.model,
                max_tokens=max_tokens,
                messages=[SystemMessage(system_prompt), UserMessage(user_prompt + self.extra_prompting)],
                temperature=temperature,
            )
            result = response.choices[0].message.content
            return result
                
except ImportError:
    pass

