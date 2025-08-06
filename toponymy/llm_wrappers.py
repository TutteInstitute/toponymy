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
import asyncio

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


class AsyncLLMWrapper(ABC):

    @abstractmethod
    async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
        """
        Call the LLM with a batch of prompts and temperature.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def _call_llm_with_system_prompt_batch(
        self, 
        system_prompts: List[str], 
        user_prompts: List[str], 
        temperature: float, 
        max_tokens: int
    ) -> List[str]:
        """
        Call the LLM with batches of system prompts and user prompts.
        This method should be implemented by subclasses.
        """
        pass


    async def generate_topic_names(
        self, 
        prompts: List[Union[str, Dict[str, str]]], 
        temperature: float = 0.4
    ) -> List[str]:
        """
        Generate topic names for a batch of prompts.
        Returns a list of topic names matching the input prompts.
        """
        if not prompts:
            return []
        
        # Check the first prompt to determine type
        if isinstance(prompts[0], str):
            responses = await self._call_llm_batch(prompts, temperature, max_tokens=128)
        elif isinstance(prompts[0], dict) and self.supports_system_prompts:
            system_prompts = [p["system"] for p in prompts]
            user_prompts = [p["user"] for p in prompts]
            responses = await self._call_llm_with_system_prompt_batch(
                system_prompts, user_prompts, temperature, max_tokens=128
            )
        else:
            raise InvalidLLMInputError(
                f"Prompts must be strings or dictionaries, got {type(prompts[0])}"
            )
        
        # Parse responses
        results = []
        for response in responses:
            if not response:
                results.append("")
                continue

            # Attempt to parse the response
            try:
                topic_name_info = llm_output_to_result(response, GET_TOPIC_NAME_REGEX)
                results.append(topic_name_info["topic_name"])
            except Exception as e:
                warn(f"Failed to generate topic name with {self.__class__.__name__}: {e}")
                results.append("")  # Fallback to empty string if parsing fails
        
        return results


    async def generate_topic_cluster_names(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        old_names_list: List[List[str]],
        temperature: float = 0.4,
    ) -> List[List[str]]:
        """
        Generate topic cluster names for a batch of prompts.
        Returns a list of lists of topic names matching the input prompts.
        """
        if len(prompts) != len(old_names_list):
            raise ValueError("Number of prompts must match number of old_names lists")
        
        if not prompts:
            return []
        
        # Check the first prompt to determine type
        if isinstance(prompts[0], str):
            responses = await self._call_llm_batch(prompts, temperature, max_tokens=1024)
        elif isinstance(prompts[0], dict) and self.supports_system_prompts:
            system_prompts = [prompt["system"] for prompt in prompts]
            user_prompts = [prompt["user"] for prompt in prompts]
            responses = await self._call_llm_with_system_prompt_batch(
                system_prompts, user_prompts, temperature, max_tokens=1024
            )
        else:
            raise InvalidLLMInputError(
                f"Prompts must be strings or dictionaries, got {type(prompts[0])}"
            )
        
        # Parse responses
        results = []
        for response, old_names in zip(responses, old_names_list):
            results.append(self._parse_cluster_response(response, old_names))
        
        return results

    def _parse_cluster_response(self, response: str, old_names: List[str]) -> List[str]:
        """Parse a single cluster response."""
        try:
            topic_name_info = llm_output_to_result(response, GET_TOPIC_CLUSTER_NAMES_REGEX)
            mapping = topic_name_info["new_topic_name_mapping"]
            
            if len(mapping) == len(old_names):
                result = []
                for i, old_name_val in enumerate(old_names, start=1):
                    key_with_val = f"{i}. {old_name_val}"
                    key_just_index = f"{i}."
                    if key_with_val in mapping:
                        result.append(mapping[key_with_val])
                    elif key_just_index in mapping:
                        result.append(mapping[key_just_index])
                    else:
                        result.append(old_name_val)
                return result
            else:
                # Fallback parsing
                mapping_str = re.findall(
                    r'"new_topic_name_mapping":\s*\{(.*?)\}',
                    response,
                    re.DOTALL,
                )[0]
                new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping_str, re.DOTALL)
                if len(new_names) == len(old_names):
                    return new_names
                else:
                    raise ValueError(f"Failed to generate enough names; got {mapping}")
        except Exception as e:
            warn(f"Failed to parse cluster names: {e}")
            return old_names

    @property
    def supports_system_prompts(self) -> bool:
        """
        Check if the LLM wrapper supports system prompts.
        By default, it does. Override in subclasses if not supported.
        """
        return True

class LLMWrapperImportError(ImportError):
    """A custom exception for missing package dependencies required by LLM wrappers. In these cases we do not want to retry, as the error will not resolve until the required package is installed."""
    pass

class FailedImportLLMWrapper(LLMWrapper):

    @classmethod
    def _import_error_message(cls):
        return f"Failed to import LLMWrapper for {cls.__name__}. This is likely because the required package is not installed. Please install the required package and try again."

    def __init__(self, *args, **kwds):
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        raise LLMWrapperImportError(self._import_error_message())
    

class FailedImportAsyncLLMWrapper(AsyncLLMWrapper):
    @classmethod
    def _import_error_message(cls):
        return f"Failed to import AsyncLLMWrapper for {cls.__name__}. This is likely because the required package is not installed. Please install the required package and try again."

    def __init__(self, *args, **kwds):
        raise LLMWrapperImportError(self._import_error_message())
    
    async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())

    async def _call_llm_with_system_prompt_batch(self, system_prompt: str, user_prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())

try:
    import llama_cpp

    class LlamaCpp(LLMWrapper):
        """
        Provides Access to LlamaCpp models with the Toponymy framework. For more information on LlamaCpp, see
        https://github.com/abetlen/llama-cpp-python. You will need llamma-cpp-python installed to make use of 
        this wrapper, and you will need a local model file downloaded to use it. This Wrapper allows you
        to use local models, rather than requiring a service API key. However this does require you to have the model
        and suitable hardware to run it.
        
        Note: This wrapper does not support system prompts, as LlamaCpp does not support them.
        
        Parameters:
        -----------
        
        model_path: str
            The path to the local LlamaCpp model file.
            
        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.
            
        **kwargs: dict, optional
            Additional keyword arguments passed to the LlamaCpp model initialization.
        
        Attributes:
        -----------
        model_path: str
            The path to the local LlamaCpp model file.
            
        llm: llama_cpp.Llama
            The LlamaCpp model instance.
            
        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.
            
        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For LlamaCpp, this is always False.
        """

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
    class LlamaCpp(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

try:
    import huggingface_hub
    import transformers

    class HuggingFace(LLMWrapper):
        """
        Provides access to Huggingface models from Huggingface Hub with the Toponymy framework. 
        For more information on Huggingface, see https://huggingface.co/docs/transformers/index.
        You will need the transformers library installed to make use of this wrapper, and you will need a model
        available on Huggingface Hub. This wrapper allows you to use models hosted on Huggingface Hub,
        rather than requiring a service API key. However, this does require you to have access to the model
        and suitable hardware to run it.

        Parameters:
        -----------
        model: str
            The name of the Huggingface model to use, e.g. "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-3-1b-it", etc.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        **kwargs: dict, optional
            Additional keyword arguments passed to the Huggingface model initialization.

        Attributes:
        -----------
        model: str
            The name of the Huggingface model to use.

        llm: transformers.pipeline
            The Huggingface model instance.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Huggingface, this is always True.
        """


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
        
    class AsyncHuggingFace(AsyncLLMWrapper):
        """This class is essentially for testing purposes only, allowing testing of the Async API with local models."""

        def __init__(self, model: str, llm_specific_instructions: Optional[str] = None, max_concurrent_requests: int = 10, **kwargs):
            self.model = model
            self.llm = transformers.pipeline("text-generation", model=model, **kwargs)
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.max_concurrent_requests = max_concurrent_requests

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            responses = []
            for prompt in prompts:
                response = self.llm(
                    [{"role": "user", "content": prompt + self.extra_prompting}],
                    return_full_text=False,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.eos_token_id,
                )
                responses.append(response[0]["generated_text"])
            return responses

        async def _call_llm_with_system_prompt_batch(
            self, system_prompts: List[str], user_prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            responses = []
            for system_prompt, user_prompt in zip(system_prompts, user_prompts):
                response = self.llm(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt + self.extra_prompting}],
                    return_full_text=False,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.eos_token_id,
                )
                responses.append(response[0]["generated_text"])
            return responses
except:
    class HuggingFace(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncHuggingFace(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

try:
    import vllm
    import vllm.v1.engine.exceptions

    class VLLM(LLMWrapper):
        """
        Provides access to Huggingface models from Huggingface Hub ran via vLLM, with the Toponymy framework. 
        For more information on vLLM, see https://docs.vllm.ai/en/latest/.
        You will need the vllm library installed to make use of this wrapper, and you will need a model
        available on Huggingface Hub. This wrapper allows you to use models hosted on Huggingface Hub,
        rather than requiring a service API key. However, this does require you to have access to the model
        and suitable hardware to run it.

        Parameters:
        -----------
        model: str
            The name of the Huggingface model to use, e.g. "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-3-1b-it", etc.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt.

        **kwargs: dict, optional
            Additional keyword arguments passed to the vLLM model initialization.

        Attributes:
        -----------
        model: str
            The name of the Huggingface model to use.

        llm: transformers.pipeline
            The vLLM model instance.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Huggingface, this is always True.
        """


        def __init__(self, model: str, llm_specific_instructions=None, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self._start_engine()
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

        def _start_engine(self):
            """
            Start the VLLM engine. This is necessary to initialize the model.
            """
            self.llm = vllm.LLM(model=self.model, **self.kwargs)

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens)
            message =  [{"role": "user", "content": prompt + self.extra_prompting}]
            try:
                outputs = self.llm.chat(message, sampling_params=sampling_params)
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()
                # Retry after restarting the engine
                outputs = self.llm.chat(message, sampling_params=sampling_params)
            result = outputs[0].outputs[0].text
            return result
        
        def _call_llm_with_system_prompt(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
            sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens)
            messages = [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt + self.extra_prompting}]
            
            try:
                outputs = self.llm.chat(messages, sampling_params=sampling_params)
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()
                outputs = self.llm.chat(messages, sampling_params=sampling_params)
            
            result = outputs[0].outputs[0].text
            return result
        
    class AsyncVLLM(AsyncLLMWrapper):
        """This class is essentially for testing purposes only, allowing testing of the Async API with local models."""

        def __init__(self, model: str, llm_specific_instructions: Optional[str] = None, max_concurrent_requests: int = 10, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self._start_engine()
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.max_concurrent_requests = max_concurrent_requests

        def _start_engine(self):
            self.llm = vllm.LLM(model=self.model, **self.kwargs)

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            messages = [[{"role": "user", "content": prompt + self.extra_prompting}] for prompt in prompts]
            sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens)
                
            try:
                outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()  # Restart the engine if it fails
                outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
            
            return [output.outputs[0].text for output in outputs]

        async def _call_llm_with_system_prompt_batch(
            self, system_prompts: List[str], user_prompts: List[str], temperature: float, max_tokens: int
        ) -> List[str]:
            messages = []
            for system_prompt, user_prompt in zip(system_prompts, user_prompts):
                messages.append([{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt + self.extra_prompting}])
            sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens)
            
            try:
                outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
            except vllm.v1.engine.exceptions.EngineDeadError:
                self._start_engine()  # Restart the engine if it fails
                outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)

            return [output.outputs[0].text for output in outputs]

except ImportError:
    class VLLM(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncVLLM(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

try:
    import cohere

    class Cohere(LLMWrapper):
        """
        Provides access to Cohere's LLMs with the Toponymy framework. For more information on Cohere, see
        https://docs.cohere.com/docs/llm-overview. You will need a Cohere API key to use this wrapper. The
        default model is "command-r-08-2024", which is a sufficiently powerful to do a good job of generating 
        topic names and clusters, but inexpensive in terms of dollars per token. You can use more advanced
        models, but they have diminishing returns for this task, and are more expensive.

        Parameters:
        -----------

        api_key: str
            Your Cohere API key. You can set this as an environment variable CO_API_KEY or pass it directly

        model: str, optional
            The name of the Cohere model to use. Default is "command-r-08-2024". You can use any model available
            in the Cohere API, but this is a good balance of performance and cost.

        base_url: str, optional
            The base URL for the Cohere API. Default is "https://api.cohere.com". You can set this as an environment
            variable CO_API_URL to use a different endpoint, such as for Cohere's private cloud.

        httpx_client: httpx.Client, optional
            An optional httpx client to use for making requests. If not provided, a default client will be created.
            This can be useful when using Cohere's private cloud or when you need to customize the HTTP client settings.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        Attributes:
        -----------

        llm: cohere.ClientV2
            The Cohere LLM client instance.

        model: str
            The name of the Cohere model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Cohere, this is always True.

        Note:
        -----
        This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
        consider using the AsyncCohere wrapper instead.
        """

        def __init__(self, api_key: str, model: str = "command-r-08-2024", base_url: str = None, httpx_client: Optional[httpx.Client] = None, llm_specific_instructions=None):
            if base_url is None:
                base_url = os.getenv("CO_API_URL", "https://api.cohere.com")

            api_key = api_key or os.getenv("CO_API_KEY")
            if not api_key:
                raise ValueError("Cohere API key is required. Set it as an environment variable CO_API_KEY or pass it directly to the constructor.")
            
            self.llm = cohere.ClientV2(api_key=api_key, base_url=base_url, httpx_client=httpx_client)

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

    class AsyncCohere(AsyncLLMWrapper):
        """
        Provides access to Cohere's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        For more information on Cohere, see https://docs.cohere.com/docs/llm-overview. You will need a Cohere API key to use this wrapper.
        The default model is "command-r-08-2024", which is a sufficiently powerful model for generating topic names and clusters,
        but inexpensive in terms of dollars per token. You can use more advanced models, but they have diminishing returns for this task,
        and are more expensive.
        
        As an asynchronous wrapper this will potentially speed up topic naming, particlarly when you have a large number of topics. If, 
        however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls."

        Parameters:
        -----------
        api_key: str
            Your Cohere API key. You can set this as an environment variable CO_API_KEY or pass it directly

        model: str, optional
            The name of the Cohere model to use. Default is "command-r-08-2024". You can use any model available
            in the Cohere API, but this is a good balance of performance and cost.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Cohere API. Default is 10. This can be adjusted based on your
            application's needs and the rate limits of the Cohere API. Higher values may improve throughput but could lead to rate limiting.

        base_url: str, optional
            The base URL for the Cohere API. Default is "https://api.cohere.com". You can set this as an environment
            variable CO_API_URL to use a different endpoint, such as for Cohere's private cloud.

        httpx_client: httpx.Client, optional
            An optional httpx client to use for making requests. If not provided, a default client will be created.

        Attributes:
        -----------
        llm: cohere.AsyncClientV2
            The Cohere asynchronous LLM client instance.

        model: str
            The name of the Cohere model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Cohere, this is always True.

        """
        
        def __init__(self, api_key: str, model: str = "command-r-08-2024", 
                     llm_specific_instructions=None, max_concurrent_requests: int = 10, base_url: str = None, httpx_client: Optional[httpx.Client] = None):
            if base_url is None:
                base_url = os.getenv("CO_API_URL", "https://api.cohere.com")

            api_key = api_key or os.getenv("CO_API_KEY")
            if not api_key:
                raise ValueError("Cohere API key is required. Set it as an environment variable CO_API_KEY or pass it directly to the constructor.")
            
            self.llm = cohere.AsyncClientV2(api_key=api_key, base_url=base_url, httpx_client=httpx_client)             
            self.model = model
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            """Call the LLM for a single prompt."""
            try:
                async with self.semaphore:
                    response = await self.llm.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.message.content[0].text
            except Exception as e:
                warn(f"Cohere API call failed: {str(e)[:100]}...")
                return ""

        async def _call_single_llm_with_system(
            self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            try:
                async with self.semaphore:
                    response = await self.llm.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt + self.extra_prompting},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.message.content[0].text
            except Exception as e:
                warn(f"Cohere API call failed: {str(e)[:100]}...")
                return ""

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self, 
            system_prompts: List[str], 
            user_prompts: List[str], 
            temperature: float, 
            max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Number of system prompts must match number of user prompts")
            
            tasks = [
                self._call_single_llm_with_system(sys_prompt, user_prompt, temperature, max_tokens)
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

        async def close(self):
            """Close the client connection."""
            await self.llm.close()

except:
    class Cohere(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncCohere(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

try:
    import anthropic
    import time

    class Anthropic(LLMWrapper):
        """
        Provides access to Anthropic's LLMs with the Toponymy framework. For more information on Anthropic, see
        https://docs.anthropic.com/docs/overview. You will need an Anthropic API key to use this wrapper.
        The default model is "claude-3-haiku-20240307", which is the smallest model available, but is generally
        more than sufficient for generating topic names and clusters. You can use more advanced
        models, but they have diminishing returns for this task, and are more expensive.
        
        Parameters:
        -----------
        api_key: str
            Your Anthropic API key. You can set this as an environment variable ANTHROPIC_API_KEY or pass it directly.

        model: str, optional
            The name of the Anthropic model to use. Default is "claude-3-haiku-20240307". You can use any model available
            in the Anthropic API, but this is a good balance of performance and cost.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        Attributes:
        -----------

        llm: anthropic.Anthropic
            The Anthropic LLM client instance.

        model: str
            The name of the Anthropic model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Anthropic, this is always True.

        Note:
        -----
        This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
        consider using the AsyncAnthropic wrapper instead.
        """

        def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", llm_specific_instructions=None):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required. Set it as an environment variable ANTHROPIC_API_KEY or pass it directly to the constructor.")
            
            self.llm = anthropic.Anthropic(api_key=api_key)
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
        
    class AsyncAnthropic(AsyncLLMWrapper):
        """
        Provides access to Anthropic's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        For more information on Anthropic, see https://docs.anthropic.com/docs/overview. You will need an Anthropic API key to use this wrapper.
        The default model is "claude-3-haiku-20240307", which is the smallest model available, but is generally
        more than sufficient for generating topic names and clusters. You can use more advanced models, but they have diminishing returns for this task,
        and are more expensive.
        
        As an asynchronous wrapper this will potentially speed up topic naming, particularly when you have a large number of topics. If,
        however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls.
        
        Parameters:
        -----------
        
        api_key: str
            Your Anthropic API key. You can set this as an environment variable ANTHROPIC_API_KEY or pass it directly.
            
        model: str, optional
            The name of the Anthropic model to use. Default is "claude-3-haiku-20240307". You can use any model available
            in the Anthropic API, but this is a good balance of performance and cost.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Anthropic API. Default is 10. This can be adjusted based on your
            application's needs and the rate limits of the Anthropic API. Higher values may improve throughput but could lead to rate limiting.

        Attributes:
        -----------
        llm: anthropic.AsyncAnthropic
            The Anthropic asynchronous LLM client instance.

        model: str
            The name of the Anthropic model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Anthropic, this is always True.
        """
        
        def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", 
                     llm_specific_instructions=None, max_concurrent_requests: int = 10):
            
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required. Set it as an environment variable ANTHROPIC_API_KEY or pass it directly to the constructor.")
            
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            self.model = model
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            """Call the LLM for a single prompt."""
            async with self.semaphore:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                    temperature=temperature,
                )
                return response.content[0].text

        async def _call_single_llm_with_system(
            self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            async with self.semaphore:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + self.extra_prompting},
                    ],
                    temperature=temperature,
                )
                return response.content[0].text

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self, 
            system_prompts: List[str], 
            user_prompts: List[str], 
            temperature: float, 
            max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Number of system prompts must match number of user prompts")
            
            tasks = [
                self._call_single_llm_with_system(sys_prompt, user_prompt, temperature, max_tokens)
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

    class BatchAnthropic(AsyncLLMWrapper):
        """
        Provides access to Anthropic's Batch Processing API with asynchronous support.
        This allows for processing large batches of prompts over an extended period.
        For more information on Anthropic's Batch Processing, see https://docs.anthropic.com/docs/batch-processing.

        This wrapper conforms to the AsyncLLMWrapper interface, but note that it uses Anthropic's batch API,
        which processes jobs over hours rather than in real-time. The async methods will block until the batch job completes.

        This class provides a different tradeoff between speed and cost compared to the AsyncAnthropic wrapper.
        It is designed for scenarios where you have a large number of prompts to process and can afford to wait for the results.
        Anthropic's batch processing is more cost-effective (half the cost per token) for large volumes of data, but it does 
        not provide immediate responses.

        Parameters:
        -----------
        api_key: str
            Your Anthropic API key. You can set this as an environment variable ANTHROPIC_API_KEY or pass it directly.

        model: str, optional
            The name of the Anthropic model to use. Default is "claude-3-haiku-20240307". You can use any model available
            in the Anthropic API, but this is a good balance of performance and cost.
        
        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        polling_interval: int, optional
            The interval (in seconds) to poll the batch job status. Default is 60 seconds. This controls how often
            the wrapper checks the status of the batch job. A lower value will check more frequently, but may increase API usage.

        timeout: int, optional
            The maximum time (in seconds) to wait for the batch job to complete. Default is 7200 seconds (2 hours). If
            the job does not complete within this time, it will raise a RuntimeError. This is useful to prevent indefinite blocking
            if the batch job takes too long to process. You can adjust this based on your expected processing time.

        Attributes:
        -----------
        client: anthropic.Anthropic
            The Anthropic client instance for batch processing.

        model: str
            The name of the Anthropic model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Anthropic, this is always True.
        
        """
        
        def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", 
                     llm_specific_instructions=None, polling_interval: int = 60, 
                     timeout: int = 7200):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.polling_interval = polling_interval
            self.timeout = timeout

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            """
            Submit a batch job and wait for completion.
            This is a blocking operation that could take hours.
            """
            # Create batch requests
            requests = []
            for i, prompt in enumerate(prompts):
                requests.append({
                    "custom_id": str(i),
                    "params": {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt + self.extra_prompting}],
                        "temperature": temperature,
                    }
                })
            
            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id
            
            # Wait for completion (with async sleep)
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _call_llm_with_system_prompt_batch(
            self, 
            system_prompts: List[str], 
            user_prompts: List[str], 
            temperature: float, 
            max_tokens: int
        ) -> List[str]:
            """
            Submit a batch job with system prompts and wait for completion.
            """
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Number of system prompts must match number of user prompts")
            
            # Create batch requests
            requests = []
            for i, (sys_prompt, user_prompt) in enumerate(zip(system_prompts, user_prompts)):
                requests.append({
                    "custom_id": str(i),
                    "params": {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt + self.extra_prompting}
                        ],
                        "temperature": temperature,
                    }
                })
            
            # Submit batch
            batch = self.client.beta.messages.batches.create(requests=requests)
            batch_id = batch.id
            
            # Wait for completion
            if await self._wait_for_completion_async(batch_id):
                return await self._retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        async def _wait_for_completion_async(self, batch_id: str) -> bool:
            """
            Wait for a batch job to complete, using async sleep.
            Returns True if completed successfully, False if failed or timed out.
            """
            start_time = time.time()
            
            while time.time() - start_time < self.timeout:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
                
                if batch.processing_status == "ended":
                    return True
                elif batch.processing_status in ["canceling", "canceled", "expired"]:
                    warn(f"Batch job {batch_id} ended with status: {batch.processing_status}")
                    return False
                
                # Use async sleep to not block the event loop
                await asyncio.sleep(self.polling_interval)
            
            warn(f"Batch job {batch_id} timed out after {self.timeout} seconds")
            return False

        async def _retrieve_batch_results(self, batch_id: str) -> List[str]:
            """
            Retrieve raw text results from a completed batch job.
            """
            # Run the synchronous API call in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            results_page = await loop.run_in_executor(
                None, 
                self.client.beta.messages.batches.results, 
                batch_id
            )
            
            # Sort by custom_id to maintain order
            sorted_results = sorted(results_page.results, key=lambda x: int(x.custom_id))
            
            responses = []
            for result in sorted_results:
                if result.result.type == "succeeded":
                    responses.append(result.result.message.content[0].text)
                else:
                    warn(f"Request {result.custom_id} failed: {result.result.error}")
                    responses.append("")  # Empty string for failed requests
            
            return responses

        # Additional methods for non-blocking usage
        def submit_batch(
            self, 
            prompts: List[Union[str, Dict[str, str]]], 
            temperature: float, 
            max_tokens: int
        ) -> str:
            """
            Submit a batch job without waiting. Returns batch ID.
            This is for users who want to manage batch jobs manually.
            """
            requests = []
            
            for i, prompt in enumerate(prompts):
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt + self.extra_prompting}]
                elif isinstance(prompt, dict):
                    messages = [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"] + self.extra_prompting}
                    ]
                else:
                    raise InvalidLLMInputError(f"Prompt must be string or dict")
                
                requests.append({
                    "custom_id": str(i),
                    "params": {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "temperature": temperature,
                    }
                })
            
            batch = self.client.beta.messages.batches.create(requests=requests)
            return batch.id

        def get_batch_status(self, batch_id: str) -> str:
            """Check the status of a batch job."""
            batch = self.client.beta.messages.batches.retrieve(batch_id)
            return batch.processing_status

        async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
            """Retrieve raw text results from a completed batch."""
            return await self._retrieve_batch_results(batch_id)

        def cancel_batch(self, batch_id: str):
            """Cancel a running batch job."""
            self.client.beta.messages.batches.cancel(batch_id)

except:
    class Anthropic(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncAnthropic(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class BatchAnthropic(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

try:
    import openai

    class OpenAI(LLMWrapper):
        """
        Provides access to OpenAI's LLMs with the Toponymy framework. For more information on OpenAI, see
        https://platform.openai.com/docs/models/overview. You will need an OpenAI API key to use this wrapper.
        The default model is "gpt-4o-mini", which is a sufficiently powerful model for generating topic names and clusters,
        but inexpensive in terms of dollars per token. You can use more advanced models, but they have diminishing returns
        for this task, and are more expensive.
        
        Parameters:
        -----------
        
        api_key: str
            Your OpenAI API key. You can set this as an environment variable OPENAI_API_KEY or pass it directly
            
        model: str, optional
            The name of the OpenAI model to use. Default is "gpt-4o-mini". You can use any model available
            in the OpenAI API, but this is a good balance of performance and cost.

        base_url: str, optional
            The base URL for the OpenAI API. Default is None, which uses the default OpenAI endpoint.
            You can set this as an environment variable OPENAI_API_BASE to use a different endpoint, such as 
            a hosted model supporting the openAI API.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        Attributes:
        -----------
        llm: openai.OpenAI
            The OpenAI LLM client instance.

        model: str
            The name of the OpenAI model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For OpenAI, this is always True.

        Note:
        -----
        This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
        consider using the AsyncOpenAI wrapper instead.
        """
        def __init__(
            self,
            api_key: str,
            model: str = "gpt-4o-mini",
            base_url: str = None,
            llm_specific_instructions=None,
        ):
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required. Set it as an environment variable OPENAI_API_KEY or pass it directly to the constructor.")
            
            self.llm = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.extra_prompting =  "\n\n" + llm_specific_instructions if llm_specific_instructions else ""

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


    class AsyncOpenAI(AsyncLLMWrapper):
        """
        Provides access to OpenAI's LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        For more information on OpenAI, see https://platform.openai.com/docs/models/overview. You will need an OpenAI API key to use this wrapper.
        The default model is "gpt-4o-mini", which is a sufficiently powerful model for generating topic names and clusters,
        but inexpensive in terms of dollars per token. You can use more advanced models, but they have diminishing returns for this task,
        and are more expensive.
        
        As an asynchronous wrapper this will potentially speed up topic naming, particularly when you have a large number of topics. If,
        however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls.
        
        Parameters:
        -----------

        api_key: str
            Your OpenAI API key. You can set this as an environment variable OPENAI_API_KEY or pass it directly
        
        model: str, optional
            The name of the OpenAI model to use. Default is "gpt-4o-mini". You can use any model available
            in the OpenAI API, but this is a good balance of performance and cost.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the OpenAI API. Default is 10. This can be adjusted based on your
            application's needs and the rate limits of the OpenAI API. Higher values may improve throughput but could lead to rate limiting.

        organization: str, optional
            The OpenAI organization ID to use for the API requests. If not provided, the default organization will be used.

        Attributes:
        -----------

        client: openai.AsyncOpenAI
            The OpenAI asynchronous LLM client instance.

        model: str
            The name of the OpenAI model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For OpenAI, this is always True.
        """
        
        def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                     llm_specific_instructions=None, max_concurrent_requests: int = 10,
                     organization: str = None):
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required. Set it as an environment variable OPENAI_API_KEY or pass it directly to the constructor.")
            
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                organization=organization
            )
            self.model = model
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            """Call the LLM for a single prompt."""
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content
            except Exception as e:
                warn(f"OpenAI API call failed: {str(e)[:100]}...")
                return ""

        async def _call_single_llm_with_system(
            self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt + self.extra_prompting}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content
            except Exception as e:
                warn(f"OpenAI API call failed: {str(e)[:100]}...")
                return ""

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self, 
            system_prompts: List[str], 
            user_prompts: List[str], 
            temperature: float, 
            max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Number of system prompts must match number of user prompts")
            
            tasks = [
                self._call_single_llm_with_system(sys_prompt, user_prompt, temperature, max_tokens)
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

        async def close(self):
            """Close the client connection."""
            await self.client.close()

except:

    class OpenAI(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

    class AsyncOpenAI(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)


try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    class AzureAI(LLMWrapper):
        """
        Provides access to the Azure AI Foundry LLMs with the Toponymy framework. For more information on Azure AI, see
        https://learn.microsoft.com/en-us/azure/ai-services/overview. You will need an Azure API key for your Foundry model 
        to use this wrapper. You will need to provide both the endpoint, and the model name per the instiated model on
        AI Foundry. For more information on creating models with Azure AI Foundry, see https://learn.microsoft.com/en-us/azure/ai-services/ai-foundry/create-models.

        Parameters:
        -----------
        api_key: str
            Your Azure API key. You can set this as an environment variable AZURE_API_KEY or pass it directly

        endpoint: str
            The endpoint URL for your Azure AI Foundry model. This is typically in the format "https://<your-resource-name>.openai.azure.com/".

        model: str
            The name of the Azure AI Foundry model to use. This should match the model name you created in Azure AI Foundry.

        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.

        Attributes:
        -----------

        llm: azure.ai.inference.ChatCompletionsClient
            The Azure AI Foundry LLM client instance.

        model: str
            The name of the Azure AI Foundry model being used.

        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.

        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Azure AI Foundry, this is always True.

        Note:
        -----
        This wrapper does not support batch processing. If you need to process multiple prompts concurrently,
        consider using the AsyncAzureAI wrapper instead. 
        
        """

        def __init__(self, api_key: str, endpoint: str, model: str, llm_specific_instructions=None):
            self.endpoint = endpoint
            self.model = model

            api_key = api_key or os.getenv("AZURE_API_KEY")
            if not api_key:
                raise ValueError("Azure API key is required. Set it as an environment variable AZURE_API_KEY or pass it directly to the constructor.")
            
            if not endpoint:
                raise ValueError("Azure endpoint is required. Provide the endpoint URL for your Azure AI Foundry model.")
            
            if not model:
                raise ValueError("Azure model name is required. Provide the name of the Azure AI Foundry model to use.")
            
            self.llm = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
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
                
    class AsyncAzureAI(AsyncLLMWrapper):
        """
        Provides access to the Azure AI Foundry LLMs with asynchronous support. This allows for concurrent processing of multiple prompts.
        For more information on Azure AI, see https://learn.microsoft.com/en-us/azure/ai-services/overview. You will need an Azure API key for your Foundry model
        to use this wrapper. You will need to provide both the endpoint, and the model name per the instiated model on
        AI Foundry. For more information on creating models with Azure AI Foundry, see https://learn.microsoft.com/en-us/azure/ai-services/ai-foundry/create-models.

        This wrapper conforms to the AsyncLLMWrapper interface, and is designed for scenarios where you need to process multiple prompts concurrently.
        This is particularly useful for applications that require high throughput or need to process large volumes of data quickly.
        As an asynchronous wrapper this will potentially speed up topic naming, particularly when you have a large number of topics. If,
        however, there are quirks in your data, or bugs in Toponymy's prompt generation, you will potentially quickly spend money on API calls.
        
        Parameters:
        -----------
        api_key: str
            Your Azure API key. You can set this as an environment variable AZURE_API_KEY or pass it directly
            
        endpoint: str
            The endpoint URL for your Azure AI Foundry model. This is typically in the format "https://<your-resource-name>.openai.azure.com/".
            
        model: str
            The name of the Azure AI Foundry model to use. This should match the model name you created in Azure AI Foundry.
            
        llm_specific_instructions: str, optional
            Additional instructions specific to the LLM, appended to the prompt. This can be used to provide
            model-specific instructions or context that may help improve the quality of the generated text.
        
        max_concurrent_requests: int, optional
            The maximum number of concurrent requests to the Azure AI Foundry API. Default is 10. This can be adjusted based on your
            application's needs and the rate limits of the Azure AI Foundry API. Higher values may improve throughput but could lead to rate limiting.
            
        Attributes:
        -----------
        client: azure.ai.inference.aio.ChatCompletionsClient
            The Azure AI Foundry asynchronous LLM client instance.
            
        model: str
            The name of the Azure AI Foundry model being used.
            
        extra_prompting: str
            Additional instructions specific to the LLM, appended to the prompt.
            
        supports_system_prompts: bool
            Indicates whether the wrapper supports system prompts. For Azure AI Foundry, this is always True.
            
        """
        
        def __init__(self, api_key: str, endpoint: str, model: str, 
                     llm_specific_instructions=None, max_concurrent_requests: int = 10):
            api_key = api_key or os.getenv("AZURE_API_KEY")
            if not api_key:
                raise ValueError("Azure API key is required. Set it as an environment variable AZURE_API_KEY or pass it directly to the constructor.")
            
            if not endpoint:
                raise ValueError("Azure endpoint is required. Provide the endpoint URL for your Azure AI Foundry model.")
            
            if not model:
                raise ValueError("Azure model name is required. Provide the name of the Azure AI Foundry model to use.")
            
            self.endpoint = endpoint
            self.model = model
            self.client = AsyncChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
            self.extra_prompting = "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        async def _call_single_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            """Call the LLM for a single prompt."""
            async with self.semaphore:
                try:
                    response = await self.client.complete(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[UserMessage(prompt + self.extra_prompting)],
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return "" # Return empty string on error to avoid breaking batch processing

        async def _call_single_llm_with_system(
            self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
        ) -> str:
            """Call the LLM for a single prompt with system prompt."""
            async with self.semaphore:
                try:
                    response = await self.client.complete(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[
                            SystemMessage(system_prompt),
                            UserMessage(user_prompt + self.extra_prompting)
                        ],
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    return ""  # Return empty string on error to avoid breaking batch processing

        async def _call_llm_batch(self, prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
            """Process a batch of prompts concurrently."""
            tasks = [
                self._call_single_llm(prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

        async def _call_llm_with_system_prompt_batch(
            self, 
            system_prompts: List[str], 
            user_prompts: List[str], 
            temperature: float, 
            max_tokens: int
        ) -> List[str]:
            """Process a batch of prompts with system prompts concurrently."""
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Number of system prompts must match number of user prompts")
            
            tasks = [
                self._call_single_llm_with_system(sys_prompt, user_prompt, temperature, max_tokens)
                for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
            return await asyncio.gather(*tasks)

        async def close(self):
            """Close the client connection."""
            await self.client.close()

except ImportError:
    
    class AzureAI(FailedImportLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)
    
    class AsyncAzureAI(FailedImportAsyncLLMWrapper):

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)

