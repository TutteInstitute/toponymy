import string
from warnings import warn

import tokenizers
import transformers

from templates import GET_TOPIC_CLUSTER_NAMES_REGEX, GET_TOPIC_NAME_REGEX
import re

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
                topic_name_info = re.findall(
                    GET_TOPIC_NAME_REGEX, topic_name_info, re.DOTALL
                )[0]
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
                    GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text
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

        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return "\nThe short distinguising topic name is:\n"
            elif kind == "intermediate_layer":
                return "\nThe short topic name that encompasses the sub-topics is:\n"
            elif kind == "remedy":
                return "\nA better and more specific name that still captures the topic of these article titles is:\n"
            else:
                raise ValueError(
                    f"Invalid llm_imnstruction kind; should be one of 'base_layer', 'intermediate_layer', or 'remedy' not '{kind}'"
                )

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
                    GET_TOPIC_NAME_REGEX, topic_name_info_text, re.DOTALL
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
                    GET_TOPIC_CLUSTER_NAMES_REGEX, topic_name_info_text, re.DOTALL
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

try:

    import json

    import cohere

    class CohereWrapper:

        def __init__(self, API_KEY, model="command-r-08-2024", local_tokenizer=None):
            self.llm = cohere.Client(api_key=API_KEY)

            try:
                self.llm.models.get(model)
            except cohere.errors.not_found_error.NotFoundError:
                models = [x.name for x in self.llm.models.list().models]
                msg = f"Model '{model}' not found, try one of {models}"
                raise ValueError(msg)
            self.model = model

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

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                )
                topic_name_info_text = topic_name_info_raw.text
                topic_name_info = json.loads(topic_name_info_text)
            except Exception as e:
                warn(f"Failed to generate topic cluster names with Cohere: {e}")
                return old_names

            result = []
            for old_name, name_mapping in zip(old_names, topic_name_info):
                try:
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values())[0])
                    else:
                        warn(
                            f"Old name {old_name} does not match the new name {list(name_mapping.keys())[0]}"
                        )
                        # use old_name?
                        result.append(list(name_mapping.values())[0])
                except:
                    result.append(old_name)

            return result

        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return """
You are to give a brief (five to ten word) name describing this group.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "intermediate_layer":
                return """
You are to give a brief (three to five word) name describing this group of papers.
The topic should be the most specific topic that encompasses the breadth of sub-topics, with a focus on the major sub-topics.
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "remedy":
                return """
You are to give a brief (three to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be in JSON formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""
            else:
                raise ValueError(
                    f"Invalid llm_imnstruction kind; should be one of 'base_layer', 'intermediate_layer', or 'remedy' not '{kind}'"
                )

except:
    pass

try:

    import json

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
                result = []
                for old_name, name_mapping in zip(old_names, topic_name_info):
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values()[0]))
                    else:
                        result.append(old_name)

                return result
            except:
                return old_names

        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return """
You are to give a brief (five to ten word) name describing this group.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "intermediate_layer":
                return """
You are to give a brief (three to five word) name describing this group of papers.
The topic should be the most specific topic that encompasses the breadth of sub-topics, with a focus on the major sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "remedy":
                return """
You are to give a brief (five to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""
            else:
                raise ValueError(
                    f"Invalid llm_imnstruction kind; should be one of 'base_layer', 'intermediate_layer', or 'remedy' not '{kind}'"
                )

except:
    pass

try:
    import json

    import openai

    class OpenAIWrapper:

        def __init__(self, API_KEY, model="gpt-4o-mini", base_url=None, verbose=False):
            self.llm = openai.OpenAI(api_key=API_KEY, base_url=base_url)
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
                result = []
                for old_name, name_mapping in zip(old_names, topic_name_info):
                    if old_name.lower() == list(name_mapping.keys())[0].lower():
                        result.append(list(name_mapping.values()[0]))
                    else:
                        result.append(old_name)

                return result
            except:
                return old_names

        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return """
You are to give a brief (five to ten word) name describing this group.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response must be **ONLY** JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "intermediate_layer":
                return """
You are to give a brief (three to five word) name describing this group of papers.
The topic should be the most specific topic that encompasses the breadth of sub-topics, with a focus on the major sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "remedy":
                return """
You are to give a brief (five to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""
            else:
                raise ValueError(
                    f"Invalid llm_imnstruction kind; should be one of 'base_layer', 'intermediate_layer', or 'remedy' not '{kind}'"
                )

except:
    pass
