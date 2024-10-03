import tokenizers
import transformers
import string

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
            topic_name = self.llm(prompt, temperature=temperature)['choices'][0]['text']
            if "\n" in topic_name:
                topic_name = topic_name.lstrip("\n ")
                topic_name = topic_name.split("\n")[0]
            topic_name = string.capwords(topic_name.strip(string.punctuation + string.whitespace))
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm(prompt, temperature=temperature)
                topic_name_info_text = topic_name_info_raw['choices'][0]['text']
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
        
        def tokenize(self, text):
            return self.llm.tokenize(text.encode('utf-8'))
        
        def detokenize(self, tokens):
            return self.llm.detokenize(tokens)
        
        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return "\nThe short distinguising topic name is:\n"
            elif kind == "intermediate_layer":
                return "\nThe short topic name that encompasses the subtopics is:\n"
            elif kind == "remedy":
                return "\nA better and more specific name that still captures the topic of these article titles is:\n"
            else:
                raise ValueError(f"Invalid llm_imnstruction kind; should be one of \'base_layer\', \'intermediate_layer\', or \'remedy\' not \'{kind}\'")
            
        def n_ctx(self):
            return self.llm.n_ctx()

except ImportError:
     pass

try:
     
    import cohere
    import json

    class CohereWrapper:
        def __init__(self, API_KEY, local_tokenizer=None):
            self.llm = cohere.Client(API_KEY)
            if local_tokenizer is not None:
                self.tokenizer = local_tokenizer
            else:
                self.tokenizer = self.llm
        
        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(message=prompt, temperature=temperature).text
                topic_name_info = json.loads(topic_name_info_raw)
                topic_name = topic_name_info["topic_name"]
            except:
                topic_name = ""
            return topic_name

        def generate_topic_cluster_names(self, prompt, old_names, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.chat(
                    message=prompt, 
                    temperature=temperature
                )
                topic_name_info_text = topic_name_info_raw.text
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
                    
        def tokenize(self, text):
            return self.tokenizer.tokenize(text)
        
        def detokenize(self, tokens):
            return self.tokenizer.detokenize(tokens)
        
        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return """
You are to give a brief (five to ten word) name describing this group and distinguishing it from other nearby groups.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "intermediate_layer":
                return """
You are to give a brief (three to five word) name describing this group of papers and distinguishing it from other nearby groups.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be in JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "remedy":
                return """
You are to give a brief (five to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be in JSON formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""
            else:
                raise ValueError(f"Invalid llm_imnstruction kind; should be one of \'base_layer\', \'intermediate_layer\', or \'remedy\' not \'{kind}\'")
            
        def n_ctx(self):
            return 128_000
except:
    pass


try:
     
    import anthropic
    import json

    class AnthropicWrapper:
        def __init__(self, API_KEY, model="claude-3-haiku-20240307", local_tokenizer=None):
            self.llm = anthropic.Anthropic(api_key=API_KEY)
            self.model = model
            if local_tokenizer is not None:
                self.tokenizer = local_tokenizer
            else:
                raise ValueError("Anthropic does not provide a tokenizer")
        
        def generate_topic_name(self, prompt, temperature=0.5):
            try:
                topic_name_info_raw = self.llm.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}], 
                    temperature=temperature
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
                    temperature=temperature
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
        
        def tokenize(self, text):
            return self.tokenizer.tokenize(text)
        
        def detokenize(self, tokens):
            return self.tokenizer.detokenize(tokens)
        
        def llm_instruction(self, kind="base_layer"):
            if kind == "base_layer":
                return """
You are to give a brief (five to ten word) name describing this group and distinguishing it from other nearby groups.
The topic name should be as specific as you can reasonably make it, while still describing the all example texts.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "intermediate_layer":
                return """
You are to give a brief (three to five word) name describing this group of papers and distinguishing it from other nearby groups.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
                """
            elif kind == "remedy":
                return """
You are to give a brief (five to ten word) name describing this group of papers that better captures the specific details of this group.
The topic should be the most specific topic that encompasses the full breadth of sub-topics.
The response should be only JSON with no preamble formatted as {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} where SCORE is a value in the range 0 to 1.
"""
            else:
                raise ValueError(f"Invalid llm_imnstruction kind; should be one of \'base_layer\', \'intermediate_layer\', or \'remedy\' not \'{kind}\'")
            
        def n_ctx(self):
            return 128_000
except:
    pass