import itertools
import json
import re
import zlib

import numpy as np

from toponymy.llm_wrappers import AsyncLLMWrapper, LLMWrapper

SINGLE_RESPONSE = '{"topic_name": "single-ok", "topic_specificity": 0.9}'
SYSTEM_RESPONSE = '{"topic_name": "single-system-ok", "topic_specificity": 0.9}'


class DummySingleWrapper(LLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    def _call_llm(self, prompt, temperature, max_tokens):
        return SINGLE_RESPONSE

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        return SYSTEM_RESPONSE


class DummyAsyncSingleWrapper(AsyncLLMWrapper):
    _supports_debug_callback = True
    model = "dummy-model"

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return SINGLE_RESPONSE

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens):
        return SYSTEM_RESPONSE


class CountingNamer(DummySingleWrapper):
    """Names each topic "Topic <n>" in call order, or ``name`` if given.

    A disambiguation prompt gets "Topic <n>.<i>" for its i-th topic, or ``name``
    for every topic unless ``rename``. The calls in ``fail_calls`` get empty names.
    """

    def __init__(self, fail_calls=(), name=None, rename=True):
        self.calls = itertools.count()
        self.fail_calls = set(fail_calls)
        self.name = name
        self.rename = rename
        self.prompts = []
        self.disambiguation_prompts = []

    def _respond(self, prompt):
        self.prompts.append(prompt)
        call = next(self.calls)
        failed = call in self.fail_calls
        text = "\n".join(prompt.values())
        if "new_topic_name_mapping" in text:
            self.disambiguation_prompts.append(prompt)
            n_topics = max(int(i) for i in re.findall(r'^"(\d+)\. ', text, re.M))
            if failed:
                new_names = [""] * n_topics
            elif self.rename:
                new_names = [f"Topic {call}.{i}" for i in range(1, n_topics + 1)]
            else:
                new_names = [self.name] * n_topics
            mapping = {f"{i}.": name for i, name in enumerate(new_names, start=1)}
            return json.dumps(
                {
                    "new_topic_name_mapping": mapping,
                    "topic_specificities": [0.9] * n_topics,
                }
            )
        name = "" if failed else self.name or f"Topic {call}"
        return json.dumps({"topic_name": name, "topic_specificity": 0.9})

    def _call_llm(self, prompt, temperature, max_tokens):
        return self._respond(prompt)

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        return self._respond(prompt)


class DummyAsyncProviderError(Exception):
    pass


class NameEmbedder:
    """Embeds each text as a random vector seeded by the text, so equal texts embed equally."""

    def __init__(self, dimension=8):
        self.dimension = dimension
        self.calls = 0

    def encode(self, texts, show_progress_bar=None):
        self.calls += 1
        # crc32 is a fast hash that, unlike hash(), is stable across runs
        return np.array(
            [
                np.random.default_rng(zlib.crc32(text.encode())).normal(
                    size=self.dimension
                )
                for text in texts
            ]
        )


class DummyAsyncFailFastWrapper(AsyncLLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    FAIL_FAST_EXCEPTIONS = (DummyAsyncProviderError,)

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        raise DummyAsyncProviderError("bad config")

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens):
        raise DummyAsyncProviderError("bad config")
