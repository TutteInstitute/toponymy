from toponymy.llm_wrappers import AsyncLLMWrapper, LLMWrapper


class DummySingleWrapper(LLMWrapper):
    model = "dummy-model"
    _supports_debug_callback = True

    def _call_llm(self, prompt, temperature, max_tokens):
        return '{"topic_name": "single-ok", "topic_specificity": 0.9}'

    def _call_llm_with_system_prompt(self, prompt, temperature, max_tokens):
        return '{"topic_name": "single-system-ok", "topic_specificity": 0.9}'


class DummyAsyncSingleWrapper(AsyncLLMWrapper):
    _supports_debug_callback = True
    model = "dummy-model"

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return '{"topic_name": "single-ok", "topic_specificity": 0.9}'

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens):
        return '{"topic_name": "single-system-ok", "topic_specificity": 0.9}'
