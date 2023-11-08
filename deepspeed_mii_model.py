from typing import Any

import mii
from llama_index.llms import CustomLLM
from llama_index.llms.base import llm_completion_callback, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.bridge.pydantic import Field


class DeepSpeedMiiLLM(CustomLLM):
    model_name: str = Field()
    max_tokens: int = Field(default=4096)
    client: Any = Field(default=None, exclude=True)

    def __init__(self, model_name: str, max_tokens: int = 4096, with_server: bool = False, **kwargs: Any):
        super().__init__(model_name=model_name, max_tokens=max_tokens, **kwargs)

        if with_server:
            mii.serve("lmsys/vicuna-13b-v1.5-16k", deployment_name=model_name)

        self.client = mii.client(model_name)

    @classmethod
    def class_name(cls) -> str:
        return "deepseed_mii"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=16384,  # Total number of tokens the model can be input and output for one response.
            num_output=self.max_tokens or -1,  # Number of tokens the model can output when generating a response.
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.client.generate(prompt, max_new_tokens=self.max_tokens)
        assert response is not None

        return CompletionResponse(text=response.response[0])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def gen():
            response = self.client.generate(prompt, max_new_tokens=self.max_tokens)
            assert response is not None

            # DeepSpeed Mii does not support streaming.
            yield CompletionResponse(text=response.response[0])

        return gen()

    def terminate_server(self) -> None:
        self.client.terminate_server()
