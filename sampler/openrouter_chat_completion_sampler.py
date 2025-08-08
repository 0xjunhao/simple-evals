import os
import requests
import time
from typing import Any


from ..custom_types import MessageList, SamplerBase, SamplerResponse

OPENROUTER_SYSTEM_MESSAGE_API = "You are a helpful assistant."


class OpenRouterChatCompletionSampler(SamplerBase):
    """
    Sample from OpenRouter's chat completion API
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        provider: str | None = None,
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise Exception(
                "OPENROUTER_API_KEY environment variable is not set.")
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        self.provider = provider
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                request = {
                    'model': self.model,
                    'messages': message_list,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                }
                if self.provider:
                    request['provider'] = {
                        'order': [
                            self.provider
                        ],
                        'allow_fallbacks': False
                    }
                response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                                         headers=self.headers, json=request)
                print(response.json())
                content = response.json().get('choices', [{}])[
                    0].get('message', {}).get('content')
                if content is None:
                    raise Exception(
                        "OpenRouter API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.json().get('usage')},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
