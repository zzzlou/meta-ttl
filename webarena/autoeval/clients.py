import os
import base64
import openai
import numpy as np
from PIL import Image
from typing import Union, Optional
from openai import OpenAI, ChatCompletion

# Configure API key
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")

def _is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI model (should use OpenAI API directly)"""
    return model_name.startswith(("gpt-", "o1-", "text-", "davinci", "curie", "babbage", "ada"))

def _get_client(model_name: str) -> OpenAI:
    """Get the appropriate OpenAI client based on model name"""
    if _is_openai_model(model_name):
        # Use OpenAI API for GPT models
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
        return OpenAI(api_key=api_key)
    else:
        # Use OpenRouter for other models (Claude, Gemini, etc.)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for non-OpenAI models")
        openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(
            api_key=api_key,
            base_url=openrouter_base_url
        )

# Global client for backward compatibility (defaults to OpenAI)
client = OpenAI()


class LM_Client:
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name
        # Get the appropriate client based on model name
        self.client = _get_client(model_name)

    def chat(self, messages, json_mode: bool = False) -> tuple[str, ChatCompletion]:
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0,
        )
        response = chat_completion.choices[0].message.content
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False
    ) -> tuple[str, ChatCompletion]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


class GPT4V_Client:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens
        # Get the appropriate client based on model name
        self.client = _get_client(model_name)

    def encode_image(self, path: str):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def one_step_chat(
        self, text, image: Union[Image.Image, np.ndarray],
        system_msg: Optional[str] = None,
    ) -> tuple[str, ChatCompletion]:
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content, response


CLIENT_DICT = {
    "gpt-3.5-turbo": LM_Client,
    "gpt-4": LM_Client,
    # "gpt-4o": GPT4V_Client,
    "gpt-4o": LM_Client,
}