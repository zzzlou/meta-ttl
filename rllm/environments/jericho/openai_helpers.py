import time
from typing import Mapping
import openai
import json
from openai import OpenAI
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()
encoding = tiktoken.get_encoding("cl100k_base")
CLIENT = None

def init_global_client(base_url: str, api_key: str = "EMPTY"):
    """
    Call this once at program startup to initialize the global singleton `CLIENT`.
    """
    global CLIENT
    

    if CLIENT is None:
        print(f"Initializing Global OpenAI Client with Base URL: {base_url}")
        CLIENT = OpenAI(
            api_key = api_key,
            base_url = base_url,
            timeout=120.0,
        )
    else:
        pass
        # print("⚠️ Global Client already initialized, skipping.")

#http://localhost:30004/v1"
#https://openrouter.ai/api/v1

import time
import openai
from colorama import Fore, Style
from typing import Mapping, List, Dict, Optional
import re
import json

class TokenLimitExceededError(Exception):
    """Raised when the context length exceeds the model's token limit"""
    pass

def _fix_json_control_characters(json_text: str) -> str:
    """Fix unescaped control characters in JSON string values."""
    result = []
    i = 0
    in_string = False
    string_char = None
    escaped = False

    while i < len(json_text):
        char = json_text[i]
        if escaped:
            result.append(char)
            escaped = False
            i += 1
            continue
        if char == '\\':
            result.append(char)
            escaped = True
            i += 1
            continue
        if not in_string:
            if char in ('"', "'"):
                in_string = True
                string_char = char
            result.append(char)
            i += 1
            continue
        if char == string_char:
            in_string = False
            string_char = None
            result.append(char)
            i += 1
            continue
        if char == '\n':
            result.append('\\n')
        elif char == '\r':
            result.append('\\r')
        elif char == '\t':
            result.append('\\t')
        elif char == '\b':
            result.append('\\b')
        elif char == '\f':
            result.append('\\f')
        else:
            result.append(char)
        i += 1
    return ''.join(result)

def extract_json_from_response(response_text: str) -> dict:
    if not response_text:
        return {}

    def try_parse(text: str, fix_control_chars: bool = False) -> dict:
        if fix_control_chars:
            text = _fix_json_control_characters(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if not fix_control_chars and 'control character' in str(e).lower():
                return None
            return None

    result = try_parse(response_text, fix_control_chars=False)
    if result is not None and isinstance(result, dict): return result
    result = try_parse(response_text, fix_control_chars=True)
    if result is not None and isinstance(result, dict): return result

    json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        extracted = json_match.group(1)
        result = try_parse(extracted, fix_control_chars=False) or try_parse(extracted, fix_control_chars=True)
        if result is not None and isinstance(result, dict): return result

    json_match = re.search(r'```\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        extracted = json_match.group(1)
        result = try_parse(extracted, fix_control_chars=False) or try_parse(extracted, fix_control_chars=True)
        if result is not None and isinstance(result, dict): return result

    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        extracted = json_match.group(0)
        result = try_parse(extracted, fix_control_chars=False) or try_parse(extracted, fix_control_chars=True)
        if result is not None and isinstance(result, dict): return result

    print(f"Warning: Could not extract JSON from response: {response_text[:200]}...")
    return {}

def chat_completion_with_retries(model: str, messages: Optional[List[Dict[str, str]]] = None, sys_prompt: Optional[str] = None, prompt: Optional[str] = None, image_content=None, max_retries: int = 5, retry_interval_sec: int = 5, top_logprobs: int = 0, response_format=None, **kwargs) -> Mapping:
    global CLIENT
    if CLIENT is None:
        raise RuntimeError("OpenAI CLIENT is not initialized! Please call 'init_global_client' in your main script first.")
        
    if messages is None:
        if image_content:
            if isinstance(image_content, list):
                user_content = [{"type": "text", "text": prompt}] + image_content
            else:
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_content}", "detail": "high"}}
                ]
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        
    # Inject logprobs and response_format into kwargs
    if top_logprobs > 0:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs
    if response_format:
        kwargs["response_format"] = response_format

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            res = CLIENT.chat.completions.create(model=model, messages=messages, **kwargs)
            
            has_choices = res and getattr(res, 'choices', None) and len(res.choices) > 0
            has_message = has_choices and res.choices[0].message
            if not has_message:
                raise openai.APIError("Invalid or empty response received (empty choices or no message)", request=None, body=None)
            
            content = res.choices[0].message.content
            reasoning = getattr(res.choices[0].message, 'reasoning', None)
            
            if not content and reasoning:
                raise openai.APIError(f"TRUNCATED_REASONING: Model returned reasoning but empty content. Finish reason: {res.choices[0].finish_reason}", request=None, body=None)
            elif not content:
                raise openai.APIError("Invalid or empty response received (empty content without reasoning)", request=None, body=None)

            return res

        except openai.BadRequestError as e:
            error_message = str(e)
            if "maximum context length" in error_message.lower() or "reduce the length" in error_message.lower():
                print(f"\n{'='*80}\nERROR: Context length exceeded!\nError: {error_message}\n{'='*80}\n")
                raise TokenLimitExceededError(f"Context length exceeded: {error_message}") from e
            else:
                print(e)
                print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
                time.sleep(retry_interval_sec)
        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
            json.JSONDecodeError,
            ConnectionError,
        ) as e:
            print(e)
            print(f"Hit exception when calling model {model}. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    return {}

def truncate_text(text, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"WARNING: Maximum token length exceeded ({len(tokens)} > {max_tokens})")
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text
