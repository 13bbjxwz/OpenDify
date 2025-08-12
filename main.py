# -*- coding: utf-8 -*-
import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import List, Dict, Any, Optional, Generator, Tuple

import httpx
from dotenv import load_dotenv
from flask import Flask, Request, Response, jsonify, stream_with_context, request

# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================

class Config:
    """
    Manages application configuration by loading settings from environment variables.
    """
    def __init__(self):
        load_dotenv()
        self.DIFY_API_BASE: str = self._get_env("DIFY_API_BASE", required=True)
        self.DIFY_API_KEYS: List[str] = self._get_env_list("DIFY_API_KEYS", required=True)
        self.VALID_API_KEYS: List[str] = self._get_env_list("VALID_API_KEYS", required=True)
        self.SERVER_HOST: str = self._get_env("SERVER_HOST", "127.0.0.1")
        self.SERVER_PORT: int = int(self._get_env("SERVER_PORT", "5000"))
        self.CONVERSATION_MEMORY_MODE: int = self._get_conversation_memory_mode()
        self.LOG_LEVEL: str = self._get_env("LOG_LEVEL", "INFO").upper()

    def _get_env(self, key: str, default: str = "", required: bool = False) -> str:
        value = os.getenv(key, default)
        if required and not value:
            raise ValueError(f"FATAL: Required environment variable '{key}' is not set.")
        return value

    def _get_env_list(self, key: str, required: bool = False) -> List[str]:
        value_str = self._get_env(key, required=required)
        if not value_str:
            return []
        return [item.strip() for item in value_str.split(',') if item.strip()]

    def _get_conversation_memory_mode(self) -> int:
        try:
            return int(self._get_env('CONVERSATION_MEMORY_MODE', '1'))
        except (ValueError, TypeError):
            log.warning("Invalid CONVERSATION_MEMORY_MODE, defaulting to 1.")
            return 1

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize configuration
try:
    config = Config()
    log.setLevel(config.LOG_LEVEL)
except ValueError as e:
    log.fatal(e)
    exit(1)


# ======================================================================================
# 2. UTILITY FUNCTIONS
# ======================================================================================

def remove_think_tags(text: str) -> str:
    """Removes <think>...</think> blocks from text."""
    return re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL)


class ConversationIdManager:
    """
    Encodes and decodes conversation IDs using zero-width characters
    to embed them invisibly in assistant messages. This is a workaround
    to maintain state with certain clients.

    The encoding maps base64 characters to pairs of zero-width chars.
    """
    _BASE64_TO_ZERO_WIDTH_MAP = {
        '0': '\u200b', '1': '\u200c', '2': '\u200d', '3': '\ufeff',
        '4': '\u2060', '5': '\u180e', '6': '\u2061', '7': '\u2062',
    }
    _ZERO_WIDTH_TO_BASE64_MAP = {v: k for k, v in _BASE64_TO_ZERO_WIDTH_MAP.items()}

    @classmethod
    def encode(cls, conversation_id: str) -> str:
        """Encodes a conversation ID into a zero-width character string."""
        if not conversation_id:
            return ""
        try:
            b64_encoded = base64.b64encode(conversation_id.encode('utf-8')).decode('utf-8')
            zero_width_chars = []
            for char in b64_encoded.rstrip('='):
                if 'a' <= char <= 'z':
                    val = ord(char) - ord('a')
                elif 'A' <= char <= 'Z':
                    val = ord(char) - ord('A') + 26
                elif '0' <= char <= '9':
                    val = ord(char) - ord('0') + 52
                elif char == '+':
                    val = 62
                elif char == '/':
                    val = 63
                else:
                    continue  # Should not happen for base64

                high_bits = (val >> 3) & 7
                low_bits = val & 7
                zero_width_chars.append(cls._BASE64_TO_ZERO_WIDTH_MAP[str(high_bits)])
                zero_width_chars.append(cls._BASE64_TO_ZERO_WIDTH_MAP[str(low_bits)])

            return "".join(zero_width_chars)
        except Exception as e:
            log.error(f"Failed to encode conversation ID '{conversation_id}': {e}")
            return ""

    @classmethod
    def decode(cls, text: str) -> Optional[str]:
        """Decodes a conversation ID from zero-width characters in a string."""
        try:
            # Extract potential zero-width characters from the end of the string
            zero_width_sequence = []
            for char in reversed(text):
                if char in cls._ZERO_WIDTH_TO_BASE64_MAP:
                    zero_width_sequence.append(char)
                else:
                    break

            if not zero_width_sequence:
                return None

            zero_width_sequence.reverse()

            # Decode the sequence
            b64_chars = []
            for i in range(0, len(zero_width_sequence), 2):
                if i + 1 >= len(zero_width_sequence):
                    continue # Incomplete pair

                high_char = zero_width_sequence[i]
                low_char = zero_width_sequence[i+1]

                high_bits = int(cls._ZERO_WIDTH_TO_BASE64_MAP[high_char])
                low_bits = int(cls._ZERO_WIDTH_TO_BASE64_MAP[low_char])

                val = (high_bits << 3) | low_bits

                if 0 <= val < 26:
                    b64_chars.append(chr(ord('a') + val))
                elif 26 <= val < 52:
                    b64_chars.append(chr(ord('A') + val - 26))
                elif 52 <= val < 62:
                    b64_chars.append(chr(ord('0') + val - 52))
                elif val == 62:
                    b64_chars.append('+')
                elif val == 63:
                    b64_chars.append('/')

            b64_string = "".join(b64_chars)
            # Add padding
            padding = -len(b64_string) % 4
            if padding:
                b64_string += '=' * padding

            return base64.b64decode(b64_string).decode('utf-8')
        except Exception as e:
            log.debug(f"Failed to decode conversation ID from text: {e}")
            return None


# ======================================================================================
# 3. DIFY API INTEGRATION
# ======================================================================================

class DifyModelManager:
    """
    Manages Dify "models" (which are Dify apps) by fetching their
    information and mapping app names to their respective API keys.
    """
    def __init__(self, app_config: Config):
        self.config = app_config
        self.name_to_api_key: Dict[str, str] = {}
        self.api_key_to_name: Dict[str, str] = {}
        log.info(f"Loaded {len(self.config.DIFY_API_KEYS)} Dify API keys.")

    async def _fetch_app_name(self, client: httpx.AsyncClient, api_key: str) -> Optional[str]:
        """Fetches the application name for a given Dify API key."""
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{self.config.DIFY_API_BASE}/info"
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            app_info = response.json()
            return app_info.get("app", {}).get("name")
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error fetching app info for key ...{api_key[-4:]}: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            log.error(f"Error fetching app info for key ...{api_key[-4:]}: {e}")
        return None

    async def refresh_model_list(self):
        """
        Refreshes the mapping of Dify app names to API keys.
        This should be called periodically or on startup.
        """
        log.info("Refreshing Dify model list...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [self._fetch_app_name(client, key) for key in self.config.DIFY_API_KEYS]
            results = await asyncio.gather(*tasks)

        new_name_to_api_key = {}
        new_api_key_to_name = {}
        for api_key, app_name in zip(self.config.DIFY_API_KEYS, results):
            if app_name:
                new_name_to_api_key[app_name] = api_key
                new_api_key_to_name[api_key] = app_name
                log.info(f"Mapped app '{app_name}' -> ...{api_key[-4:]}")
            else:
                log.warning(f"Could not retrieve app name for key ...{api_key[-4:]}. It will be unavailable.")

        self.name_to_api_key = new_name_to_api_key
        self.api_key_to_name = new_api_key_to_name
        log.info(f"Model list refresh complete. {len(self.name_to_api_key)} models available.")

    def get_api_key(self, model_name: str) -> Optional[str]:
        """Retrieves the Dify API key for a given model (app) name."""
        return self.name_to_api_key.get(model_name)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns a list of available models in OpenAI format."""
        return [
            {"id": name, "object": "model", "created": int(time.time()), "owned_by": "dify"}
            for name in self.name_to_api_key
        ]

# Initialize the model manager
model_manager = DifyModelManager(config)


# ======================================================================================
# 4. OPENAI TO DIFY PAYLOAD TRANSFORMATION
# ======================================================================================

def transform_openai_to_dify_payload(openai_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms an OpenAI-compatible chat completion request to a Dify chat request payload.
    """
    messages = openai_request.get("messages", [])
    user_query = ""
    system_prompt = ""
    conversation_id = None

    # Extract system prompt
    for message in messages:
        if message.get("role") == "system":
            system_prompt = message.get("content", "")
            if system_prompt:
                log.info(f"Found system prompt: '{system_prompt[:100]}...'")
            break # Use the first system message

    # Extract user query and conversation history
    if config.CONVERSATION_MEMORY_MODE == 2:
        # Mode 2: Use zero-width char encoding for conversation_id
        if messages and messages[-1].get("role") == "user":
            user_query = messages[-1].get("content", "")

        # Find the last assistant message to decode the conversation_id
        for message in reversed(messages[:-1]):
            if message.get("role") == "assistant":
                cid = ConversationIdManager.decode(message.get("content", ""))
                if cid:
                    conversation_id = cid
                    log.info(f"Resuming conversation with ID: {conversation_id}")
                    break

        if system_prompt and not conversation_id:
            user_query = f"System Prompt: {system_prompt}\n\nUser Query: {user_query}"
            log.info("Applying system prompt to the first message of the conversation.")

    else: # Default to Mode 1: History injection
        if messages and messages[-1].get("role") == "user":
            user_query = messages[-1].get("content", "")

        history = []
        has_system_in_history = False
        for msg in messages[:-1]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history.append(f"{role}: {content}")
                if role == "system":
                    has_system_in_history = True

        if system_prompt and not has_system_in_history:
            history.insert(0, f"system: {system_prompt}")

        if history:
            history_str = "\n\n".join(history)
            user_query = f"<history>\n{history_str}\n</history>\n\nUser's current query: {user_query}"

    return {
        "inputs": {},
        "query": user_query,
        "response_mode": "streaming", # We always use streaming from Dify
        "conversation_id": conversation_id if conversation_id else "",
        "user": openai_request.get("user", "default_user"),
    }


# ======================================================================================
# 5. FLASK APPLICATION
# ======================================================================================

app = Flask(__name__)

def _get_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """Extracts the bearer token from the Authorization header."""
    if not auth_header:
        log.warning("Authorization header is missing.")
        return None
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        log.warning(f"Invalid Authorization header format.")
        return None
    return parts[1]

def _stream_dify_chat_completion(dify_api_key: str, payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Streams from Dify's chat-messages endpoint and yields parsed JSON events.
    """
    headers = {
        "Authorization": f"Bearer {dify_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    url = f"{config.DIFY_API_BASE}/chat-messages"

    try:
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=600) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.startswith("data:"):
                        try:
                            data = line[len("data:"):].strip()
                            if data:
                                yield json.loads(data)
                        except json.JSONDecodeError:
                            log.warning(f"Failed to decode JSON from stream line: {line}")
    except httpx.HTTPStatusError as e:
        log.error(f"Dify API request failed: {e.response.status_code} - {e.response.text}")
        yield {"error": "Dify API error", "details": e.response.text}
    except Exception as e:
        log.error(f"An unexpected error occurred during streaming: {e}")
        yield {"error": "Streaming failed", "details": str(e)}


@app.route('/v1/models', methods=['GET'])
def list_models():
    """Handles OpenAI-compatible '/v1/models' endpoint."""
    log.info("Request received for /v1/models")
    # Refresh is expensive, consider doing it in a background thread periodically
    # For now, we do it on-demand but this can be slow.
    asyncio.run(model_manager.refresh_model_list())
    available_models = model_manager.get_available_models()
    log.info(f"Returning {len(available_models)} available models.")
    return jsonify({"object": "list", "data": available_models})


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handles OpenAI-compatible '/v1/chat/completions' endpoint."""
    # 1. Authentication
    auth_token = _get_bearer_token(request.headers.get('Authorization'))
    if not auth_token or auth_token not in config.VALID_API_KEYS:
        return jsonify({"error": {"message": "Invalid API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}), 401

    # 2. Request Parsing
    try:
        openai_request = request.get_json()
        model_name = openai_request.get("model")
        is_stream = openai_request.get("stream", False)
        if not model_name:
            return jsonify({"error": {"message": "Request must include a 'model' parameter.", "type": "invalid_request_error"}}), 400
    except Exception as e:
        log.error(f"Failed to parse request JSON: {e}")
        return jsonify({"error": {"message": f"Invalid request body: {e}", "type": "invalid_request_error"}}), 400

    log.info(f"Request for model '{model_name}', stream={is_stream}")

    # 3. Get Dify API Key for the requested model
    dify_api_key = model_manager.get_api_key(model_name)
    if not dify_api_key:
        log.error(f"Model '{model_name}' not found. Available models: {list(model_manager.name_to_api_key.keys())}")
        return jsonify({"error": {"message": f"Model '{model_name}' not found.", "type": "invalid_request_error", "code": "model_not_found"}}), 404

    # 4. Transform Payload and Call Dify API
    dify_payload = transform_openai_to_dify_payload(openai_request)

    # 5. Handle Streaming vs. Non-Streaming Response
    if is_stream:
        # Streaming Response
        def stream_generator():
            message_id = ""
            for event in _stream_dify_chat_completion(dify_api_key, dify_payload):
                event_type = event.get("event")
                if event_type == "message" or event_type == "agent_message":
                    message_id = event.get("message_id", message_id)
                    answer_chunk = remove_think_tags(event.get("answer", ""))
                    if answer_chunk:
                        chunk = {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": answer_chunk}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                elif event_type == "message_end":
                    log.info(f"Stream finished for message ID: {message_id}")
                    final_chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    break # Stop after message_end

            yield "data: [DONE]\n\n"

        return Response(stream_with_context(stream_generator()), content_type='text/event-stream')

    else:
        # Non-Streaming Response
        full_answer = ""
        message_id = ""
        conversation_id = ""

        for event in _stream_dify_chat_completion(dify_api_key, dify_payload):
            event_type = event.get("event")
            if event_type == "message" or event_type == "agent_message":
                message_id = event.get("message_id", message_id)
                full_answer += remove_think_tags(event.get("answer", ""))

            elif event_type == "message_end":
                conversation_id = event.get("conversation_id", "")
                break

        # Embed conversation ID if needed
        if config.CONVERSATION_MEMORY_MODE == 2 and conversation_id:
            encoded_cid = ConversationIdManager.encode(conversation_id)
            full_answer += encoded_cid
            log.info(f"Embedded conversation ID {conversation_id} into response.")

        response_payload = {
            "id": message_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop"
            }],
            "usage": { # Note: Dify doesn't provide token usage, so this is a placeholder
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        return jsonify(response_payload)


# ======================================================================================
# 6. APPLICATION STARTUP
# ======================================================================================

if __name__ == '__main__':
    log.info("Starting Dify to OpenAI Adapter...")

    # Perform initial model list refresh on startup
    try:
        log.info("Performing initial model list refresh...")
        asyncio.run(model_manager.refresh_model_list())
    except Exception as e:
        log.error(f"Initial model list refresh failed: {e}. The service might not work correctly.")

    log.info(f"Server starting on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    log.info(f"Log level set to: {config.LOG_LEVEL}")
    log.info(f"Conversation memory mode: {config.CONVERSATION_MEMORY_MODE}")

    # Use a production-ready WSGI server like Gunicorn or Waitress instead of app.run in production
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)
