import os
import httpx
import json
import asyncio
import logging
import base64
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

# --- Configuration ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
DIFY_API_KEYS = os.getenv("DIFY_API_KEYS", "").split(',')
DIFY_API_BASE = os.getenv("DIFY_API_BASE")
VALID_API_KEYS = os.getenv("VALID_API_KEYS", "").split(',')
CONVERSATION_MEMORY_MODE = int(os.getenv("CONVERSATION_MEMORY_MODE", 1))
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))

# Zero-width characters for encoding
ZERO_WIDTH_CHARS = [
    '\u200b', '\u200c', '\u200d', '\u2060', '\u2061', '\u2062', '\u2063', '\u2064'
]

# --- Dify Model Manager ---
class DifyModelManager:
    """Manages Dify API keys and their corresponding model names."""
    def __init__(self):
        self.api_keys: List[str] = []
        self.model_info: Dict[str, str] = {}  # {model_name: api_key}
        self.load_api_keys()

    def load_api_keys(self):
        """Loads Dify API keys from environment variables."""
        self.api_keys = [key.strip() for key in DIFY_API_KEYS if key.strip()]
        if not self.api_keys:
            logging.warning("DIFY_API_KEYS environment variable is not set or empty.")

    async def fetch_app_info(self, api_key: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """Fetches application information using a Dify API key."""
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = await client.get(f"{DIFY_API_BASE}/meta", headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logging.error(f"Error fetching app info for key ending in ...{api_key[-4:]}: {e}")
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error fetching app info for key ...{api_key[-4:]}: {e.response.status_code}")
        return None

    async def refresh_model_info(self):
        """Refreshes the mapping of model names to API keys."""
        logging.info("Refreshing Dify model info...")
        async with httpx.AsyncClient() as client:
            tasks = [self.fetch_app_info(key, client) for key in self.api_keys]
            results = await asyncio.gather(*tasks)

            new_model_info = {}
            for i, data in enumerate(results):
                if data and data.get("app"):
                    model_name = data["app"]["title"]
                    new_model_info[model_name] = self.api_keys[i]
                    logging.info(f"Loaded model '{model_name}'")
            self.model_info = new_model_info
        logging.info("Dify model info refreshed.")

    def get_api_key(self, model_name: str) -> Optional[str]:
        """Gets the API key for a given model name."""
        return self.model_info.get(model_name)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Gets a list of available models in OpenAI format."""
        return [
            {"id": name, "object": "model", "created": 1677610602, "owned_by": "dify"}
            for name in self.model_info.keys()
        ]

# --- Session Memory Encoding/Decoding ---
def encode_conversation_id(conversation_id: str) -> str:
    """Encodes a conversation ID into zero-width characters."""
    if not conversation_id:
        return ""
    b64_bytes = base64.b64encode(conversation_id.encode('utf-8'))
    b64_string = b64_bytes.decode('utf-8')

    encoded_chars = []
    for char in b64_string:
        binary_representation = format(ord(char), '08b')
        for i in range(0, 8, 3):
            chunk = binary_representation[i:i+3]
            if len(chunk) < 3:
                chunk = chunk.ljust(3, '0')
            decimal_value = int(chunk, 2)
            encoded_chars.append(ZERO_WIDTH_CHARS[decimal_value])

    return "".join(encoded_chars)

def decode_conversation_id(content: str) -> (Optional[str], str):
    """Decodes a conversation ID from a message content string."""
    binary_string = ""
    decoded_chars = []

    for char in content:
        if char in ZERO_WIDTH_CHARS:
            index = ZERO_WIDTH_CHARS.index(char)
            binary_string += format(index, '03b')

    if not binary_string:
        return None, content

    original_content = content
    for char in ZERO_WIDTH_CHARS:
        original_content = original_content.replace(char, '')

    try:
        byte_array = bytearray()
        for i in range(0, len(binary_string), 8):
            byte_chunk = binary_string[i:i+8]
            if len(byte_chunk) == 8:
                byte_array.append(int(byte_chunk, 2))

        b64_decoded_bytes = base64.b64decode(byte_array)
        conversation_id = b64_decoded_bytes.decode('utf-8')
        return conversation_id, original_content
    except Exception as e:
        logging.warning(f"Failed to decode conversation ID: {e}")
        return None, original_content

# --- Request/Response Transformers ---
def transform_openai_to_dify(openai_request: Dict[str, Any], conversation_id: Optional[str]) -> Dict[str, Any]:
    """Transforms an OpenAI-formatted request to a Dify-formatted request."""
    dify_request = {
        "inputs": {},
        "query": "",
        "user": "openai-proxy-user",
        "response_mode": "streaming" if openai_request.get("stream") else "blocking",
    }

    if conversation_id:
        dify_request["conversation_id"] = conversation_id

    user_query = ""
    system_message = ""
    history = []

    for message in openai_request["messages"]:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            user_query = content
        elif role == "system":
            system_message = content
        elif role == "assistant":
            history.append({"role": "assistant", "content": content})
        elif role == "tool": # Dify might use this for tool outputs
            history.append({"role": "tool", "content": content})

    if CONVERSATION_MEMORY_MODE == 1 and history:
        history_str = "\n".join([f"<history>{msg['content']}</history>" for msg in history])
        user_query = f"{history_str}\n{user_query}"

    if system_message:
        dify_request["inputs"]["system_prompt"] = system_message

    dify_request["query"] = user_query

    return dify_request

def transform_dify_to_openai(dify_response: Dict[str, Any], model: str, stream: bool) -> str:
    """Transforms a Dify response to an OpenAI-formatted response string."""
    if stream:
        # For streaming, we assume the raw SSE data is passed
        return dify_response
    else:
        openai_response = {
            "id": f"chatcmpl-{dify_response.get('metadata', {}).get('usage', {}).get('completion_tokens', 0)}",
            "object": "chat.completion",
            "created": int(dify_response.get('created_at', 0)),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": dify_response.get("answer", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": dify_response.get('metadata', {}).get('usage', {}).get('prompt_tokens', 0),
                "completion_tokens": dify_response.get('metadata', {}).get('usage', {}).get('completion_tokens', 0),
                "total_tokens": dify_response.get('metadata', {}).get('usage', {}).get('total_tokens', 0)
            }
        }
        return json.dumps(openai_response)


# --- Flask App ---
app = Flask(__name__)
model_manager = DifyModelManager()

@app.before_request
def check_api_key():
    """Validates the API key before processing any request."""
    if request.path in ["/v1/models"]:
        return

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": {"message": "Authorization header is missing or invalid.", "type": "invalid_api_key", "code": "invalid_api_key"}}), 401

    client_api_key = auth_header.split(" ")[1]
    if client_api_key not in VALID_API_KEYS:
        return jsonify({"error": {"message": "Invalid API Key.", "type": "invalid_api_key", "code": "invalid_api_key"}}), 401

@app.route("/v1/models", methods=["GET"])
def get_models():
    """Returns a list of available models."""
    return jsonify({"object": "list", "data": model_manager.get_available_models()})

@app.route("/v1/chat/completions", methods=["POST"])
async def chat_completions():
    """Handles chat completion requests."""
    openai_request = request.json
    model_name = openai_request.get("model")
    stream = openai_request.get("stream", False)

    if not model_name:
        return jsonify({"error": {"message": "Model name is required.", "type": "invalid_request_error", "code": "model_not_found"}}), 400

    dify_api_key = model_manager.get_api_key(model_name)
    if not dify_api_key:
        return jsonify({"error": {"message": f"Model '{model_name}' not found.", "type": "invalid_request_error", "code": "model_not_found"}}), 404

    # --- Conversation ID Handling ---
    conversation_id = None
    if CONVERSATION_MEMORY_MODE == 2:
        # Look for conversation ID in the last user message
        last_message = openai_request["messages"][-1]
        if last_message["role"] == "user":
            cid, content = decode_conversation_id(last_message["content"])
            if cid:
                conversation_id = cid
                openai_request["messages"][-1]["content"] = content

    dify_request = transform_openai_to_dify(openai_request, conversation_id)
    headers = {
        "Authorization": f"Bearer {dify_api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            dify_url = f"{DIFY_API_BASE}/chat-messages"
            if stream:
                async def stream_response():
                    async with client.stream("POST", dify_url, json=dify_request, headers=headers, timeout=600) as r:
                        r.raise_for_status()
                        async for chunk in r.aiter_bytes():
                            yield chunk
                return Response(stream_response(), content_type='text/event-stream')
            else:
                r = await client.post(dify_url, json=dify_request, headers=headers, timeout=600)
                r.raise_for_status()
                dify_response = r.json()

                # Add conversation ID for zero-width mode
                if CONVERSATION_MEMORY_MODE == 2 and 'conversation_id' in dify_response:
                    encoded_id = encode_conversation_id(dify_response['conversation_id'])
                    if 'answer' in dify_response:
                        dify_response['answer'] += encoded_id

                openai_response = transform_dify_to_openai(dify_response, model_name, stream=False)
                return Response(openai_response, content_type='application/json')

        except httpx.HTTPStatusError as e:
            logging.error(f"Dify API error: {e.response.status_code} - {e.response.text}")
            return jsonify({"error": {"message": "Dify API request failed.", "type": "internal_error", "code": "internal_error"}}), 500
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return jsonify({"error": {"message": "An internal server error occurred.", "type": "internal_error", "code": "internal_error"}}), 500


async def main():
    """Main function to initialize and run the application."""
    if not all([DIFY_API_KEYS, DIFY_API_BASE, VALID_API_KEYS]):
        logging.error("One or more required environment variables are missing. Please check your .env file.")
        return

    await model_manager.refresh_model_info()

    # Note: Running Flask with 'app.run' is not ideal for async.
    # For production, a proper ASGI server like Gunicorn with Uvicorn workers is recommended.
    # For simplicity in this script, we'll run it this way.
    # To run this properly: `uvicorn main:app --host 127.0.0.1 --port 5000`

    # This part is tricky because app.run is blocking and not async-friendly.
    # The 'main' function is kept async to allow for the initial async model refresh.
    # The user should run this with an ASGI server.
    logging.info("Initialization complete. The server is ready to be run with an ASGI server like Uvicorn.")
    logging.info(f"Example: uvicorn main:app --host {SERVER_HOST} --port {SERVER_PORT}")


if __name__ == "__main__":
    # This setup allows for the async main function to be run.
    # In a real-world scenario, you'd just run `uvicorn main:app`.
    # We run the async setup here to fetch models before starting.
    try:
        asyncio.run(main())
        logging.info("To start the server, run:")
        logging.info(f"uvicorn main:app --host {SERVER_HOST} --port {SERVER_PORT}")
    except Exception as e:
        logging.fatal(f"Failed to initialize the application: {e}")

# If you want to run directly with Python (for simple testing, not for production)
# you would need to use a library like 'asgiref' to serve the app.
# Example:
#
# if __name__ == "__main__":
#     import uvicorn
#     asyncio.run(model_manager.refresh_model_info())
#     uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
#
# The provided structure is more to show the async nature of the setup.
# I will leave the final run command to the user as instructed by the logging messages.
