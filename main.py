import json
import logging
import asyncio
from flask import Flask, request, Response, stream_with_context, jsonify
import httpx
import time
from dotenv import load_dotenv
import os
import ast
import re
import base64

# 1. 导入和初始化部分
# ==============================================================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

# 加载环境变量
load_dotenv()
VALID_API_KEYS = os.getenv("VALID_API_KEYS", "").split(',')
CONVERSATION_MEMORY_MODE = int(os.getenv("CONVERSATION_MEMORY_MODE", 1))
DIFY_API_BASE = os.getenv("DIFY_API_BASE")

# 2. DifyModelManager 类定义
# ==============================================================================

class DifyModelManager:
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}
        self.api_key_to_name = {}
        self.load_api_keys()

    def load_api_keys(self):
        dify_api_keys = os.getenv("DIFY_API_KEYS")
        if dify_api_keys:
            self.api_keys = [key.strip() for key in dify_api_keys.split(',')]
            logging.info(f"Loaded {len(self.api_keys)} Dify API keys.")
        else:
            logging.warning("DIFY_API_KEYS environment variable is not set.")

    async def fetch_app_info(self, api_key):
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{DIFY_API_BASE}/info", headers=headers, timeout=600)
                if response.status_code == 200:
                    app_info = response.json().get('app')
                    return app_info.get('title') if app_info else None
                else:
                    logging.warning(f"Failed to fetch app info for a key: Status {response.status_code}")
                    return None
        except Exception as e:
            logging.error(f"Error fetching app info: {e}")
            return None

    async def refresh_model_info(self):
        self.name_to_api_key = {}
        self.api_key_to_name = {}
        tasks = [self.fetch_app_info(key) for key in self.api_keys]
        results = await asyncio.gather(*tasks)
        for api_key, name in zip(self.api_keys, results):
            if name:
                self.name_to_api_key[name] = api_key
                self.api_key_to_name[api_key] = name
        logging.info(f"Refreshed model info. Mapped models: {list(self.name_to_api_key.keys())}")

    def get_api_key(self, model_name):
        return self.name_to_api_key.get(model_name)

    def get_available_models(self):
        return [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify",
            }
            for name in self.name_to_api_key.keys()
        ]

# 3. 全局变量和实例
# ==============================================================================

model_manager = DifyModelManager()
app = Flask(__name__)

# 4. 辅助函数
# ==============================================================================

def get_api_key(model_name):
    api_key = model_manager.get_api_key(model_name)
    if not api_key:
        logging.warning(f"API key not found for model: {model_name}")
    return api_key

def encode_conversation_id(conversation_id):
    if not conversation_id:
        return ""

    b64_encoded = base64.b64encode(conversation_id.encode('utf-8')).decode('utf-8')

    zero_width_chars = [
        '\u200b', '\u200c', '\u200d', '\u2060',
        '\u2061', '\u2062', '\u2063', '\u2064'
    ]

    char_to_zwc = {str(i): zero_width_chars[i] for i in range(8)}

    octal_string = ""
    for char in b64_encoded:
        octal_string += oct(ord(char))[2:].zfill(3)

    zwc_sequence = ""
    for digit in octal_string:
        zwc_sequence += char_to_zwc[digit]

    return zwc_sequence

def decode_conversation_id(content):
    zero_width_chars = [
        '\u200b', '\u200c', '\u200d', '\u2060',
        '\u2061', '\u2062', '\u2063', '\u2064'
    ]
    zwc_to_char = {zwc: str(i) for i, zwc in enumerate(zero_width_chars)}

    extracted_zwc = ""
    for char in reversed(content):
        if char in zwc_to_char:
            extracted_zwc = char + extracted_zwc
        else:
            break

    if not extracted_zwc:
        return None

    octal_string = ""
    for zwc in extracted_zwc:
        octal_string += zwc_to_char[zwc]

    b64_encoded = ""
    try:
        for i in range(0, len(octal_string), 3):
            octal_chunk = octal_string[i:i+3]
            if len(octal_chunk) < 3: continue
            decimal_value = int(octal_chunk, 8)
            b64_encoded += chr(decimal_value)

        padding_needed = -len(b64_encoded) % 4
        b64_encoded += "=" * padding_needed

        decoded_bytes = base64.b64decode(b64_encoded)
        return decoded_bytes.decode('utf-8')
    except Exception:
        return None

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def transform_openai_to_dify(openai_request):
    messages = openai_request.get("messages", [])
    stream = openai_request.get("stream", False)

    conversation_id = None
    system_message_content = ""

    if CONVERSATION_MEMORY_MODE == 2: # Zero-width character mode
        for message in reversed(messages):
            if message.get("role") == "assistant":
                conversation_id = decode_conversation_id(message.get("content", ""))
                if conversation_id:
                    break

        user_query = messages[-1].get("content") if messages and messages[-1].get("role") == "user" else ""

        if messages and messages[0].get("role") == "system":
            system_message_content = messages[0].get("content")
            if not conversation_id: # First turn of a conversation
                 user_query = f"{system_message_content}\n{user_query}"

        dify_request = {
            "inputs": {},
            "query": user_query,
            "user": "openai-proxy-user",
            "response_mode": "streaming" if stream else "blocking",
        }
        if conversation_id:
            dify_request["conversation_id"] = conversation_id
        if system_message_content:
            dify_request["inputs"] = {"system_prompt": system_message_content}

    else: # History message mode
        user_query = messages[-1].get("content") if messages and messages[-1].get("role") == "user" else ""

        history = []
        has_system_in_history = False
        for msg in messages[:-1]:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                has_system_in_history = True
            history.append({"role": role, "content": content})

        if messages and messages[0].get("role") == "system" and not has_system_in_history:
             system_message_content = messages[0].get("content")
             history.insert(0, {"role": "system", "content": system_message_content})

        if history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            user_query = f"<history>\n{history_str}\n</history>\n{user_query}"

        dify_request = {
            "inputs": {},
            "query": user_query,
            "user": "openai-proxy-user",
            "response_mode": "streaming" if stream else "blocking",
        }

    return dify_request


def transform_dify_to_openai(dify_response, model, stream, conversation_id, messages_history):
    if not stream:
        answer = ""
        if "answer" in dify_response:
            answer = dify_response.get("answer", "")
        elif "agent_thoughts" in dify_response:
            thoughts = dify_response.get("agent_thoughts", [])
            final_answer_thought = next((t for t in thoughts if t.get('final_answer')), None)
            if final_answer_thought:
                answer = final_answer_thought.get('final_answer', {}).get('answer', '')

        if CONVERSATION_MEMORY_MODE == 2:
            has_conv_id_in_history = any(decode_conversation_id(m.get("content", "")) for m in messages_history if m.get("role") == "assistant")
            is_new_conversation = not any(m.get("role") == "assistant" for m in messages_history)

            if conversation_id and (is_new_conversation or not has_conv_id_in_history):
                answer += encode_conversation_id(conversation_id)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": dify_response.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": dify_response.get("metadata", {}).get("usage", {}).get("completion_tokens", 0),
                "total_tokens": dify_response.get("metadata", {}).get("usage", {}).get("total_tokens", 0),
            }
        }
    else: # streaming is handled elsewhere
        return dify_response

def create_openai_stream_response(content, message_id, model):
    return {
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": content
            },
            "finish_reason": None
        }]
    }

# 5. Flask路由定义
# ==============================================================================

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": {"message": "Authorization header is missing or invalid.", "type": "invalid_request_error", "code": "auth_error"}}), 401

    api_key = auth_header.split(' ')[1]
    if api_key not in VALID_API_KEYS:
        return jsonify({"error": {"message": "Invalid API key.", "type": "invalid_request_error", "code": "invalid_api_key"}}), 401

    try:
        data = request.json
        model_name = data.get('model')
        stream = data.get('stream', False)
        messages = data.get('messages', [])

        dify_api_key = get_api_key(model_name)
        if not dify_api_key:
            return jsonify({"error": {"message": f"Model '{model_name}' not found.", "type": "invalid_request_error", "code": "model_not_found"}}), 404

        dify_request = transform_openai_to_dify(data)
        if not dify_request:
             return jsonify({"error": "Failed to transform request"}), 500

        headers = {
            "Authorization": f"Bearer {dify_api_key}",
            "Content-Type": "application/json"
        }

        dify_request["response_mode"] = "streaming" # Force streaming

        def generate():
            full_content = ""
            conversation_id = None
            message_id = f"chatcmpl-stream-{int(time.time())}"
            buffer = ""
            last_sent_time = time.time()

            with httpx.stream("POST", f"{DIFY_API_BASE}/chat-messages", json=dify_request, headers=headers, timeout=600) as r:
                for chunk in r.iter_bytes():
                    chunk_str = chunk.decode('utf-8')
                    for line in chunk_str.splitlines():
                        if line.startswith("data:"):
                            try:
                                content_str = line[len("data:"):].strip()
                                if content_str == "[DONE]":
                                    if buffer:
                                        response_data = create_openai_stream_response(buffer, message_id, model_name)
                                        yield f"data: {json.dumps(response_data)}\n\n"
                                    if CONVERSATION_MEMORY_MODE == 2 and conversation_id:
                                        encoded_id = encode_conversation_id(conversation_id)
                                        id_response = create_openai_stream_response(encoded_id, message_id, model_name)
                                        yield f"data: {json.dumps(id_response)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                                data = json.loads(content_str)
                                if data.get('event') == 'message':
                                    text = data.get('answer', '')
                                    text_no_think = remove_think_tags(text)
                                    full_content += text_no_think
                                    buffer += text_no_think

                                    if conversation_id is None:
                                        conversation_id = data.get('conversation_id')

                                    current_time = time.time()
                                    if buffer and (current_time - last_sent_time > 0.05 or len(buffer) > 5):
                                        response_data = create_openai_stream_response(buffer, message_id, model_name)
                                        yield f"data: {json.dumps(response_data)}\n\n"
                                        buffer = ""
                                        last_sent_time = current_time

                            except json.JSONDecodeError:
                                logging.warning(f"Could not decode JSON from line: {line}")
                                continue

            if buffer: # Send any remaining buffer content
                response_data = create_openai_stream_response(buffer, message_id, model_name)
                yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"


        def collect_stream():
            full_content = ""
            conversation_id = None
            final_dify_response = {}
            with httpx.stream("POST", f"{DIFY_API_BASE}/chat-messages", json=dify_request, headers=headers, timeout=600) as r:
                for chunk in r.iter_bytes():
                     chunk_str = chunk.decode('utf-8')
                     for line in chunk_str.splitlines():
                        if line.startswith("data:"):
                            content_str = line[len("data:"):].strip()
                            if content_str == "[DONE]":
                                continue
                            try:
                                data = json.loads(content_str)
                                if data.get('event') == 'message':
                                    full_content += data.get('answer', '')
                                if conversation_id is None:
                                    conversation_id = data.get('conversation_id')
                                # The last data event will contain metadata
                                final_dify_response = data
                            except json.JSONDecodeError:
                                continue

            final_dify_response['answer'] = remove_think_tags(full_content)
            return transform_dify_to_openai(final_dify_response, model_name, False, conversation_id, messages)

        if stream:
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            openai_response = collect_stream()
            return jsonify(openai_response)

    except Exception as e:
        logging.error(f"An error occurred in chat_completions: {e}")
        return jsonify({"error": {"message": "An internal server error occurred.", "type": "server_error", "code": "internal_error"}}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    logging.info("Refreshing and listing available models.")
    asyncio.run(model_manager.refresh_model_info())
    models = model_manager.get_available_models()
    response = {"object": "list", "data": models}
    logging.info(f"Returning {len(models)} models.")
    return jsonify(response)

# 6. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    if not all([os.getenv("VALID_API_KEYS"), os.getenv("DIFY_API_KEYS"), os.getenv("DIFY_API_BASE")]):
        logging.error("FATAL: Essential environment variables (VALID_API_KEYS, DIFY_API_KEYS, DIFY_API_BASE) are not set.")
    else:
        logging.info("Starting initial model info refresh...")
        asyncio.run(model_manager.refresh_model_info())

        host = os.getenv("SERVER_HOST", "127.0.0.1")
        port = int(os.getenv("SERVER_PORT", 5000))

        logging.info(f"Starting Flask server on {host}:{port}")
        app.run(host=host, port=port)
