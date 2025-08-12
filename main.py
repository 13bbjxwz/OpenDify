# 导入所需的标准库和第三方库
import json  # 用于处理 JSON 数据
import logging  # 用于日志记录
import asyncio  # 用于异步编程
from flask import Flask, request, Response, stream_with_context, jsonify  # Flask Web 框架相关
import httpx  # 用于 HTTP 请求
import time  # 用于时间相关操作
from dotenv import load_dotenv  # 用于加载 .env 环境变量
import os  # 用于操作系统相关功能
import ast  # 用于抽象语法树处理
import re  # 用于正则表达式处理

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,  # 日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志输出格式
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 设置 httpx 的日志级别为 WARNING，减少不必要的输出
logging.getLogger("httpx").setLevel(logging.WARNING)

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量读取有效的 API 密钥（以逗号分隔）
VALID_API_KEYS = [key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key]

# 获取会话记忆功能模式配置
# 1: 构造 history_message 附加到消息中的模式(默认)
# 2: 零宽字符模式
CONVERSATION_MEMORY_MODE = int(os.getenv('CONVERSATION_MEMORY_MODE', '1'))

class DifyModelManager:
    """Dify 模型管理器，负责管理 API Key 与模型名称的映射"""
    def __init__(self):
        self.api_keys = []  # 存储所有 Dify API Key
        self.name_to_api_key = {}  # 应用名称到 API Key 的映射
        self.api_key_to_name = {}  # API Key 到应用名称的映射
        self.load_api_keys()  # 初始化时加载 API Key

    def load_api_keys(self):
        """从环境变量加载 Dify API Keys"""
        api_keys_str = os.getenv('DIFY_API_KEYS', '')  # 读取 DIFY_API_KEYS 环境变量
        if api_keys_str:
            self.api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
            logger.info(f"Loaded {len(self.api_keys)} API keys")

    async def fetch_app_info(self, api_key):
        """通过 API Key 获取 Dify 应用信息"""
        try:
            async with httpx.AsyncClient(timeout=600) as client:  # 设置超时时间为 600 秒
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = await client.get(
                    f"{DIFY_API_BASE}/info",
                    headers=headers,
                    params={"user": "default_user"}
                )
                
                if response.status_code == 200:
                    app_info = response.json()
                    return app_info.get("name", "Unknown App")  # 返回应用名称
                else:
                    logger.error(f"Failed to fetch app info for API key: {api_key[:8]}...")
                    return None
        except Exception as e:
            logger.error(f"Error fetching app info: {str(e)}")
            return None

    async def refresh_model_info(self):
        """刷新所有应用信息，建立名称与 API Key 的映射"""
        self.name_to_api_key.clear()
        self.api_key_to_name.clear()
        
        for api_key in self.api_keys:
            app_name = await self.fetch_app_info(api_key)
            if app_name:
                self.name_to_api_key[app_name] = api_key
                self.api_key_to_name[api_key] = app_name
                logger.info(f"Mapped app '{app_name}' to API key: {api_key[:8]}...")

    def get_api_key(self, model_name):
        """根据模型名称获取 API Key"""
        return self.name_to_api_key.get(model_name)

    def get_available_models(self):
        """获取可用模型列表，返回 OpenAI 风格结构"""
        return [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dify"
            }
            for name in self.name_to_api_key.keys()
        ]

# 创建模型管理器实例
model_manager = DifyModelManager()

# 从环境变量获取 Dify API 基础 URL
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")

# 创建 Flask 应用实例
app = Flask(__name__)

def get_api_key(model_name):
    """根据模型名称获取对应的 API 密钥"""
    api_key = model_manager.get_api_key(model_name)
    if not api_key:
        logger.warning(f"No API key found for model: {model_name}")
    return api_key

def transform_openai_to_dify(openai_request, endpoint):
    """将 OpenAI 格式的请求转换为 Dify 格式"""
    
    if endpoint == "/chat/completions":
        messages = openai_request.get("messages", [])  # 获取消息列表
        stream = openai_request.get("stream", False)  # 是否为流式请求
        
        # 尝试从历史消息中提取 conversation_id
        conversation_id = None
        
        # 提取 system 消息内容
        system_content = ""
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        if system_messages:
            system_content = system_messages[0].get("content", "")
            # 记录找到的 system 消息
            logger.info(f"Found system message: {system_content[:100]}{'...' if len(system_content) > 100 else ''}")
        
        if CONVERSATION_MEMORY_MODE == 2:  # 零宽字符模式
            if len(messages) > 1:
                # 遍历历史消息，找到最近的 assistant 消息
                for msg in reversed(messages[:-1]):  # 除了最后一条消息
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # 尝试解码 conversation_id
                        conversation_id = decode_conversation_id(content)
                        if conversation_id:
                            break
            
            # 获取最后一条用户消息
            user_query = messages[-1]["content"] if messages and messages[-1].get("role") != "system" else ""
            
            # 如果有 system 消息且是首次对话(没有 conversation_id)，则将 system 内容添加到用户查询前
            if system_content and not conversation_id:
                user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"
                logger.info(f"[零宽字符模式] 首次对话，添加system内容到查询前")
            
            dify_request = {
                "inputs": {},
                "query": user_query,
                "response_mode": "streaming" if stream else "blocking",
                "conversation_id": conversation_id,
                "user": openai_request.get("user", "default_user")
            }
        else:  # history_message 模式(默认)
            # 获取最后一条用户消息
            user_query = messages[-1]["content"] if messages and messages[-1].get("role") != "system" else ""
            
            # 构造历史消息
            if len(messages) > 1:
                history_messages = []
                has_system_in_history = False
                
                # 检查历史消息中是否已经包含 system 消息
                for msg in messages[:-1]:  # 除了最后一条消息
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        if role == "system":
                            has_system_in_history = True
                        history_messages.append(f"{role}: {content}")
                
                # 如果历史中没有 system 消息但现在有 system 消息，则添加到历史的最前面
                if system_content and not has_system_in_history:
                    history_messages.insert(0, f"system: {system_content}")
                    logger.info(f"[history_message模式] 添加system内容到历史消息前")
                
                # 将历史消息添加到查询中
                if history_messages:
                    history_context = "\n\n".join(history_messages)
                    user_query = f"<history>\n{history_context}\n</history>\n\n用户当前问题: {user_query}"
            elif system_content:  # 没有历史消息但有 system 消息
                user_query = f"系统指令: {system_content}\n\n用户问题: {user_query}"
                logger.info(f"[history_message模式] 首次对话，添加system内容到查询前")
            
            dify_request = {
                "inputs": {},
                "query": user_query,
                "response_mode": "streaming" if stream else "blocking",
                "user": openai_request.get("user", "default_user")
            }

        return dify_request
    
    return None

def transform_dify_to_openai(dify_response, model="claude-3-5-sonnet-v2", stream=False):
    """将Dify格式的响应转换为OpenAI格式"""
    
    if not stream:
        # 首先获取回答内容，支持不同的响应模式
        answer = ""
        mode = dify_response.get("mode", "")
        
        # 普通聊天模式
        if "answer" in dify_response:
            answer = dify_response.get("answer", "")
        
        # 如果是Agent模式，需要从agent_thoughts中提取回答
        elif "agent_thoughts" in dify_response:
            # Agent模式下通常最后一个thought包含最终答案
            agent_thoughts = dify_response.get("agent_thoughts", [])
            if agent_thoughts:
                for thought in agent_thoughts:
                    if thought.get("thought"):
                        answer = thought.get("thought", "")
        
        # 只在零宽字符会话记忆模式时处理conversation_id
        if CONVERSATION_MEMORY_MODE == 2:
            conversation_id = dify_response.get("conversation_id", "")
            history = dify_response.get("conversation_history", [])
            
            # 检查历史消息中是否已经有会话ID
            has_conversation_id = False
            if history:
                for msg in history:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if decode_conversation_id(content) is not None:
                            has_conversation_id = True
                            break
            
            # 只在新会话且历史消息中没有会话ID时插入
            if conversation_id and not has_conversation_id:
                logger.info(f"[Debug] Inserting conversation_id: {conversation_id}, history_length: {len(history)}")
                encoded = encode_conversation_id(conversation_id)
                answer = answer + encoded
                logger.info(f"[Debug] Response content after insertion: {repr(answer)}")
        
        return {
            "id": dify_response.get("message_id", ""),
            "object": "chat.completion",
            "created": dify_response.get("created", int(time.time())),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }]
        }
    else:
        # 流式响应的转换在stream_response函数中处理
        return dify_response

def create_openai_stream_response(content, message_id, model="claude-3-5-sonnet-v2"):
    """创建OpenAI格式的流式响应"""
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

def encode_conversation_id(conversation_id):
    """将conversation_id编码为不可见的字符序列"""
    if not conversation_id:
        return ""
    
    # 使用Base64编码减少长度
    import base64
    encoded = base64.b64encode(conversation_id.encode()).decode()
    
    # 使用8种不同的零宽字符表示3位数字
    # 这样可以将编码长度进一步减少
    char_map = {
        '0': '\u200b',  # 零宽空格
        '1': '\u200c',  # 零宽非连接符
        '2': '\u200d',  # 零宽连接符
        '3': '\ufeff',  # 零宽非断空格
        '4': '\u2060',  # 词组连接符
        '5': '\u180e',  # 蒙古语元音分隔符
        '6': '\u2061',  # 函数应用
        '7': '\u2062',  # 不可见乘号
    }
    
    # 将Base64字符串转换为八进制数字
    result = []
    for c in encoded:
        # 将每个字符转换为8进制数字（0-7）
        if c.isalpha():
            if c.isupper():
                val = ord(c) - ord('A')
            else:
                val = ord(c) - ord('a') + 26
        elif c.isdigit():
            val = int(c) + 52
        elif c == '+':
            val = 62
        elif c == '/':
            val = 63
        else:  # '='
            val = 0
            
        # 每个Base64字符可以产生2个3位数字
        first = (val >> 3) & 0x7
        second = val & 0x7
        result.append(char_map[str(first)])
        if c != '=':  # 不编码填充字符的后半部分
            result.append(char_map[str(second)])
    
    return ''.join(result)

def decode_conversation_id(content):
    """从消息内容中解码conversation_id"""
    try:
        # 零宽字符到3位数字的映射
        char_to_val = {
            '\u200b': '0',  # 零宽空格
            '\u200c': '1',  # 零宽非连接符
            '\u200d': '2',  # 零宽连接符
            '\ufeff': '3',  # 零宽非断空格
            '\u2060': '4',  # 词组连接符
            '\u180e': '5',  # 蒙古语元音分隔符
            '\u2061': '6',  # 函数应用
            '\u2062': '7',  # 不可见乘号
        }
        
        # 提取最后一段零宽字符序列
        space_chars = []
        for c in reversed(content):
            if c not in char_to_val:
                break
            space_chars.append(c)
        
        if not space_chars:
            return None
            
        # 将零宽字符转换回Base64字符串
        space_chars.reverse()
        base64_chars = []
        for i in range(0, len(space_chars), 2):
            first = int(char_to_val[space_chars[i]], 8)
            if i + 1 < len(space_chars):
                second = int(char_to_val[space_chars[i + 1]], 8)
                val = (first << 3) | second
            else:
                val = first << 3
                
            # 转换回Base64字符
            if val < 26:
                base64_chars.append(chr(val + ord('A')))
            elif val < 52:
                base64_chars.append(chr(val - 26 + ord('a')))
            elif val < 62:
                base64_chars.append(str(val - 52))
            elif val == 62:
                base64_chars.append('+')
            else:
                base64_chars.append('/')
                
        # 添加Base64填充
        padding = len(base64_chars) % 4
        if padding:
            base64_chars.extend(['='] * (4 - padding))
            
        # 解码Base64字符串
        import base64
        base64_str = ''.join(base64_chars)
        return base64.b64decode(base64_str).decode()
        
    except Exception as e:
        logger.debug(f"Failed to decode conversation_id: {e}")
        return None

def remove_think_tags(text):
    """去除所有 <think>...</think> 及其内容"""
    return re.sub(r'<think>[\s\S]*?</think>', '', text)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        # 新增：验证API密钥
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({
                "error": {
                    "message": "缺少 Authorization 头部信息",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }), 401

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({
                "error": {
                    "message": "Authorization 头部格式无效，格式应为: Bearer <API_KEY>",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }), 401

        provided_api_key = parts[1]
        if provided_api_key not in VALID_API_KEYS:
            return jsonify({
                "error": {
                    "message": "API 密钥无效",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }), 401

        # 继续处理原始逻辑
        openai_request = request.get_json()
        logger.info(f"Received request: {json.dumps(openai_request, ensure_ascii=False)}")
        
        model = openai_request.get("model", "claude-3-5-sonnet-v2")
        
        # 验证模型是否支持
        api_key = get_api_key(model)
        if not api_key:
            error_msg = f"模型 {model} 不被支持。可用模型: {', '.join(model_manager.name_to_api_key.keys())}"
            logger.error(error_msg)
            return jsonify({
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_not_found"
                }
            }), 404
            
        dify_request = transform_openai_to_dify(openai_request, "/chat/completions")
        
        if not dify_request:
            logger.error("请求转换失败")
            return jsonify({
                "error": {
                    "message": "请求格式无效",
                    "type": "invalid_request_error",
                    "param": None
                }
            }), 400

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stream = openai_request.get("stream", False)
        dify_endpoint = f"{DIFY_API_BASE}/chat-messages"
        logger.info(f"Sending request to Dify endpoint: {dify_endpoint}, stream={stream}")

        # 强制所有 Dify 请求都用 streaming，规避 Dify 超时
        dify_request['response_mode'] = 'streaming'

        if stream:
            logger.info("[流式处理] 当前为流式输出模式，后端始终用 Dify streaming 接口")
            def generate():
                client = httpx.Client(timeout=600)
                def flush_chunk(chunk_data):
                    return chunk_data.encode('utf-8')
                def calculate_delay(buffer_size):
                    if buffer_size > 30:
                        return 0.001
                    elif buffer_size > 20:
                        return 0.002
                    elif buffer_size > 10:
                        return 0.01
                    else:
                        return 0.02
                def send_char(char, message_id):
                    openai_chunk = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": char},
                            "finish_reason": None
                        }]
                    }
                    chunk_data = f"data: {json.dumps(openai_chunk)}\n\n"
                    return flush_chunk(chunk_data)
                output_buffer = []
                try:
                    with client.stream(
                        'POST',
                        dify_endpoint,
                        json=dify_request,
                        headers={
                            **headers,
                            'Accept': 'text/event-stream',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive'
                        }
                    ) as response:
                        generate.message_id = None
                        buffer = ""
                        for raw_bytes in response.iter_raw():
                            if not raw_bytes:
                                continue
                            try:
                                buffer += raw_bytes.decode('utf-8')
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    line = line.strip()
                                    if not line or not line.startswith('data: '):
                                        continue
                                    try:
                                        json_str = line[6:]
                                        dify_chunk = json.loads(json_str)
                                        if dify_chunk.get("event") == "message" and "answer" in dify_chunk:
                                            current_answer = dify_chunk["answer"]
                                            if not current_answer:
                                                continue
                                            message_id = dify_chunk.get("message_id", "")
                                            if not generate.message_id:
                                                generate.message_id = message_id
                                            # 先拼到大 buffer，去除 <think> 块后再输出
                                            filtered_answer = remove_think_tags(current_answer)
                                            for char in filtered_answer:
                                                output_buffer.append((char, generate.message_id))
                                            while output_buffer:
                                                char, msg_id = output_buffer.pop(0)
                                                yield send_char(char, msg_id)
                                                delay = calculate_delay(len(output_buffer))
                                                time.sleep(delay)
                                            continue
                                        elif dify_chunk.get("event") == "agent_message" and "answer" in dify_chunk:
                                            current_answer = dify_chunk["answer"]
                                            if not current_answer:
                                                continue
                                            message_id = dify_chunk.get("message_id", "")
                                            if not generate.message_id:
                                                generate.message_id = message_id
                                            filtered_answer = remove_think_tags(current_answer)
                                            for char in filtered_answer:
                                                output_buffer.append((char, generate.message_id))
                                            while output_buffer:
                                                char, msg_id = output_buffer.pop(0)
                                                yield send_char(char, msg_id)
                                                delay = calculate_delay(len(output_buffer))
                                                time.sleep(delay)
                                            continue
                                        elif dify_chunk.get("event") == "message_end":
                                            while output_buffer:
                                                char, msg_id = output_buffer.pop(0)
                                                yield send_char(char, msg_id)
                                                time.sleep(0.001)
                                            final_chunk = {
                                                "id": generate.message_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": "stop"
                                                }]
                                            }
                                            yield flush_chunk(f"data: {json.dumps(final_chunk)}\n\n")
                                            yield flush_chunk("data: [DONE]\n\n")
                                    except json.JSONDecodeError as e:
                                        logger.error(f"JSON decode error: {str(e)}")
                                        continue
                            except Exception as e:
                                logger.error(f"Error processing chunk: {str(e)}")
                                continue
                finally:
                    client.close()
            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache, no-transform',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                    'Content-Encoding': 'none'
                },
                direct_passthrough=True
            )
        else:
            def collect_stream():
                client = httpx.Client(timeout=600)
                content = ""
                message_id = None
                try:
                    with client.stream(
                        'POST',
                        dify_endpoint,
                        json=dify_request,
                        headers={
                            **headers,
                            'Accept': 'text/event-stream',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive'
                        }
                    ) as response:
                        buffer = ""
                        for raw_bytes in response.iter_raw():
                            if not raw_bytes:
                                continue
                            buffer += raw_bytes.decode('utf-8')
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                if not line or not line.startswith('data: '):
                                    continue
                                try:
                                    json_str = line[6:]
                                    dify_chunk = json.loads(json_str)
                                    if dify_chunk.get("event") == "message" and "answer" in dify_chunk:
                                        current_answer = dify_chunk["answer"]
                                        if not current_answer:
                                            continue
                                        if not message_id:
                                            message_id = dify_chunk.get("message_id", "")
                                        content += current_answer
                                    elif dify_chunk.get("event") == "agent_message" and "answer" in dify_chunk:
                                        current_answer = dify_chunk["answer"]
                                        if not current_answer:
                                            continue
                                        if not message_id:
                                            message_id = dify_chunk.get("message_id", "")
                                        content += current_answer
                                    elif dify_chunk.get("event") == "message_end":
                                        break
                                except Exception:
                                    continue
                finally:
                    client.close()
                # 最终统一去除 <think> 块
                content = remove_think_tags(content)
                openai_response = {
                    "id": message_id or "",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }]
                }
                return openai_response
            return jsonify(collect_stream())

    except Exception as e:
        logger.exception("发生未知错误")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "param": None
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回可用的模型列表"""
    logger.info("Listing available models")
    
    # 刷新模型信息
    asyncio.run(model_manager.refresh_model_info())
    
    # 获取可用模型列表
    available_models = model_manager.get_available_models()
    
    response = {
        "object": "list",
        "data": available_models
    }
    logger.info(f"Available models: {json.dumps(response, ensure_ascii=False)}")
    return response

# 在main.py的最后初始化时添加环境变量检查：
if __name__ == '__main__':
    if not VALID_API_KEYS:
        print("Warning: No API keys configured. Set the VALID_API_KEYS environment variable with comma-separated keys.")
    
    # 启动时初始化模型信息
    asyncio.run(model_manager.refresh_model_info())
    
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", 5000))
    logger.info(f"Starting server on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
