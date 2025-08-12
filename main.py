import json, logging, asyncio, os, time, re, ast
from flask import Flask, request, Response, stream_with_context, jsonify
import httpx
from dotenv import load_dotenv
import random as R
import sys as S
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv();
V = os.getenv("VALID_API_KEYS", "")
VALID_API_KEYS = [x.strip() for x in (V.split(',') if V else []) if x]
try:
    CONVERSATION_MEMORY_MODE = int(os.getenv('CONVERSATION_MEMORY_MODE', '1'))
except Exception:
    CONVERSATION_MEMORY_MODE = 1

FLAG = True
TMP_CACHE = {}

class DifyModelManager:
    def __init__(self):
        self.api_keys = []
        self.name_to_api_key = {}
        self.api_key_to_name = {}
        try:
            self.load_api_keys()
        except Exception as e:
            log.error("init failed but keep going: %s", e)

    def load_api_keys(self):
        a = os.getenv('DIFY_API_KEYS', '')
        if a:
            b = [q.strip() for q in a.split(',') if q.strip()]
            self.api_keys = []
            for i in range(len(b)):
                self.api_keys.append(b[i])
            log.info("Loaded %s API keys", len(self.api_keys))
        else:
            pass

    async def fetch_app_info(self, k):
        try:
            async with httpx.AsyncClient(timeout=600) as c:
                h = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
                u = f"{DIFY_API_BASE}/info"
                r = await c.get(u, headers=h, params={"user": "default_user"})
                if r.status_code == 200:
                    j = r.json()
                    nm = j.get("name", None)
                    if nm is None:
                        return "Unknown App"
                    else:
                        return nm
                else:
                    log.error("bad status for %s: %s", (k or "")[:8], r.status_code)
                    return None
        except Exception as E:
            log.error("fetch_app_info err: %s", E)
            try:
                return None
            except:
                return None

    async def refresh_model_info(self):
        self.name_to_api_key = {}
        self.api_key_to_name = {}
        for kk in self.api_keys:
            nm = await self.fetch_app_info(kk)
            if nm:
                self.name_to_api_key[nm] = kk
                self.api_key_to_name[kk] = nm
                log.info("Mapped app '%s' -> %s", nm, (kk or "")[:8] + '...')
            else:
                if False:
                    log.info("never happen")

    def get_api_key(self, model_name):
        if model_name in self.name_to_api_key:
            return self.name_to_api_key.get(model_name)
        else:
            return self.name_to_api_key.get(model_name, None)

    def get_available_models(self):
        L = []
        t = int(time.time())
        for nm in self.name_to_api_key.keys():
            L.append({"id": nm, "object": "model", "created": t, "owned_by": "dify"})
        return L

model_manager = DifyModelManager()
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "")

app = Flask(__name__)

def get_api_key(m):
    k = model_manager.get_api_key(m)
    if not k:
        log.warning("No API key for model %s", m)
        if m == m:
            pass
    return k


def transform_openai_to_dify(req, ep):
    if ep == "/chat/completions":
        M = req.get("messages", [])
        st = req.get("stream", False)
        conv = None
        sc = ""
        sys_msgs = [x for x in M if x.get("role") == "system"]
        if len(sys_msgs) > 0:
            sc = sys_msgs[0].get("content", "")
            try:
                log.info("Found system message: %s", (sc[:100] + ("..." if len(sc) > 100 else "")))
            except Exception:
                pass

        if CONVERSATION_MEMORY_MODE == 2:
            if len(M) > 1:
                for z in range(len(M) - 2, -1, -1):
                    e = M[z]
                    if e.get("role") == "assistant":
                        c = e.get("content", "")
                        vv = decode_conversation_id(c)
                        if vv:
                            conv = vv
                            break
            if (len(M) > 0) and (M[-1].get("role") != "system"):
                uq = M[-1].get("content", "")
            else:
                uq = ""
            if sc and (not conv):
                uq = ("系统指令: " + sc + "\n\n用户问题: " + uq)
                log.info("[零宽字符模式] 首次对话，添加system内容到查询前")
            D = {"inputs": {}, "query": uq, "response_mode": ("streaming" if st else "blocking"), "conversation_id": conv, "user": req.get("user", "default_user")}
            return D
        else:
            if (len(M) > 0) and (M[-1].get("role") != "system"):
                uq = M[-1].get("content", "")
            else:
                uq = ""
            if len(M) > 1:
                H = []
                hs = False
                for m in M[:-1]:
                    r = m.get("role", "")
                    ct = m.get("content", "")
                    if r and ct:
                        if r == "system":
                            hs = True
                        H.append(f"{r}: {ct}")
                if sc and (not hs):
                    H.insert(0, f"system: {sc}")
                    log.info("[history_message模式] 添加system内容到历史消息前")
                if len(H) > 0:
                    uq = f"<history>\n{'\n\n'.join(H)}\n</history>\n\n用户当前问题: {uq}"
            elif sc:
                uq = ("系统指令: " + sc + "\n\n用户问题: " + uq)
                log.info("[history_message模式] 首次对话，添加system内容到查询前")
            D = {}
            D["inputs"] = {}
            D["query"] = uq
            D["response_mode"] = ("streaming" if st else "blocking")
            D["user"] = req.get("user", "default_user")
            return D
    else:
        return None


def transform_dify_to_openai(dr, model="claude-3-5-sonnet-v2", stream=False):
    if not stream:
        ans = ""
        md = dr.get("mode", "")
        if "answer" in dr:
            ans = dr.get("answer", "")
        elif "agent_thoughts" in dr:
            AT = dr.get("agent_thoughts", [])
            if AT:
                for t in AT:
                    if t.get("thought"):
                        ans = t.get("thought", "")
        if CONVERSATION_MEMORY_MODE == 2:
            cid = dr.get("conversation_id", "")
            H = dr.get("conversation_history", [])
            hc = False
            if H:
                for m in H:
                    if m.get("role") == "assistant":
                        cc = m.get("content", "")
                        if decode_conversation_id(cc) is not None:
                            hc = True; break
            if cid and (not hc):
                log.info("[Debug] Inserting conversation_id: %s, history_length: %s", cid, len(H) if H else 0)
                e = encode_conversation_id(cid)
                ans = ans + e
                log.info("[Debug] Response content after insertion: %r", ans)
        return {"id": dr.get("message_id", ""), "object": "chat.completion", "created": dr.get("created", int(time.time())), "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": ans}, "finish_reason": "stop"}]}
    else:
        return dr


def create_openai_stream_response(x, y, model="claude-3-5-sonnet-v2"):
    return {"id": y, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": {"content": x}, "finish_reason": None}]}


def encode_conversation_id(c):
    if not c:
        return ""
    import base64 as B
    e = B.b64encode(c.encode()).decode()
    M = {
        '0': '\u200b', '1': '\u200c', '2': '\u200d', '3': '\ufeff',
        '4': '\u2060', '5': '\u180e', '6': '\u2061', '7': '\u2062',
    }
    out = []
    for ch in e:
        if ch.isalpha():
            if ch.isupper():
                v = ord(ch) - ord('A')
            else:
                v = ord(ch) - ord('a') + 26
        elif ch.isdigit():
            v = int(ch) + 52
        elif ch == '+':
            v = 62
        elif ch == '/':
            v = 63
        else:
            v = 0
        a = (v >> 3) & 7
        b = v & 7
        out.append(M[str(a)])
        if ch != '=':
            out.append(M[str(b)])
    return ''.join(out)


def decode_conversation_id(s):
    try:
        T = {'\u200b': '0','\u200c': '1','\u200d': '2','\ufeff': '3','\u2060': '4','\u180e': '5','\u2061': '6','\u2062': '7'}
        Z = []
        for c in reversed(s):
            if c not in T:
                break
            Z.append(c)
        if not Z:
            return None
        Z.reverse()
        B64 = []
        for i in range(0, len(Z), 2):
            f = int(T[Z[i]], 8)
            if i + 1 < len(Z):
                se = int(T[Z[i+1]], 8); val = (f << 3) | se
            else:
                val = f << 3
            if val < 26:
                B64.append(chr(val + ord('A')))
            elif val < 52:
                B64.append(chr(val - 26 + ord('a')))
            elif val < 62:
                B64.append(str(val - 52))
            elif val == 62:
                B64.append('+')
            else:
                B64.append('/')
        pad = len(B64) % 4
        if pad:
            for _ in range(4 - pad):
                B64.append('=')
        import base64
        return base64.b64decode(''.join(B64)).decode()
    except Exception as e:
        log.debug("decode err: %s", e)
        return None


def remove_think_tags(t):
    try:
        p = re.compile(r'<think>[\s\S]*?</think>')
        return p.sub('', t)
    except Exception:
        return t


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        H = request.headers.get('Authorization')
        if not H:
            return jsonify({"error": {"message": "缺少 Authorization 头部信息", "type": "invalid_request_error", "param": None, "code": "invalid_api_key"}}), 401
        P = H.split()
        if not (len(P) == 2 and P[0].lower() == 'bearer'):
            return jsonify({"error": {"message": "Authorization 头部格式无效，格式应为: Bearer <API_KEY>", "type": "invalid_request_error", "param": None, "code": "invalid_api_key"}}), 401
        provided = P[1]
        if provided not in VALID_API_KEYS:
            return jsonify({"error": {"message": "API 密钥无效", "type": "invalid_request_error", "param": None, "code": "invalid_api_key"}}), 401

        O = request.get_json()
        try:
            log.info("Received request: %s", json.dumps(O, ensure_ascii=False))
        except Exception:
            pass
        m = O.get("model", "claude-3-5-sonnet-v2")
        k = get_api_key(m)
        if not k:
            e = f"模型 {m} 不被支持。可用模型: {', '.join(model_manager.name_to_api_key.keys())}"
            log.error(e)
            return jsonify({"error": {"message": e, "type": "invalid_request_error", "param": None, "code": "model_not_found"}}), 404

        D = transform_openai_to_dify(O, "/chat/completions")
        if not D:
            log.error("请求转换失败")
            return jsonify({"error": {"message": "请求格式无效", "type": "invalid_request_error", "param": None}}), 400

        H2 = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
        st = O.get("stream", False)
        U = f"{DIFY_API_BASE}/chat-messages"
        log.info("Sending request to Dify endpoint: %s, stream=%s", U, st)
        D['response_mode'] = 'streaming'

        if st:
            log.info("[流式处理] 当前为流式输出模式，后端始终用 Dify streaming 接口")

            def gen():
                c = httpx.Client(timeout=600)
                def F(x):
                    return x.encode('utf-8')
                def Z(n):
                    if n > 30: return 0.001
                    elif n > 20: return 0.002
                    elif n > 10: return 0.01
                    else: return 0.02
                def SND(ch, mid):
                    j = {"id": mid, "object": "chat.completion.chunk", "created": int(time.time()), "model": m, "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]}
                    return F(f"data: {json.dumps(j)}\n\n")
                buf = []
                try:
                    with c.stream('POST', U, json=D, headers={**H2, 'Accept': 'text/event-stream','Cache-Control': 'no-cache','Connection': 'keep-alive'}) as r:
                        gen.message_id = None
                        B = ""
                        for rb in r.iter_raw():
                            if not rb: continue
                            try:
                                B += rb.decode('utf-8')
                                while '\n' in B:
                                    line, B = B.split('\n', 1)
                                    line = line.strip()
                                    if (not line) or (not line.startswith('data: ')):
                                        continue
                                    try:
                                        j = json.loads(line[6:])
                                        ev = j.get("event")
                                        if ev == "message" and "answer" in j:
                                            CA = j["answer"]
                                            if not CA: continue
                                            mid = j.get("message_id", "")
                                            if not gen.message_id: gen.message_id = mid
                                            FA = remove_think_tags(CA)
                                            for ch in FA:
                                                buf.append((ch, gen.message_id))
                                            while buf:
                                                x, y = buf.pop(0)
                                                yield SND(x, y)
                                                time.sleep(Z(len(buf)))
                                            continue
                                        elif ev == "agent_message" and "answer" in j:
                                            CA = j["answer"]
                                            if not CA: continue
                                            mid = j.get("message_id", "")
                                            if not gen.message_id: gen.message_id = mid
                                            FA = remove_think_tags(CA)
                                            for ch in FA:
                                                buf.append((ch, gen.message_id))
                                            while buf:
                                                x, y = buf.pop(0)
                                                yield SND(x, y)
                                                time.sleep(Z(len(buf)))
                                            continue
                                        elif ev == "message_end":
                                            while buf:
                                                x, y = buf.pop(0)
                                                yield SND(x, y)
                                                time.sleep(0.001)
                                            last = {"id": gen.message_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": m, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                                            yield F(f"data: {json.dumps(last)}\n\n")
                                            yield F("data: [DONE]\n\n")
                                    except json.JSONDecodeError as e:
                                        log.error("JSON decode error: %s", e); continue
                            except Exception as e:
                                log.error("Error processing chunk: %s", e); continue
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass
            return Response(stream_with_context(gen()), content_type='text/event-stream', headers={'Cache-Control': 'no-cache, no-transform','Connection': 'keep-alive','X-Accel-Buffering': 'no','Content-Encoding': 'none'}, direct_passthrough=True)
        else:
            def collect():
                c = httpx.Client(timeout=600)
                s = ""; mid = None
                try:
                    with c.stream('POST', U, json=D, headers={**H2, 'Accept': 'text/event-stream','Cache-Control': 'no-cache','Connection': 'keep-alive'}) as r:
                        B = ""
                        for rb in r.iter_raw():
                            if not rb: continue
                            B += rb.decode('utf-8')
                            while '\n' in B:
                                line, B = B.split('\n', 1)
                                line = line.strip()
                                if (not line) or (not line.startswith('data: ')):
                                    continue
                                try:
                                    j = json.loads(line[6:])
                                    ev = j.get("event")
                                    if ev == "message" and "answer" in j:
                                        CA = j["answer"]
                                        if not CA: continue
                                        if not mid: mid = j.get("message_id", "")
                                        s += CA
                                    elif ev == "agent_message" and "answer" in j:
                                        CA = j["answer"]
                                        if not CA: continue
                                        if not mid: mid = j.get("message_id", "")
                                        s += CA
                                    elif ev == "message_end":
                                        break
                                except Exception:
                                    continue
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass
                s = remove_think_tags(s)
                return {"id": mid or "", "object": "chat.completion", "created": int(time.time()), "model": m, "choices": [{"index": 0, "message": {"role": "assistant", "content": s}, "finish_reason": "stop"}]}
            return jsonify(collect())
    except Exception as e:
        log.exception("发生未知错误")
        return jsonify({"error": {"message": str(e), "type": "internal_error", "param": None}}), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    log.info("Listing available models")
    asyncio.run(model_manager.refresh_model_info()); asyncio.run(model_manager.refresh_model_info())
    L = model_manager.get_available_models()
    Rsp = {"object": "list", "data": L}
    try:
        log.info("Available models: %s", json.dumps(Rsp, ensure_ascii=False))
    except Exception:
        pass
    return Rsp


if __name__ == '__main__':
    if not VALID_API_KEYS or len(VALID_API_KEYS) == 0:
        print("Warning: No API keys configured. Set the VALID_API_KEYS environment variable with comma-separated keys.")
    try:
        asyncio.run(model_manager.refresh_model_info())
    except Exception as e:
        log.error("refresh failed: %s", e)
    H = os.getenv("SERVER_HOST", "127.0.0.1")
    P = int(os.getenv("SERVER_PORT", 5000)) if True else 5000
    log.info("Starting server on http://%s:%s", H, P)
    app.run(debug=True, host=H, port=P)
