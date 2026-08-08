import os  # 导入 os，用来从环境变量读取 DeepSeek 的配置。
import hashlib  # 导入 hashlib，用来生成不包含敏感内容的请求指纹。
import time  # 导入 time，用来记录请求耗时。
from functools import lru_cache  # 导入 lru_cache，让同一进程复用 DeepSeek 客户端。

from openai import OpenAI  # 导入 OpenAI SDK，因为 DeepSeek 提供 OpenAI 兼容接口。

import env_loader  # 导入 env_loader，确保同目录下的 .env 会先被加载进环境变量。  # noqa: F401
from config import DEEPSEEK_MAX_TOKENS  # 从配置读取回答最大 token 数。
from config import DEEPSEEK_MAX_RETRIES  # 从配置读取 DeepSeek 可重试网络错误的次数。
from config import DEEPSEEK_TIMEOUT_SECONDS  # 从配置读取 DeepSeek 请求超时秒数。
from config import DEEPSEEK_TEMPERATURE  # 从配置读取回答温度。
from config import DEEPSEEK_THINKING  # 从配置读取是否启用 DeepSeek 思考模式。
from app_logging import get_logger  # 导入统一日志器，用来记录请求开始、成功和异常。
from request_governance import RequestDeadlineExceeded  # 导入总截止时间异常，保证它不会被普通模型错误吞掉。
from request_governance import CircuitOpenError  # 导入熔断异常，向上层报告服务被保护性阻断。
from request_governance import classify_exception  # 导入异常分类函数，日志中区分可重试和不可重试。
from request_governance import estimate_tokens  # 导入 token 估算函数，记录没有 usage 时的保守输入成本。
from request_governance import ensure_request_budget  # 导入剩余时间检查函数。
from request_governance import tracked_service_call  # 导入外部调用追踪上下文。


logger = get_logger(__name__)  # 创建当前模块日志器。


@lru_cache(maxsize=1)  # 一个进程只创建一个客户端，避免每个子问题重复建立连接对象。
def make_deepseek_client() -> OpenAI:  # 定义创建 DeepSeek 客户端的函数。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 从 .env 或系统环境变量读取 DeepSeek API Key。
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()  # 读取 DeepSeek 接口地址。
    if not api_key or api_key == "your_deepseek_api_key_here":  # 判断用户是否还没有填写真实 Key。
        raise RuntimeError("请先在 code/.env 里填写 DEEPSEEK_API_KEY。")  # 抛出清晰错误，避免后面出现难懂的鉴权失败。
    return OpenAI(api_key=api_key, base_url=base_url, timeout=DEEPSEEK_TIMEOUT_SECONDS, max_retries=DEEPSEEK_MAX_RETRIES)  # 创建带超时和有限重试的 OpenAI 兼容客户端。


def call_deepseek(prompt: str) -> str:  # 定义调用 DeepSeek 生成回答的函数。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()  # 从 .env 读取模型名，没填就用 DeepSeek 默认聊天模型名。
    request_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]  # 根据提示词生成短指纹，日志可关联但不会泄露正文。
    started = time.perf_counter()  # 记录请求开始时间。
    logger.info("DeepSeek request_start request_id=%s model=%s prompt_chars=%d timeout_seconds=%s max_retries=%d", request_id, model, len(prompt), DEEPSEEK_TIMEOUT_SECONDS, DEEPSEEK_MAX_RETRIES)  # 记录请求元数据，不记录 API Key 和完整提示词。
    try:  # 开始保护客户端创建和网络请求。
        with tracked_service_call("deepseek", "answer", estimate_tokens(prompt)) as tracker:  # 让回答请求共享总截止时间、熔断器和 token 统计。
            client = make_deepseek_client()  # 创建或复用带超时的 DeepSeek 客户端。
            remaining = ensure_request_budget("deepseek_answer")  # 计算本次调用还能使用的最长时间。
            timeout_client = client.with_options(timeout=min(DEEPSEEK_TIMEOUT_SECONDS, remaining)) if hasattr(client, "with_options") else client  # 优先把剩余预算传给支持 with_options 的 OpenAI 客户端，测试替身则继续使用原客户端。
            response = timeout_client.chat.completions.create(  # 调用聊天补全接口，让 DeepSeek 根据提示词生成答案。
            model=model,  # 指定使用哪个 DeepSeek 模型。
            messages=[  # 组织消息列表，这是聊天模型最常见的输入格式。
                {  # 第一条是 system 消息，用来规定模型的角色和边界。
                    "role": "system",  # system 表示系统级指令。
                    "content": "你是严谨的信息系统项目管理师考试辅导老师。只能依据用户提供的教材原文回答；如果原文没有直接依据，必须明确说知识库依据不足，不能凭常识或训练记忆补充。",  # 要求模型严格按证据回答。
                },  # system 消息结束。
                {  # 第二条是 user 消息，放入本次 RAG 拼好的完整提示词。
                    "role": "user",  # user 表示用户消息。
                    "content": prompt,  # 把检索到的原文和用户问题一起交给模型。
                },  # user 消息结束。
            ],  # 消息列表结束。
            temperature=DEEPSEEK_TEMPERATURE,  # temperature 越低越稳定，考试知识库更需要稳而不是发散。
            max_tokens=DEEPSEEK_MAX_TOKENS,  # 限制回答长度，既省成本，也避免模型把答案写得过长。
            extra_body={"thinking": {"type": DEEPSEEK_THINKING}},  # 显式关闭默认思考模式，避免思考内容耗尽结构化答案的输出预算。
            )  # DeepSeek 请求结束。
            usage = getattr(response, "usage", None)  # 尝试读取 DeepSeek 返回的真实 token 用量。
            tracker.set_usage(input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "completion_tokens", None))  # 优先记录真实 usage，没有时保留输入估算和输出默认值。
        content = response.choices[0].message.content  # 取出模型返回的正文。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)  # 计算请求耗时毫秒数。
        logger.info("DeepSeek request_success request_id=%s elapsed_ms=%s output_chars=%d", request_id, elapsed_ms, len(content or ""))  # 记录成功请求的耗时和输出长度。
        return content or ""  # 如果返回 None，就兜底成空字符串，避免后续拼接报错。
    except (RequestDeadlineExceeded, CircuitOpenError) as exc:  # 截止时间和熔断是治理层主动决定的结果。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)  # 计算治理阻断耗时。
        logger.warning("DeepSeek request_blocked request_id=%s elapsed_ms=%s failure_type=%s", request_id, elapsed_ms, type(exc).__name__)  # 记录稳定阻断类型，不输出问题正文。
        raise  # 保留治理异常，让主流程生成明确的保守降级提示。
    except Exception as exc:  # 捕获超时、连接、鉴权、响应格式等普通请求异常。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)  # 计算失败请求耗时。
        failure_kind, retryable = classify_exception(exc)  # 对异常分类，便于判断是否需要修复重试配置。
        logger.exception("DeepSeek request_failed request_id=%s elapsed_ms=%s error_type=%s failure_kind=%s retryable=%s", request_id, elapsed_ms, type(exc).__name__, failure_kind, retryable)  # 写入带堆栈的分类日志，但不写入密钥或完整提示词。
        raise RuntimeError(f"DeepSeek 请求失败或超时，详情请查看 {request_id} 对应的 logs/rag.log。") from exc  # 抛出不泄露敏感信息的上层错误。
