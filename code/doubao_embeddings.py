import hashlib  # 导入 hashlib，用来给 query 生成稳定的缓存 key。
import json  # 导入 json，用来读写 JSONL 格式的 embedding 缓存。
import os  # 导入 os，用来读取环境变量里的豆包配置。
import time  # 导入 time，用来在可重试请求之间做短暂退避。
from typing import Any  # 导入 Any，标注 OpenAI 兼容客户端和多模态 JSON 响应。

from langchain_core.embeddings import Embeddings  # 导入 LangChain Embeddings 基类，方便接入 Chroma。
from openai import OpenAI  # 导入 OpenAI SDK，因为火山方舟提供 OpenAI 兼容接口。
import requests  # 导入 requests，用来调用豆包多模态 embedding 专用接口。

from config import EMBEDDING_CACHE_JSONL  # 从配置读取查询 embedding 缓存文件路径。
from config import DOUBAO_MAX_RETRIES  # 从配置读取豆包 OpenAI 兼容客户端最大重试次数。
from config import DOUBAO_TIMEOUT_SECONDS  # 从配置读取豆包请求超时秒数。
from app_logging import get_logger  # 导入统一日志器，用来记录 embedding 请求和缓存异常。
from request_governance import CircuitOpenError  # 导入熔断异常，让 embedding 服务连续失败时停止撞击。
from request_governance import RequestDeadlineExceeded  # 导入总截止时间异常。
from request_governance import classify_exception  # 导入异常分类函数，区分可重试和不可重试错误。
from request_governance import ensure_request_budget  # 导入剩余时间检查函数。
from request_governance import estimate_tokens  # 导入 token 估算函数，记录 embedding 输入成本。
from request_governance import record_cache_hit  # 导入缓存命中统计函数。
from request_governance import tracked_service_call  # 导入外部调用追踪上下文。
from file_locks import locked_file  # 导入文件锁，保护 query embedding 缓存的并发读写。
import env_loader  # 导入 env_loader，确保 .env 已经被加载。  # noqa: F401


logger = get_logger(__name__)  # 创建当前模块日志器。


class DoubaoEmbeddings(Embeddings):  # 定义豆包 embedding 适配器，让它符合 LangChain 的 Embeddings 接口。
    def __init__(self) -> None:  # 定义初始化函数。
        self.api_key = os.environ.get("DOUBAO_API_KEY", "").strip()  # 从环境变量读取豆包 API Key。
        raw_base_url = os.environ.get("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()  # 读取火山方舟接口地址。
        self.model = os.environ.get("DOUBAO_EMBEDDING_MODEL", "doubao-embedding-text-240715").strip()  # 读取 embedding 模型名。
        self.client: Any = None  # 先声明兼容客户端，普通和多模态分支会分别赋值。
        if not self.api_key or self.api_key == "your_doubao_api_key_here":  # 检查 Key 是否还没填写。
            raise RuntimeError("请先在 code/.env 里填写 DOUBAO_API_KEY。")  # 抛出清晰错误，提醒用户补 Key。
        self.is_multimodal = "multimodal" in raw_base_url or "vision" in self.model  # 判断是否使用多模态 embedding 接口。
        if self.is_multimodal:  # 如果是多模态接口。
            root = raw_base_url.split("/embeddings", 1)[0].rstrip("/")  # 提取 API 根地址。
            self.multimodal_url = root + "/embeddings/multimodal"  # 拼出多模态 embedding endpoint。
            self.client = None  # 多模态模式不使用 OpenAI SDK 客户端。
        else:  # 如果是普通文本 OpenAI 兼容 embedding 接口。
            self.base_url = raw_base_url.split("/embeddings", 1)[0].rstrip("/")  # 截回 OpenAI 兼容根地址。
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=DOUBAO_TIMEOUT_SECONDS, max_retries=DOUBAO_MAX_RETRIES)  # 创建带超时和有限重试的 OpenAI 兼容客户端。
        self.query_cache = self._load_query_cache()  # 加载本地 query embedding 缓存，重复问题直接复用向量。

    def _cache_key(self, text: str) -> str:  # 定义生成缓存 key 的函数。
        normalized = " ".join((text or "").split())  # 把连续空白压成单个空格，避免同一句因为空格不同而缓存 miss。
        raw_key = f"{self.model}\n{normalized}"  # 把模型名也放进 key，避免换模型后误用旧向量。
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()  # 返回 SHA256 哈希，适合做字典 key。

    def _load_query_cache(self) -> dict[str, list[float]]:  # 定义加载缓存文件的函数。
        cache: dict[str, list[float]] = {}  # 准备内存缓存字典。
        if not EMBEDDING_CACHE_JSONL.exists():  # 如果缓存文件还不存在。
            return cache  # 直接返回空缓存。
        try:  # 保护缓存文件读取，磁盘或权限异常不应该阻断整个查询流程。
            with locked_file(EMBEDDING_CACHE_JSONL.with_suffix(EMBEDDING_CACHE_JSONL.suffix + ".lock")), EMBEDDING_CACHE_JSONL.open("r", encoding="utf-8") as file:  # 读取时加锁，避免读到正在追加的半条记录。
                for line in file:  # 逐行读取缓存。
                    if not line.strip():  # 如果是空行。
                        continue  # 跳过空行。
                    try:  # 保护单条缓存解析，避免一条损坏记录阻断整个知识库。
                        item = json.loads(line)  # 解析一条缓存记录。
                    except json.JSONDecodeError:  # 如果缓存文件中有损坏 JSON。
                        logger.exception("Doubao embedding cache_invalid path=%s", EMBEDDING_CACHE_JSONL)  # 记录损坏文件和堆栈，继续加载其他缓存。
                        continue  # 跳过损坏记录，保留其余可用缓存。
                    try:  # 保护单条缓存字段读取，避免缺字段记录阻断整个缓存加载。
                        if item.get("model") == self.model:  # 只加载当前模型的缓存，避免混用不同维度向量。
                            cache[item["key"]] = item["embedding"]  # 把 key 和向量放入内存缓存。
                    except (KeyError, TypeError):  # 捕获缺少 key 或记录不是对象的异常。
                        logger.exception("Doubao embedding cache_record_invalid path=%s", EMBEDDING_CACHE_JSONL)  # 记录损坏记录并继续读取其他缓存。
        except OSError:  # 如果缓存文件无法读取。
            logger.exception("Doubao embedding cache_read_failed path=%s", EMBEDDING_CACHE_JSONL)  # 记录读取异常，并让本次查询重新请求 embedding。
        return cache  # 返回缓存字典。

    def _save_query_cache_item(self, key: str, text: str, embedding: list[float]) -> None:  # 定义追加一条缓存的函数。
        record = {"key": key, "model": self.model, "text": text, "embedding": embedding}  # 组织缓存记录。
        try:  # 保护缓存写入，避免磁盘问题阻断已经成功的 embedding 请求。
            EMBEDDING_CACHE_JSONL.parent.mkdir(parents=True, exist_ok=True)  # 确保缓存目录存在。
            with locked_file(EMBEDDING_CACHE_JSONL.with_suffix(EMBEDDING_CACHE_JSONL.suffix + ".lock")), EMBEDDING_CACHE_JSONL.open("a", encoding="utf-8") as file:  # 锁住缓存追加，避免多进程同时写入交错 JSONL。
                file.write(json.dumps(record, ensure_ascii=False) + "\n")  # 写入一行 JSON，方便后续增量追加。
        except OSError:  # 如果缓存目录或磁盘写入失败。
            logger.exception("Doubao embedding cache_write_failed path=%s", EMBEDDING_CACHE_JSONL)  # 记录缓存写入异常。

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:  # 定义批量请求 embedding 的内部函数。
        if self.is_multimodal:  # 如果使用多模态 embedding 接口。
            return [self._embed_multimodal_text(text) for text in texts]  # 多模态接口逐条请求文本向量。
        request_id = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:12]  # 为批次生成不包含正文的请求指纹，便于关联日志。
        started = time.perf_counter()  # 记录普通 embedding 请求开始时间。
        logger.info("Doubao embedding_start request_id=%s model=%s batch_size=%d timeout_seconds=%s max_retries=%d", request_id, self.model, len(texts), DOUBAO_TIMEOUT_SECONDS, DOUBAO_MAX_RETRIES)  # 记录请求元数据，不记录 API Key 和正文。
        try:  # 保护 OpenAI 兼容 embedding 请求。
            with tracked_service_call("doubao", "embedding_batch", sum(estimate_tokens(text) for text in texts)) as tracker:  # 让普通 embedding 请求共享总截止时间、熔断器和成本统计。
                remaining = ensure_request_budget("doubao_embedding_batch")  # 计算本批请求的剩余总预算。
                client = self.client.with_options(timeout=min(DOUBAO_TIMEOUT_SECONDS, remaining)) if hasattr(self.client, "with_options") else self.client  # 优先把剩余预算传给支持 with_options 的客户端，离线替身继续使用原对象。
                response = client.embeddings.create(model=self.model, input=texts, encoding_format="float")  # 调用火山方舟 embeddings 接口。
                vectors = [item.embedding for item in response.data]  # 按返回顺序取出每条文本的向量。
                usage = getattr(response, "usage", None)  # 尝试读取兼容接口返回的真实 token 用量。
                tracker.set_usage(input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "total_tokens", None))  # embedding 没有 usage 时保留输入估算。
            logger.info("Doubao embedding_success request_id=%s model=%s batch_size=%d elapsed_ms=%s", request_id, self.model, len(texts), round((time.perf_counter() - started) * 1000, 2))  # 记录成功请求耗时和批量大小。
            return vectors  # 返回批量向量。
        except (RequestDeadlineExceeded, CircuitOpenError) as exc:  # 截止时间或熔断是治理层主动阻断。
            logger.warning("Doubao embedding_blocked request_id=%s model=%s failure_type=%s", request_id, self.model, type(exc).__name__)  # 记录稳定阻断类型。
            raise  # 保留治理异常，让上层保守降级。
        except Exception as exc:  # 捕获超时、鉴权、限流和响应格式异常。
            failure_kind, retryable = classify_exception(exc)  # 对异常分类，便于诊断是否应该重试。
            logger.exception("Doubao embedding_failed request_id=%s model=%s batch_size=%d error_type=%s failure_kind=%s retryable=%s", request_id, self.model, len(texts), type(exc).__name__, failure_kind, retryable)  # 记录异常堆栈和分类，不记录 API Key。
            raise RuntimeError(f"豆包 embedding 请求失败或超时，详情请查看 {request_id} 对应的 logs/rag.log。") from exc  # 抛出不泄露敏感信息的上层错误。

    def _embed_multimodal_text(self, text: str) -> list[float]:  # 定义多模态接口下的纯文本向量化函数。
        payload: dict[str, Any] = {"model": self.model, "input": [{"type": "text", "text": text}]}  # 按多模态接口要求组织文本输入。
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}  # 组织鉴权和 JSON 请求头。
        for attempt in range(1, DOUBAO_MAX_RETRIES + 2):  # 按配置执行首次请求加有限次数重试。
            started = time.perf_counter()  # 记录当前尝试开始时间。
            try:  # 保护多模态 HTTP 请求和响应解析。
                with tracked_service_call("doubao", "embedding_multimodal", estimate_tokens(text)) as tracker:  # 让每次多模态请求都进入统一熔断、截止时间和成本统计。
                    remaining = ensure_request_budget("doubao_embedding_multimodal")  # 计算当前尝试还剩多少时间。
                    response = requests.post(self.multimodal_url, headers=headers, json=payload, timeout=min(DOUBAO_TIMEOUT_SECONDS, remaining))  # 发送不能超过总预算的 HTTP 请求。  # nosec B113
                    response.raise_for_status()  # 如果接口返回错误，就抛出异常，方便分类和有限重试。
                    body: dict[str, Any] = response.json()  # 解析 JSON 响应。
                    vector = body["data"]["embedding"]  # 多模态接口的向量在 data.embedding 字段里。
                    tracker.set_usage(input_tokens=estimate_tokens(text))  # 多模态接口没有稳定 usage 时记录输入估算。
                logger.info("Doubao multimodal_embedding_success model=%s attempt=%d elapsed_ms=%s", self.model, attempt, round((time.perf_counter() - started) * 1000, 2))  # 记录成功尝试耗时。
                return vector  # 返回当前文本向量。
            except (RequestDeadlineExceeded, CircuitOpenError) as exc:  # 总预算耗尽或熔断时不再重试。
                logger.warning("Doubao multimodal_embedding_blocked model=%s attempt=%d failure_type=%s", self.model, attempt, type(exc).__name__)  # 记录稳定阻断类型。
                raise  # 保留治理异常。
            except requests.RequestException as exc:  # 捕获超时、连接和 HTTP 状态异常。
                failure_kind, retryable = classify_exception(exc)  # 根据状态码和异常类型判断是否允许重试。
                logger.warning("Doubao multimodal_embedding_failed model=%s attempt=%d/%d error_type=%s failure_kind=%s retryable=%s", self.model, attempt, DOUBAO_MAX_RETRIES + 1, type(exc).__name__, failure_kind, retryable)  # 记录分类结果，不记录请求体和密钥。
                if not retryable or attempt > DOUBAO_MAX_RETRIES:  # 不可重试或已达到上限时直接结束。
                    logger.exception("Doubao multimodal_embedding_failed model=%s error_type=%s", self.model, type(exc).__name__)  # 记录最终失败堆栈。
                    raise RuntimeError("豆包多模态 embedding 请求失败或超时，详情请查看 logs/rag.log。") from exc  # 抛出安全错误。
                time.sleep(min(2 ** (attempt - 1), 4, max(0.05, ensure_request_budget("doubao_embedding_backoff") / 2)))  # 使用不超过剩余预算的短暂指数退避。
            except (KeyError, TypeError, ValueError) as exc:  # 捕获响应结构或 JSON 内容异常。
                logger.exception("Doubao multimodal_embedding_response_invalid model=%s error_type=%s", self.model, type(exc).__name__)  # 记录响应格式异常。
                raise RuntimeError("豆包 embedding 返回内容格式异常，详情请查看 logs/rag.log。") from exc  # 响应格式错误不盲目重试。
        raise RuntimeError("豆包 embedding 请求未返回结果，详情请查看 logs/rag.log。")  # 理论上不会到达，作为静态兜底。

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # 实现 LangChain 要求的文档向量化方法。
        cleaned = [text if text is not None else "" for text in texts]  # 防御性处理 None，避免 API 报错。
        vectors: list[list[float]] = []  # 准备保存所有向量。
        batch_size = 32  # 设置批量大小，避免单次请求太大。
        for start in range(0, len(cleaned), batch_size):  # 按 batch_size 分批遍历文本。
            batch = cleaned[start : start + batch_size]  # 取出当前批次文本。
            vectors.extend(self._embed_batch(batch))  # 调用接口并追加向量结果。
            if self.is_multimodal and (len(vectors) % 25 == 0 or len(vectors) == len(cleaned)):  # 多模态逐条请求时输出进度。
                print(f"豆包 embedding 进度：{len(vectors)}/{len(cleaned)}", flush=True)  # 打印当前向量化进度。
        return vectors  # 返回所有文档向量。

    def embed_query(self, text: str) -> list[float]:  # 实现 LangChain 要求的查询向量化方法。
        key = self._cache_key(text)  # 为这条 query 生成稳定缓存 key。
        if key in self.query_cache:  # 如果本地缓存里已经有这条 query 的向量。
            record_cache_hit("doubao", "embedding_query", estimate_tokens(text))  # 记录一次没有产生网络调用的缓存命中。
            print("豆包 query embedding 命中缓存，未调用 API。", flush=True)  # 打印提示，方便确认省钱逻辑生效。
            return self.query_cache[key]  # 直接返回缓存向量，不再请求豆包。
        vector = self._embed_batch([text])[0]  # 缓存未命中时调用豆包 API 生成 query 向量。
        self.query_cache[key] = vector  # 把新向量写入内存缓存。
        self._save_query_cache_item(key, text, vector)  # 把新向量追加到本地 JSONL 缓存文件。
        print("豆包 query embedding 已写入缓存。", flush=True)  # 打印提示，方便用户知道这次产生了新缓存。
        return vector  # 返回新生成的 query 向量。


def make_doubao_embeddings() -> DoubaoEmbeddings:  # 定义工厂函数，方便其他脚本创建豆包 embedding。
    return DoubaoEmbeddings()  # 返回豆包 embedding 适配器实例。
