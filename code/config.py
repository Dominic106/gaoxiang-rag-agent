from pathlib import Path  # 导入 Path，用它来安全地拼接 macOS 文件路径。
import os  # 导入 os，用来读取 .env 里加载后的环境变量。

import env_loader  # 导入 env_loader，确保 .env 配置在 config 初始化时被加载。  # noqa: F401


def _read_positive_float(name: str, default: str) -> float:  # 定义读取正浮点配置的函数，避免错误配置直接变成难懂的 ValueError。
    raw_value = os.environ.get(name, default).strip()  # 读取环境变量并去掉首尾空白。
    try:  # 保护配置转换过程。
        value = float(raw_value)  # 把配置转换成浮点数。
    except ValueError as exc:  # 捕获无法转换的配置值。
        raise RuntimeError(f"{name} 必须是正数，当前值为：{raw_value!r}") from exc  # 抛出带配置名的可诊断错误。
    if value <= 0:  # 检查超时配置是否真的大于零。
        raise RuntimeError(f"{name} 必须大于 0，当前值为：{raw_value!r}")  # 拒绝会导致请求立即失败的超时值。
    return value  # 返回经过校验的超时数值。


def _read_non_negative_int(name: str, default: str) -> int:  # 定义读取非负整数配置的函数，用于重试次数和日志轮转参数。
    raw_value = os.environ.get(name, default).strip()  # 读取环境变量并去掉首尾空白。
    try:  # 保护配置转换过程。
        value = int(raw_value)  # 把配置转换成整数。
    except ValueError as exc:  # 捕获无法转换的配置值。
        raise RuntimeError(f"{name} 必须是非负整数，当前值为：{raw_value!r}") from exc  # 抛出带配置名的可诊断错误。
    if value < 0:  # 检查重试或保留数量不能为负数。
        raise RuntimeError(f"{name} 必须大于等于 0，当前值为：{raw_value!r}")  # 拒绝无意义的负配置。
    return value  # 返回经过校验的整数。


def _read_positive_int(name: str, default: str) -> int:  # 定义读取正整数配置的函数，用于日志文件大小等必须为整数的参数。
    value = _read_non_negative_int(name, default)  # 先复用非负整数的格式和范围校验。
    if value == 0:  # 检查日志文件大小不能为零。
        raise RuntimeError(f"{name} 必须大于 0，当前值为：{value}")  # 拒绝无法轮转日志的配置。
    return value  # 返回经过校验的正整数。

PROJECT_ROOT = Path(os.environ.get("RAG_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()  # 默认按当前代码位置推断项目根目录，也支持部署时显式覆盖。

SOURCE_ROOT = PROJECT_ROOT / "source_docs" / "信息系统项目管理师辅导教程第3版全版"  # 定义复制后的教材 Word 源文档目录。

INDEX_ROOT = PROJECT_ROOT / "indexes"  # 定义所有索引文件的存放目录。

OUTPUT_ROOT = PROJECT_ROOT / "outputs"  # 定义查询结果、调试报告等输出文件的存放目录。

LOG_ROOT = PROJECT_ROOT / "logs"  # 定义运行日志目录，专门保存异常和性能诊断信息。

LOG_FILE = LOG_ROOT / "rag.log"  # 定义主运行日志文件路径。

CHUNKS_JSONL = INDEX_ROOT / "chunks.jsonl"  # 定义切分后的 chunk 明细文件，一行一个 JSON，方便审计和重建索引。

MANIFEST_JSON = INDEX_ROOT / "manifest.json"  # 定义本次建库的统计清单文件。

CHROMA_DIR = INDEX_ROOT / "chroma"  # 定义 Chroma 向量数据库的本地持久化目录。

BM25_PICKLE = INDEX_ROOT / "bm25_retriever.pkl"  # 定义 BM25 关键词检索器的本地保存文件。

EMBEDDING_CACHE_JSONL = INDEX_ROOT / "embedding_query_cache.jsonl"  # 定义查询 embedding 缓存文件，避免同一个问题反复花 API token。

COLLECTION_NAME = "信息系统项目管理师辅导教程第3版全版"  # 定义知识库名称，后续检索和元数据都会使用它。

CHROMA_COLLECTION_NAME = "xg_rag_book"  # 定义 Chroma 内部集合名，Chroma 新版本要求英文数字等安全字符。

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "doubao").strip().lower()  # 定义当前使用的 embedding 提供方。

CHUNK_SIZE = 900  # 定义每个 chunk 的目标长度，中文教材建议 600 到 1000 字之间。

CHUNK_OVERLAP = 140  # 定义相邻 chunk 的重叠长度，避免知识点刚好被切断。

CHUNK_SEPARATORS = ("\n\n", "\n", "。", "；", "，", " ", "")  # 固定切分边界并纳入索引版本指纹，避免只改切分规则却复用旧索引。

INDEX_MANIFEST_SCHEMA_VERSION = 2  # 定义索引清单结构版本，清单字段变化时可被健康检查识别。

CHUNK_METADATA_SCHEMA_VERSION = 2  # 定义 chunk 元数据结构版本，定位字段变化时强制重新检查索引。

TOP_K_VECTOR = 12  # 定义向量检索初筛返回数量，先多拿一些，后面再合并排序。

TOP_K_KEYWORD = 12  # 定义关键词检索初筛返回数量，用来补强固定术语命中。

FINAL_CONTEXTS = 6  # 定义最终送给大模型的原文片段数量，太多会稀释重点。

MIN_EVIDENCE_SCORE = 6  # 定义最低证据分，低于这个分数就认为“检索证据不足”，禁止模型硬答。

MAX_RETRIEVAL_ATTEMPTS = 3  # 定义最多检索自救次数，避免查不到时无限循环和浪费 token。

SIMPLE_CONTEXTS = 4  # 定义简单问题最多送入模型的片段数，兼顾多项服务模式和跨 chunk 定义的完整性。

MEDIUM_CONTEXTS = 5  # 定义中等置信度问题最多送入模型的片段数。

MAX_CONTEXT_CHARS = 720  # 定义每个引用片段送入模型的最大字符数，补回定义和流程起始段，同时控制 token 成本。

DEEPSEEK_TEMPERATURE = 0  # 定义回答模型温度，0 表示最保守、最稳定。

DEEPSEEK_MAX_TOKENS = 1200  # 定义单次回答最大输出长度，避免模型把答案写得过长。

DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()  # 定义回答请求是否启用 DeepSeek 思考模式，备考回答默认关闭以避免思考 token 挤占正文预算。
if DEEPSEEK_THINKING not in {"enabled", "disabled"}:  # 检查思考模式配置只能使用接口支持的两个值。
    raise RuntimeError("DEEPSEEK_THINKING 必须是 enabled 或 disabled。")  # 提前阻止拼写错误配置进入真实请求。

DEEPSEEK_TIMEOUT_SECONDS = _read_positive_float("DEEPSEEK_TIMEOUT_SECONDS", "60")  # 定义 DeepSeek 单次请求的连接和读取超时秒数。

DEEPSEEK_MAX_RETRIES = _read_non_negative_int("DEEPSEEK_MAX_RETRIES", "2")  # 定义 DeepSeek SDK 对可重试网络错误的最大重试次数。

DOUBAO_TIMEOUT_SECONDS = _read_positive_float("DOUBAO_TIMEOUT_SECONDS", "60")  # 定义豆包 embedding 单次 HTTP 请求超时秒数。

DOUBAO_MAX_RETRIES = _read_non_negative_int("DOUBAO_MAX_RETRIES", "2")  # 定义豆包 OpenAI 兼容 embedding 客户端的最大重试次数。

REQUEST_DEADLINE_SECONDS = _read_positive_float("REQUEST_DEADLINE_SECONDS", "180")  # 定义一条完整用户查询的总截止时间，覆盖多问题、检索自救和引用修复。

CIRCUIT_FAILURE_THRESHOLD = _read_positive_int("CIRCUIT_FAILURE_THRESHOLD", "3")  # 定义同一进程内连续可重试失败多少次后打开熔断。

CIRCUIT_RECOVERY_SECONDS = _read_positive_float("CIRCUIT_RECOVERY_SECONDS", "30")  # 定义熔断打开后等待多久才允许一次恢复探测请求。

LOG_MAX_BYTES = _read_positive_int("LOG_MAX_BYTES", str(5 * 1024 * 1024))  # 定义单个日志文件最大字节数，避免日志无限增长。

LOG_BACKUP_COUNT = _read_non_negative_int("LOG_BACKUP_COUNT", "3")  # 定义日志轮转后保留的历史文件数量。

SESSION_ROOT = PROJECT_ROOT / "sessions"  # 定义会话目录，用来保存每次学习过程的问题和答案。

MEMORY_ROOT = PROJECT_ROOT / "memory"  # 定义长期记忆目录，用来保存学习画像、错题本和问题历史。

EVAL_SAMPLES_PATH = PROJECT_ROOT / "notes" / "eval_questions_v1.json"  # 定义当前正式检索评估集路径，避免只用少量样例判断质量。

BM25_RANK_WEIGHT = 1.25  # 定义 BM25 排名得分权重，固定考试术语通常更适合关键词精确命中。

VECTOR_RANK_WEIGHT = 1.0  # 定义向量排名得分权重，保留语义召回能力。

TERM_HIT_WEIGHT = 2.5  # 定义查询术语在标题或正文命中时的加分，帮助结果更贴近用户真正问的词。

SECTION_MATCH_WEIGHT = 12.0  # 定义完整小节标题出现在增强 query 中时的加分，优先保证教材明确指向的小节进入答案上下文。
