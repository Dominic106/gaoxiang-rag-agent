"""索引版本、artifact 指纹和健康检查的共享实现。"""  # 说明本模块只处理索引治理，不参与问答编排。

import hashlib  # 导入 hashlib，用来计算源文件、chunk 和索引 artifact 指纹。
import csv  # 导入 csv，用来从历史评估明细中恢复检索指标。
import json  # 导入 json，用来读写机器可审计的索引清单和健康报告。
import os  # 导入 os，用来读取当前 embedding 模型配置。
import pickle  # 导入 pickle，用来验证 BM25 文件是否可以正常反序列化。
import re  # 导入 re，用来从 Markdown 评估报告提取汇总数字。
from datetime import datetime  # 导入 datetime，用来记录清单和健康报告时间。
from pathlib import Path  # 导入 Path，用来处理项目路径。

from langchain_chroma import Chroma  # 导入 Chroma，用来只读检查本地向量集合，不执行 embedding。

from config import CHROMA_COLLECTION_NAME  # 从配置读取 Chroma 集合名。
from config import CHROMA_DIR  # 从配置读取 Chroma 目录。
from config import CHUNK_METADATA_SCHEMA_VERSION  # 从配置读取 chunk 元数据版本。
from config import CHUNK_OVERLAP  # 从配置读取 chunk 重叠长度。
from config import CHUNK_SEPARATORS  # 从配置读取切分边界。
from config import CHUNK_SIZE  # 从配置读取 chunk 长度。
from config import CHUNKS_JSONL  # 从配置读取 chunk 文件路径。
from config import COLLECTION_NAME  # 从配置读取教材集合名。
from config import EMBEDDING_PROVIDER  # 从配置读取 embedding 提供方。
from config import INDEX_MANIFEST_SCHEMA_VERSION  # 从配置读取清单结构版本。
from config import INDEX_ROOT  # 从配置读取索引目录。
from config import MANIFEST_JSON  # 从配置读取清单路径。
from config import BM25_PICKLE  # 从配置读取 BM25 文件路径。
from config import OUTPUT_ROOT  # 从配置读取评估报告输出目录。
from config import SOURCE_ROOT  # 从配置读取源 Word 目录。


def sha256_file(path: Path) -> str:  # 定义文件 SHA256 计算函数，用于识别源文档和 artifact 是否变化。
    digest = hashlib.sha256()  # 创建 SHA256 累加器。
    with path.open("rb") as file:  # 以二进制方式打开文件，避免文本编码影响指纹。
        for block in iter(lambda: file.read(1024 * 1024), b""):  # 按 1 MB 分块读取，避免一次性占用过多内存。
            digest.update(block)  # 把当前块加入摘要。
    return digest.hexdigest()  # 返回完整十六进制摘要。


def canonical_hash(value: object) -> str:  # 定义 JSON 规范化哈希函数，保证字段顺序不影响版本指纹。
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # 用稳定格式序列化对象。
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()  # 返回规范化对象的 SHA256。


def embedding_snapshot() -> dict:  # 定义当前 embedding 配置快照函数，不读取或记录 API Key。
    if EMBEDDING_PROVIDER == "doubao":  # 如果当前使用豆包 embedding。
        model = os.environ.get("DOUBAO_EMBEDDING_MODEL", "doubao-embedding-text-240715").strip()  # 读取豆包模型名。
        base_url = os.environ.get("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()  # 读取服务地址用于识别接口版本。
    else:  # 如果当前使用其他兼容 embedding 提供方。
        model = "text-embedding-3-large"  # 与 retrieval_resources 中的 OpenAI 默认模型保持一致。
        base_url = "openai-compatible"  # 不记录用户的自定义密钥或完整私有地址。
    return {"provider": EMBEDDING_PROVIDER, "model": model, "base_url_kind": "multimodal" if "multimodal" in base_url or "vision" in model else "openai-compatible"}  # 返回不含敏感信息的 embedding 快照。


def source_inventory() -> list[dict]:  # 定义扫描源 Word 文件并建立清单的函数。
    files = []  # 准备保存每个源文件的可审计信息。
    for path in sorted(SOURCE_ROOT.rglob("*.docx")):  # 按稳定路径顺序遍历全部 Word 文件。
        relative = str(path.relative_to(SOURCE_ROOT))  # 计算相对于教材根目录的路径。
        files.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})  # 保存路径、大小和内容指纹。
    return files  # 返回完整源文件清单。


def source_inventory_hash(files: list[dict]) -> str:  # 定义源文件清单总指纹函数。
    return canonical_hash(files)  # 对排序后的文件清单做规范化哈希。


def artifact_info(path: Path) -> dict:  # 定义索引 artifact 信息函数。
    if not path.exists():  # 如果 artifact 不存在。
        return {"path": str(path), "exists": False}  # 返回明确的缺失状态，不抛出模糊异常。
    if path.is_file():  # 如果 artifact 是普通文件。
        return {"path": str(path), "exists": True, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}  # 保存文件大小和指纹。
    files = sorted(item for item in path.rglob("*") if item.is_file())  # 如果是目录，收集内部文件路径和大小。
    directory_fingerprint = canonical_hash([{ "relative_path": str(item.relative_to(path)), "size_bytes": item.stat().st_size } for item in files])  # 目录只记录路径和大小，避免健康检查重复读取大型向量文件。
    return {"path": str(path), "exists": True, "file_count": len(files), "fingerprint": directory_fingerprint}  # 返回目录 artifact 摘要。


def build_index_version(source_hash: str) -> str:  # 定义确定性索引版本函数，同一内容和配置生成同一版本。
    version_input = {"collection": COLLECTION_NAME, "source_hash": source_hash, "chunk_metadata_schema": CHUNK_METADATA_SCHEMA_VERSION, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "separators": list(CHUNK_SEPARATORS), "embedding": embedding_snapshot()}  # 汇总会影响索引语义的关键因素。
    return f"idx-{canonical_hash(version_input)[:16]}"  # 使用短指纹作为人类可读索引版本号。


def create_manifest(doc_count: int, chunk_count: int) -> dict:  # 定义从当前源目录和配置创建索引清单的函数。
    files = source_inventory()  # 扫描当前教材源文件。
    source_hash = source_inventory_hash(files)  # 计算源文件总指纹。
    return {"manifest_schema_version": INDEX_MANIFEST_SCHEMA_VERSION, "chunk_metadata_schema_version": CHUNK_METADATA_SCHEMA_VERSION, "collection": COLLECTION_NAME, "source_version": COLLECTION_NAME, "source_root": str(SOURCE_ROOT), "source_files": files, "source_inventory_hash": source_hash, "index_version": build_index_version(source_hash), "generated_at": datetime.now().isoformat(timespec="seconds"), "docx_files": doc_count, "chunks": chunk_count, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "chunk_separators": list(CHUNK_SEPARATORS), "embedding": embedding_snapshot(), "artifacts": {}, "build_status": {"chunks": False, "bm25": False, "chroma": False}}  # 返回完整清单骨架。


def write_manifest(manifest: dict) -> None:  # 定义写入索引清单的函数。
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")  # 用可读 JSON 保存清单。


def finalize_manifest(vector_ready: bool) -> dict:  # 定义 BM25/Chroma 建库后补齐 artifact 状态的函数。
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))  # 读取第一阶段生成的清单骨架。
    manifest["artifacts"] = {"chunks_jsonl": artifact_info(CHUNKS_JSONL), "bm25_pickle": artifact_info(BM25_PICKLE), "chroma_directory": artifact_info(CHROMA_DIR)}  # 写入三个核心 artifact 摘要。
    manifest["build_status"] = {"chunks": CHUNKS_JSONL.exists(), "bm25": BM25_PICKLE.exists(), "chroma": bool(vector_ready and CHROMA_DIR.exists())}  # 根据实际结果记录每个索引是否就绪。
    manifest["index_ready"] = all(manifest["build_status"].values())  # 只有三类 artifact 都存在才标记为可查询。
    manifest["finalized_at"] = datetime.now().isoformat(timespec="seconds")  # 记录最后完成索引构建的时间。
    write_manifest(manifest)  # 保存更新后的清单。
    return manifest  # 返回最终清单供命令行打印。


def load_metric_summary(path: Path) -> dict:  # 定义评估指标读取函数，兼容 JSON、CSV 和 Markdown 三种历史报告格式。
    if path.suffix.lower() == ".json":  # 如果输入是机器可读 JSON 报告。
        payload = json.loads(path.read_text(encoding="utf-8"))  # 读取 JSON 内容。
        summary = payload.get("summary", payload)  # 兼容完整报告和直接保存汇总字典两种格式。
        return {key: value for key, value in summary.items() if isinstance(value, (int, float))}  # 只返回可比较的数字指标。
    if path.suffix.lower() == ".csv":  # 如果输入是逐题 CSV 明细。
        with path.open("r", newline="", encoding="utf-8") as file:  # 打开 CSV 文件。
            rows = list(csv.DictReader(file))  # 读取所有评估行。
        positive = [row for row in rows if row.get("should_refuse", "False").lower() != "true"]  # 排除拒答题，避免把拒答混入召回率。
        summary = {"total": len(rows), "positive_total": len(positive), "negative_total": len(rows) - len(positive)}  # 初始化通用数量指标。
        for prefix in ["bm25", "vector", "hybrid"] if rows and "hybrid_hit_rank" in rows[0] else ["bm25"]:  # 根据 CSV 字段判断评估类型。
            ranks = [int(row.get(f"{prefix}_hit_rank", row.get("hit_rank", "0")) or 0) for row in positive]  # 读取每题命中排名。
            summary[f"{prefix}_top1"] = sum(rank == 1 for rank in ranks)  # 统计 Top1 命中数量。
            summary[f"{prefix}_top5"] = sum(rank > 0 for rank in ranks)  # 统计 Top5 命中数量。
        return summary  # 返回从 CSV 恢复出的数字指标。
    text = path.read_text(encoding="utf-8")  # 读取 Markdown 报告文本。
    summary = {}  # 准备保存 Markdown 中识别出的数字。
    patterns = {"total": r"问题数：\s*(\d+)", "positive_total": r"正向检索题：\s*(\d+)", "negative_total": r"拒答题：\s*(\d+)", "bm25_top1": r"BM25[^\n]*Top1[^：:]*[：:]\s*(\d+)\s*/", "bm25_top5": r"BM25[^\n]*Top1/Top5[^：:]*：\s*\d+\s*/\d+，\s*(\d+)\s*/", "vector_top1": r"Vector[^\n]*Top1[^：:]*[：:]\s*(\d+)\s*/", "vector_top5": r"Vector[^\n]*Top1/Top5[^：:]*：\s*\d+\s*/\d+，\s*(\d+)\s*/", "hybrid_top1": r"Hybrid[^\n]*Top1[^：:]*[：:]\s*(\d+)\s*/", "hybrid_top5": r"Hybrid[^\n]*Top1/Top5[^：:]*：\s*\d+\s*/\d+，\s*(\d+)\s*/"}  # 定义两类报告的汇总行模式。
    for key, pattern in patterns.items():  # 遍历所有数字指标模式。
        match = re.search(pattern, text)  # 在报告中查找当前指标。
        if match:  # 如果识别到当前指标。
            summary[key] = int(match.group(1))  # 保存为整数，供前后版本比较。
    return summary  # 返回从 Markdown 恢复出的指标。


def compare_metric_summaries(before: dict, after: dict) -> dict:  # 定义前后评估指标比较函数。
    keys = sorted(set(before) & set(after))  # 只比较两个报告都存在的指标。
    comparison = {key: {"before": before[key], "after": after[key], "delta": after[key] - before[key]} for key in keys}  # 计算每个共同指标的变化。
    regressions = [key for key in keys if ("top1" in key or "top5" in key) and after[key] < before[key]]  # 检索命中数下降视为回归。
    return {"passed": bool(keys) and not regressions, "compared_keys": keys, "metrics": comparison, "regressions": regressions}  # 返回可审计的比较结果。


def write_metric_comparison(before_path: Path, after_path: Path) -> tuple[str, str]:  # 定义保存评估前后比较报告的函数。
    before = load_metric_summary(before_path)  # 读取重建前指标。
    after = load_metric_summary(after_path)  # 读取重建后指标。
    result = compare_metric_summaries(before, after)  # 执行指标比较。
    result.update({"before_report": str(before_path), "after_report": str(after_path), "index_version": json.loads(MANIFEST_JSON.read_text(encoding="utf-8")).get("index_version", "")})  # 补充报告来源和当前索引版本。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成唯一报告时间戳。
    json_path = OUTPUT_ROOT / f"index_metric_compare_{safe_time}.json"  # 定义 JSON 比较报告路径。
    md_path = OUTPUT_ROOT / f"index_metric_compare_{safe_time}.md"  # 定义 Markdown 比较报告路径。
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存机器可读比较结果。
    lines = ["# 索引重建前后指标比较", "", f"- 状态：{'通过' if result['passed'] else '发现回归'}", f"- 重建前：{before_path}", f"- 重建后：{after_path}", f"- 索引版本：{result['index_version']}", "", "| 指标 | 重建前 | 重建后 | 变化 |", "| --- | ---: | ---: | ---: |"]  # 初始化 Markdown 表格。
    for key in result["compared_keys"]:  # 遍历共同指标。
        metric = result["metrics"][key]  # 取当前指标的前后数值。
        lines.append(f"| {key} | {metric['before']} | {metric['after']} | {metric['delta']:+d} |")  # 写入指标变化。
    if result["regressions"]:  # 如果发现命中指标下降。
        lines.extend(["", "回归指标：" + ", ".join(result["regressions"])])  # 在报告中明确列出回归项。
    md_path.write_text("\n".join(lines), encoding="utf-8")  # 保存 Markdown 比较报告。
    return str(json_path), str(md_path)  # 返回两个比较报告路径。


def _check(name: str, passed: bool, detail: str, severity: str = "error") -> dict:  # 定义统一健康检查结果格式。
    return {"name": name, "passed": passed, "detail": detail, "severity": severity}  # 返回可写入 JSON/Markdown 的检查项。


def run_health_check() -> dict:  # 定义完整索引健康检查函数。
    checks = []  # 准备保存所有检查结果。
    if not MANIFEST_JSON.exists():  # 先检查清单是否存在。
        checks.append(_check("manifest_exists", False, f"清单不存在：{MANIFEST_JSON}"))  # 没有清单时后续无法判断版本。
        return {"passed": False, "checked_at": datetime.now().isoformat(timespec="seconds"), "checks": checks}  # 直接返回明确失败。
    try:  # 保护清单 JSON 解析。
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))  # 读取清单。
    except (OSError, json.JSONDecodeError) as exc:  # 捕获文件读取或 JSON 损坏。
        checks.append(_check("manifest_parse", False, f"清单无法解析：{type(exc).__name__}"))  # 记录清单损坏。
        return {"passed": False, "checked_at": datetime.now().isoformat(timespec="seconds"), "checks": checks}  # 终止后续依赖检查。
    checks.append(_check("manifest_schema", manifest.get("manifest_schema_version") == INDEX_MANIFEST_SCHEMA_VERSION, f"清单版本={manifest.get('manifest_schema_version')}，运行时需要={INDEX_MANIFEST_SCHEMA_VERSION}"))  # 检查清单结构版本。
    checks.append(_check("collection", manifest.get("collection") == COLLECTION_NAME, f"清单={manifest.get('collection')}，运行时={COLLECTION_NAME}"))  # 检查教材集合一致性。
    current_files = source_inventory()  # 重新扫描源文件，识别源教材是否发生变化。
    current_source_hash = source_inventory_hash(current_files)  # 计算当前源文件总指纹。
    checks.append(_check("source_inventory", current_source_hash == manifest.get("source_inventory_hash"), f"清单={manifest.get('source_inventory_hash')}，当前={current_source_hash}"))  # 检查源文档是否与索引版本一致。
    checks.append(_check("chunk_config", manifest.get("chunk_size") == CHUNK_SIZE and manifest.get("chunk_overlap") == CHUNK_OVERLAP and manifest.get("chunk_separators") == list(CHUNK_SEPARATORS), "chunk_size、overlap 和 separators 与运行时一致" if manifest.get("chunk_size") == CHUNK_SIZE and manifest.get("chunk_overlap") == CHUNK_OVERLAP and manifest.get("chunk_separators") == list(CHUNK_SEPARATORS) else "chunk 切分配置与运行时不一致"))  # 检查切分配置一致性。
    current_embedding = embedding_snapshot()  # 读取当前 embedding 提供方和模型快照。
    checks.append(_check("embedding_config", manifest.get("embedding") == current_embedding, f"清单={manifest.get('embedding')}，当前={current_embedding}"))  # 检查向量索引的 embedding 配置没有漂移。
    try:  # 保护 chunks JSONL 完整性检查。
        rows = [json.loads(line) for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]  # 读取所有 chunk 记录。
        ids = [row.get("metadata", {}).get("chunk_id") for row in rows]  # 提取 chunk ID。
        required = all(row.get("metadata", {}).get(field) is not None for row in rows for field in ["source_file_id", "section_chunk_index", "section_chunk_count", "source_char_start", "source_char_end", "source_anchor"])  # 检查第三阶段定位字段是否齐全。
        chunks_ok = len(rows) == manifest.get("chunks") and len(ids) == len(set(ids)) and required and artifact_info(CHUNKS_JSONL).get("sha256") == manifest.get("artifacts", {}).get("chunks_jsonl", {}).get("sha256")  # 综合检查 chunk 数量、唯一性、字段和文件指纹。
        checks.append(_check("chunks_integrity", chunks_ok, f"记录={len(rows)}，清单={manifest.get('chunks')}，定位字段={'完整' if required else '缺失'}"))  # 记录 chunk 健康状态。
    except (OSError, json.JSONDecodeError, TypeError):  # 捕获文件缺失、JSON 损坏和字段类型异常。
        rows = []  # 为后续检查提供空集合。
        checks.append(_check("chunks_integrity", False, "chunks.jsonl 缺失、损坏或字段无法读取"))  # 记录 chunk 失败。
    try:  # 保护 BM25 反序列化和数量检查。
        with BM25_PICKLE.open("rb") as file:  # 打开 BM25 artifact。
            retriever = pickle.load(file)  # 尝试反序列化。
        bm25_ids = [doc.metadata.get("chunk_id") for doc in retriever.docs]  # 提取 BM25 文档 ID。
        bm25_ok = len(bm25_ids) == manifest.get("chunks") and set(bm25_ids) == {row.get("metadata", {}).get("chunk_id") for row in rows} and artifact_info(BM25_PICKLE).get("sha256") == manifest.get("artifacts", {}).get("bm25_pickle", {}).get("sha256")  # 检查 BM25 是否覆盖全部 chunk 且文件未被替换。
        checks.append(_check("bm25_integrity", bm25_ok, f"BM25 记录={len(bm25_ids)}，期望={manifest.get('chunks')}"))  # 记录 BM25 状态。
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, TypeError):  # 捕获 BM25 缺失或损坏。
        checks.append(_check("bm25_integrity", False, "BM25 文件缺失或无法反序列化"))  # 记录 BM25 失败。
    try:  # 保护 Chroma 本地元数据检查，不创建 embedding 客户端。
        vector_store = Chroma(collection_name=CHROMA_COLLECTION_NAME, persist_directory=str(CHROMA_DIR))  # 只读打开 Chroma 集合。
        count = vector_store._collection.count()  # 读取集合记录数，不发起向量请求。
        metadata_rows = vector_store._collection.get(limit=min(5, max(1, count))).get("metadatas", []) or [] if count else []  # 读取少量元数据样本。
        metadata_ok = bool(metadata_rows) and all(item.get("source_file_id") and item.get("source_char_start") is not None for item in metadata_rows)  # 检查向量元数据同步了定位字段。
        chroma_fingerprint_ok = artifact_info(CHROMA_DIR).get("fingerprint") == manifest.get("artifacts", {}).get("chroma_directory", {}).get("fingerprint")  # 比较目录内文件路径和大小，识别向量目录被替换或部分写入。
        checks.append(_check("chroma_integrity", count == manifest.get("chunks") and metadata_ok and chroma_fingerprint_ok, f"Chroma 记录={count}，期望={manifest.get('chunks')}，定位元数据={'完整' if metadata_ok else '缺失'}，目录指纹={'一致' if chroma_fingerprint_ok else '不一致'}"))  # 记录 Chroma 状态。
    except Exception as exc:  # 捕获 Chroma 数据库缺失、损坏或集合不存在。  # noqa: BLE001
        checks.append(_check("chroma_integrity", False, f"Chroma 无法读取：{type(exc).__name__}"))  # 记录向量库失败。
    status_ok = manifest.get("index_ready") is True and all(manifest.get("build_status", {}).values())  # 检查清单是否明确标记三个 artifact 都已就绪。
    checks.append(_check("build_status", status_ok, f"build_status={manifest.get('build_status')}，index_ready={manifest.get('index_ready')}"))  # 记录构建状态。
    passed = all(item["passed"] for item in checks)  # 所有检查通过才允许继续使用当前索引。
    return {"passed": passed, "checked_at": datetime.now().isoformat(timespec="seconds"), "index_version": manifest.get("index_version", ""), "manifest": str(MANIFEST_JSON), "checks": checks}  # 返回完整健康结果。


def write_health_report(result: dict) -> tuple[str, str]:  # 定义保存健康检查 JSON/Markdown 报告的函数。
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)  # 确保索引目录存在。
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成唯一报告时间戳。
    json_path = INDEX_ROOT / f"index_health_{safe_time}.json"  # 定义 JSON 报告路径。
    md_path = INDEX_ROOT / f"index_health_{safe_time}.md"  # 定义 Markdown 报告路径。
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存机器可读报告。
    lines = ["# 索引健康检查报告", "", f"- 状态：{'通过' if result.get('passed') else '失败'}", f"- 索引版本：{result.get('index_version', '未知')}", f"- 检查时间：{result.get('checked_at', '')}", "", "| 检查项 | 状态 | 详情 |", "| --- | --- | --- |"]  # 初始化 Markdown 报告。
    for item in result.get("checks", []):  # 遍历检查项。
        detail = str(item.get("detail", "")).replace("|", "\\|")  # 转义 Markdown 表格符号。
        lines.append(f"| {item.get('name')} | {'通过' if item.get('passed') else '失败'} | {detail} |")  # 写入检查结果。
    md_path.write_text("\n".join(lines), encoding="utf-8")  # 保存 Markdown 报告。
    return str(json_path), str(md_path)  # 返回两个报告路径。
