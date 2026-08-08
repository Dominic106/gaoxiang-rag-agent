"""RAG 核心性能基准。

默认只测试不产生 embedding API 调用的 BM25 和门控判断。
增加 --with-vector 后才执行向量延迟测试；运行前应确认查询 embedding 已经缓存。
"""

import argparse  # 导入 argparse，用来控制是否执行向量基准。
import json  # 导入 json，用来读取正式评估问题。
import statistics  # 导入 statistics，用来计算平均耗时。
import time  # 导入 time，用来记录每次检索耗时。
from datetime import datetime  # 导入 datetime，用来生成基准报告时间戳。

from config import EVAL_SAMPLES_PATH  # 从配置读取正式评估集路径。
from config import OUTPUT_ROOT  # 从配置读取输出目录。
from rag_evidence import score_evidence  # 导入证据评分函数，估算 BM25 能否直接挡住向量调用。
from retrieval_resources import get_bm25_retriever  # 导入缓存的全量 BM25 检索器。
from retrieval_resources import get_vector_store  # 导入缓存的 Chroma 向量库。


def percentile(values: list[float], ratio: float) -> float:  # 定义百分位耗时函数，观察长尾延迟。
    ordered = sorted(values)  # 从小到大排序所有耗时。
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * ratio)))  # 计算不越界的百分位索引。
    return ordered[index]  # 返回指定百分位耗时。


def load_questions() -> list[str]:  # 定义读取评估问题函数。
    samples = json.loads(EVAL_SAMPLES_PATH.read_text(encoding="utf-8"))  # 读取正式评估集 JSON。
    return [item["question"] for item in samples]  # 只保留问题文本用于性能测试。


def benchmark_bm25(questions: list[str]) -> dict:  # 定义 BM25 和首轮门控基准函数。
    retriever = get_bm25_retriever()  # 第一次获取 BM25，记录磁盘加载后的可复用对象。
    times: list[float] = []  # 准备保存每次关键词检索耗时。
    enough_count = 0  # 统计无需向量检索的题目数。
    for question in questions:  # 遍历所有正式评估问题。
        started = time.perf_counter()  # 记录当前 BM25 查询开始时间。
        docs = retriever.invoke(question)  # 执行一次低成本关键词检索。
        times.append((time.perf_counter() - started) * 1000)  # 记录毫秒级耗时。
        if score_evidence(question, docs) >= 6:  # 使用当前证据阈值判断是否可以跳过向量检索。
            enough_count += 1  # 记录 BM25 已经足够的题目。
    return {  # 返回 BM25 基准指标。
        "count": len(questions),  # 保存问题数量。
        "avg_ms": round(statistics.mean(times), 2),  # 保存平均耗时。
        "p95_ms": round(percentile(times, 0.95), 2),  # 保存 95 分位耗时。
        "bm25_enough_count": enough_count,  # 保存可直接使用 BM25 的题目数。
        "estimated_vector_skip_rate": round(enough_count / len(questions), 3) if questions else 0,  # 保存预计可跳过向量的比例。
    }  # 返回 BM25 指标。


def benchmark_vector(questions: list[str]) -> dict:  # 定义向量检索基准函数。
    vector_store = get_vector_store()  # 获取缓存的 Chroma 和 embedding 对象。
    times: list[float] = []  # 准备保存每次向量检索耗时。
    for question in questions:  # 遍历所有正式评估问题。
        started = time.perf_counter()  # 记录当前向量检索开始时间。
        vector_store.similarity_search(question, k=5)  # 执行一次向量检索。
        times.append((time.perf_counter() - started) * 1000)  # 记录毫秒级耗时。
    return {  # 返回向量基准指标。
        "count": len(questions),  # 保存问题数量。
        "avg_ms": round(statistics.mean(times), 2),  # 保存平均耗时。
        "p95_ms": round(percentile(times, 0.95), 2),  # 保存 95 分位耗时。
    }  # 返回向量指标。


def main() -> None:  # 定义性能基准主函数。
    parser = argparse.ArgumentParser(description="运行 RAG 核心性能基准")  # 创建命令行参数解析器。
    parser.add_argument("--with-vector", action="store_true", help="执行向量延迟测试，可能消耗未缓存的 embedding API")  # 增加可选向量测试开关。
    args = parser.parse_args()  # 解析命令行参数。
    questions = load_questions()  # 读取正式评估问题。
    bm25_result = benchmark_bm25(questions)  # 运行 BM25 和向量门控基准。
    vector_result = benchmark_vector(questions) if args.with_vector else {}  # 只有显式要求时才运行向量基准。
    cache_started = time.perf_counter()  # 记录缓存复用检查开始时间。
    bm25_same = get_bm25_retriever() is get_bm25_retriever()  # 检查 BM25 是否复用同一对象。
    cache_ms = (time.perf_counter() - cache_started) * 1000  # 记录缓存检查耗时。
    vector_text = f"- 向量平均耗时：{vector_result['avg_ms']} ms\n- 向量 P95 耗时：{vector_result['p95_ms']} ms" if vector_result else "- 未执行。需要明确增加 `--with-vector`，并确认查询 embedding 已经缓存。"  # 把可选向量指标提前格式化，避免 f-string 表达式中出现换行转义。
    report = f"""# RAG 核心性能基准报告

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 基准数据

- 评估问题数：{len(questions)}
- BM25 平均耗时：{bm25_result['avg_ms']} ms
- BM25 P95 耗时：{bm25_result['p95_ms']} ms
- BM25 证据已足够：{bm25_result['bm25_enough_count']}/{bm25_result['count']}
- 预计可跳过向量比例：{bm25_result['estimated_vector_skip_rate']:.1%}
- BM25 对象复用：{bm25_same}
- BM25 缓存检查耗时：{cache_ms:.2f} ms

## 向量基准

{vector_text}

## 结论

当前最有效的性能策略是：先执行 BM25，证据已经达到阈值时跳过向量；只有 BM25 不足时才调用一次向量检索。这样能够保留语义召回兜底，同时减少固定术语问题的 embedding API 消耗。
"""  # 组织 Markdown 性能报告。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成微秒级报告时间戳。
    report_path = OUTPUT_ROOT / f"performance_benchmark_{timestamp}.md"  # 拼出性能报告路径。
    report_path.write_text(report, encoding="utf-8")  # 写入性能报告。
    print(f"性能基准报告：{report_path}")  # 输出报告路径。
    print(json.dumps({"bm25": bm25_result, "vector": vector_result, "bm25_same": bm25_same}, ensure_ascii=False))  # 输出机器可读摘要。


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    main()  # 直接运行时执行性能基准。
