import csv  # 导入 csv，用来输出评估明细。
from datetime import datetime  # 导入 datetime，用来生成报告时间戳。
import json  # 导入 json，用来读取样例问题。
import pickle  # 导入 pickle，用来加载 BM25 检索器。

from langchain_chroma import Chroma  # 导入 Chroma，用来加载本地向量索引。

from config import BM25_PICKLE  # 从配置读取 BM25 索引路径。
from config import BM25_RANK_WEIGHT  # 从配置读取 BM25 排名权重。
from config import CHROMA_COLLECTION_NAME  # 从配置读取 Chroma 集合名。
from config import CHROMA_DIR  # 从配置读取 Chroma 索引目录。
from config import EVAL_SAMPLES_PATH  # 从配置读取正式评估集路径。
from config import OUTPUT_ROOT  # 从配置读取输出目录。
from config import VECTOR_RANK_WEIGHT  # 从配置读取向量排名权重。
from doubao_embeddings import make_doubao_embeddings  # 导入豆包 embedding，用来做向量查询。
from query_understanding import understand_query  # 导入问题理解器，用来生成增强 query。
from rag_tokenizers import tokenize_for_bm25  # 导入 BM25 分词函数，确保 pickle 加载正常。  # noqa: F401


SAMPLES_PATH = EVAL_SAMPLES_PATH  # 使用正式评估集检验 BM25、向量和混合检索。


def source_of(doc) -> str:  # 定义取文档来源路径的函数。
    return doc.metadata.get("relative_path", "")  # 返回相对路径，评估时用它判断章节命中。


def hit_rank(sources: list[str], expected_values: list[str]) -> int:  # 定义计算命中排名的函数。
    for index, source in enumerate(sources, start=1):  # 遍历来源列表。
        if any(expected in source for expected in expected_values):  # 如果来源包含任意一个合法章节关键词。
            return index  # 返回命中的排名。
    return 0  # 没命中就返回 0。


def merge_hybrid(bm25_docs: list, vector_docs: list, limit: int = 8) -> list:  # 定义混合检索合并函数。
    merged: dict[str, object] = {}  # 用 chunk_id 去重保存文档。
    scores: dict[str, float] = {}  # 保存每个 chunk 的融合分数。
    for rank, doc in enumerate(bm25_docs, start=1):  # 遍历 BM25 结果。
        chunk_id = doc.metadata["chunk_id"]  # 取 chunk_id。
        merged[chunk_id] = doc  # 保存文档。
        scores[chunk_id] = scores.get(chunk_id, 0) + (20 - rank) * BM25_RANK_WEIGHT  # BM25 排名分，固定术语略加权。
    for rank, doc in enumerate(vector_docs, start=1):  # 遍历向量结果。
        chunk_id = doc.metadata["chunk_id"]  # 取 chunk_id。
        merged[chunk_id] = doc  # 保存文档。
        scores[chunk_id] = scores.get(chunk_id, 0) + (20 - rank) * VECTOR_RANK_WEIGHT  # 向量排名分，保留语义召回能力。
    ordered = sorted(scores, key=lambda chunk_id: scores.get(chunk_id, 0.0), reverse=True)  # 按融合分从高到低排序。
    return [merged[chunk_id] for chunk_id in ordered[:limit]]  # 返回前 limit 条文档。


def evaluate() -> tuple[list[dict], dict]:  # 定义主评估函数。
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))  # 读取样例问题。
    with BM25_PICKLE.open("rb") as file:  # 打开 BM25 索引文件。
        bm25 = pickle.load(file)  # 加载 BM25 检索器。
    bm25.k = 8  # 设置 BM25 返回数量。
    vector_store = Chroma(  # 加载 Chroma 向量库。
        collection_name=CHROMA_COLLECTION_NAME,  # 指定集合名。
        embedding_function=make_doubao_embeddings(),  # 指定豆包 embedding 查询函数。
        persist_directory=str(CHROMA_DIR),  # 指定向量库目录。
    )  # Chroma 加载完成。
    rows: list[dict] = []  # 准备评估明细列表。
    for sample in samples:  # 遍历样例问题。
        understanding = understand_query(sample["question"])  # 做问题理解和 query 增强。
        query = understanding.expanded_query  # 取增强后的 query。
        bm25_docs = bm25.invoke(query)  # 执行 BM25 检索。
        should_refuse = bool(sample.get("should_refuse", False))  # 标记这条题是否属于知识库应该拒答的负向样例。
        vector_docs = [] if should_refuse else vector_store.similarity_search(query, k=8)  # 拒答题不额外调用 embedding，正向题执行向量检索。
        hybrid_docs = [] if should_refuse else merge_hybrid(bm25_docs, vector_docs)  # 拒答题不参与正向章节融合，正向题执行混合结果融合。
        expected_values = [] if should_refuse else sample.get("expected_contains_any", [sample["expected_contains"]])  # 拒答题没有期望章节，正向题支持多个合法章节。
        row = {  # 组织一条评估明细。
            "question": sample["question"],  # 保存问题。
            "expected_contains": " / ".join(expected_values),  # 保存所有合法章节。
            "should_refuse": should_refuse,  # 保存这条题是否应该拒答。
            "question_type": understanding.question_type,  # 保存问题类型。
            "bm25_hit_rank": hit_rank([source_of(doc) for doc in bm25_docs[:5]], expected_values),  # 保存 BM25 Top5 命中排名。
            "vector_hit_rank": hit_rank([source_of(doc) for doc in vector_docs[:5]], expected_values),  # 保存向量 Top5 命中排名。
            "hybrid_hit_rank": hit_rank([source_of(doc) for doc in hybrid_docs[:5]], expected_values),  # 保存混合 Top5 命中排名。
            "bm25_top1": source_of(bm25_docs[0]) if bm25_docs else "",  # 保存 BM25 Top1 来源。
            "vector_top1": source_of(vector_docs[0]) if vector_docs else "",  # 保存向量 Top1 来源。
            "hybrid_top1": source_of(hybrid_docs[0]) if hybrid_docs else "",  # 保存混合 Top1 来源。
        }  # 评估明细结束。
        rows.append(row)  # 保存评估明细。
    summary = {}  # 准备汇总指标。
    positive_rows = [row for row in rows if not row["should_refuse"]]  # 只用正向题统计章节命中率。
    summary["positive_total"] = len(positive_rows)  # 保存正向检索题数量。
    summary["negative_total"] = len(rows) - len(positive_rows)  # 保存拒答题数量。
    for name in ["bm25", "vector", "hybrid"]:  # 依次统计三种检索模式。
        ranks = [row[f"{name}_hit_rank"] for row in positive_rows]  # 只取正向题的命中排名。
        summary[f"{name}_top1"] = sum(1 for rank in ranks if rank == 1)  # 统计 Top1 命中数。
        summary[f"{name}_top5"] = sum(1 for rank in ranks if rank > 0)  # 统计 Top5 命中数。
    summary["total"] = len(rows)  # 保存总问题数。
    return rows, summary  # 返回明细和汇总。


def write_reports(rows: list[dict], summary: dict) -> tuple[str, str]:  # 定义写报告函数。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成时间戳。
    csv_path = OUTPUT_ROOT / f"hybrid_eval_{safe_time}.csv"  # 定义 CSV 路径。
    md_path = OUTPUT_ROOT / f"hybrid_eval_{safe_time}.md"  # 定义 Markdown 路径。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 打开 CSV。
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))  # 创建 writer。
        writer.writeheader()  # 写表头。
        writer.writerows(rows)  # 写明细。
    total = summary["total"]  # 取问题数。
    lines = [  # 准备 Markdown 内容。
        "# 混合检索评估报告",  # 标题。
        "",  # 空行。
        f"- 问题数：{total}",  # 问题数。
        f"- 正向检索题：{summary['positive_total']}，拒答题：{summary['negative_total']}",  # 正负向题汇总。
        f"- BM25 Top1/Top5：{summary['bm25_top1']}/{summary['positive_total']}，{summary['bm25_top5']}/{summary['positive_total']}",  # BM25 汇总。
        f"- Vector Top1/Top5：{summary['vector_top1']}/{summary['positive_total']}，{summary['vector_top5']}/{summary['positive_total']}",  # 向量汇总。
        f"- Hybrid Top1/Top5：{summary['hybrid_top1']}/{summary['positive_total']}，{summary['hybrid_top5']}/{summary['positive_total']}",  # 混合汇总。
        "",  # 空行。
    ]  # Markdown 头结束。
    for row in rows:  # 遍历明细。
        lines.append(f"## {row['question']}")  # 问题标题。
        lines.append(f"- 类型：{'拒答题' if row['should_refuse'] else '正向检索题'}")  # 写入评估方向。
        lines.append(f"- 期望：{row['expected_contains'] or '知识库不应提供教材依据'}")  # 期望章节或拒答说明。
        lines.append(f"- BM25：rank={row['bm25_hit_rank'] or '未命中'}，Top1={row['bm25_top1']}")  # BM25 结果。
        lines.append(f"- Vector：rank={row['vector_hit_rank'] or '未命中'}，Top1={row['vector_top1']}")  # 向量结果。
        lines.append(f"- Hybrid：rank={row['hybrid_hit_rank'] or '未命中'}，Top1={row['hybrid_top1']}")  # 混合结果。
        lines.append("")  # 空行。
    md_path.write_text("\n".join(lines), encoding="utf-8")  # 写 Markdown。
    json_path = OUTPUT_ROOT / f"hybrid_eval_{safe_time}.json"  # 定义机器可读汇总报告路径，供索引重建验收使用。
    json_path.write_text(json.dumps({"report_type": "hybrid_retrieval", "generated_at": datetime.now().isoformat(timespec="seconds"), "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存 BM25、Vector 和 Hybrid 汇总指标。
    return str(csv_path), str(md_path)  # 返回报告路径。


def main() -> None:  # 定义主函数。
    rows, summary = evaluate()  # 执行评估。
    csv_path, md_path = write_reports(rows, summary)  # 写报告。
    print(f"CSV 评估表：{csv_path}")  # 打印 CSV 路径。
    print(f"Markdown 报告：{md_path}")  # 打印 Markdown 路径。
    print(f"汇总：{summary}")  # 打印汇总。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行主函数。
