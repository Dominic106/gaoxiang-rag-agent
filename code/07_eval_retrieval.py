import csv  # 导入 csv，用来输出评估明细表。
from datetime import datetime  # 导入 datetime，用来生成评估报告时间。
import json  # 导入 json，用来读取示例问题集。
import pickle  # 导入 pickle，用来加载 BM25 检索器。

from config import BM25_PICKLE  # 从配置文件读取 BM25 索引路径。
from config import EVAL_SAMPLES_PATH  # 从配置读取正式评估集路径。
from config import OUTPUT_ROOT  # 从配置文件读取输出目录。
from query_understanding import understand_query  # 导入问题理解函数。
from rag_tokenizers import tokenize_for_bm25  # 导入分词函数，确保 BM25 pickle 能加载。  # noqa: F401


SAMPLES_PATH = EVAL_SAMPLES_PATH  # 使用 30 到 50 条正式评估问题，而不是只测试少量示例。


def load_samples() -> list[dict]:  # 定义读取样例问题的函数。
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))  # 读取 JSON 并解析成列表。


def evaluate_one(retriever, sample: dict) -> dict:  # 定义评估单个问题的函数。
    understanding = understand_query(sample["question"])  # 对问题做类型判断和 query 增强。
    docs = retriever.invoke(understanding.expanded_query)  # 用增强 query 检索。
    top_sources = [doc.metadata.get("relative_path", "") for doc in docs[:5]]  # 取前 5 条来源路径。
    should_refuse = bool(sample.get("should_refuse", False))  # 标记这条题是否属于知识库应该拒答的负向样例。
    expected_values = [] if should_refuse else sample.get("expected_contains_any", [sample["expected_contains"]])  # 拒答题没有期望章节，正向题支持多个合法章节。
    hit_rank = 0  # 初始化命中排名，0 表示未命中。
    for index, source in enumerate(top_sources, start=1):  # 遍历前 5 条来源。
        if any(expected in source for expected in expected_values):  # 如果来源路径包含任意一个合法章节关键词。
            hit_rank = index  # 记录命中排名。
            break  # 找到后停止。
    return {  # 返回评估结果字典。
        "question": sample["question"],  # 保存问题。
        "expected_contains": " / ".join(expected_values),  # 保存所有合法章节关键词。
        "should_refuse": should_refuse,  # 保存这条题是否应该拒答。
        "question_type": understanding.question_type,  # 保存问题类型。
        "expected_question_type": sample.get("question_type", ""),  # 保存评估集标注的问题类型。
        "question_type_correct": should_refuse or understanding.question_type == sample.get("question_type", understanding.question_type),  # 拒答题不把普通分类当作质量指标，正向题才比较人工标注。
        "expanded_query": understanding.expanded_query,  # 保存增强 query。
        "hit_rank": hit_rank,  # 保存命中排名。
        "top1_source": top_sources[0] if top_sources else "",  # 保存 top1 来源。
        "top5_sources": " || ".join(top_sources),  # 保存 top5 来源。
    }  # 评估结果结束。


def write_reports(rows: list[dict]) -> tuple[str, str]:  # 定义写评估报告的函数。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成时间戳。
    csv_path = OUTPUT_ROOT / f"retrieval_eval_{safe_time}.csv"  # 定义 CSV 路径。
    md_path = OUTPUT_ROOT / f"retrieval_eval_{safe_time}.md"  # 定义 Markdown 路径。
    with csv_path.open("w", newline="", encoding="utf-8") as file:  # 打开 CSV 文件。
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))  # 创建 CSV writer。
        writer.writeheader()  # 写入表头。
        writer.writerows(rows)  # 写入所有评估行。
    positive_rows = [row for row in rows if not row["should_refuse"]]  # 只用正向题统计章节检索命中率。
    negative_rows = [row for row in rows if row["should_refuse"]]  # 单独保留拒答题数量，避免混淆检索指标。
    total = len(rows)  # 统计全部问题总数。
    positive_total = len(positive_rows)  # 统计正向检索题数量。
    top1 = sum(1 for row in positive_rows if row["hit_rank"] == 1)  # 统计正向题 top1 命中数。
    top5 = sum(1 for row in positive_rows if row["hit_rank"] > 0)  # 统计正向题 top5 命中数。
    type_rows = positive_rows  # 问题类型准确率只统计需要进入教材检索的正向题。
    type_correct = sum(1 for row in type_rows if row["question_type_correct"])  # 统计正向题问题类型分类正确数。
    lines = [  # 准备 Markdown 报告内容。
        "# 检索评估报告",  # 报告标题。
        "",  # 空行。
        f"- 问题数：{total}",  # 写入问题数。
        f"- 正向检索题：{positive_total}",  # 写入正向题数量。
        f"- 拒答题：{len(negative_rows)}",  # 写入负向题数量。
        f"- Top1 命中：{top1}/{positive_total}",  # 写入正向题 top1 命中。
        f"- Top5 命中：{top5}/{positive_total}",  # 写入正向题 top5 命中。
        f"- 问题类型分类正确：{type_correct}/{len(type_rows)}",  # 写入正向题问题分类准确数。
        "",  # 空行。
    ]  # 报告头结束。
    for row in rows:  # 遍历评估行。
        lines.append(f"## {row['question']}")  # 写入问题标题。
        lines.append(f"- 类型：{'拒答题' if row['should_refuse'] else '正向检索题'}")  # 写入评估方向。
        lines.append(f"- 期望：{row['expected_contains'] or '知识库不应提供教材依据'}")  # 写入期望关键词或拒答说明。
        lines.append(f"- 问题类型：{row['question_type']}")  # 写入问题类型。
        lines.append(f"- 命中排名：{row['hit_rank'] or '未命中'}")  # 写入命中排名。
        lines.append(f"- Top1：{row['top1_source']}")  # 写入 top1 来源。
        lines.append("")  # 空行。
    md_path.write_text("\n".join(lines), encoding="utf-8")  # 写入 Markdown 报告。
    json_path = OUTPUT_ROOT / f"retrieval_eval_{safe_time}.json"  # 定义机器可读汇总报告路径，供重建前后自动比较。
    json_path.write_text(json.dumps({"report_type": "retrieval", "generated_at": datetime.now().isoformat(timespec="seconds"), "summary": {"total": total, "positive_total": positive_total, "negative_total": len(negative_rows), "top1": top1, "top5": top5, "question_type_correct": type_correct}}, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存评估汇总，避免比较工具依赖 Markdown 文案。
    return str(csv_path), str(md_path)  # 返回两个报告路径。


def main() -> None:  # 定义主函数。
    with BM25_PICKLE.open("rb") as file:  # 打开 BM25 索引文件。
        retriever = pickle.load(file)  # 加载 BM25 检索器。
    retriever.k = 8  # 设置返回数量。
    samples = load_samples()  # 读取样例问题。
    rows = [evaluate_one(retriever, sample) for sample in samples]  # 逐个问题评估。
    csv_path, md_path = write_reports(rows)  # 写出报告。
    print(f"CSV 评估表：{csv_path}")  # 打印 CSV 路径。
    print(f"Markdown 报告：{md_path}")  # 打印 Markdown 路径。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行主函数。
