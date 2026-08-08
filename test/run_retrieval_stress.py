"""对外部真题、知识点和边界题执行全量 BM25 检索压力测试。"""

import json  # 导入 json，用来读取测试集和输出结果。
import sys  # 导入 sys，用来把项目 code 目录加入模块搜索路径。
from collections import defaultdict  # 导入 defaultdict，汇总测试组结果。
from pathlib import Path  # 导入 Path，用来处理测试文件路径。


TEST_ROOT = Path(__file__).resolve().parent  # 获取 test 目录。
PROJECT_ROOT = TEST_ROOT.parent  # 获取项目根目录。
sys.path.insert(0, str(PROJECT_ROOT / "code"))  # 让测试脚本复用现有问题理解和检索资源模块。

from query_understanding import understand_query  # noqa: E402  # 导入现有问题理解器。
from retrieval_resources import get_bm25_retriever  # noqa: E402  # 导入缓存后的 BM25 检索资源。


DATASET_PATH = TEST_ROOT / "datasets" / "rag_stress_questions.json"  # 定义压力测试集路径。
REPORT_ROOT = TEST_ROOT / "reports"  # 定义临时测试报告目录。


def expected_hit(documents: list, expected_chapters: list[str]) -> int:  # 定义按章节候选计算命中排名的函数。
    if not expected_chapters:  # 负向题和没有章节标注的题没有正向命中目标。
        return 0  # 返回未命中。
    for rank, document in enumerate(documents, start=1):  # 遍历 BM25 返回的文档。
        chapter = str(document.metadata.get("chapter", ""))  # 读取 chunk 的标准章节名。
        relative_path = str(document.metadata.get("relative_path", ""))  # 读取源文件相对路径。
        if any(candidate in chapter or candidate in relative_path for candidate in expected_chapters):  # 任一候选章节命中即记录排名。
            return rank  # 返回首次命中排名。
    return 0  # 前五条没有命中时返回 0。


def main() -> None:  # 定义检索压力测试主函数。
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))  # 读取合并后的压力测试集。
    retriever = get_bm25_retriever()  # 在当前进程中只加载一次 BM25，模拟常驻服务资源复用。
    retriever.k = 8  # 返回 8 条候选，观察 Top5 和 Top8 的章节覆盖。
    rows: list[dict] = []  # 准备保存每题检索结果。
    for index, sample in enumerate(samples, start=1):  # 逐题执行检索。
        understanding = understand_query(sample["question"])  # 使用真实问题理解和 query 增强逻辑。
        documents = retriever.invoke(understanding.expanded_query)  # 执行 BM25 检索，不调用外部 embedding API。
        expected_chapters = sample.get("expected_chapters", [])  # 读取题目预标注的候选章节。
        rank = expected_hit(documents[:8], expected_chapters)  # 计算章节首次命中排名。
        rows.append({  # 保存可审计的检索明细。
            "index": index,  # 保存批次序号。
            "question_id": sample.get("question_id", f"stress-{index:03d}"),  # 保存稳定题目标识。
            "question": sample["question"],  # 保存问题文本。
            "test_group": sample.get("test_group", "unknown"),  # 保存测试组。
            "expected_chapters": expected_chapters,  # 保存人工或规则预标注章节。
            "question_type": understanding.question_type,  # 保存实际分类结果。
            "expanded_query": understanding.expanded_query,  # 保存实际增强 query。
            "hit_rank": rank,  # 保存首次命中排名。
            "top8_chapters": [document.metadata.get("chapter", "") for document in documents[:8]],  # 保存 Top8 章节供失败复核。
            "top8_sources": [document.metadata.get("relative_path", "") for document in documents[:8]],  # 保存 Top8 来源路径供定位。
        })  # 单题结果结束。
        if index % 25 == 0:  # 每 25 题输出进度。
            print(f"检索压力测试进度：{index}/{len(samples)}", flush=True)  # 输出当前批次进度。
    by_group: dict[str, list[dict]] = defaultdict(list)  # 准备按测试组汇总。
    for row in rows:  # 遍历所有检索结果。
        by_group[row["test_group"]].append(row)  # 把结果放入对应测试组。
    summary: dict[str, dict] = {}  # 准备保存各测试组指标。
    for group, group_rows in sorted(by_group.items()):  # 按测试组稳定排序汇总。
        positive = [row for row in group_rows if row["expected_chapters"]]  # 只有有章节目标的题统计命中率。
        summary[group] = {  # 计算当前组的可比指标。
            "total": len(group_rows),  # 保存题目总数。
            "positive_target": len(positive),  # 保存有正向章节目标的题数。
            "top1": sum(1 for row in positive if row["hit_rank"] == 1),  # 保存 Top1 命中数。
            "top5": sum(1 for row in positive if 0 < row["hit_rank"] <= 5),  # 保存 Top5 命中数。
            "top8": sum(1 for row in positive if row["hit_rank"] > 0),  # 保存 Top8 命中数。
            "unmatched": sum(1 for row in positive if row["hit_rank"] == 0),  # 保存没有命中候选章节的数量。
        }  # 当前组汇总结束。
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    report_path = REPORT_ROOT / "retrieval_stress_bm25.json"  # 定义机器可读报告路径。
    report_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入完整检索明细。
    markdown_lines = ["# 外部资料检索压力测试", "", f"总题数：{len(rows)}", "", "## 分组汇总", ""]  # 准备 Markdown 报告。
    for group, metrics in summary.items():  # 遍历分组指标。
        markdown_lines.append(f"- {group}：总数 {metrics['total']}，Top1 {metrics['top1']}/{metrics['positive_target']}，Top5 {metrics['top5']}/{metrics['positive_target']}，Top8 {metrics['top8']}/{metrics['positive_target']}，未命中 {metrics['unmatched']}")  # 写入组指标。
    markdown_lines.extend(["", "## 未命中题目", ""])  # 添加失败复核区块。
    for row in rows:  # 遍历检索结果。
        if row["expected_chapters"] and row["hit_rank"] == 0:  # 只列出有目标但未命中的题。
            markdown_lines.append(f"- `{row['question_id']}` {row['question']}；期望：{' / '.join(row['expected_chapters'])}；Top8：{' / '.join(row['top8_chapters'])}")  # 记录未命中原因线索。
    markdown_path = REPORT_ROOT / "retrieval_stress_bm25.md"  # 定义可读报告路径。
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")  # 写入 Markdown 报告。
    print(f"检索 JSON：{report_path}")  # 输出 JSON 路径。
    print(f"检索 Markdown：{markdown_path}")  # 输出 Markdown 路径。
    print(json.dumps(summary, ensure_ascii=False))  # 输出汇总指标。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行检索压力测试。
