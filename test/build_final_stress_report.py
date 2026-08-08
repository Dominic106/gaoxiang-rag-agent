"""合并全量压力测试和修复后定向回归，生成最终临时测试报告。"""

import json  # 导入 json，用来读取各批次机器报告。
from collections import Counter  # 导入 Counter，用来统计结果和失败原因。
from pathlib import Path  # 导入 Path，用来处理测试报告路径。

import run_rag_stress  # 导入真实压力测试模块，复用统一汇总和 Markdown 格式化逻辑。


TEST_ROOT = Path(__file__).resolve().parent  # 获取 test 目录。
DATASET_PATH = TEST_ROOT / "datasets" / "rag_stress_questions.json"  # 定义完整压力测试集路径。
BASE_PATH = TEST_ROOT / "reports" / "rag_stress_results_20260805_192110_117769_reclassified.json"  # 定义原始 145 题离线重解析结果。
KNOWLEDGE_PATH = TEST_ROOT / "reports" / "rag_stress_results_20260805_192603_078717.json"  # 定义拆分修复后的 35 题回归结果。
NEGATIVE_PATH = TEST_ROOT / "reports" / "rag_stress_results_20260805_192620_667079.json"  # 定义边界回归结果。
RETRIEVAL_PATH = TEST_ROOT / "reports" / "retrieval_stress_bm25.json"  # 定义检索层压力测试结果。
OUTPUT_PATH = TEST_ROOT / "reports" / "外部资料RAG压力测试最终报告.md"  # 定义最终可读报告路径。


def read_json(path: Path) -> dict:  # 定义读取机器报告的函数。
    return json.loads(path.read_text(encoding="utf-8"))  # 读取并解析 JSON 报告。


def main() -> None:  # 定义报告合并主函数。
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))  # 读取原始压力测试集，确保报告按固定题目顺序输出。
    base = read_json(BASE_PATH)  # 读取 145 题全量结果的离线重解析版本。
    knowledge = read_json(KNOWLEDGE_PATH)  # 读取拆分规则修复后的知识点回归结果。
    negative = read_json(NEGATIVE_PATH)  # 读取边界问题回归结果。
    retrieval = read_json(RETRIEVAL_PATH)  # 读取 BM25 检索压力测试结果。
    base_by_id = {str(row["question_id"]): row for row in base["results"]}  # 按题目标识建立全量结果索引。
    knowledge_by_id = {str(row["question_id"]): row for row in knowledge["results"]}  # 按题目标识建立知识点回归索引。
    negative_by_id = {str(row["question_id"]): row for row in negative["results"]}  # 按题目标识建立边界回归索引。
    merged: list[dict] = []  # 准备保存合并后的最终结果。
    for sample in dataset:  # 按测试集稳定顺序遍历全部题目。
        question_id = str(sample["question_id"])  # 读取当前题目标识。
        if question_id in knowledge_by_id:  # 如果当前题目属于修复后的知识点回归。
            merged.append(knowledge_by_id[question_id])  # 使用修复后真实重跑结果覆盖旧结果。
        elif question_id in negative_by_id:  # 如果当前题目属于边界回归。
            merged.append(negative_by_id[question_id])  # 使用边界回归真实结果覆盖旧结果。
        else:  # 其他题目使用 145 题批次已保存的真实结果。
            merged.append(base_by_id[question_id])  # 保留外部真题和内部可靠性样本结果。
    summary = run_rag_stress.build_summary(merged)  # 使用统一函数汇总最终结果。
    outcome_counter = Counter(str(row["outcome"]) for row in merged)  # 统计全量结果分类。
    group_counter = {group: Counter(str(row["outcome"]) for row in merged if row["test_group"] == group) for group in sorted({row["test_group"] for row in merged})}  # 统计每个测试组的结果分类。
    external_rows = [row for row in merged if row["test_group"] == "external_true_question"]  # 筛选公开真题摘要。
    external_refusals = [row for row in external_rows if row["outcome"] == "safe_refusal"]  # 筛选公开真题中被保守拒答的题目。
    low_evidence = sum(1 for row in external_refusals if (row.get("evidence_score") or 0) < 6)  # 统计证据分极低的公开题。
    medium_evidence = sum(1 for row in external_refusals if 6 <= (row.get("evidence_score") or 0) < 27)  # 统计证据分中等的公开题。
    high_evidence = sum(1 for row in external_refusals if (row.get("evidence_score") or 0) >= 27)  # 统计有候选片段但模型仍保守拒答的公开题。
    retrieval_summary = retrieval["summary"]  # 读取检索层分组摘要。
    lines = [  # 初始化最终报告正文。
        "# 外部资料 RAG 压力测试最终报告",  # 写入报告标题。
        "",  # 添加 Markdown 空行。
        "生成时间：2026-08-05",  # 记录本次测试日期。
        "",  # 添加 Markdown 空行。
        "## 测试结论",  # 写入结论标题。
        "",  # 添加 Markdown 空行。
        "核心查询链路已经完成了真实 API 批量测试，但本次结果证明‘核心可用’不等于‘已经覆盖新版软考全题库’。当前第三版教材知识库对原教材范围内的学习问法具备可用能力；面对 2025 公开真题摘要、政策时事、运筹计算和新版考纲术语时，系统会大量保守拒答，这是安全策略生效，同时也暴露出资料版本覆盖和题型能力不足。",  # 给出面向产品目标的总判断。
        "",  # 添加 Markdown 空行。
        "## 测试规模",  # 写入规模标题。
        "",  # 添加 Markdown 空行。
        "- 全量真实 API 批次：145 题，3 个独立 CLI 并发，单题超时 240 秒。",  # 说明主批次规模和压力参数。
        "- 定向回归：拆分修复后重跑知识点 35 题，边界安全题 4 题。",  # 说明修复后的补充验证。
        "- 样本构成：公开真题题干摘要 75 题、已有可靠性样本 31 题、章节知识点探针 35 题、越界边界题 4 题。",  # 说明样本组成。
        "- 外部题目不是完整试卷或标准答案集；很多网页题干是摘要或截断文本，章节标签也是规则预标注。",  # 说明测试集边界。
        "",  # 添加 Markdown 空行。
        "## 全链路结果",  # 写入全链路结果标题。
        "",  # 添加 Markdown 空行。
        f"- 总体结果分类：{dict(outcome_counter)}。",  # 输出全量分类。
        f"- 平均端到端耗时：{summary['average_seconds']} 秒；P95：{summary['p95_seconds']} 秒。",  # 输出总体延迟。
        "- 全量测试未出现查询未生成报告、进程启动失败或单题超时。",  # 输出稳定性结论。
        "",  # 添加 Markdown 空行。
        "| 测试组 | 总数 | 正常回答且引用有效 | 部分回答且引用有效 | 保守拒答 | 章节代理需复核 | 正确拒答 | 不安全回答 |",  # 写入分组结果表头。
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",  # 写入表格分隔线。
    ]  # 基础报告内容结束。
    for group in sorted(group_counter):  # 遍历所有测试组。
        counts = group_counter[group]  # 读取当前测试组结果计数。
        lines.append(f"| {group} | {sum(counts.values())} | {counts['answered_with_valid_citation']} | {counts['partial_answer_with_valid_citation']} | {counts['safe_refusal']} | {counts['chapter_proxy_miss']} | {counts['correct_refusal']} | {counts['unsafe_answer']} |")  # 写入分组结果。
    lines.extend([  # 追加检索层指标。
        "",  # 添加 Markdown 空行。
        "## 检索层结果",  # 写入检索层标题。
        "",  # 添加 Markdown 空行。
        f"- 公开真题摘要：Top1 {retrieval_summary['external_true_question']['top1']}/75，Top5 {retrieval_summary['external_true_question']['top5']}/75，Top8 {retrieval_summary['external_true_question']['top8']}/75，规则预标注未命中 {retrieval_summary['external_true_question']['unmatched']}。",  # 输出外部题目检索结果。
        f"- 教材知识点探针：Top1 {retrieval_summary['knowledge_point']['top1']}/35，Top5 {retrieval_summary['knowledge_point']['top5']}/35，Top8 {retrieval_summary['knowledge_point']['top8']}/35，规则预标注未命中 {retrieval_summary['knowledge_point']['unmatched']}。",  # 输出教材知识点检索结果。
        "- 公开真题章节命中率不能直接当答案准确率，因为章节标签来自题干关键词规则，且题目网页通常只提供摘要；它主要用于发现候选召回和章节映射问题。",  # 解释检索指标边界。
        "",  # 添加 Markdown 空行。
        "## 答不上来的问题归因",  # 写入归因标题。
        "",  # 添加 Markdown 空行。
        "公开真题摘要 75 题中，61 题被系统保守拒答，2 题属于部分回答并明确声明部分内容没有教材依据，11 题回答有有效引用但未命中规则预标注章节，1 题正常回答且引用有效。也就是说，按‘完整回答’口径，外部样本中至少 63/75 题没有得到完整答案；但这不是 63 题代码全错，其中很多题属于当前第三版教材未覆盖、2025 新考纲或题干截断。",  # 输出外部题目核心结论。
        f"- 证据分低于 6：{low_evidence} 题，优先怀疑检索召回不足、术语变体、公式/表格缺失或当前教材没有对应片段。",  # 归类低证据问题。
        f"- 证据分 6 至 26：{medium_evidence} 题，属于弱证据候选，优先人工检查章节映射、切分和是否需要补充资料。",  # 归类中等证据问题。
        f"- 证据分至少 27 但仍保守拒答：{high_evidence} 题，说明候选片段存在但不一定直接支持题目选项，可能是题干截断、标准答案在新版资料中、或严格引用门禁拒绝模型推断。",  # 归类高证据但拒答问题。
        "- 章节代理未命中：13 题，不能直接判定答案错误，应人工对照原始引用；现有规则把很多新版治理、绩效、采购、IT 审计题粗略归入质量或项目基础章节，映射过宽。",  # 归类章节代理问题。
        "- 稳定性异常：0 题。没有发现 API 超时、进程失败、报告丢失或并发导致的查询异常。",  # 输出稳定性归因。
        "",  # 添加 Markdown 空行。
        "## 已发现并修复的问题",  # 写入修复标题。
        "",  # 添加 Markdown 空行。
        "- 修复 `question_splitter.py`：把‘什么是信息系统的生命周期？通常包括哪些阶段？’合并为一个连续问句，避免丢失前文语境。",  # 记录真实核心缺陷修复。
        "- 修复测试器对多问题总报告的解析：现在会读取子问题元数据和子报告的引用校验，不再把有效多问题回答误记为空答案。",  # 记录测试基础设施修复。
        "- 修正压力测试的拒答识别：能识别‘没有找到与……相关依据’和‘没有足够依据’等保守表达；4 条越界题定向回归全部正确拒答。",  # 记录安全测试修复。
        "",  # 添加 Markdown 空行。
        "## 需要改造的优先级",  # 写入改造建议标题。
        "",  # 添加 Markdown 空行。
        "1. 建立‘第三版教材核心库’与‘新版考纲/真题补充库’两个明确版本，检索时显示资料版本；不要把 2025 新题强行判为第三版教材能够回答。",  # 建议资料分层。
        "2. 增加选择题专用链路：完整保存题干、选项、题型、标准答案和解析授权依据；支持选项抽取、考点定位、逐项排除和答案引用。当前外部样本只有题干摘要，无法做真正的选择题准确率评估。",  # 建议题型能力。
        "3. 建立正式章节别名和考点映射表，替代当前测试专用的关键词规则；映射要有人工确认状态，才能把章节命中率变成可靠指标。",  # 建议章节治理。
        "4. 对公式、运筹题、表格题建立结构化解析和计算器工具；纯文本 BM25 对运输问题、指派问题、关键路径和选项表格不够稳定。",  # 建议计算题能力。
        "5. 保留严格引用门禁和保守拒答，不为了提高回答率放宽无依据回答；产品层应把‘资料中没有’和‘资料版本未覆盖’明确展示给用户。",  # 建议安全边界。
        "",  # 添加 Markdown 空行。
        "## 复核入口",  # 写入文件入口标题。
        "",  # 添加 Markdown 空行。
        "- `datasets/external_true_questions_2025_batch1.json`：75 条外部题干摘要和来源链接。",  # 指向外部样本。
        "- `datasets/external_knowledge_tree.json`：26 个一级主题、121 个子主题、题量合计 3364。",  # 指向考点树。
        "- `reports/retrieval_stress_bm25.md`：检索层逐题结果。",  # 指向检索报告。
        "- `reports/rag_stress_results_20260805_192110_117769.json`：145 题原始全链路结果。",  # 指向原始结果。
        "- `reports/rag_stress_results_20260805_192110_117769_reclassified.json`：修正测试器解析后的离线结果。",  # 指向解析修正版。
        "- `reports/rag_stress_results_20260805_192603_078717.json`：拆分修复后的 35 题知识点回归。",  # 指向定向回归。
        "- `reports/rag_stress_results_20260805_192620_667079.json`：4 题安全边界回归。",  # 指向安全回归。
        "- `source_inventory.md`：外部资料、来源、版权边界和采集范围。",  # 指向资料清单。
        "",  # 添加 Markdown 空行。
        "本报告是临时压力测试，不替代人工答案审校，也不代表已经完成全部历年真题覆盖。",  # 明确最终边界。
    ])  # 报告正文结束。
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")  # 写入最终报告。
    print(OUTPUT_PATH)  # 输出报告路径。
    print(json.dumps(summary, ensure_ascii=False))  # 输出机器可读的最终摘要。


if __name__ == "__main__":  # 判断是否由命令行直接运行。
    main()  # 执行报告合并主函数。
