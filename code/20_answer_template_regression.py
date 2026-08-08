"""五类备考回答模板的结构和集成回归测试。"""  # 说明本文件验证模板结构，不调用真实回答 API。

import importlib  # 导入 importlib，用来加载数字开头的完整问答模块。
from typing import Any  # 导入 Any，标注通过 importlib 动态加载的测试模块。

from answer_templates import build_template_instructions  # 导入主提示词模板说明函数。
from answer_templates import build_template_repair_instructions  # 导入修复提示词模板说明函数。
from answer_templates import get_answer_template  # 导入模板映射函数。
from answer_templates import normalize_repaired_template  # 导入只补结构标签的修复函数。
from answer_templates import validate_template_structure  # 导入模板结构校验函数。
from citation_validator import validate_answer  # 导入严格引用校验，验证模板不会替代原文安全门。
from langchain_core.documents import Document  # 导入 Document，构造最小教材证据。
from rag_state import make_initial_state  # 导入初始状态工厂，测试回答节点集成。


VALID_STRUCTURES = {  # 为五类模板准备最小结构样例。
    "定义解释": "定义结论：教材定义。[1]\n核心要点：教材要点。[1]\n教材依据：教材原文位置。[1]",  # 定义模板样例。
    "区别对比": "概念A：概念A定义。[1]\n概念B：概念B定义。[1]\n差异表：\n| 对比维度 | 概念A | 概念B |\n| 范围 | A的范围 | B的范围 [1]\n结论提醒：教材明确区分二者。[1]",  # 对比模板样例。
    "流程步骤": "过程顺序：第一步完成事项。[1]\n输入：教材列出的输入。[1]\n输出：教材列出的输出。[1]\n注意事项：按教材顺序执行。[1]",  # 流程模板样例。
    "输入输出工具技术": "过程顺序：工具与技术用于分析。[1]\n输入：项目管理计划。[1]\n输出：项目文件。[1]\n注意事项：依据教材边界使用。[1]",  # ITTO 映射样例。
    "公式计算": "公式或指标：SPI=EV/PV。[1]\n变量含义：EV表示挣值，PV表示计划价值。[1]\n计算步骤：将EV除以PV。[1]\n结果判读：SPI小于1表示进度落后。[1]",  # 公式模板样例。
    "章节复习": "本章重点：本章核心知识范围。[1]\n考点清单：教材列出的考点。[1]\n易错点：两个相邻概念边界不同。[1]\n原文依据：本次引用对应教材片段。[1]",  # 章节复习模板样例。
    "考点记忆": "本章重点：本章核心知识范围。[1]\n考点清单：教材列出的考点。[1]\n易错点：两个相邻概念边界不同。[1]\n原文依据：本次引用对应教材片段。[1]",  # 考点类型映射到章节复习模板。
}  # 样例字典结束。


def test_template_mapping_and_structure() -> None:  # 验证五类模板都能被映射并识别。
    for question_type, answer in VALID_STRUCTURES.items():  # 遍历所有问题类型样例。
        result = validate_template_structure(answer, question_type)  # 执行模板结构校验。
        assert result["passed"], f"模板结构不通过：{question_type} -> {result}"  # 合法结构必须通过。
        prompt = build_template_instructions(question_type)  # 生成主回答提示词约束。
        repair = build_template_repair_instructions(question_type)  # 生成修复提示词约束。
        template = get_answer_template(question_type)  # 读取模板对象。
        assert template.name in prompt and template.name in repair, f"模板名称没有进入提示词：{question_type}"  # 主回答和修复必须使用同一模板。
        assert all(section in prompt for section in template.required_sections), f"主提示词缺少区块：{question_type}"  # 主提示词必须列出所有区块。
    assert "关键路径" in build_template_instructions("公式计算"), "公式模板没有提醒关键路径题覆盖核心考点"  # 确认高频 CPM 题的教学约束真正进入提示词。
    assert "可能性" in build_template_instructions("区别对比"), "对比模板没有提醒风险分析的可能性维度"  # 确认风险分析对比题的核心维度真正进入提示词。
    missing = validate_template_structure("定义结论：只有定义。[1]", "定义解释")  # 构造缺少两个区块的回答。
    assert not missing["passed"] and "核心要点" in missing["missing_sections"], "缺少模板区块没有被拦截"  # 结构门必须拦截不完整回答。


def test_table_header_and_citation_gate() -> None:  # 验证对比表表头不会污染引用校验，事实行仍受校验。
    citation_text = "[1] 原文：概念A和概念B的范围、目标和管理方式存在差异。"  # 构造最小引用原文。
    answer = "概念A：原文中的概念A。[1]\n概念B：原文中的概念B。[1]\n差异表：\n| 对比维度 | 概念A | 概念B |\n| 范围 | 项目范围 | 产品范围 [1]\n结论提醒：二者存在差异。[1]"  # 构造带表头和事实行的对比回答。
    validation = validate_answer(answer, citation_text)  # 执行严格引用校验。
    assert validation["passed"], f"对比表结构行不应导致引用校验失败：{validation}"  # 表头跳过、事实行有引用时应通过。
    unsupported = answer.replace("项目范围 | 产品范围 [1]", "火星芯片 | 量子航天 [1]")  # 构造与教材原文没有共享关键术语的表格事实。
    assert not validate_answer(unsupported, citation_text)["passed"], "表格事实没有被原文支持时应该拦截"  # 表格不能绕过引用安全门。
    repair_without_labels = "概念A的教材定义。[1]。概念B的教材定义。[1]。\n\n| 对比维度 | 概念A | 概念B |\n| 范围 | A范围 | B范围 [1]\n\n教材明确支持二者存在范围差异。[1]"  # 模拟引用修复模型保留事实但删除区块标签的结果。
    normalized = normalize_repaired_template(repair_without_labels, "区别对比")  # 执行纯结构标签修复。
    assert validate_template_structure(normalized, "区别对比")["passed"], "修复草稿丢失对比区块标签时没有恢复结构"  # 确认兜底不改事实也能恢复模板。
    headed_answer = "### 概念A：J2EE 的教材定义。[1]\n### 概念B：.NET 的教材定义。[1]\n差异表：\n| 对比维度 | J2EE | .NET |\n| 跨平台 | 能力强 | 仅支持 Windows [1]\n结论提醒：教材只比较平台特点。[1]"  # 模拟模型使用标题和实际平台名称作为表头的合法对比答案。
    headed_structure = validate_template_structure(headed_answer, "区别对比")  # 验证兼容标题和实际概念名的结构识别。
    assert headed_structure["passed"], f"兼容模型标题和实际概念表头失败：{headed_structure}"  # 结构门不应因展示风格差异把有依据答案降级。


def test_deterministic_structure_repair_and_neutral_boundaries() -> None:  # 验证非对比模板可以安全补齐结构，且边界说明不被当成事实。
    raw_answer = "第一步完成任务。[1]\n第二步检查结果。[1]"  # 模拟回答模型只给出事实、没有输出模板标签的情况。
    normalized = normalize_repaired_template(raw_answer, "流程步骤")  # 执行确定性结构修复。
    structure = validate_template_structure(normalized, "流程步骤")  # 检查补齐后的流程模板。
    assert structure["passed"], f"流程模板缺少标签时没有被安全补齐：{structure}"  # 结构修复必须完整。
    assert "第一步完成任务。[1]" in normalized and "第二步检查结果。[1]" in normalized, "结构修复不应改写原始事实"  # 确认事实和引用仍保持原样。
    citation_text = "[1] 原文：第一步完成任务，第二步检查结果。"  # 构造能支持两条事实的教材原文。
    assert validate_answer(normalized, citation_text)["passed"], "补结构后的流程回答不应绕过引用安全门"  # 补结构不能降低引用要求。
    boundary_answer = "本章重点：SPI 是进度指标。[1]\n易错点：当前片段未明确给出易错提示"  # 构造带诚实证据边界说明的复习回答。
    boundary_citation = "[1] 原文：SPI 是进度指标。"  # 只为事实句提供原文引用。
    assert validate_answer(boundary_answer, boundary_citation)["passed"], "诚实的证据边界说明不应被判为无引用事实"  # 边界说明应被安全门忽略。
    formula_boundary = "公式或指标：SPI=EV/PV。[1]\n计算步骤：本题未提供数值，暂不进行代入计算"  # 构造没有用户数值的公式题回答。
    formula_citation = "[1] 原文：SPI=EV/PV。"  # 构造公式原文引用。
    assert validate_answer(formula_boundary, formula_citation)["passed"], "公式题的暂不代入说明不应导致引用校验失败"  # 计算边界提示不是教材事实。


def test_generate_answer_integration() -> None:  # 验证主回答节点真正执行模板校验。
    query_graph: Any = importlib.import_module("03_query_graph")  # 加载完整回答节点模块。
    original_call = query_graph.call_deepseek  # 保存真实模型调用函数。
    captured: dict = {}  # 准备保存主提示词，确认模板约束被传入模型。

    def fake_call(prompt: str) -> str:  # 定义不访问网络的回答替身。
        captured["prompt"] = prompt  # 保存提示词正文供断言。
        return "定义结论：范围基准是经过批准的项目范围说明书、工作分解结构和工作分解结构词汇表。[1]\n核心要点：范围基准由三部分组成。[1]\n教材依据：教材原文直接给出了范围基准定义。[1]"  # 返回符合定义模板且有引用的回答。

    query_graph.call_deepseek = fake_call  # 临时替换真实模型调用。
    state = make_initial_state("什么是范围基准？")  # 创建初始状态。
    state["resolved_question"] = "什么是范围基准？"  # 写入完整检索问题。
    state["question_type"] = "定义解释"  # 指定定义模板。
    state["evidence_enough"] = True  # 模拟已经通过证据门。
    state["evidence_score"] = 10  # 写入足够证据分。
    state["contexts"] = [Document(page_content="范围基准是经过批准的项目范围说明书、工作分解结构和工作分解结构词汇表。", metadata={"chapter": "第18章 项目范围管理", "section": "范围基准", "chunk_id": "template-001"})]  # 构造最小教材片段。
    try:  # 保护模块替换。
        result = query_graph.generate_answer(state)  # 执行真实回答节点。
    finally:  # 无论测试结果如何恢复真实调用。
        query_graph.call_deepseek = original_call  # 恢复真实模型调用。
    assert result["citation_validation"]["passed"], "模板集成回答的引用校验没有通过"  # 引用安全门必须通过。
    assert result["template_validation"]["passed"], "模板集成回答的结构校验没有通过"  # 模板结构门必须通过。
    assert "当前回答模板：定义解释" in captured["prompt"], "主提示词没有收到定义模板约束"  # 确认模板真正进入模型提示词。


def main() -> None:  # 定义模板回归测试入口。
    test_template_mapping_and_structure()  # 执行五类模板结构测试。
    test_table_header_and_citation_gate()  # 执行对比表引用安全测试。
    test_deterministic_structure_repair_and_neutral_boundaries()  # 执行通用结构修复和边界说明测试。
    test_generate_answer_integration()  # 执行主回答节点集成测试。
    print("回答模板回归通过：定义、对比、流程/ITTO、公式、章节复习及引用安全门。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断脚本是否被直接运行。
    main()  # 直接运行时执行全部模板回归。
