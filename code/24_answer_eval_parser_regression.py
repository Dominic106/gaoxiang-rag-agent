"""答案可靠性评估报告解析回归测试。"""  # 说明本文件验证评估器可以正确读取新版报告的两个校验区块。

import importlib  # 导入 importlib，用来加载数字开头的评估脚本。
import tempfile  # 导入 tempfile，创建隔离的测试报告文件。
from pathlib import Path  # 导入 Path，用来处理临时报告路径。


answer_eval = importlib.import_module("16_eval_answer_reliability")  # 加载待测的答案评估器。


def test_parser_stops_before_template_section() -> None:  # 验证引用校验字典不会把模板校验字典一起解析。
    report = """# 报告

## 回答与引用

定义结论：测试内容 [1]

引用依据：
[1] 第1章 测试原文

## 引用校验结果

{'passed': True, 'cited_numbers': [1], 'invalid_numbers': [], 'sentence_results': [], 'reason': '通过'}

## 回答模板校验结果

{'passed': True, 'template': '定义解释'}

## 检索日志

第 1 次检索
"""  # 构造与真实新版报告相同区块顺序的最小 Markdown。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 创建一次性测试目录。
        report_path = Path(temporary_directory) / "report.md"  # 定义测试报告路径。
        report_path.write_text(report, encoding="utf-8")  # 写入模拟报告。
        parsed = answer_eval.parse_answer_report(report_path)  # 调用评估器的报告解析函数。
    assert parsed["validation"].get("passed") is True, "评估器没有正确解析引用校验结果"  # 确认引用校验状态被读取。
    assert parsed["validation"].get("template") is None, "引用校验结果不应混入模板校验字段"  # 确认区块边界正确。


def test_parser_reads_degraded_citations_without_heading() -> None:  # 验证安全降级报告没有“引用依据”标题时仍能提取原文。
    report = """## 回答与引用

模型草稿没有通过严格引用校验，因此不直接采用。

[1] 第6章 Web Service 技术 / 原文

## 引用校验结果

{'passed': False, 'reason': '未通过'}

## 检索日志

第 1 次检索
"""  # 构造没有固定引用标题但保留引用记录的降级报告。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 创建一次性测试目录。
        report_path = Path(temporary_directory) / "degraded-report.md"  # 定义降级报告路径。
        report_path.write_text(report, encoding="utf-8")  # 写入模拟降级报告。
        parsed = answer_eval.parse_answer_report(report_path)  # 调用评估器的报告解析函数。
    assert "第6章 Web Service 技术" in parsed["citations"], "降级报告丢失了有效章节引用"  # 确认章节检查可以继续工作。


def test_key_term_alias_coverage() -> None:  # 验证少量人工确认的中文词形别名不会降低评估结果。
    coverage = answer_eval.calculate_key_term_coverage("需要评估风险发生的可能，并分析其影响。", ["可能性", "影响"])  # 模拟回答使用“可能”表达“可能性”的情况。
    assert coverage["coverage"] == 1.0, f"中文词形别名没有被识别：{coverage}"  # 可能性和影响都应被计入覆盖。


def main() -> None:  # 定义解析回归测试入口。
    test_parser_stops_before_template_section()  # 执行新版报告解析测试。
    test_parser_reads_degraded_citations_without_heading()  # 执行降级报告解析测试。
    test_key_term_alias_coverage()  # 执行中文词形别名覆盖测试。
    print("答案评估解析回归通过：引用校验、降级引用和关键词词形别名均可正确处理。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 直接运行时执行解析回归测试。
