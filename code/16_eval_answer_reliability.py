"""评估 RAG 回答是否忠于教材、引用是否有效以及拒答是否正确。

这个脚本使用真实 CLI 运行标准问题，再读取每条问答报告进行离线判定。
它不使用另一个大模型评价答案，避免评价模型把成本和不确定性继续引入核心测试。
关键词覆盖只是保守的自动化下限，最终的人工抽查仍然有价值。
"""

import argparse  # 导入 argparse，用来支持只运行离线报告或重新调用 API。
import ast  # 导入 ast，用来安全解析报告中保存的 Python 字典字符串。
import json  # 导入 json，用来读取评估数据并输出结构化结果。
import re  # 导入 re，用来解析 CLI 输出和 Markdown 报告区块。
import subprocess  # 导入 subprocess，用真实命令行方式执行完整问答流程。
import sys  # 导入 sys，用当前虚拟环境解释器执行主查询脚本。
import time  # 导入 time，用来记录每条答案评估耗时。
from datetime import datetime  # 导入 datetime，用来生成本次评估的唯一会话和报告名称。
from pathlib import Path  # 导入 Path，用来安全处理项目文件路径。

from config import OUTPUT_ROOT  # 导入统一输出目录，保存答案评估报告。


CODE_ROOT = Path(__file__).resolve().parent  # 获取 code 目录，子进程统一从这里启动。
PROJECT_ROOT = CODE_ROOT.parent  # 获取 RAG 项目根目录，用来定位 notes 和输出文件。
ANSWER_EVAL_PATH = PROJECT_ROOT / "notes" / "answer_eval_v1.json"  # 定义答案可靠性评估数据路径。
MIN_POSITIVE_PASS_RATE = 0.80  # 定义正向问题整题通过率最低门槛，避免偶然一两题通过就判定可靠。
MIN_KEY_TERM_COVERAGE = 0.75  # 定义正向问题关键词平均覆盖率最低门槛。
MIN_CITATION_ACCURACY = 0.90  # 定义正向问题引用准确率最低门槛。
MIN_REFUSAL_ACCURACY = 0.90  # 定义拒答问题正确拒答率最低门槛。
KEY_TERM_ALIASES = {"可能性": ("可能", "概率")}  # 定义少量教材语境中稳定的词形别名，避免“可能性”写成“可能”时被错误计为完全缺失。


def normalize_text(value: str) -> str:  # 定义文本归一化函数，减少大小写、空格和标点差异造成的误判。
    lowered = value.casefold()  # 统一英文术语大小写，例如 SPI 和 spi 视为相同。
    return re.sub(r"[\s，。！？、；：：；,.!?;:（）()「」『』\[\]{}]", "", lowered)  # 去掉常见空白和标点，保留中文和关键字母数字。


def load_samples() -> list[dict]:  # 定义读取和校验答案评估集的函数。
    samples = json.loads(ANSWER_EVAL_PATH.read_text(encoding="utf-8"))  # 读取答案评估 JSON。
    if not isinstance(samples, list) or not samples:  # 确保文件最外层是非空数组。
        raise RuntimeError("答案评估集必须是非空数组。")  # 用清晰错误阻止空评估误报通过。
    questions = set()  # 准备检查问题是否重复。
    for sample in samples:  # 遍历每条评估样例。
        required = {"question", "question_type", "reference_points", "key_terms"}  # 定义所有样例都必须具备的字段。
        missing = required - sample.keys()  # 计算当前样例缺失字段。
        if missing:  # 如果存在缺失字段。
            raise RuntimeError(f"答案评估题缺少字段：{sample.get('question', '<未知问题>')} -> {sorted(missing)}")  # 提示具体坏数据。
        question = str(sample["question"]).strip()  # 取出并清理问题文本。
        if not question or question in questions:  # 检查空问题和重复问题。
            raise RuntimeError(f"答案评估题为空或重复：{question!r}")  # 阻止重复题稀释统计结果。
        questions.add(question)  # 记录当前问题。
        if not isinstance(sample["key_terms"], list):  # 确保关键词字段是列表。
            raise TypeError(f"答案评估题 key_terms 必须是列表：{question}")  # 提示数据结构问题。
        if sample.get("should_refuse", False) and sample.get("expected_contains"):  # 拒答题不应伪造教材章节。
            raise RuntimeError(f"拒答题不应配置 expected_contains：{question}")  # 阻止拒答样例被错误当成正向题。
    return samples  # 返回经过结构检查的评估样例。


def extract_report_path(output: str) -> Path | None:  # 定义从 CLI 输出中读取问答报告路径的函数。
    match = re.search(r"报告已保存：(.+)", output)  # 查找主查询脚本打印的报告路径。
    if not match:  # 如果没有找到报告路径。
        return None  # 返回空值，由调用方记录为执行失败。
    path = Path(match.group(1).strip())  # 把路径文本转换成 Path。
    return path if path.exists() else None  # 只有报告实际存在时才返回。


def extract_section(report: str, start_marker: str, end_marker: str) -> str:  # 定义读取 Markdown 两个标题之间内容的函数。
    pattern = rf"{re.escape(start_marker)}\n\n(.*?)(?=\n\n{re.escape(end_marker)}|\Z)"  # 用非贪婪匹配只取当前区块。
    match = re.search(pattern, report, flags=re.DOTALL)  # 在完整报告中搜索区块。
    return match.group(1).strip() if match else ""  # 找到则返回正文，否则返回空字符串。


def parse_answer_report(report_path: Path) -> dict:  # 定义解析单条问答报告的函数。
    report = report_path.read_text(encoding="utf-8")  # 读取完整 Markdown 报告。
    answer_section = extract_section(report, "## 回答与引用", "## 引用校验结果")  # 读取回答和引用区块。
    if "引用依据：" in answer_section:  # 正常答案会用固定标题分隔回答正文和引用原文。
        answer_text, citation_section = answer_section.split("引用依据：", 1)  # 按固定标题拆分回答和引用。
    else:  # 引用校验失败、证据不足或模型异常降级时可能没有固定引用标题。
        first_citation = re.search(r"(?m)^\[\d+\]\s+", answer_section)  # 查找第一条真实引用记录作为降级分界。
        if first_citation:  # 如果报告仍然保留了可复核的原文片段。
            answer_text = answer_section[: first_citation.start()]  # 只保留引用记录之前的系统提示或回答正文。
            citation_section = answer_section[first_citation.start() :]  # 保留所有引用记录供章节和引用检查。
        else:  # 报告没有任何引用记录。
            answer_text, citation_section = answer_section, ""  # 保留整个回答并把引用置空。
    answer_text = answer_text.strip()  # 清理回答正文首尾空白。
    citation_section = citation_section.strip()  # 清理引用正文首尾空白。
    validation_text = extract_section(report, "## 引用校验结果", "## 回答模板校验结果")  # 新版报告在引用校验和检索日志之间还有模板校验区块，因此必须先在模板标题处截断。
    if not validation_text:  # 兼容旧版没有回答模板校验区块的历史报告。
        validation_text = extract_section(report, "## 引用校验结果", "## 检索日志")  # 回退到旧版报告的区块边界。
    try:  # 保护报告字典解析，坏报告不能导致整个批次崩溃。
        validation = ast.literal_eval(validation_text) if validation_text and validation_text != "证据不足时未调用回答模型" else {}  # 安全解析校验字典。
    except (SyntaxError, ValueError):  # 捕获格式异常而不是执行任意字符串。
        validation = {"passed": False, "reason": "引用校验结果格式无法解析"}  # 将格式错误计为不通过。
    return {  # 返回评估器需要的结构化字段。
        "answer": answer_text,  # 保存不含引用正文的回答。
        "citations": citation_section,  # 保存系统实际提供的引用。
        "validation": validation,  # 保存引用校验细节。
        "report_path": str(report_path),  # 保存原始报告路径，便于人工复核。
        "raw_report": report,  # 保存完整报告文本，检查章节和检索日志时使用。
    }  # 返回解析结果。


def refusal_marker_present(answer: str) -> bool:  # 定义判断回答是否明确拒答的函数。
    markers = ["当前知识库没有找到足够依据", "不直接回答", "无法依据教材回答", "不能依据教材回答", "模型草稿没有通过严格引用校验", "回答模型本次不可用"]  # 定义系统保守拒答和安全降级的关键表达。
    normalized = normalize_text(answer)  # 归一化回答以减少标点差异。
    return any(normalize_text(marker) in normalized for marker in markers)  # 任一明确拒答表达出现即认为有拒答信号。


def model_failure_marker_present(answer: str) -> bool:  # 定义判断回答是否只是错误降级提示的函数。
    markers = ["回答模型本次不可用"]  # 只有回答服务不可用才算模型故障；引用校验失败属于系统安全拒答，不能误记为服务故障。
    normalized = normalize_text(answer)  # 归一化回答文本。
    return any(normalize_text(marker) in normalized for marker in markers)  # 任一错误降级提示出现即视为没有正常生成答案。


def calculate_key_term_coverage(answer: str, key_terms: list[str]) -> dict:  # 定义计算关键知识点覆盖率的函数。
    normalized_answer = normalize_text(answer)  # 归一化模型回答文本。
    matched = [term for term in key_terms if any(normalize_text(alias) in normalized_answer for alias in KEY_TERM_ALIASES.get(term, (term,)))]  # 统计原词或经过人工确认的词形别名，避免把合理中文表达误判为缺失。
    coverage = len(matched) / len(key_terms) if key_terms else 1.0  # 没有关键词的拒答题按不适用处理为满覆盖。
    return {"matched": matched, "missing": [term for term in key_terms if term not in matched], "coverage": round(coverage, 4)}  # 返回可审计的覆盖明细。


def evaluate_sample(sample: dict, execution: dict) -> dict:  # 定义对单条样例进行可靠性判定的函数。
    result = {  # 先保存问题和执行基本信息。
        "question": sample["question"],  # 保存原始问题。
        "question_type": sample["question_type"],  # 保存问题类型。
        "should_refuse": bool(sample.get("should_refuse", False)),  # 保存该题是否应该拒答。
        "expected_chapter": sample.get("expected_contains", ""),  # 保存期望章节。
        "elapsed_seconds": execution.get("elapsed_seconds", 0.0),  # 保存单题耗时。
        "cli_returncode": execution.get("returncode", 1),  # 保存 CLI 退出码。
        "report_path": execution.get("report_path", ""),  # 保存原始问答报告路径。
        "error": execution.get("error", ""),  # 保存执行错误。
    }  # 基础结果初始化结束。
    if not execution.get("report_path"):  # 如果查询没有生成报告。
        result.update({"passed": False, "reason": "查询未生成报告", "key_term_coverage": 0.0, "citation_accurate": False, "refusal_correct": False})  # 把执行失败计为不通过。
        return result  # 直接返回，不再解析空报告。
    parsed = parse_answer_report(Path(execution["report_path"]))  # 读取报告中的回答和引用校验结果。
    answer = parsed["answer"]  # 取出模型回答正文。
    validation = parsed["validation"]  # 取出引用校验结果。
    key_terms = [str(term) for term in sample.get("key_terms", [])]  # 规范化关键词列表。
    coverage = calculate_key_term_coverage(answer, key_terms)  # 计算关键知识点覆盖率。
    citation_passed = validation.get("passed") is True  # 只有严格引用校验明确通过才算引用校验通过。
    chapter_present = bool(sample.get("expected_contains")) and sample["expected_contains"] in parsed["citations"]  # 检查实际引用是否来自期望章节。
    refusal_correct = refusal_marker_present(answer) if sample.get("should_refuse", False) else (not refusal_marker_present(answer) or citation_passed)  # 正向题允许在已引用的部分答案后附带“其余证据不足”说明，但纯拒答仍不通过。
    answer_available = bool(answer) and not model_failure_marker_present(answer)  # 正常回答不能只是模型失败或引用失败提示。
    if sample.get("should_refuse", False):  # 处理应该拒答的越界问题。
        passed = refusal_correct and not model_failure_marker_present(answer)  # 拒答可以不要求引用通过，但必须明确说明不能回答。
        reason = "正确拒答" if passed else "没有明确拒答或模型异常降级"  # 组织拒答判定说明。
    else:  # 处理应该回答的教材问题。
        passed = answer_available and refusal_correct and coverage["coverage"] >= MIN_KEY_TERM_COVERAGE and citation_passed and chapter_present  # 正向题必须同时满足回答、要点、引用和章节条件。
        reasons = []  # 准备保存正向题的失败原因。
        if not answer_available:  # 正常答案缺失时记录原因。
            reasons.append("没有正常生成答案")  # 添加回答可用性问题。
        if not refusal_correct:  # 正向题被错误拒答时记录原因。
            reasons.append("误判为证据不足")  # 添加误拒答问题。
        if coverage["coverage"] < MIN_KEY_TERM_COVERAGE:  # 关键点覆盖不足时记录原因。
            reasons.append(f"关键点覆盖率不足 {coverage['coverage']:.0%}")  # 添加覆盖率问题。
        if not citation_passed:  # 引用校验失败时记录原因。
            reasons.append("引用校验未通过")  # 添加引用校验问题。
        if not chapter_present:  # 引用章节不正确时记录原因。
            reasons.append("引用未落到期望章节")  # 添加章节相关性问题。
        reason = "通过" if passed else "；".join(reasons)  # 组织最终判定说明。
    result.update({  # 写入所有可审计指标。
        "passed": passed,  # 保存该题是否整体通过。
        "reason": reason,  # 保存通过或失败原因。
        "answer_available": answer_available,  # 保存回答是否正常生成。
        "refusal_correct": refusal_correct,  # 保存拒答判定。
        "key_term_coverage": coverage["coverage"],  # 保存关键点覆盖率。
        "matched_key_terms": coverage["matched"],  # 保存已命中的关键词。
        "missing_key_terms": coverage["missing"],  # 保存缺失的关键词。
        "citation_accurate": citation_passed and chapter_present if not sample.get("should_refuse", False) else True,  # 正向题要求校验和章节都正确，拒答题此项不适用。
        "citation_validation": validation,  # 保存完整引用校验结果。
        "answer_preview": answer[:500],  # 保存短预览，避免评估报告无限膨胀。
    })  # 单题结果写入结束。
    return result  # 返回单题评估结果。


def run_query(sample: dict, session: str) -> dict:  # 定义通过真实 CLI 执行单条问题的函数。
    started = time.perf_counter()  # 记录单题执行开始时间。
    try:  # 保护单题子进程，单条失败不应丢失整个批次的其他结果。
        completed = subprocess.run(  # 启动完整 RAG 查询流程。
            [sys.executable, "03_query_graph.py", "--session", session, sample["question"]],  # 使用当前虚拟环境和独立评估会话。
            cwd=CODE_ROOT,  # 统一从 code 目录执行，确保项目配置加载一致。
            capture_output=True,  # 捕获输出，解析报告路径和诊断信息。
            text=True,  # 让输出直接以字符串返回。
            timeout=240,  # 单题设置总超时，防止异常网络请求卡住批次。
            check=False,  # 保留退出码，让单题失败可以进入评估结果而不是中断整批测试。
        )  # 子进程执行结束。
        output = f"{completed.stdout}\n{completed.stderr}"  # 合并标准输出和错误输出供诊断使用。
        report_path = extract_report_path(output)  # 从 CLI 输出读取本题报告路径。
        return {  # 返回单题执行结果。
            "returncode": completed.returncode,  # 保存退出码。
            "report_path": str(report_path) if report_path else "",  # 保存存在的报告路径。
            "elapsed_seconds": round(time.perf_counter() - started, 3),  # 保存单题耗时。
            "error": "" if completed.returncode == 0 else output[-1000:],  # 失败时只保留末尾诊断，避免报告过大。
        }  # 返回执行结果。
    except subprocess.TimeoutExpired as exc:  # 捕获单题总超时。
        return {"returncode": 124, "report_path": "", "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"查询超时：{exc}"}  # 将超时转成可审计失败。
    except OSError as exc:  # 捕获 Python 解释器或工作目录启动失败。
        return {"returncode": 1, "report_path": "", "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"查询进程启动失败：{exc}"}  # 返回启动错误。


def build_summary(results: list[dict]) -> dict:  # 定义汇总所有指标的函数。
    positive = [item for item in results if not item["should_refuse"]]  # 筛出应该正常回答的题。
    negative = [item for item in results if item["should_refuse"]]  # 筛出应该拒答的题。
    positive_passed = sum(1 for item in positive if item["passed"])  # 统计正向题整体通过数。
    negative_passed = sum(1 for item in negative if item["passed"])  # 统计拒答题通过数。
    citation_passed = sum(1 for item in positive if item.get("citation_accurate"))  # 统计正向题引用准确数。
    coverage_values = [item.get("key_term_coverage", 0.0) for item in positive]  # 收集正向题关键点覆盖率。
    average_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0  # 计算平均关键点覆盖率。
    answer_pass_rate = positive_passed / len(positive) if positive else 0.0  # 计算正向题整体通过率。
    citation_accuracy = citation_passed / len(positive) if positive else 0.0  # 计算正向题引用准确率。
    refusal_accuracy = negative_passed / len(negative) if negative else 0.0  # 计算拒答准确率。
    passed = bool(positive and negative and answer_pass_rate >= MIN_POSITIVE_PASS_RATE and average_coverage >= MIN_KEY_TERM_COVERAGE and citation_accuracy >= MIN_CITATION_ACCURACY and refusal_accuracy >= MIN_REFUSAL_ACCURACY)  # 所有关键门槛满足才允许阶段通过。
    return {  # 返回可供核心回归读取的摘要。
        "total": len(results),  # 保存总题数。
        "positive": len(positive),  # 保存正向题数量。
        "negative": len(negative),  # 保存拒答题数量。
        "answer_passed": positive_passed,  # 保存正向题通过数。
        "answer_pass_rate": round(answer_pass_rate, 4),  # 保存正向题通过率。
        "citation_passed": citation_passed,  # 保存引用准确数。
        "citation_accuracy": round(citation_accuracy, 4),  # 保存引用准确率。
        "average_key_term_coverage": round(average_coverage, 4),  # 保存平均关键点覆盖率。
        "refusal_passed": negative_passed,  # 保存拒答题通过数。
        "refusal_accuracy": round(refusal_accuracy, 4),  # 保存拒答准确率。
        "passed": passed,  # 保存阶段门槛是否通过。
    }  # 返回摘要。


def format_percent(value: float) -> str:  # 定义百分比格式化函数，让 Markdown 报告更易读。
    return f"{value:.1%}"  # 把小数比例格式化为一位小数百分比。


def build_markdown(summary: dict, results: list[dict], mode: str) -> str:  # 定义生成 Markdown 评估报告的函数。
    status = "通过" if summary["passed"] else "未通过"  # 将摘要状态转成中文。
    lines = [  # 初始化报告正文。
        "# 答案可靠性评估报告",  # 写入报告标题。
        "",  # 添加 Markdown 空行。
        f"评估模式：{mode}",  # 记录离线还是真实 API 模式。
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",  # 记录生成时间。
        "",  # 添加 Markdown 空行。
        f"## 结论：{status}",  # 写入阶段门槛结论。
        "",  # 添加 Markdown 空行。
        "本评估使用参考关键点、关键词覆盖、系统引用校验和期望章节四类信号；它是自动化下限，不等同于人工语义审阅。",  # 明确评估器边界。
        "",  # 添加 Markdown 空行。
        "## 汇总指标",  # 写入汇总标题。
        "",  # 添加 Markdown 空行。
        f"- 总题数：{summary['total']}，正向题：{summary['positive']}，拒答题：{summary['negative']}",  # 输出样例规模。
        f"- 答案整体通过：{summary['answer_passed']}/{summary['positive']}（{format_percent(summary['answer_pass_rate'])}）",  # 输出正向题整体通过率。
        f"- 关键要点覆盖率：{format_percent(summary['average_key_term_coverage'])}",  # 输出平均关键词覆盖率。
        f"- 引用准确率：{summary['citation_passed']}/{summary['positive']}（{format_percent(summary['citation_accuracy'])}）",  # 输出引用准确率。
        f"- 拒答准确率：{summary['refusal_passed']}/{summary['negative']}（{format_percent(summary['refusal_accuracy'])}）",  # 输出拒答准确率。
        "",  # 添加 Markdown 空行。
        "## 通过门槛",  # 写入门槛标题。
        "",  # 添加 Markdown 空行。
        f"- 正向题整体通过率 >= {MIN_POSITIVE_PASS_RATE:.0%}",  # 输出正向题门槛。
        f"- 关键要点平均覆盖率 >= {MIN_KEY_TERM_COVERAGE:.0%}",  # 输出关键点门槛。
        f"- 引用准确率 >= {MIN_CITATION_ACCURACY:.0%}",  # 输出引用门槛。
        f"- 拒答准确率 >= {MIN_REFUSAL_ACCURACY:.0%}",  # 输出拒答门槛。
        "",  # 添加 Markdown 空行。
        "## 逐题结果",  # 写入明细标题。
        "",  # 添加 Markdown 空行。
        "| 状态 | 类型 | 问题 | 要点覆盖 | 引用准确 | 拒答正确 | 说明 |",  # 写入表头。
        "| --- | --- | --- | ---: | --- | --- | --- |",  # 写入表格分隔线。
    ]  # 报告基础内容初始化结束。
    for item in results:  # 遍历每条评估结果。
        status_text = "通过" if item["passed"] else "失败"  # 将单题状态转成中文。
        question = item["question"].replace("|", "\\|")  # 转义问题中的 Markdown 表格符号。
        reason = item.get("reason", "").replace("|", "\\|")  # 转义原因中的表格符号。
        lines.append(f"| {status_text} | {item['question_type']} | {question} | {format_percent(item.get('key_term_coverage', 0.0))} | {'是' if item.get('citation_accurate') else '否'} | {'是' if item.get('refusal_correct') else '否'} | {reason} |")  # 写入单题表格行。
    lines.extend(["", "## 复核入口", "", "每条结果都保留了原始问答报告路径；对于失败题，应优先检查原文引用、模型回答和评估关键词是否需要人工修订。", ""])  # 添加人工复核说明。
    return "\n".join(lines)  # 返回完整 Markdown 报告。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="运行 RAG 答案可靠性评估")  # 创建参数解析器。
    parser.add_argument("--offline", action="store_true", help="只校验答案评估集结构，不调用 API")  # 提供无 API 的数据结构检查模式。
    return parser.parse_args()  # 返回解析结果。


def main() -> None:  # 定义脚本主函数。
    args = parse_args()  # 解析命令行参数。
    samples = load_samples()  # 读取并校验答案评估集。
    if args.offline:  # 如果只要求离线数据检查。
        positive = sum(1 for item in samples if not item.get("should_refuse", False))  # 统计正向样例数。
        negative = len(samples) - positive  # 统计拒答样例数。
        print(f"答案评估集质量通过：总题数={len(samples)}，正向题={positive}，拒答题={negative}。")  # 输出离线检查结果。
        return  # 离线模式不执行 API。
    session = f"answer_eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"  # 创建独立评估会话，避免污染用户日常学习记录。
    results = []  # 准备保存所有逐题结果。
    for index, sample in enumerate(samples, start=1):  # 按顺序执行每条标准问题。
        print(f"答案可靠性评估进度：{index}/{len(samples)} {sample['question']}", flush=True)  # 输出进度，长时间运行时让用户知道程序仍在工作。
        execution = run_query(sample, session)  # 调用真实 RAG CLI。
        results.append(evaluate_sample(sample, execution))  # 解析报告并保存单题评估结果。
    summary = build_summary(results)  # 汇总答案、引用和拒答指标。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成报告时间戳。
    json_path = OUTPUT_ROOT / f"answer_reliability_eval_{timestamp}.json"  # 拼出结构化结果路径。
    markdown_path = OUTPUT_ROOT / f"answer_reliability_eval_{timestamp}.md"  # 拼出 Markdown 报告路径。
    payload = {"summary": summary, "results": results, "session": session, "generated_at": datetime.now().isoformat(timespec="seconds")}  # 组织完整 JSON 结果。
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入结构化评估结果。
    markdown_path.write_text(build_markdown(summary, results, "真实 API"), encoding="utf-8")  # 写入人类可读报告。
    print(f"答案可靠性 JSON：{json_path}")  # 输出 JSON 报告路径。
    print(f"答案可靠性 Markdown：{markdown_path}")  # 输出 Markdown 报告路径。
    status = "通过" if summary["passed"] else "未通过"  # 把布尔门槛状态转换成中文。
    print(f"答案可靠性总体状态：{status}")  # 输出阶段门槛状态。
    raise SystemExit(0 if summary["passed"] else 1)  # 阶段门槛未通过时返回失败码，接入核心回归后不能误报通过。


if __name__ == "__main__":  # 判断脚本是否被直接执行。
    main()  # 直接执行时运行答案可靠性评估。
