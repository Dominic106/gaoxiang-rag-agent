"""使用真实 RAG CLI 执行全量压力测试，并保留每题的可审计归因。"""

import argparse  # 导入 argparse，用来支持并发数和小批量试跑参数。
import json  # 导入 json，用来读取测试集和写入结构化结果。
import re  # 导入 re，用来解析 CLI 输出和问答报告中的关键字段。
import subprocess  # 导入 subprocess，用独立进程执行完整 RAG CLI。
import sys  # 导入 sys，用当前虚拟环境解释器启动子进程。
import time  # 导入 time，用来计算每道题的端到端耗时。
from concurrent.futures import ThreadPoolExecutor  # 导入线程池，让多个独立 CLI 请求模拟并发访问。
from datetime import datetime  # 导入 datetime，用来生成本次压力测试的稳定会话名。
from importlib import import_module  # 导入 import_module，用来复用现有答案报告解析器。
from pathlib import Path  # 导入 Path，用来处理项目内的测试文件路径。


TEST_ROOT = Path(__file__).resolve().parent  # 获取 test 目录。
PROJECT_ROOT = TEST_ROOT.parent  # 获取项目根目录。
CODE_ROOT = PROJECT_ROOT / "code"  # 获取核心代码目录。
DATASET_PATH = TEST_ROOT / "datasets" / "rag_stress_questions.json"  # 定义全量压力测试数据集路径。
RETRIEVAL_PATH = TEST_ROOT / "reports" / "retrieval_stress_bm25.json"  # 定义检索层压力测试结果路径。
REPORT_ROOT = TEST_ROOT / "reports"  # 定义压力测试报告输出目录。
sys.path.insert(0, str(CODE_ROOT))  # 允许复用项目配置和答案报告解析函数。

answer_eval = import_module("16_eval_answer_reliability")  # 复用已有的报告解析和拒答判断逻辑。
parse_answer_report = answer_eval.parse_answer_report  # 取出统一的问答报告解析函数。
refusal_marker_present = answer_eval.refusal_marker_present  # 取出统一的安全拒答判断函数。
model_failure_marker_present = answer_eval.model_failure_marker_present  # 取出统一的模型故障判断函数。


def unsupported_answer_marker_present(answer: str) -> bool:  # 定义识别“有引用但仍无法回答”表达的函数。
    markers = ["当前知识库没有找到关于", "当前知识库没有找到足够依据", "没有找到与", "没有提供任何可直接依据", "无法判断选项", "知识库依据不足", "无法依据教材回答", "不直接回答"]  # 覆盖模型在保守拒答和部分回答中的常见表达。
    normalized = answer.casefold().replace(" ", "")  # 归一化大小写和普通空格。
    return any(marker.casefold().replace(" ", "") in normalized for marker in markers)  # 任一表达出现即认为答案声明了证据不足。


def partial_answer_signal_present(answer: str) -> bool:  # 定义识别“回答了部分内容但对部分结论保守声明不足”的函数。
    signals = ["概念A：", "概念B：", "差异表：", "过程顺序：", "输入：", "输出：", "公式：", "建议："]  # 定义统一回答模板中代表已有实质内容的结构信号。
    return any(signal in answer for signal in signals)  # 任一实质结构出现即认为不是纯拒答。


def parse_any_answer_report(report_path: Path) -> dict:  # 定义同时支持单问题和多问题总报告的解析函数。
    report = report_path.read_text(encoding="utf-8")  # 读取原始问答报告。
    if "# 多问题 RAG 总报告" not in report:  # 普通单问题报告直接复用正式解析器。
        return parse_answer_report(report_path)  # 返回单问题报告结构。
    unified_start = report.find("## 统一回答")  # 定位多问题统一回答区块。
    unified_text = report[unified_start:] if unified_start >= 0 else report  # 截取统一回答及其后续内容。
    child_paths = re.findall(r"report_path': '([^']+)'", report)  # 从子问题元数据读取每个子查询的原始报告路径。
    child_reports = [parse_answer_report(Path(path)) for path in child_paths if Path(path).exists()]  # 解析存在的子问题报告。
    citations = "\n\n".join(child["citations"] for child in child_reports)  # 合并所有子问题引用，供章节和数量检查。
    validation = {"passed": bool(child_reports) and all(child["validation"].get("passed") is True for child in child_reports), "child_count": len(child_reports)}  # 只有所有子问题引用校验都通过才算多问题引用通过。
    return {"answer": unified_text, "citations": citations, "validation": validation, "report_path": str(report_path), "raw_report": report + "\n" + "\n".join(child["raw_report"] for child in child_reports)}  # 返回统一回答和子问题证据。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="运行全量 RAG 真实 API 压力测试")  # 创建参数解析器。
    parser.add_argument("--workers", type=int, default=3, help="并发的独立 RAG CLI 数量，默认 3")  # 提供保守的并发参数。
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0 表示全量")  # 提供小批量试跑开关。
    parser.add_argument("--group", default="", help="只运行指定测试组，例如 knowledge_point")  # 提供按测试组回归的开关。
    parser.add_argument("--timeout", type=int, default=240, help="单题 CLI 最大耗时秒数")  # 提供单题超时参数。
    return parser.parse_args()  # 返回解析后的参数。


def load_rows() -> list[dict]:  # 定义读取并校验压力测试集的函数。
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))  # 读取 JSON 测试集。
    if not isinstance(rows, list) or not rows:  # 检查数据集必须是非空数组。
        raise RuntimeError("压力测试集必须是非空数组。")  # 用清晰错误阻止空测试误报通过。
    question_ids = [str(row.get("question_id", "")) for row in rows]  # 收集所有题目标识。
    if len(question_ids) != len(set(question_ids)):  # 检查题目标识不能重复。
        raise RuntimeError("压力测试集存在重复 question_id，无法安全归档单题结果。")  # 阻止报告互相覆盖或难以追踪。
    return rows  # 返回经过基础校验的数据集。


def load_retrieval_rows() -> dict[str, dict]:  # 定义读取检索层结果的函数。
    payload = json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))  # 读取 BM25 压力测试结果。
    return {str(row["question_id"]): row for row in payload.get("rows", [])}  # 按题目标识建立快速查询索引。


def extract_report_path(output: str) -> Path | None:  # 定义从 CLI 输出中读取问答报告路径的函数。
    match = re.search(r"报告已保存：(.+)", output)  # 查找主查询脚本打印的报告路径。
    if not match:  # 如果没有输出报告路径。
        return None  # 返回空值，让调用方记录执行失败。
    report_path = Path(match.group(1).strip())  # 把路径文本转换为 Path。
    return report_path if report_path.exists() else None  # 只有文件确实存在时才返回。


def extract_evidence_score(report: str) -> int | None:  # 定义从问答报告读取最高证据分的函数。
    scores = [int(value) for value in re.findall(r"证据分[=：](\d+)", report)]  # 提取检索日志里的所有证据分。
    return max(scores) if scores else None  # 返回最高分，报告没有检索日志时返回空值。


def extract_attempts(report: str) -> int:  # 定义读取检索尝试次数的函数。
    matches = re.findall(r"第 (\d+) 次检索：", report)  # 查找每一轮检索日志的编号。
    return max((int(value) for value in matches), default=0)  # 返回最大编号作为实际尝试次数。


def execute_query(row: dict, session: str, timeout: int) -> dict:  # 定义执行单题真实 RAG 查询的函数。
    started = time.perf_counter()  # 记录单题开始时间。
    question_id = str(row["question_id"])  # 读取稳定题目标识。
    command = [sys.executable, "03_query_graph.py", "--session", session, str(row["question"])]  # 组装完整查询 CLI 命令。
    try:  # 保护单题子进程，单题故障不能中止剩余批次。
        completed = subprocess.run(command, cwd=CODE_ROOT, capture_output=True, text=True, timeout=timeout, check=False)  # 执行完整 RAG 流程并捕获诊断输出。
        output = f"{completed.stdout}\n{completed.stderr}"  # 合并标准输出与错误输出。
        report_path = extract_report_path(output)  # 尝试定位本题问答报告。
        return {  # 返回单题执行结果。
            "question_id": question_id,  # 保存题目标识。
            "returncode": completed.returncode,  # 保存子进程退出码。
            "elapsed_seconds": round(time.perf_counter() - started, 3),  # 保存端到端耗时。
            "report_path": str(report_path) if report_path else "",  # 保存原始报告路径。
            "error": "" if completed.returncode == 0 else output[-1500:],  # 失败时保留末尾诊断信息。
        }  # 单题执行结果结束。
    except subprocess.TimeoutExpired as exc:  # 捕获单题总超时。
        return {"question_id": question_id, "returncode": 124, "elapsed_seconds": round(time.perf_counter() - started, 3), "report_path": "", "error": f"查询超时：{exc}"}  # 把超时转成结构化失败。
    except OSError as exc:  # 捕获解释器、工作目录或进程启动错误。
        return {"question_id": question_id, "returncode": 1, "elapsed_seconds": round(time.perf_counter() - started, 3), "report_path": "", "error": f"查询进程启动失败：{exc}"}  # 把启动错误转成结构化失败。


def classify_result(row: dict, execution: dict, retrieval: dict) -> dict:  # 定义单题结果解析和失败归因函数。
    result = {  # 初始化单题基础结果。
        "question_id": row["question_id"],  # 保存稳定题目标识。
        "question": row["question"],  # 保存原始问题文本。
        "test_group": row.get("test_group", "unknown"),  # 保存测试分组。
        "source_type": row.get("source_type", "unknown"),  # 保存样本来源类型。
        "should_answer": bool(row.get("should_answer", False)),  # 保存预期是否回答。
        "expected_chapters": row.get("expected_chapters", []),  # 保存章节预标注。
        "retrieval_hit_rank": retrieval.get("hit_rank", 0),  # 保存 BM25 章节预标注命中排名。
        "retrieval_top8_chapters": retrieval.get("top8_chapters", []),  # 保存 BM25 Top8 章节。
        "returncode": execution.get("returncode", 1),  # 保存 CLI 退出码。
        "elapsed_seconds": execution.get("elapsed_seconds", 0.0),  # 保存端到端耗时。
        "report_path": execution.get("report_path", ""),  # 保存原始问答报告路径。
        "error": execution.get("error", ""),  # 保存执行错误。
    }  # 基础结果初始化结束。
    if not execution.get("report_path"):  # 如果没有生成问答报告。
        result.update({"outcome": "execution_failure", "reason": "查询未生成报告或进程异常"})  # 归因为接口、超时或启动异常。
        return result  # 无报告时无法继续做答案和引用判定。
    parsed = parse_any_answer_report(Path(execution["report_path"]))  # 读取单问题或多问题原始回答、引用和校验结果。
    answer = parsed["answer"]  # 取出回答正文。
    citations = parsed["citations"]  # 取出引用部分。
    validation = parsed["validation"]  # 取出严格引用校验结果。
    unsupported = unsupported_answer_marker_present(answer)  # 判断回答是否包含证据不足声明。
    partial_answer = unsupported and partial_answer_signal_present(answer)  # 判断回答是否属于部分回答加保守声明。
    refusal = refusal_marker_present(answer) or (unsupported and not partial_answer)  # 纯证据不足才算拒答，部分回答保留为独立结果。
    model_failure = model_failure_marker_present(answer)  # 判断回答是否只是模型故障提示。
    citation_passed = validation.get("passed") is True  # 只有严格校验明确通过才算引用有效。
    expected_chapters = [str(value) for value in row.get("expected_chapters", [])]  # 规范化章节预标注。
    chapter_in_citation = any(chapter in citations for chapter in expected_chapters) if expected_chapters else None  # 检查引用是否落到预期章节。
    evidence_score = extract_evidence_score(parsed["raw_report"])  # 提取最高证据分。
    attempts = extract_attempts(parsed["raw_report"])  # 提取实际检索次数。
    result.update({  # 写入报告级指标。
        "answer_preview": answer[:800],  # 保存回答短预览，方便失败复核。
        "refusal": refusal,  # 保存是否安全拒答。
        "model_failure": model_failure,  # 保存是否模型故障。
        "citation_passed": citation_passed,  # 保存引用校验是否通过。
        "chapter_in_citation": chapter_in_citation,  # 保存预期章节是否出现在引用中。
        "evidence_score": evidence_score,  # 保存最高证据分。
        "attempts": attempts,  # 保存检索尝试次数。
        "citation_count": len(re.findall(r"(?m)^\[\d+\]", citations)),  # 保存引用数量。
    })  # 报告指标写入结束。
    if not row.get("should_answer", False):  # 处理预期越界拒答的边界题。
        if model_failure:  # 模型故障不能算正确拒答。
            result.update({"outcome": "model_failure", "reason": "模型服务异常降级"})  # 单独归类模型故障。
        elif refusal:  # 明确拒答说明安全门生效。
            result.update({"outcome": "correct_refusal", "reason": "越界问题被明确拒答"})  # 归为正确拒答。
        else:  # 没有拒答信号表示越界回答风险。
            result.update({"outcome": "unsafe_answer", "reason": "越界问题未明确拒答"})  # 记录安全治理缺陷。
        return result  # 边界题判定完成。
    if model_failure:  # 正向题出现模型故障。
        result.update({"outcome": "model_failure", "reason": "正向题未生成正常回答"})  # 记录模型层故障。
    elif refusal:  # 正向题被证据门禁拒答。
        result.update({"outcome": "safe_refusal", "reason": "正向题因证据不足被保守拒答"})  # 先记录安全但影响可用性的结果。
    elif partial_answer and citation_passed:  # 正向题回答了部分内容且引用通过，但明确声明部分内容没有教材依据。
        result.update({"outcome": "partial_answer_with_valid_citation", "reason": "部分回答有有效引用，但仍有结论声明证据不足"})  # 记录可用性和完整性之间的折中结果。
    elif not citation_passed:  # 正常回答但引用校验没有通过。
        result.update({"outcome": "citation_gate_failure", "reason": "回答生成但严格引用校验未通过"})  # 记录引用或模型表达问题。
    elif chapter_in_citation is False:  # 引用通过但没有落到章节预标注。
        result.update({"outcome": "chapter_proxy_miss", "reason": "引用通过但未落到规则预标注章节"})  # 标记为需要人工复核的章节代理指标。
    else:  # 正常回答、引用有效且章节代理命中。
        result.update({"outcome": "answered_with_valid_citation", "reason": "正常回答且引用校验通过"})  # 归为自动化通过。
    return result  # 返回单题完整结果。


def build_summary(results: list[dict]) -> dict:  # 定义按测试组和结果类型汇总的函数。
    by_group: dict[str, dict] = {}  # 准备保存测试组指标。
    groups = sorted({str(result["test_group"]) for result in results})  # 收集所有测试组并稳定排序。
    for group in groups:  # 遍历每个测试组。
        group_rows = [result for result in results if result["test_group"] == group]  # 筛选当前测试组。
        outcomes: dict[str, int] = {}  # 准备统计当前组的结果类型。
        for result in group_rows:  # 遍历当前组结果。
            outcome = str(result["outcome"])  # 读取结果类型。
            outcomes[outcome] = outcomes.get(outcome, 0) + 1  # 累计结果类型数量。
        latencies = sorted(float(result["elapsed_seconds"]) for result in group_rows)  # 收集耗时并排序。
        p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1)) if latencies else 0  # 计算近似 P95 下标。
        by_group[group] = {"total": len(group_rows), "outcomes": outcomes, "average_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0.0, "p95_seconds": round(latencies[p95_index], 3) if latencies else 0.0}  # 写入分组统计。
    outcomes: dict[str, int] = {}  # 准备保存全量结果类型统计。
    for result in results:  # 遍历全量结果。
        outcome = str(result["outcome"])  # 读取结果类型。
        outcomes[outcome] = outcomes.get(outcome, 0) + 1  # 累计结果类型数量。
    durations = sorted(float(result["elapsed_seconds"]) for result in results)  # 收集全量耗时。
    p95_index = min(len(durations) - 1, max(0, int(len(durations) * 0.95) - 1)) if durations else 0  # 计算全量 P95 下标。
    return {"total": len(results), "outcomes": outcomes, "by_group": by_group, "average_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0, "p95_seconds": round(durations[p95_index], 3) if durations else 0.0}  # 返回汇总结果。


def build_markdown(summary: dict, results: list[dict], workers: int) -> str:  # 定义生成可读压力测试报告的函数。
    lines = ["# 全量 RAG 真实 API 压力测试", "", f"生成时间：{datetime.now().isoformat(timespec='seconds')}", f"并发独立 CLI 数：{workers}", "", "## 总体结果", "", f"- 总题数：{summary['total']}", f"- 平均端到端耗时：{summary['average_seconds']} 秒", f"- P95 端到端耗时：{summary['p95_seconds']} 秒", f"- 结果分类：{summary['outcomes']}", "", "## 分组结果", "", "| 测试组 | 总数 | 平均秒数 | P95秒数 | 结果分类 |", "| --- | ---: | ---: | ---: | --- |"]  # 初始化报告正文。
    for group, metrics in summary["by_group"].items():  # 遍历分组汇总。
        lines.append(f"| {group} | {metrics['total']} | {metrics['average_seconds']} | {metrics['p95_seconds']} | {metrics['outcomes']} |")  # 写入分组统计。
    lines.extend(["", "## 需要复核或改造的题目", "", "| 题目 | 分组 | 结果 | 原因 | 证据分 | 检索次数 | 原始报告 |", "| --- | --- | --- | --- | ---: | ---: | --- |"])  # 添加失败复核表头。
    for result in results:  # 遍历单题结果。
        if result["outcome"] in {"answered_with_valid_citation", "correct_refusal"}:  # 通过项不进入问题清单。
            continue  # 跳过自动化通过项。
        question = str(result["question"]).replace("|", "\\|")  # 转义 Markdown 表格分隔符。
        reason = str(result.get("reason", "")).replace("|", "\\|")  # 转义失败原因中的分隔符。
        lines.append(f"| {question} | {result['test_group']} | {result['outcome']} | {reason} | {result.get('evidence_score', '')} | {result.get('attempts', '')} | {result.get('report_path', '')} |")  # 写入复核入口。
    lines.extend(["", "## 解释边界", "", "公开真题题干只有规则预标注章节，不包含经人工确认的标准答案；外部真题的章节代理未命中不能单独证明答案错误。应优先结合原始问答报告和公开来源人工复核。", ""])  # 明确自动化测试的证据边界。
    return "\n".join(lines)  # 返回完整 Markdown 文本。


def main() -> None:  # 定义压力测试主函数。
    args = parse_args()  # 解析命令行参数。
    if args.workers < 1:  # 检查并发数必须为正数。
        raise RuntimeError("--workers 必须大于等于 1。")  # 拒绝无效并发配置。
    rows = load_rows()  # 读取全量压力测试集。
    if args.group:  # 如果指定了测试组过滤。
        rows = [row for row in rows if row.get("test_group") == args.group]  # 只保留指定测试组，避免回归时重复消耗无关请求。
        if not rows:  # 检查过滤后不能是空集。
            raise RuntimeError(f"测试集不存在指定分组：{args.group}")  # 用清晰错误提示拼写或数据问题。
    if args.limit > 0:  # 如果用户指定小批量运行。
        rows = rows[: args.limit]  # 截取前 N 条用于试跑。
    retrieval_rows = load_retrieval_rows()  # 读取相同数据集的检索层结果。
    session = f"stress_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"  # 为本次批量测试创建独立会话前缀。
    executions: dict[str, dict] = {}  # 准备按题目标识保存执行结果。
    with ThreadPoolExecutor(max_workers=args.workers) as executor:  # 创建保守并发的独立 CLI 调度器。
        futures = {executor.submit(execute_query, row, f"{session}_{row['question_id']}", args.timeout): row for row in rows}  # 提交所有题目并保留题目映射。
        for index, future in enumerate(futures, start=1):  # 按提交顺序收集结果，进度更稳定。
            row = futures[future]  # 取出当前 Future 对应题目。
            executions[str(row["question_id"])] = future.result()  # 收集单题执行结果，异常由 execute_query 转为结构化数据。
            if index % 10 == 0 or index == len(rows):  # 每 10 题或最后一题报告进度。
                print(f"RAG 压力测试进度：{index}/{len(rows)}", flush=True)  # 输出当前批次进度。
    results = [classify_result(row, executions[str(row["question_id"])], retrieval_rows.get(str(row["question_id"]), {})) for row in rows]  # 解析全部执行结果并做归因。
    summary = build_summary(results)  # 汇总全量指标。
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成报告时间戳。
    json_path = REPORT_ROOT / f"rag_stress_results_{timestamp}.json"  # 定义结构化报告路径。
    markdown_path = REPORT_ROOT / f"rag_stress_results_{timestamp}.md"  # 定义 Markdown 报告路径。
    payload = {"summary": summary, "results": results, "session": session, "workers": args.workers, "dataset": str(DATASET_PATH)}  # 组织完整报告数据。
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入结构化结果。
    markdown_path.write_text(build_markdown(summary, results, args.workers), encoding="utf-8")  # 写入人类可读报告。
    print(f"RAG 压力测试 JSON：{json_path}")  # 输出结构化报告路径。
    print(f"RAG 压力测试 Markdown：{markdown_path}")  # 输出 Markdown 报告路径。
    print(json.dumps(summary, ensure_ascii=False))  # 输出汇总摘要供终端和自动化读取。


if __name__ == "__main__":  # 判断脚本是否由命令行直接启动。
    main()  # 执行压力测试主函数。
