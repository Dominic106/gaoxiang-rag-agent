"""核心 RAG 总结测试入口。

这个脚本把分散的检查统一成一份可重复的报告，方便每次核心代码修改后复盘。
默认执行不需要 API 的检查；增加 --with-api 后，再执行向量 Hybrid 和端到端冒烟。
"""

import argparse  # 导入 argparse，用来支持是否执行 API 测试的命令行选项。
import re  # 导入 re，用来从评估报告中提取可比较的数字指标。
import subprocess  # 导入 subprocess，用来以真实命令行方式测试各个脚本。
import sys  # 导入 sys，用来复用当前虚拟环境中的 Python 解释器。
import time  # 导入 time，用来记录缓存资源的首次加载和再次获取耗时。
from datetime import datetime  # 导入 datetime，用来生成本次测试报告的唯一名称。
from pathlib import Path  # 导入 Path，用来安全处理项目文件路径。

from config import OUTPUT_ROOT  # 从配置读取测试输出目录。
from retrieval_resources import get_bm25_retriever  # 导入缓存的 BM25 资源。
from retrieval_resources import get_embedding_model  # 导入缓存的 embedding 资源。
from retrieval_resources import get_vector_store  # 导入缓存的 Chroma 资源。


CODE_ROOT = Path(__file__).resolve().parent  # 获取 code 目录，所有测试命令都从这里执行。
PROJECT_ROOT = CODE_ROOT.parent  # 获取 RAG 项目根目录，用来组织报告中的相对路径。


def run_command(label: str, arguments: list[str], timeout: int = 180) -> dict:  # 定义统一命令执行函数，记录状态但不泄露环境变量。
    started = time.perf_counter()  # 记录命令开始时间。
    completed = subprocess.run(  # 启动一个子进程，模拟用户真实执行脚本的方式。
        [sys.executable, *arguments],  # 使用当前虚拟环境解释器，避免系统 Python 和项目环境不一致。
        cwd=CODE_ROOT,  # 统一在 code 目录执行，保证相对导入和 .env 加载一致。
        capture_output=True,  # 捕获输出，便于提取结果并写入报告。
        text=True,  # 让 stdout 和 stderr 直接以字符串返回。
        timeout=timeout,  # 防止某个网络调用异常时整个总结测试无限等待。
        check=False,  # 保留退出码，统一写入总结报告而不是抛出未诊断异常。
    )  # 子进程执行结束。
    elapsed = time.perf_counter() - started  # 计算本条命令耗时。
    output = f"{completed.stdout}\n{completed.stderr}".strip()  # 合并输出，方便后续识别报告路径。
    return {  # 返回结构化结果，避免主流程到处解析 subprocess 对象。
        "label": label,  # 保存测试名称。
        "passed": completed.returncode == 0,  # 退出码为 0 才算命令级通过。
        "returncode": completed.returncode,  # 保存原始退出码，方便排查失败。
        "elapsed": round(elapsed, 3),  # 保存秒级耗时。
        "output": output,  # 保存输出，仅用于本地报告，不包含 API Key。
    }  # 返回测试结果。


def extract_report_path(output: str, marker: str) -> Path | None:  # 定义从脚本输出中提取报告路径的函数。
    match = re.search(rf"{re.escape(marker)}：(.+)", output)  # 查找形如“Markdown 报告：/path”的输出行。
    if not match:  # 如果没有找到对应路径。
        return None  # 返回空值，让调用方把它记录为诊断缺失。
    return Path(match.group(1).strip())  # 把文本路径转换为 Path 对象。


def extract_line_path(output: str, marker: str) -> Path | None:  # 定义从任意带冒号输出行中读取路径的函数。
    match = re.search(rf"{re.escape(marker)}：(.+)", output)  # 查找指定标记后的完整路径。
    if not match:  # 如果没有找到路径。
        return None  # 返回空值，让调用方记录断言失败。
    return Path(match.group(1).strip())  # 把文本路径转换成 Path 对象。


def read_metrics(report_path: Path | None) -> dict[str, str]:  # 定义评估指标读取函数。
    if not report_path or not report_path.exists():  # 如果报告路径缺失或文件不存在。
        return {}  # 返回空指标，报告会明确显示缺失。
    text = report_path.read_text(encoding="utf-8")  # 读取 Markdown 评估报告。
    metrics: dict[str, str] = {}  # 准备保存匹配到的指标。
    patterns = {  # 定义当前评估报告需要固定提取的指标。
        "top1": r"Top1 命中：([^\n]+)",  # 提取单检索器 Top1。
        "top5": r"Top5 命中：([^\n]+)",  # 提取单检索器 Top5。
        "type_accuracy": r"问题类型分类正确：([^\n]+)",  # 提取问题分类准确数。
        "hybrid": r"Hybrid Top1/Top5：([^\n]+)",  # 提取 Hybrid Top1/Top5。
    }  # 指标模式定义结束。
    for key, pattern in patterns.items():  # 遍历所有需要提取的指标。
        match = re.search(pattern, text)  # 在报告正文里搜索当前指标。
        if match:  # 如果找到指标。
            metrics[key] = match.group(1).strip()  # 保存指标文本。
    return metrics  # 返回指标字典。


def read_answer_metrics(report_path: Path | None) -> dict[str, str]:  # 定义读取答案可靠性评估指标的函数。
    if not report_path or not report_path.exists():  # 如果答案评估没有生成报告。
        return {}  # 返回空指标，报告会明确显示缺失。
    text = report_path.read_text(encoding="utf-8")  # 读取答案可靠性 Markdown 报告。
    patterns = {  # 定义答案可靠性报告中的固定指标模式。
        "answer_passed": r"答案整体通过：([^（\n]+)",  # 提取正向题整体通过数量。
        "key_term_coverage": r"关键要点覆盖率：([^\n]+)",  # 提取关键要点覆盖率。
        "citation_accuracy": r"引用准确率：[^（\n]+（([^）]+)）",  # 提取引用准确率百分比。
        "refusal_accuracy": r"拒答准确率：[^（\n]+（([^）]+)）",  # 提取拒答准确率百分比。
    }  # 答案指标模式定义结束。
    metrics: dict[str, str] = {}  # 准备保存读取到的答案指标。
    for key, pattern in patterns.items():  # 遍历所有指标模式。
        match = re.search(pattern, text)  # 在报告正文中查找当前指标。
        if match:  # 如果匹配成功。
            metrics[key] = match.group(1).strip()  # 保存指标文本。
    return metrics  # 返回答案可靠性指标。


def is_perfect_metric(value: str) -> bool:  # 定义判断 x/x 指标是否满分的函数，避免把评估集数量写死。
    match = re.match(r"(\d+)/(\d+)", value or "")  # 从指标开头读取分子和分母。
    return bool(match and match.group(1) == match.group(2))  # 分子等于分母时认为该指标满分。


def check_resource_cache() -> dict:  # 定义运行时缓存检查，验证同一进程不会重复创建昂贵对象。
    started = time.perf_counter()  # 记录首次加载开始时间。
    bm25_first = get_bm25_retriever()  # 第一次获取 BM25，预期会从磁盘加载。
    bm25_first_elapsed = time.perf_counter() - started  # 记录 BM25 首次获取耗时。
    started = time.perf_counter()  # 记录第二次获取开始时间。
    bm25_second = get_bm25_retriever()  # 第二次获取 BM25，预期直接命中缓存。
    bm25_second_elapsed = time.perf_counter() - started  # 记录 BM25 第二次获取耗时。
    embedding_first = get_embedding_model()  # 第一次获取 embedding 适配器。
    embedding_second = get_embedding_model()  # 第二次获取 embedding 适配器。
    vector_first = get_vector_store()  # 第一次获取 Chroma 向量库对象。
    vector_second = get_vector_store()  # 第二次获取 Chroma 向量库对象。
    return {  # 返回缓存检查结果。
        "passed": bm25_first is bm25_second and embedding_first is embedding_second and vector_first is vector_second,  # 三类资源都必须是同一对象。
        "bm25_first_ms": round(bm25_first_elapsed * 1000, 2),  # 记录 BM25 首次加载毫秒数。
        "bm25_second_ms": round(bm25_second_elapsed * 1000, 2),  # 记录 BM25 缓存命中毫秒数。
        "bm25_same_object": bm25_first is bm25_second,  # 记录 BM25 是否复用。
        "embedding_same_object": embedding_first is embedding_second,  # 记录 embedding 是否复用。
        "vector_same_object": vector_first is vector_second,  # 记录 Chroma 是否复用。
    }  # 返回缓存结果。


def run_e2e_smoke() -> list[dict]:  # 定义 API 模式下的端到端冒烟测试集合。
    session = f"core_audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"  # 为本次冒烟创建独立会话，避免污染日常学习会话。
    cases = [  # 准备覆盖核心用户路径的命令。
        ("E2E 可回答问题", ["03_query_graph.py", "--session", session, "什么是范围基准？"]),  # 验证正常检索、回答和保存。
        ("E2E 章节过滤", ["03_query_graph.py", "--session", session, "--chapter", "第17章", "项目章程的作用是什么？"]),  # 验证章节范围仍然生效。
        ("E2E 多问题", ["03_query_graph.py", "--session", session, "1. 什么是项目章程？ 2. 项目管理计划有什么特点？"]),  # 验证多个子问题能统一返回。
        ("E2E 追问记忆", ["03_query_graph.py", "--session", session, "它和项目范围说明书有什么区别？"]),  # 验证后续问题能读取同一会话锚点。
        ("E2E 证据不足", ["03_query_graph.py", "--session", session, "教材中关于火星殖民预算的规定是什么？"]),  # 验证没有依据时不让模型自由发挥。
    ]  # 冒烟命令定义结束。
    results: list[dict] = []  # 准备保存各条端到端结果。
    for label, command in cases:  # 逐条执行真实 CLI 流程。
        result = run_command(label, command, timeout=240)  # 给网络调用更充足的时间。
        result["has_saved_report"] = "报告已保存：" in result["output"]  # 正常 CLI 必须保存问答报告。
        result["has_saved_session"] = "会话已保存：" in result["output"]  # 正常 CLI 必须保存会话记录。
        if label == "E2E 证据不足":  # 对拒答场景额外检查保守提示。
            result["has_guardrail"] = "没有找到足够依据" in result["output"]  # 没有证据时必须出现拒答门禁提示。
        else:  # 其他场景不要求特定答案内容。
            result["has_guardrail"] = True  # 统一标记为不适用。
        result["unique_sub_reports"] = True  # 默认非多问题场景不需要检查多个单题报告。
        if label == "E2E 多问题":  # 多问题场景要额外验证单题报告不会同秒覆盖。
            combined_path = extract_line_path(result["output"], "报告已保存")  # 读取多问题总报告路径。
            combined_text = combined_path.read_text(encoding="utf-8") if combined_path and combined_path.exists() else ""  # 读取总报告内容。
            sub_reports = re.findall(r"'report_path': '([^']+)'", combined_text)  # 从子问题元数据里提取单题报告路径。
            result["unique_sub_reports"] = len(sub_reports) >= 2 and len(sub_reports) == len(set(sub_reports))  # 每个子问题必须拥有不同的报告文件。
        result["passed"] = result["passed"] and result["has_saved_report"] and result["has_saved_session"] and result["has_guardrail"] and result["unique_sub_reports"]  # 汇总命令和业务断言。
        results.append(result)  # 保存当前冒烟结果。
    return results  # 返回全部端到端结果。


def format_command_result(result: dict) -> str:  # 定义命令结果 Markdown 格式化函数。
    status = "通过" if result["passed"] else "失败"  # 将布尔值转成用户可读状态。
    return f"- {result['label']}：{status}，耗时 {result['elapsed']} 秒，退出码 {result['returncode']}"  # 返回一行摘要。


def build_report(results: list[dict], cache_result: dict, with_api: bool) -> str:  # 定义总结报告生成函数。
    retrieval_result = next((item for item in results if item["label"] == "检索评估"), {})  # 找到普通检索评估结果。
    retrieval_metrics = retrieval_result.get("metrics", {})  # 读取普通检索指标。
    hybrid_result = next((item for item in results if item["label"] == "Hybrid 检索评估"), {})  # 找到 Hybrid 评估结果。
    hybrid_metrics = hybrid_result.get("metrics", {})  # 读取 Hybrid 指标。
    answer_result = next((item for item in results if item["label"] == "答案可靠性评估"), {})  # 找到真实 API 答案可靠性评估结果。
    answer_metrics = answer_result.get("metrics", {})  # 读取答案可靠性指标。
    e2e_results = [item for item in results if item["label"].startswith("E2E ")]  # 筛出端到端结果。
    logic_score = 8.0 if is_perfect_metric(retrieval_metrics.get("type_accuracy", "")) else 7.0  # 以正式评估的分类结果作为逻辑基础分。
    if is_perfect_metric(hybrid_metrics.get("hybrid", "")):  # 如果 Hybrid Top1 达到当前评估集满分。
        logic_score += 0.3  # 给核心召回表现加分。
    if e2e_results and all(item["passed"] for item in e2e_results):  # 如果 API 模式下所有端到端场景也通过。
        logic_score += 0.4  # 给真实 CLI 链路加分。
    logic_score = round(min(logic_score, 10.0), 1)  # 把逻辑分限制在 10 分以内并统一保留一位小数。
    cleanliness_score = 8.0 if cache_result["passed"] else 7.0  # 资源边界和缓存模块通过时，整洁度达到优化后基线。
    performance_score = 8.5 if cache_result["passed"] else 6.5  # 缓存对象复用是本轮性能优化的核心验收点。
    required_labels = {"语法检查", "模块回归", "异常治理回归", "评估集质量", "答案评估集质量", "检索评估", "性能基准"}  # 定义离线回归必须通过的测试集合。
    if with_api:  # 完整 API 模式还必须包含真实答案质量评估。
        required_labels.add("答案可靠性评估")  # 将答案忠实度、引用和拒答评估纳入回归门禁。
    regression_score = 8.0 if all(any(item["label"] == label and item["passed"] for item in results) for label in required_labels) else 6.0  # 必须每个要求标签都实际存在且通过，避免缺失测试项目时误报回归通过。
    if with_api and e2e_results and not all(item["passed"] for item in e2e_results):  # 如果 API 模式下有端到端失败。
        regression_score -= 1.0  # 端到端失败要明显扣分，避免只看离线评估。
    regression_score = round(max(regression_score, 0.0), 1)  # 保证分数不会小于 0，并统一保留一位小数。
    overall_score = round((logic_score + cleanliness_score + performance_score + regression_score) / 4, 2)  # 计算四项平均分。
    test_lines = "\n".join(format_command_result(item) for item in results)  # 格式化所有命令结果。
    e2e_lines = "\n".join(format_command_result(item) for item in e2e_results) or "- 未执行：需要增加 --with-api"  # 格式化端到端结果。
    regression_basis = "静态、模块、离线检索和端到端测试结果" if with_api else "静态、模块、离线检索结果（端到端待执行）"  # 根据测试模式生成准确的评分依据。
    return f"""# RAG 核心代码审计与总结测试报告

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

当前核心功能达到“可稳定使用的基础版”水平，适合继续做知识库查询和学习记录；错题本、复习卡、Web UI，以及回退后自动重算一致性仍然属于后续增强范围。

本次总体评分：**{overall_score}/10**。

## 分项评分

| 维度 | 得分 | 依据 |
| --- | ---: | --- |
| 逻辑正确性 | {logic_score}/10 | 问题分类、混合检索、证据门禁和引用校验结果 |
| 代码整洁度 | {cleanliness_score}/10 | 检索资源集中到独立模块，并减少主流程的直接依赖 |
| 性能与成本 | {performance_score}/10 | BM25、Chroma、embedding 在同一进程内复用，减少重复初始化 |
| 功能回归 | {regression_score}/10 | {regression_basis} |

## 自动化测试结果

{test_lines}

### 缓存复用

- BM25 首次加载：{cache_result['bm25_first_ms']} ms
- BM25 再次获取：{cache_result['bm25_second_ms']} ms
- BM25 同对象复用：{cache_result['bm25_same_object']}
- embedding 同对象复用：{cache_result['embedding_same_object']}
- Chroma 同对象复用：{cache_result['vector_same_object']}

### 端到端冒烟

{e2e_lines}

## 检索指标

- 普通评估：Top1 {retrieval_metrics.get('top1', '未读取')}，Top5 {retrieval_metrics.get('top5', '未读取')}，问题类型 {retrieval_metrics.get('type_accuracy', '未读取')}
- Hybrid 评估：{hybrid_metrics.get('hybrid', '未执行')}

## 答案可靠性指标

- 正向题整体通过：{answer_metrics.get('answer_passed', '未执行')}
- 关键要点覆盖率：{answer_metrics.get('key_term_coverage', '未执行')}
- 引用准确率：{answer_metrics.get('citation_accuracy', '未执行')}
- 拒答准确率：{answer_metrics.get('refusal_accuracy', '未执行')}

## 当前明确的剩余风险

1. `03_query_graph.py` 仍然是较大的编排模块，后续可以继续拆成状态、编排、格式化三个边界，但不应在本轮引入大规模重写。
2. 回退目前能删除上一轮会话文件，但尚未自动重算长期记忆画像和问题历史；第四阶段继续挂起。
3. 当前缓存是单进程缓存，重启进程后会重新加载资源；这符合 CLI 使用方式，Web UI 阶段再评估进程级共享和并发安全。
4. 答案可靠性评估已经纳入核心回归；关键词覆盖属于自动化下限，仍需要后续增加人工抽查或更细的语义评估。

## 后续建议

先保持核心查询链路稳定，补充更多“应该拒答”和“章节边界”样例；错题本、复习卡、Web UI 等增强功能等核心测试连续通过后再进入设计。
"""  # 返回完整 Markdown 报告。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="运行 RAG 核心代码总结测试")  # 创建参数解析器。
    parser.add_argument("--with-api", action="store_true", help="增加 Hybrid 检索和 DeepSeek 端到端冒烟测试")  # 支持完整 API 测试模式。
    return parser.parse_args()  # 返回解析结果。


def main() -> None:  # 定义总结测试主函数。
    args = parse_args()  # 解析命令行参数。
    results: list[dict] = []  # 准备保存所有测试结果。
    py_files = sorted(path.name for path in CODE_ROOT.glob("*.py"))  # 收集 code 目录下的 Python 源文件。
    results.append(run_command("语法检查", ["-m", "py_compile", *py_files]))  # 执行 L0 语法检查。
    results.append(run_command("模块回归", ["11_regression_tests.py"]))  # 执行 L1 模块回归测试。
    results.append(run_command("异常治理回归", ["14_exception_regression.py"]))  # 执行超时、异常降级和日志落盘测试。
    results.append(run_command("评估集质量", ["15_eval_data_quality.py"]))  # 执行评估题数量、字段、去重和章节覆盖检查。
    results.append(run_command("答案评估集质量", ["16_eval_answer_reliability.py", "--offline"]))  # 执行答案参考点、关键词和拒答样例的结构检查。
    retrieval_result = run_command("检索评估", ["07_eval_retrieval.py"])  # 执行 L2 普通检索评估。
    retrieval_result["metrics"] = read_metrics(extract_report_path(retrieval_result["output"], "Markdown 报告"))  # 读取普通检索指标。
    results.append(retrieval_result)  # 保存普通检索结果。
    if args.with_api:  # 只有用户明确要求时才执行会产生 API 访问的测试。
        hybrid_result = run_command("Hybrid 检索评估", ["10_eval_hybrid_retrieval.py"], timeout=240)  # 执行向量和 BM25 混合评估。
        hybrid_result["metrics"] = read_metrics(extract_report_path(hybrid_result["output"], "Markdown 报告"))  # 读取 Hybrid 指标。
        results.append(hybrid_result)  # 保存 Hybrid 结果。
        answer_result = run_command("答案可靠性评估", ["16_eval_answer_reliability.py"], timeout=1800)  # 执行真实 API 答案、引用和拒答评估。
        answer_result["metrics"] = read_answer_metrics(extract_report_path(answer_result["output"], "答案可靠性 Markdown"))  # 读取答案可靠性指标。
        results.append(answer_result)  # 保存答案可靠性结果。
    performance_args = ["13_performance_benchmark.py"]  # 默认只执行不额外消耗 embedding 的性能基准。
    if args.with_api:  # 完整模式下增加已缓存向量检索的延迟基准。
        performance_args.append("--with-vector")  # 显式开启向量性能测试。
    results.append(run_command("性能基准", performance_args, timeout=240))  # 将性能基准纳入核心总结验收。
    cache_result = check_resource_cache()  # 在当前进程里验证资源对象的缓存复用。
    results.append({"label": "资源缓存检查", "passed": cache_result["passed"], "returncode": 0 if cache_result["passed"] else 1, "elapsed": 0.0, "output": ""})  # 把缓存检查纳入统一结果。
    if args.with_api:  # API 模式下继续执行真实端到端流程。
        results.extend(run_e2e_smoke())  # 把每一条端到端结果加入报告。
    report = build_report(results, cache_result, args.with_api)  # 生成总结报告正文。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成微秒级报告时间戳。
    report_path = OUTPUT_ROOT / f"core_audit_report_{timestamp}.md"  # 拼出本次总结报告路径。
    report_path.write_text(report, encoding="utf-8")  # 将总结报告写入磁盘。
    print(f"核心总结测试报告：{report_path}")  # 输出报告路径。
    print(f"总体测试状态：{'通过' if all(item['passed'] for item in results) else '存在失败'}")  # 输出整体状态。


if __name__ == "__main__":  # 判断当前脚本是否被直接执行。
    main()  # 直接执行时运行总结测试。
