"""稳定的 RAG 查询服务接口。"""  # 说明本模块只负责把现有 LangGraph 能力包装成稳定的结构化接口。

import importlib  # 导入 importlib，用来加载数字开头的现有主流程模块。
import math  # 导入 math，用来校验接口传入的超时是否为有限正数。
from collections.abc import Mapping  # 导入 Mapping，让接口同时支持字典和其他映射对象。
from dataclasses import dataclass  # 导入 dataclass，定义类型明确的 Python 调用请求。
from typing import Any  # 导入 Any，标注 JSON 友好字典中的动态值。

from config import OUTPUT_ROOT  # 导入输出目录，用来返回请求统计文件路径。
from config import REQUEST_DEADLINE_SECONDS  # 导入默认总截止时间，保持接口和 CLI 默认值一致。
from app_logging import get_logger  # 导入统一日志器，记录接口级异常但不记录完整问题正文。
from request_governance import RequestGovernanceError  # 导入请求治理异常，给接口返回稳定错误码。
from request_governance import RequestDeadlineExceeded  # 导入总截止时间异常，准确标记 timeout 字段。
from request_governance import request_id_for  # 导入已有请求指纹函数，保证接口和日志使用同一 request_id。


query_graph = importlib.import_module("03_query_graph")  # 复用现有 LangGraph 主流程，避免服务层复制检索和回答逻辑。
logger = get_logger(__name__)  # 创建服务接口专用日志器。


@dataclass(frozen=True)
class QueryRequest:  # 定义 Python 调用方可以使用的强类型请求对象。
    question: str  # 保存用户原始问题，支持单题或包含多个问题的文本。
    session_name: str | None = None  # 保存学习会话名称，未提供时使用项目现有默认会话行为。
    chapter: str | None = None  # 保存可选章节过滤条件，例如“第17章”。
    timeout_seconds: float | None = None  # 保存可选的单次查询总时限，未提供时使用配置默认值。


def _request_values(request: QueryRequest | Mapping[str, Any]) -> tuple[str, str | None, str | None, Any]:  # 把两种输入形式统一成内部字段。
    if isinstance(request, QueryRequest):  # 如果调用方传入了强类型请求。
        return request.question, request.session_name, request.chapter, request.timeout_seconds  # 直接返回类型明确的字段。
    if isinstance(request, Mapping):  # 如果调用方传入 JSON 反序列化后的字典。
        return request.get("question", ""), request.get("session_name"), request.get("chapter"), request.get("timeout_seconds")  # 读取固定接口字段。
    raise TypeError("request 必须是 QueryRequest 或字典。")  # 拒绝不明确的调用对象，避免静默读取错误字段。


def _normalize_text(value: Any) -> str | None:  # 规范化可选文本字段。
    if value is None:  # 未提供可选字段时。
        return None  # 保留缺省语义。
    return str(value).strip()  # 统一转换成去除首尾空白的字符串。


def _normalize_timeout(value: Any) -> float:  # 规范化接口总时限并执行边界校验。
    if value is None or value == "":  # 未提供超时时使用项目默认配置。
        return float(REQUEST_DEADLINE_SECONDS)  # 返回当前配置中的默认总时限。
    try:  # 保护外部 JSON 数值转换。
        timeout = float(value)  # 把整数或字符串数字转换成浮点秒数。
    except (TypeError, ValueError) as exc:  # 捕获无法转换的接口输入。
        raise ValueError("timeout_seconds 必须是正数。") from exc  # 返回稳定的参数错误，不泄露底层转换堆栈。
    if not math.isfinite(timeout) or timeout <= 0:  # 拒绝零、负数、无穷和 NaN。
        raise ValueError("timeout_seconds 必须是有限正数。")  # 避免查询立即超时或永不结束。
    return timeout  # 返回已经校验过的总时限。


def _base_response(request_id: str, question: str, timeout_seconds: float) -> dict[str, Any]:  # 创建固定返回结构的共同字段。
    return {  # 所有成功、降级和错误结果都保持相同顶层字段。
        "schema_version": "1.0",  # 固定接口契约版本，未来字段变更时通过版本演进。
        "request_id": request_id,  # 返回整条查询的稳定短指纹，便于关联日志和 metrics。
        "status": "error",  # 默认按错误初始化，成功路径会明确覆盖。
        "question": question,  # 返回原始输入问题，便于前端展示和审计。
        "answer": "",  # 保存单题或多题统一回答正文。
        "sub_answers": [],  # 保存每个拆分子问题的结构化结果。
        "citations": [],  # 保存带问题序号和局部引用编号的原文元数据。
        "report_path": "",  # 保存单题报告或多题总报告的绝对路径。
        "session_turn_path": "",  # 保存会话 turn Markdown 的绝对路径。
        "chapter_filter": "",  # 保存最终解析后的标准章节名。
        "request": {  # 保存请求治理层字段，不把治理信息散落到多个顶层字段。
            "deadline_seconds": timeout_seconds,  # 返回本次查询使用的总时限。
            "timed_out": False,  # 标记是否因总时限主动结束。
            "error_type": "",  # 保存稳定治理异常类型，正常完成时为空。
        },  # 请求治理字段结束。
        "metrics": {},  # 保存 request_metrics.jsonl 中同口径的统计快照。
        "warnings": [],  # 保存报告或会话写入失败等不影响答案的提示。
        "error": None,  # 成功和安全降级返回 None，接口级异常时填充错误对象。
    }  # 固定返回结构结束。


def _error_response(request_id: str, question: str, timeout_seconds: float, code: str, error_type: str, message: str) -> dict[str, Any]:  # 构造不抛异常的稳定错误响应。
    response = _base_response(request_id, question, timeout_seconds)  # 先创建完整字段集合。
    response["error"] = {"code": code, "type": error_type, "message": message}  # 只返回安全、稳定、可展示的错误信息。
    return response  # 返回错误响应。


def _citation_metadata(result: dict[str, Any], question_index: int) -> list[dict[str, Any]]:  # 把单题上下文转成前端可直接使用的引用元数据。
    records: list[dict[str, Any]] = []  # 准备保存当前子问题的局部引用。
    question = result.get("resolved_question") or result.get("question", "")  # 优先围绕结合记忆后的问题裁剪引用窗口。
    for citation_index, doc in enumerate(result.get("contexts", []), start=1):  # 为每个上下文分配当前子问题内的引用编号。
        metadata = doc.metadata  # 读取索引阶段保存的稳定元数据。
        chapter = metadata.get("chapter", "")  # 读取章节名。
        section = metadata.get("section", "")  # 读取小节名。
        snippet = query_graph.extract_relevant_window(doc.page_content, query_graph.extract_terms(question + chapter + section), query_graph.MAX_CONTEXT_CHARS)  # 复用报告使用的窗口裁剪逻辑，避免接口返回整段无关正文。
        records.append({  # 组织一个可审计的引用对象。
            "question_index": question_index,  # 标记引用属于哪个子问题。
            "citation_id": citation_index,  # 保存答案中使用的局部 `[n]` 编号。
            "chapter": chapter,  # 返回教材章节。
            "section": section,  # 返回教材小节。
            "source": metadata.get("relative_path", metadata.get("source_docx", "")),  # 返回源 Word 相对路径。
            "source_file_id": metadata.get("source_file_id", ""),  # 返回源文件稳定 ID。
            "chunk_id": metadata.get("chunk_id", ""),  # 返回 chunk 稳定 ID。
            "locator": query_graph.format_source_locator(doc),  # 返回章节、片段序号和字符范围组成的定位信息。
            "snippet": snippet,  # 返回围绕问题裁剪的原文窗口。
        })  # 一个引用对象结束。
    return records  # 返回当前子问题的引用元数据。


def _sub_answer(result: dict[str, Any], question_index: int) -> dict[str, Any]:  # 把内部 RagState 映射成稳定的子答案对象。
    citation_validation = result.get("citation_validation") or {}  # 读取严格引用校验结果。
    template_validation = result.get("template_validation") or {}  # 读取学习模板结构校验结果。
    request_error = result.get("request_error", "")  # 读取请求治理主动中断类型。
    if request_error:  # 请求超时或熔断时。
        status = "degraded"  # 仍可能保留教材证据，但不生成未经确认的回答。
    elif not result.get("evidence_enough", False):  # 证据门没有通过时。
        status = "partial"  # 表示系统返回了保守结果或近似证据。
    elif citation_validation and not citation_validation.get("passed", False):  # 模型答案存在引用安全门失败时。
        status = "degraded"  # 表示回答已降级为证据或服务异常提示。
    elif template_validation and not template_validation.get("passed", False):  # 模板结构没有通过时。
        status = "degraded"  # 表示没有返回未验证的自由回答。
    else:  # 证据、引用和模板都通过时。
        status = "ok"  # 标记为可直接消费的学习回答。
    return {  # 返回稳定子答案契约。
        "index": question_index,  # 保存子问题序号。
        "question": result.get("question", ""),  # 保存子问题原文。
        "resolved_question": result.get("resolved_question", ""),  # 保存记忆补全后的问题。
        "question_type": result.get("question_type", ""),  # 保存问题类型。
        "status": status,  # 保存子问题状态。
        "answer": result.get("answer", ""),  # 保存当前子问题回答。
        "evidence_score": result.get("evidence_score", 0),  # 保存证据分。
        "evidence_enough": bool(result.get("evidence_enough", False)),  # 保存证据门结果。
        "retrieval_log": list(result.get("retrieval_log", [])),  # 保存最多三轮检索日志。
        "citation_validation": citation_validation,  # 返回完整引用校验明细，便于前端展示可信度。
        "template_validation": template_validation,  # 返回完整模板校验明细，便于前端按题型渲染。
        "request_error": request_error,  # 返回稳定的治理异常类型。
        "report_path": result.get("report_path", ""),  # 返回单题报告路径。
        "citations": _citation_metadata(result, question_index),  # 返回当前子问题的引用元数据。
    }  # 子答案对象结束。


def _overall_status(sub_answers: list[dict[str, Any]]) -> str:  # 根据子问题状态计算统一查询状态。
    statuses = [item["status"] for item in sub_answers]  # 提取每个子问题状态。
    if statuses and all(status == "ok" for status in statuses):  # 所有子问题都通过完整安全门时。
        return "ok"  # 整条请求可以视为完整成功。
    if any(status == "ok" for status in statuses):  # 至少有一个子问题成功、其他题部分或降级时。
        return "partial"  # 表示统一回答可用但不是全部完整成功。
    return "degraded"  # 所有子问题都保守降级或证据不足时。


def query(request: QueryRequest | Mapping[str, Any]) -> dict[str, Any]:  # 定义未来 Web UI 和小程序统一调用的同步查询入口。
    raw_question = ""  # 先准备原始问题，保证参数错误也能生成完整响应。
    session_name: str | None = None  # 先准备会话名称。
    try:  # 保护请求字段读取和规范化。
        raw_question, session_name, raw_chapter, raw_timeout = _request_values(request)  # 读取固定请求字段。
        raw_question = str(raw_question or "").strip()  # 统一问题为非空白字符串。
        session_name = _normalize_text(session_name)  # 规范化会话名称。
        chapter = _normalize_text(raw_chapter)  # 规范化章节条件。
        timeout_seconds = _normalize_timeout(raw_timeout)  # 规范化总截止时间。
    except (TypeError, ValueError) as exc:  # 捕获输入类型、超时和字段格式错误。
        request_id = request_id_for(raw_question, session_name)  # 即使参数错误也返回可关联的 request_id。
        return _error_response(request_id, raw_question, float(REQUEST_DEADLINE_SECONDS), "INVALID_REQUEST", type(exc).__name__, str(exc))  # 返回稳定参数错误。
    request_id = request_id_for(raw_question, session_name)  # 在进入检索前生成整条查询指纹。
    if not raw_question:  # 拒绝空问题，避免无意义消耗 embedding 和回答 token。
        return _error_response(request_id, raw_question, timeout_seconds, "INVALID_REQUEST", "ValueError", "question 不能为空。")  # 返回明确输入错误。
    response = _base_response(request_id, raw_question, timeout_seconds)  # 初始化成功和降级路径共用的返回结构。
    runtime = None  # 准备保存请求运行时，异常时仍可读取治理统计。
    try:  # 保护完整查询编排。
        chapter_filter = query_graph.resolve_chapter_filter(chapter)  # 把短章节名解析成唯一标准章节名。
        if chapter and not chapter_filter:  # 用户指定章节但无法唯一解析时。
            return _error_response(request_id, raw_question, timeout_seconds, "INVALID_CHAPTER", "ValueError", f"没有找到唯一章节：{chapter}。")  # 返回参数错误而不静默查询全书。
        response["chapter_filter"] = chapter_filter or ""  # 返回实际使用的标准章节过滤条件。
        with query_graph.request_scope(raw_question, session_name, deadline_seconds=timeout_seconds, request_id=request_id) as runtime:  # 让所有子问题共享本次接口总时限和同一个追踪编号。
            query_graph.require_embedding_config()  # 在请求上下文中检查 embedding 配置，失败也会留下治理统计。
            sub_questions = query_graph.split_questions(raw_question)  # 使用现有规则拆分单题或多题输入。
            app = query_graph.build_graph()  # 编译一次 LangGraph，供所有子问题复用。
            memory_context = query_graph.build_memory_context(session_name, raw_question)  # 读取当前会话需要的最小历史记忆。
            results = [  # 逐题执行相同的 LangGraph 核心流程。
                query_graph.answer_one_question(  # 调用现有单题执行函数，保留其局部降级行为。
                    app,  # 传入复用的 LangGraph。
                    item,  # 传入当前拆分后的子问题。
                    resolved_question=query_graph.resolve_follow_up_question(session_name, item),  # 补全“它/刚才”等上下文追问。
                    memory_context=memory_context,  # 传入最小会话记忆。
                    chapter_filter=chapter_filter or "",  # 传入标准章节过滤条件。
                )  # 单题调用结束。
                for item in sub_questions  # 遍历所有子问题。
            ]  # 子问题执行结束。
            combined_answer = query_graph.build_combined_answer(results)  # 复用现有多问题统一回答格式。
            combined_report_path = query_graph.save_combined_report(raw_question, combined_answer, results) if len(results) > 1 else results[0]["report_path"]  # 单题返回单题报告，多题返回总报告。
            response["answer"] = combined_answer  # 写入统一回答正文。
            response["sub_answers"] = [_sub_answer(result, index) for index, result in enumerate(results, start=1)]  # 写入结构化子答案。
            response["citations"] = [citation for item in response["sub_answers"] for citation in item["citations"]]  # 汇总所有子问题引用并保留 question_index 防止编号冲突。
            response["report_path"] = combined_report_path or ""  # 返回报告路径，写入失败时保持空字符串。
            if not combined_report_path:  # 报告写入失败但回答可能仍然可用时。
                response["warnings"].append("问答报告保存失败，详情请查看 logs/rag.log。")  # 让前端显示非致命提示。
            chapters = sorted({doc.metadata.get("chapter", "") for result in results for doc in result.get("contexts", []) if doc.metadata.get("chapter", "")})  # 汇总本轮引用章节。
            question_types = sorted({result.get("question_type", "") for result in results if result.get("question_type")})  # 汇总本轮问题类型。
            session_turn_path = query_graph.safe_append_turn(session_name, {  # 保存会话历史，保持和 CLI 相同的学习记忆行为。
                "question": raw_question,  # 保存原始接口问题。
                "sub_questions": sub_questions,  # 保存拆分后的子问题。
                "answer": combined_answer,  # 保存统一回答。
                "question_types": question_types,  # 保存问题类型集合。
                "chapters": chapters,  # 保存引用章节集合。
                "report_path": combined_report_path,  # 保存报告路径。
                "chapter_filter": chapter_filter or "",  # 保存章节过滤条件。
                "memory_used": bool(memory_context),  # 保存本轮是否使用会话记忆。
                "resolved_questions": [result["resolved_question"] for result in results],  # 保存记忆补全后的问题。
                "session_name": session_name or "",  # 保存会话名称。
                "request_id": request_id,  # 保存接口 request_id，便于会话和日志关联。
            })  # 会话追加结束。
            response["session_turn_path"] = session_turn_path  # 返回会话 turn 路径。
            if not session_turn_path and session_name:  # 指定会话但保存失败时。
                response["warnings"].append("学习会话保存失败，详情请查看 logs/rag.log。")  # 让前端知道记忆没有落盘。
            response["metrics"] = runtime.summary()  # 在请求上下文结束前读取本次治理统计快照。
            response["metrics"]["metrics_path"] = str(OUTPUT_ROOT / "request_metrics.jsonl")  # 返回统一统计文件路径。
            response["status"] = _overall_status(response["sub_answers"])  # 根据全部子问题安全状态计算统一状态。
            response["request"]["timed_out"] = any(item["request_error"] == RequestDeadlineExceeded.__name__ for item in response["sub_answers"])  # 标记是否有子问题因总时限结束。
            response["request"]["error_type"] = next((item["request_error"] for item in response["sub_answers"] if item["request_error"]), "")  # 返回第一个稳定治理异常类型。
            return response  # 返回成功、部分成功或安全降级的结构化结果。
    except RequestGovernanceError as exc:  # 捕获未被单题局部逻辑消费的总时限或熔断异常。
        logger.warning("Service request_governance_degraded request_id=%s error_type=%s", request_id, type(exc).__name__)  # 记录稳定治理类型，不记录问题正文。
        response["status"] = "degraded"  # 说明接口没有生成未经确认的完整回答。
        response["request"]["timed_out"] = isinstance(exc, RequestDeadlineExceeded)  # 只有截止时间异常标记 timed_out。
        response["request"]["error_type"] = type(exc).__name__  # 返回稳定异常类型。
        response["error"] = {"code": "REQUEST_GOVERNANCE", "type": type(exc).__name__, "message": "查询被请求保护机制中止，已保留可用结果。"}  # 返回安全错误对象。
    except Exception as exc:  # 捕获接口层未预期异常，避免把 Python 堆栈暴露给前端。
        logger.exception("Service unhandled_failure request_id=%s error_type=%s", request_id, type(exc).__name__)  # 把完整堆栈写入统一日志。
        response["status"] = "error"  # 标记接口级错误。
        response["error"] = {"code": "INTERNAL_ERROR", "type": type(exc).__name__, "message": "查询未完成，详情请根据 request_id 查看 logs/rag.log。"}  # 返回不泄露敏感细节的错误。
    finally:  # 无论异常发生在哪一层，都尽量补齐治理统计。
        if runtime is not None and not response["metrics"]:  # 正常路径已经写入统计时不重复覆盖。
            response["metrics"] = runtime.summary()  # 异常路径返回截至当前的统计快照。
            response["metrics"]["metrics_path"] = str(OUTPUT_ROOT / "request_metrics.jsonl")  # 返回统计文件路径。
    return response  # 返回异常或降级响应，保持函数不抛出接口层异常。


__all__ = ["QueryRequest", "query"]  # 明确稳定公开的接口对象，避免调用方依赖内部辅助函数。
