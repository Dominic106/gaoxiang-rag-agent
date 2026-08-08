"""稳定 RAG 服务接口的离线契约回归测试。"""  # 说明本文件只测试结构化接口，不调用真实豆包或 DeepSeek API。

import importlib  # 导入 importlib，用来加载数字开头的现有主流程模块。
from unittest.mock import patch  # 导入 patch，用来替换 LangGraph 执行和文件写入边界。

from langchain_core.documents import Document  # 导入 Document，用来构造最小可引用教材片段。

import rag_service  # 导入待测稳定服务接口。
from rag_state import make_initial_state  # 导入初始状态工厂，构造符合主流程契约的模拟结果。


query_graph = importlib.import_module("03_query_graph")  # 加载现有主流程模块，供 patch 使用。


def fake_state(question: str, report_path: str) -> dict:  # 构造一个已经通过核心安全门的模拟子问题结果。
    state = make_initial_state(question)  # 先创建字段完整的 LangGraph 状态。
    state["question_type"] = "定义解释"  # 写入稳定问题类型。
    state["resolved_question"] = question  # 写入已经完成上下文补全的问题。
    state["query"] = question  # 写入模拟检索 query。
    state["evidence_score"] = 10  # 写入足够证据分。
    state["evidence_enough"] = True  # 打开证据门。
    state["answer"] = "定义结论：教材定义。[1]\n核心要点：教材要点。[1]\n教材依据：原文依据。[1]"  # 写入符合模板和引用格式的模拟答案。
    state["citation_validation"] = {"passed": True, "reason": "通过"}  # 写入引用校验通过结果。
    state["template_validation"] = {"passed": True, "template": "定义解释", "reason": "通过"}  # 写入模板校验通过结果。
    state["contexts"] = [  # 写入一条具有稳定定位字段的模拟教材片段。
        Document(  # 创建最小 LangChain 文档。
            page_content="教材定义：这是可追溯的测试原文。",  # 提供引用窗口需要的正文。
            metadata={  # 提供接口契约需要的索引元数据。
                "chapter": "第1章 测试章节",  # 保存章节名。
                "section": "1.1 测试小节",  # 保存小节名。
                "relative_path": "第1章 测试章节/1.1 测试小节.docx",  # 保存源文件相对路径。
                "source_file_id": "source-test-001",  # 保存源文件稳定 ID。
                "chunk_id": "chunk-test-001",  # 保存 chunk 稳定 ID。
                "chunk_index": 0,  # 提供片段序号。
                "chunk_count": 1,  # 提供片段总数。
                "source_char_start": 0,  # 提供原文起始字符位置。
                "source_char_end": 20,  # 提供原文结束字符位置。
            },  # 元数据结束。
        ),  # 模拟文档结束。
    ]  # 模拟上下文结束。
    state["retrieval_log"] = ["第 1 次检索：仅BM25，证据分=10，片段数=1"]  # 写入一条检索日志。
    state["report_path"] = report_path  # 写入单题报告路径。
    return dict(state)  # 转成普通字典返回，匹配服务层替身的宽松返回契约。


def patched_dependencies(split_result: list[str], states: list[dict]):  # 创建一组可复用的离线依赖替身。
    def fake_split(question: str) -> list[str]:  # 定义拆题替身。
        return split_result  # 返回测试指定的子问题列表。

    def fake_answer(app, question, resolved_question=None, memory_context="", chapter_filter=""):  # 定义单题 LangGraph 替身。
        return states[split_result.index(question)]  # 根据子问题顺序返回对应模拟状态。

    return {  # 返回 patch 所需的属性和值。
        "require_embedding_config": lambda: None,  # 跳过真实 API 配置检查。
        "build_graph": lambda: object(),  # 返回一个不访问网络的占位图。
        "split_questions": fake_split,  # 替换拆题函数。
        "build_memory_context": lambda session_name, question: "",  # 不读取真实会话文件。
        "resolve_follow_up_question": lambda session_name, question: question,  # 不做真实追问补全。
        "answer_one_question": fake_answer,  # 替换单题执行函数。
        "build_combined_answer": lambda results: "\n".join(result["answer"] for result in results),  # 返回可预测的统一答案。
        "save_combined_report": lambda question, answer, results: "/tmp/fake-combined-report.md",  # 返回可预测的总报告路径。
        "safe_append_turn": lambda session_name, turn: "/tmp/fake-turn.md",  # 返回可预测的会话路径。
    }  # 替身映射结束。


def error_code(response: dict) -> str:  # 从稳定接口错误响应中安全读取错误码，避免测试重复深入 Any 类型字典。
    error = response.get("error")  # 读取统一错误对象。
    return str(error.get("code", "")) if isinstance(error, dict) else ""  # 只有错误对象确实是字典时才读取 code。


def test_single_question_contract() -> None:  # 验证单题输入的固定返回结构。
    states = [fake_state("什么是测试定义？", "/tmp/fake-single-report.md")]  # 准备单题模拟状态。
    dependencies = patched_dependencies(["什么是测试定义？"], states)  # 创建单题替身。
    with patch.multiple(query_graph, **dependencies):  # 临时替换所有会访问真实核心资源的边界。
        response = rag_service.query({"question": "什么是测试定义？", "session_name": "contract-single", "timeout_seconds": 12})  # 调用 JSON 友好的稳定接口。
    required = {"schema_version", "request_id", "status", "question", "answer", "sub_answers", "citations", "report_path", "session_turn_path", "chapter_filter", "request", "metrics", "warnings", "error"}  # 定义接口必须返回的顶层字段。
    assert required.issubset(response), "单题响应缺少固定顶层字段"  # 确认前端不需要猜字段是否存在。
    assert response["status"] == "ok", "单题通过安全门后状态不应为错误"  # 确认成功状态映射正确。
    assert response["request_id"], "单题响应缺少 request_id"  # 确认日志关联字段存在。
    assert response["request"]["deadline_seconds"] == 12.0, "接口没有使用单次 timeout_seconds 覆盖值"  # 确认服务接口支持单次预算覆盖。
    assert response["report_path"] == "/tmp/fake-single-report.md", "单题没有返回单题报告路径"  # 确认报告可被前端打开或下载。
    assert response["sub_answers"][0]["citations"][0]["chunk_id"] == "chunk-test-001", "单题没有返回引用元数据"  # 确认引用具备稳定 chunk_id。
    assert response["error"] is None, "成功响应不应带接口错误"  # 确认错误字段保持统一的 null 语义。


def test_multi_question_contract() -> None:  # 验证多题输入可以统一返回且引用不会丢失问题序号。
    questions = ["问题一", "问题二"]  # 准备两个独立子问题。
    states = [fake_state(question, f"/tmp/fake-report-{index}.md") for index, question in enumerate(questions, start=1)]  # 为每个子问题准备独立状态。
    dependencies = patched_dependencies(questions, states)  # 创建多题替身。
    with patch.multiple(query_graph, **dependencies):  # 临时替换真实核心边界。
        response = rag_service.query(rag_service.QueryRequest(question="1. 问题一 2. 问题二", session_name="contract-multi"))  # 使用强类型请求调用接口。
    assert response["status"] == "ok", "多题全部通过安全门后状态不应降级"  # 确认多题统一状态正确。
    assert len(response["sub_answers"]) == 2, "多题响应没有保留两个子答案"  # 确认拆题结果可被前端逐题渲染。
    assert response["report_path"] == "/tmp/fake-combined-report.md", "多题没有返回总报告路径"  # 确认多题报告路径统一。
    assert [item["question_index"] for item in response["citations"]] == [1, 2], "多题引用没有保留所属问题序号"  # 确认局部引用编号不会跨题混淆。


def test_invalid_input_and_internal_error() -> None:  # 验证参数错误和内部异常都返回固定错误对象而不是抛给前端。
    invalid = rag_service.query({"question": "", "timeout_seconds": 10})  # 调用空问题场景。
    assert invalid["status"] == "error" and error_code(invalid) == "INVALID_REQUEST", "空问题没有返回 INVALID_REQUEST"  # 确认输入错误码稳定。
    with patch.object(query_graph, "resolve_chapter_filter", return_value=""):  # 模拟无法唯一解析的章节。
        chapter_error = rag_service.query({"question": "测试", "chapter": "不存在章节"})  # 调用章节参数错误场景。
    assert error_code(chapter_error) == "INVALID_CHAPTER", "无效章节没有返回 INVALID_CHAPTER"  # 确认章节错误不静默查全书。
    with patch.object(query_graph, "require_embedding_config", side_effect=RuntimeError("simulated internal failure")):  # 模拟接口编排内部异常。
        internal = rag_service.query({"question": "测试内部异常"})  # 调用内部异常场景。
    assert internal["status"] == "error" and error_code(internal) == "INTERNAL_ERROR", "内部异常没有返回 INTERNAL_ERROR"  # 确认前端不会收到未处理异常。
    assert internal["request_id"], "内部异常响应缺少 request_id"  # 确认错误也能关联日志。


def test_request_id_is_unique_per_execution() -> None:  # 验证重复问题不会复用同一个请求追踪编号。
    states = [fake_state("重复问题", "/tmp/fake-repeat-report.md")]  # 准备可重复使用的单题模拟状态。
    dependencies = patched_dependencies(["重复问题"], states)  # 创建不访问真实模型的接口替身。
    with patch.multiple(query_graph, **dependencies):  # 临时替换真实核心边界。
        first = rag_service.query({"question": "重复问题", "session_name": "contract-repeat", "timeout_seconds": 12})  # 执行第一次相同问题查询。
        second = rag_service.query({"question": "重复问题", "session_name": "contract-repeat", "timeout_seconds": 12})  # 执行第二次相同问题查询。
    assert first["request_id"] != second["request_id"], "重复请求不应共享同一个 request_id"  # 确保异常日志和指标可以区分两次真实执行。


def test_graph_is_cached_in_process() -> None:  # 验证常驻服务不会为每次请求重复编译同一个 LangGraph。
    query_graph.build_graph.cache_clear()  # 清理测试前的图缓存，保证断言从空缓存开始。
    first = query_graph.build_graph()  # 第一次创建编译图。
    second = query_graph.build_graph()  # 第二次读取同一进程中的编译图。
    assert first is second, "LangGraph 没有在同一进程内复用"  # 确认服务层不会重复承担图编译开销。


def main() -> None:  # 定义接口回归测试入口。
    test_single_question_contract()  # 执行单题契约测试。
    test_multi_question_contract()  # 执行多题契约测试。
    test_invalid_input_and_internal_error()  # 执行输入错误和内部异常测试。
    test_request_id_is_unique_per_execution()  # 执行请求追踪编号唯一性测试。
    test_graph_is_cached_in_process()  # 执行 LangGraph 缓存测试。
    print("稳定服务接口回归通过：单题、多题、引用元数据、超时字段和错误契约。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 直接运行时执行全部接口回归测试。
