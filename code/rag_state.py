from typing import TypedDict  # 导入 TypedDict，用来声明 LangGraph 状态的字段结构。

from langchain_core.documents import Document  # 导入 Document，用来标注检索文档列表的类型。


class RagState(TypedDict):  # 定义完整 RAG 流程共享的状态结构。
    question: str  # 保存用户原始问题。
    resolved_question: str  # 保存结合会话上下文后用于检索的完整问题。
    memory_context: str  # 保存本题使用的最小会话记忆。
    chapter_filter: str  # 保存标准化后的章节过滤条件。
    question_type: str  # 保存问题类型，例如定义解释、区别对比、流程步骤。
    understanding_reason: str  # 保存问题分类原因，方便调试。
    query: str  # 保存改写后的检索查询。
    retrieval_queries: list[str]  # 保存本题准备尝试的多种检索 query。
    attempt: int  # 保存当前是第几次检索尝试。
    vector_used: bool  # 保存当前检索轮次是否实际调用了向量检索。
    keyword_preloaded: bool  # 保存首轮是否已经用 BM25 预检索，避免后续重复调用。
    vector_docs: list[Document]  # 保存向量检索结果。
    keyword_docs: list[Document]  # 保存关键词检索结果。
    contexts: list[Document]  # 保存合并重排后的最终原文片段。
    candidate_docs: list[Document]  # 保存多轮检索累计得到的候选片段。
    candidate_scores: dict[str, float]  # 保存多轮检索累计分数。
    evidence_score: int  # 保存证据强度分，用来决定是否允许模型回答。
    evidence_enough: bool  # 保存证据是否足够。
    retrieval_log: list[str]  # 保存检索过程日志，方便复盘为什么查到或没查到。
    answer: str  # 保存最终回答。
    report_path: str  # 保存本次问答报告文件路径。
    citation_validation: dict  # 保存回答逐句引用校验结果。
    template_validation: dict  # 保存回答模板结构校验结果。
    request_error: str  # 保存请求治理层主动中断的异常类型，便于报告和会话复盘。


def make_initial_state(question: str, resolved_question: str | None = None, memory_context: str = "", chapter_filter: str = "") -> RagState:  # 创建单题流程的初始状态。
    return {  # 返回字段完整、可直接交给 LangGraph 的状态字典。
        "question": question,  # 写入用户问题。
        "resolved_question": resolved_question or question,  # 写入结合记忆后的完整查询问题。
        "memory_context": memory_context,  # 写入本题需要的最小历史记忆。
        "chapter_filter": chapter_filter,  # 写入标准章节过滤条件。
        "question_type": "",  # 问题类型先留空。
        "understanding_reason": "",  # 分类原因先留空。
        "query": "",  # 查询改写结果先留空。
        "retrieval_queries": [],  # 多次检索 query 先留空。
        "attempt": 0,  # 当前检索尝试次数先设为 0。
        "vector_used": False,  # 默认还没有调用向量检索。
        "keyword_preloaded": False,  # 默认还没有预加载 BM25 结果。
        "vector_docs": [],  # 向量结果先留空。
        "keyword_docs": [],  # 关键词结果先留空。
        "contexts": [],  # 最终上下文先留空。
        "candidate_docs": [],  # 多轮候选片段先留空。
        "candidate_scores": {},  # 多轮累计分数先留空。
        "evidence_score": 0,  # 证据分先设为 0。
        "evidence_enough": False,  # 默认认为证据不足，后面由检索节点更新。
        "retrieval_log": [],  # 检索日志先留空。
        "answer": "",  # 最终答案先留空。
        "report_path": "",  # 报告路径先留空。
        "citation_validation": {},  # 引用校验结果先留空。
        "template_validation": {},  # 模板结构校验结果先留空。
        "request_error": "",  # 请求治理错误先留空。
    }  # 初始状态结束。
