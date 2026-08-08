from datetime import datetime  # 导入 datetime，用来给每次问答报告生成时间戳文件名。
import argparse  # 导入 argparse，用来支持 session、back 等命令行参数。
import os  # 导入 os，用来在非豆包 embedding 模式下检查 OPENAI_API_KEY。
import re  # 导入 re，用来识别模型回答中的标题和引用格式。
from functools import lru_cache  # 导入 lru_cache，让常驻服务复用已经编译的 LangGraph。
from typing import Any  # 导入 Any，标注 Chroma 接受的混合检索参数。

from langchain_core.documents import Document  # 导入 Document，用来给检索结果做类型标注。
from langgraph.graph import END  # 导入 END，表示 LangGraph 流程结束。
from langgraph.graph import StateGraph  # 导入 StateGraph，用来搭建 RAG 流程图。

from config import BM25_RANK_WEIGHT  # 从配置读取 BM25 排名权重。
from config import EMBEDDING_PROVIDER  # 从配置读取 embedding provider。
from config import FINAL_CONTEXTS  # 从配置读取最终上下文数量。
from config import MAX_CONTEXT_CHARS  # 从配置读取单个上下文片段最大字符数。
from config import MIN_EVIDENCE_SCORE  # 从配置读取最低证据分。
from config import SECTION_MATCH_WEIGHT  # 从配置读取小节标题精确匹配加分。
from config import OUTPUT_ROOT  # 从配置读取输出目录，用来保存完整 RAG 问答报告。
from config import TOP_K_KEYWORD  # 从配置读取关键词检索数量。
from config import TOP_K_VECTOR  # 从配置读取向量检索数量。
from config import TERM_HIT_WEIGHT  # 从配置读取查询词命中加分权重。
from config import VECTOR_RANK_WEIGHT  # 从配置读取向量排名权重。
from app_logging import get_logger  # 导入统一日志器，用来记录检索、生成和文件异常。
from chapter_filter import resolve_chapter_filter  # 导入章节过滤解析函数。
from citation_validator import validate_answer  # 导入严格引用校验函数。
from answer_templates import build_template_instructions  # 导入主回答模板说明。
from answer_templates import build_template_repair_instructions  # 导入引用修复阶段的模板说明。
from answer_templates import normalize_repaired_template  # 导入只补结构标签的确定性修复函数。
from answer_templates import validate_template_structure  # 导入模板结构校验函数。
from deepseek_llm import call_deepseek  # 导入 DeepSeek 调用函数，用它负责最终答案生成。
from question_splitter import split_questions  # 导入多问题拆分函数，支持用户一次问多个知识点。
from query_understanding import understand_query  # 导入问题理解函数，让完整 RAG 也使用问题分类和 query 增强。
from rag_evidence import build_retrieval_queries  # 导入检索 query 构造函数，隔离问题理解和检索编排。
from rag_evidence import choose_context_limit  # 导入动态上下文数量函数。
from rag_evidence import extract_relevant_window  # 导入引用窗口裁剪函数。
from rag_evidence import extract_terms  # 导入查询关键词抽取函数。
from rag_evidence import score_evidence  # 导入证据评分函数。
from retrieval_resources import get_bm25_retriever  # 导入全量 BM25 文档，用来发现 query 明确点名的小节。
from retrieval_resources import get_keyword_retriever  # 导入统一的关键词检索器入口。
from retrieval_resources import get_section_documents  # 导入同小节上下文补全入口。
from retrieval_resources import get_vector_store  # 导入缓存的 Chroma 向量库。
from source_trace import build_binding_report  # 导入逐结论引用绑定报告生成函数。
from source_trace import build_trace_report  # 导入前后文追溯报告生成函数。
from source_trace import format_source_locator  # 导入稳定原文定位格式化函数。
from study_memory import append_turn  # 导入会话追加函数，用来保存长期学习记录。
from study_memory import build_memory_context  # 导入最小会话记忆函数，用来支持上下文追问。
from study_memory import latest_turn  # 导入读取上一轮函数，用来支持回看。
from study_memory import resolve_follow_up_question  # 导入追问补全函数，把“它/刚才”补成完整查询。
from study_memory import rollback_last_turn  # 导入回退函数，用来支持撤销上一轮。
from rag_state import RagState  # 导入 LangGraph 状态类型，避免主编排文件维护字段定义。
from rag_state import make_initial_state  # 导入初始状态工厂，避免主编排文件堆积状态默认值。
from request_governance import RequestGovernanceError  # 导入请求治理异常，让单个子问题可以安全降级而不拖垮整条查询。
from request_governance import current_request_summary  # 导入当前请求统计摘要，用于写入问答报告。
from request_governance import ensure_request_budget  # 导入统一总截止时间检查函数。
from request_governance import request_scope  # 导入完整用户查询上下文，覆盖多问题和引用修复。


logger = get_logger(__name__)  # 创建当前主流程日志器。


def require_embedding_config() -> None:  # 定义检查 embedding 配置的函数。
    if EMBEDDING_PROVIDER == "doubao":  # 如果使用豆包 embedding。
        if not os.environ.get("DOUBAO_API_KEY"):  # 如果豆包模式却没有配置 Key。
            raise RuntimeError("请先在 code/.env 里填写 DOUBAO_API_KEY，否则无法使用豆包 embedding 做向量检索。")  # 提前给出清晰错误，避免运行到向量调用才失败。
        return  # 豆包 embedding 不要求 OPENAI_API_KEY。
    if not os.environ.get("OPENAI_API_KEY"):  # 如果环境变量里没有 OPENAI_API_KEY。
        raise RuntimeError("请先设置 OPENAI_API_KEY，否则无法使用 OpenAI embedding 做向量检索。")  # 抛出清晰错误。


def rewrite_query(state: RagState) -> RagState:  # 定义查询改写节点。
    ensure_request_budget("rewrite_query")  # 在每个子问题开始前检查整条用户查询是否已经超时。
    question = state["resolved_question"].strip()  # 取出已经结合会话上下文补全的问题。
    understanding = understand_query(question)  # 调用规则版问题理解器。
    state["question_type"] = understanding.question_type  # 保存问题类型。
    state["understanding_reason"] = understanding.reason  # 保存分类原因。
    state["query"] = understanding.expanded_query  # 把增强后的查询写回 LangGraph 状态。
    state["retrieval_queries"] = build_retrieval_queries(question, understanding.expanded_query)  # 构造最多三次检索自救 query。
    state["attempt"] = 0  # 当前尝试从第 0 次开始。
    state["retrieval_log"] = []  # 初始化检索日志。
    return state  # 返回更新后的状态。


def retrieve_vector(state: RagState) -> RagState:  # 定义向量检索节点。
    ensure_request_budget("retrieve_vector")  # 在进入向量或 BM25 预检前检查剩余预算。
    query = state["retrieval_queries"][state["attempt"]]  # 取出当前这一轮要使用的检索 query。
    if state["attempt"] > 0:  # 如果不是第一轮检索。
        state["vector_docs"] = []  # 后续自救轮次先不做向量检索，避免重复消耗 embedding token。
        state["vector_used"] = False  # 标记本轮没有使用向量检索。
        return state  # 直接返回状态，让 BM25 继续兜底检索。
    try:  # 保护首轮 BM25 预检索，索引损坏时仍然尝试向量兜底。
        keyword_retriever = get_keyword_retriever(state["chapter_filter"])  # 首轮先获取低成本的 BM25 检索器。
        keyword_retriever.k = TOP_K_KEYWORD  # 设置预检索返回数量。
        keyword_docs = keyword_retriever.invoke(query)  # 先用 BM25 判断固定术语是否已经足够命中。
    except Exception as exc:  # 捕获 BM25 文件损坏、反序列化或检索异常。
        logger.exception("BM25 precheck_failed attempt=%d error_type=%s", state["attempt"] + 1, type(exc).__name__)  # 记录 BM25 失败堆栈。
        keyword_docs = []  # BM25 失败时用空结果继续尝试向量检索。
    state["keyword_docs"] = keyword_docs  # 暂存 BM25 结果，后面的关键词节点直接复用。
    state["keyword_preloaded"] = True  # 无论是否继续向量检索，后续关键词节点都复用这批结果。
    keyword_score = score_evidence(state["resolved_question"], keyword_docs)  # 只用 BM25 结果计算一次初步证据分。
    if keyword_score >= MIN_EVIDENCE_SCORE:  # 如果便宜的 BM25 已经有足够证据。
        state["vector_docs"] = []  # 不再调用向量模型，避免无意义的 embedding token。
        state["vector_used"] = False  # 标记本轮采用 BM25 直接回答。
        return state  # 直接进入关键词节点和合并评分节点。
    try:  # 保护向量库连接和 embedding 请求。
        vector_store = get_vector_store()  # 复用本进程已经加载的 Chroma 向量库和 embedding 对象。
        search_kwargs: dict[str, Any] = {"k": TOP_K_VECTOR}  # 准备向量检索参数。
        if state["chapter_filter"]:  # 如果用户指定了章节过滤。
            search_kwargs["filter"] = {"chapter": state["chapter_filter"]}  # 让 Chroma 直接在指定章节内做语义检索。
        docs = vector_store.similarity_search(query, **search_kwargs)  # 用当前检索 query 做语义相似检索。
    except Exception as exc:  # 捕获向量库、embedding 超时、鉴权和响应异常。
        logger.exception("Vector retrieval_failed attempt=%d error_type=%s", state["attempt"] + 1, type(exc).__name__)  # 记录向量检索失败堆栈。
        docs = []  # 向量失败时保留 BM25 结果，后续由证据门禁决定是否拒答。
    state["vector_docs"] = docs  # 把向量检索结果写回状态。
    state["vector_used"] = bool(docs)  # 只有实际拿到向量结果时才标记为使用，避免异常降级报告误称向量检索成功。
    return state  # 返回更新后的状态。


def retrieve_keyword(state: RagState) -> RagState:  # 定义关键词检索节点。
    ensure_request_budget("retrieve_keyword")  # 在关键词检索前检查剩余预算，避免自救轮次无限拖延。
    query = state["retrieval_queries"][state["attempt"]]  # 取出当前这一轮要使用的检索 query。
    if state["keyword_preloaded"]:  # 如果首轮已经完成 BM25 预检索。
        state["keyword_preloaded"] = False  # 消费这次预加载标记，下一轮自救仍会重新检索。
        return state  # 直接复用已经写入状态的关键词结果。
    try:  # 保护后续 BM25 自救检索。
        retriever = get_keyword_retriever(state["chapter_filter"])  # 按条件复用全量或章节限定检索器。
        retriever.k = TOP_K_KEYWORD  # 设置 BM25 返回数量。
        docs = retriever.invoke(query)  # 用当前检索 query 做关键词检索。
    except Exception as exc:  # 捕获自救阶段的索引或检索异常。
        logger.exception("BM25 retry_failed attempt=%d error_type=%s", state["attempt"] + 1, type(exc).__name__)  # 记录自救检索失败堆栈。
        docs = []  # 自救失败时让证据评分走保守拒答路径。
    state["keyword_docs"] = docs  # 把关键词检索结果写回状态。
    return state  # 返回更新后的状态。


def merge_and_score(state: RagState) -> RagState:  # 定义合并和简单重排节点。
    ensure_request_budget("merge_and_score")  # 在候选合并前检查剩余预算。
    merged: dict[str, Document] = {doc.metadata["chunk_id"]: doc for doc in state["candidate_docs"]}  # 载入前几轮候选片段，保证自救检索可以累计召回。
    scores: dict[str, float] = {chunk_id: score * 0.85 for chunk_id, score in state["candidate_scores"].items()}  # 对旧轮次分数轻微衰减，让新 query 的结果优先。
    query = state["retrieval_queries"][state["attempt"]]  # 取出当前检索 query。
    query_terms = extract_terms(query)  # 抽取更稳定的查询关键词，用来计算命中加分。
    for rank, doc in enumerate(state["vector_docs"]):  # 遍历向量检索结果。
        chunk_id = doc.metadata["chunk_id"]  # 取出 chunk_id。
        merged[chunk_id] = doc  # 保存文档，后出现的同 ID 会覆盖但内容一致。
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (TOP_K_VECTOR - rank) * VECTOR_RANK_WEIGHT  # 给向量结果按排名加分。
    for rank, doc in enumerate(state["keyword_docs"]):  # 遍历关键词检索结果。
        chunk_id = doc.metadata["chunk_id"]  # 取出 chunk_id。
        merged[chunk_id] = doc  # 保存文档。
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (TOP_K_KEYWORD - rank) * BM25_RANK_WEIGHT  # 给关键词结果按排名加分，固定术语稍微加权。
    for chunk_id, doc in merged.items():  # 遍历去重后的候选片段。
        title_text = f"{doc.metadata.get('chapter', '')} {doc.metadata.get('section', '')}"  # 拼出标题文本。
        hit_count = sum(1 for term in query_terms if term in title_text or term in doc.page_content)  # 统计查询词命中次数。
        scores[chunk_id] = scores.get(chunk_id, 0.0) + hit_count * TERM_HIT_WEIGHT  # 标题或正文命中查询词则加分。
        normalized_query = query.replace(" ", "").replace("\n", "").casefold()  # 去掉增强 query 的空格和换行，便于匹配完整教材小节标题。
        normalized_section = doc.metadata.get("section_title", doc.metadata.get("section", "")).replace(" ", "").replace("\n", "").casefold()  # 归一化候选片段的小节标题，优先使用不带编号的 section_title。
        if normalized_section and normalized_section in normalized_query:  # 用户问题经过术语增强后明确指向当前小节时。
            scores[chunk_id] = scores.get(chunk_id, 0.0) + SECTION_MATCH_WEIGHT  # 提升该小节候选，减少相似章节抢占上下文。
    seed_ids = sorted(scores, key=lambda chunk_id: scores.get(chunk_id, 0.0), reverse=True)[:FINAL_CONTEXTS]  # 先取当前候选中排名靠前的片段作为小节补全锚点。
    normalized_query = query.replace(" ", "").replace("\n", "").casefold()  # 归一化当前 query，用于判断是否有明确的小节标题指向。
    all_section_keys = {(  # 建立全量“章节 + 小节”索引，避免明确的小节尚未进入当前候选时无法被发现。
        doc.metadata.get("chapter", ""),  # 保存小节所属章节。
        doc.metadata.get("section", ""),  # 保存带编号的小节名称。
        doc.metadata.get("section_title", doc.metadata.get("section", "")),  # 保存不带编号的小节标题。
    ) for doc in get_bm25_retriever().docs}  # 从缓存的 BM25 文档读取元数据，不产生 embedding 请求。
    exact_section_keys = [  # 查找 query 明确点名的小节。
        (chapter, section)  # 只保留章节和带编号的小节名称供后续加载。
        for chapter, section, section_title in all_section_keys  # 遍历全量小节元数据。
        if (not state["chapter_filter"] or chapter == state["chapter_filter"])  # 章节过滤存在时不允许跨章节误命中。
        and section_title.replace(" ", "").replace("\n", "").casefold() in normalized_query  # 小节无编号标题完整出现在增强 query 中才算精确锚点。
    ]  # 全量精确小节收集结束。
    exact_section_keys.sort(key=lambda item: len(item[1]), reverse=True)  # 优先处理标题更长、更具体的小节，避免短标题抢占上下文。
    seed_section_keys = []  # 准备把当前高分候选的小节转成带章节的稳定键。
    for chunk_id in seed_ids:  # 遍历当前高分候选片段。
        key = (merged[chunk_id].metadata.get("chapter", ""), merged[chunk_id].metadata.get("section", ""))  # 读取候选片段的章节和小节。
        if key[0] and key[1] and key not in seed_section_keys:  # 只保留有效且不重复的小节。
            seed_section_keys.append(key)  # 保存当前小节键。
    exact_section_limit = 5 if state["question_type"] in {"考点记忆", "输入输出工具技术"} else 3  # 跨小节列表题允许覆盖更多明确小节，普通问题仍限制补全范围控制成本。
    sections_to_augment = exact_section_keys[:exact_section_limit] if exact_section_keys else seed_section_keys[:3]  # 明确小节优先；否则沿用高分候选的小节补全策略。
    exact_boost_base = max(scores.values(), default=0.0) + SECTION_MATCH_WEIGHT * 3  # 精确小节需要明显压过旧轮次的相似章节，避免“相关但不是答案出处”的片段抢占上下文。
    for section_index, (chapter, section) in enumerate(sections_to_augment):  # 按精确小节补全文本，列表题可以覆盖多个章节结构单元。
        section_documents = list(get_section_documents(chapter, section))  # 读取该小节按原文排序的全部 chunk。
        section_boost = exact_boost_base - section_index * SECTION_MATCH_WEIGHT * 0.5  # 按小节具体程度保持稳定优先级，而不是让后处理的小节覆盖先处理的小节。
        section_documents = sorted(  # 在精确小节内部再按当前 query 的词命中数排序，避免公式和列表总在小节后半段却因机械取开头而丢失。
            section_documents,  # 使用该小节的全部原文片段作为候选。
            key=lambda doc: (sum(term in doc.page_content for term in query_terms), -int(doc.metadata.get("chunk_index", 0))),  # 先保证问题词命中，再用原文顺序打破平局。
            reverse=True,  # 命中更多的片段优先。
        )  # 小节内重排结束。
        for position, doc in enumerate(section_documents[:FINAL_CONTEXTS]):  # 精确小节最多补回一个上下文窗口，列表题需要跨多个 chunk 才能完整回答。
            chunk_id = doc.metadata["chunk_id"]  # 读取补全文档的稳定 ID。
            if chunk_id not in merged:  # 只补充当前候选中还没有的 chunk。
                merged[chunk_id] = doc  # 将小节起始文档加入候选集合。
            scores[chunk_id] = max(scores.get(chunk_id, 0.0), section_boost - position * 0.1)  # 精确小节中的原文顺序优先，同时不降低已经更高的有效召回分。
    state["candidate_docs"] = list(merged.values())  # 保存累计候选片段，供下一轮自救继续使用。
    state["candidate_scores"] = scores  # 保存累计分数，供下一轮重排继续使用。
    ordered_ids = sorted(scores, key=lambda chunk_id: scores.get(chunk_id, 0.0), reverse=True)  # 按综合分从高到低排序 chunk_id。
    if exact_section_keys:  # 如果 query 明确点名了教材小节。
        exact_ids = [chunk_id for chunk_id in ordered_ids if (merged[chunk_id].metadata.get("chapter", ""), merged[chunk_id].metadata.get("section", "")) in exact_section_keys]  # 先取精确小节中的片段。
        other_ids = [chunk_id for chunk_id in ordered_ids if chunk_id not in exact_ids]  # 再保留其他片段作为补充对照证据。
        ordered_ids = exact_ids + other_ids  # 精确小节优先进入模型上下文，防止相似章节污染答案。
    if exact_section_keys and state["question_type"] in {"考点记忆", "输入输出工具技术"} and len(exact_section_keys) >= 4:  # 多小节列表题需要保证每个明确小节至少贡献一段，而不是被单一小节的连续片段占满。
        diversified_ids = []  # 准备按小节轮询收集首个高分片段。
        for section_key in exact_section_keys:  # 遍历用户问题明确指向的小节。
            first_id = next((chunk_id for chunk_id in ordered_ids if (merged[chunk_id].metadata.get("chapter", ""), merged[chunk_id].metadata.get("section", "")) == section_key), None)  # 取当前小节综合分最高的片段。
            if first_id and first_id not in diversified_ids:  # 只保留存在且未重复的片段。
                diversified_ids.append(first_id)  # 先放入每个小节的代表片段。
        ordered_ids = diversified_ids + [chunk_id for chunk_id in ordered_ids if chunk_id not in diversified_ids]  # 代表片段优先，其余片段继续按综合分补充。
    selected = [merged[chunk_id] for chunk_id in ordered_ids[:FINAL_CONTEXTS]]  # 先取最多 FINAL_CONTEXTS 个片段。
    evidence_score = score_evidence(state["resolved_question"], selected)  # 根据完整问题和候选片段计算证据强度。
    exact_context_hit = any((doc.metadata.get("chapter", ""), doc.metadata.get("section", "")) in exact_section_keys for doc in selected)  # 判断最终上下文是否真正包含 query 点名的小节。
    if exact_context_hit and exact_section_keys:  # 小节标题是教材结构中的高置信度证据，即使问题只有一个缩写词也不应被普通词命中分误判为证据不足。
        evidence_score = max(evidence_score, MIN_EVIDENCE_SCORE)  # 将证据分抬到最低可回答门槛，不绕过小节和引用校验。
    context_limit = choose_context_limit(state["question_type"], evidence_score)  # 根据问题类型和证据分动态决定上下文数量。
    if len(exact_section_keys) >= 4 and state["question_type"] in {"考点记忆", "输入输出工具技术"}:  # 明确点名四个以上小节时，为完整列举保留第五段上下文。
        context_limit = max(context_limit, min(FINAL_CONTEXTS, 5))  # 仅对这类低频完整清单题增加一段，避免全局 token 成本上升。
    state["contexts"] = selected[:context_limit]  # 只保留必要数量的上下文，节省 token。
    state["evidence_score"] = evidence_score  # 保存证据分。
    state["evidence_enough"] = evidence_score >= MIN_EVIDENCE_SCORE  # 判断证据是否足够。
    vector_note = "向量+BM25" if state["vector_used"] else "仅BM25"  # 根据本轮是否用了向量检索生成说明。
    log_line = f"第 {state['attempt'] + 1} 次检索：{vector_note}，query={query}，证据分={evidence_score}，片段数={len(state['contexts'])}"  # 组织检索日志。
    state["retrieval_log"].append(log_line)  # 写入检索日志。
    return state  # 返回更新后的状态。


def should_retry(state: RagState) -> str:  # 定义 LangGraph 条件判断函数，决定是否继续检索自救。
    if state["evidence_enough"]:  # 如果证据已经足够。
        return "answer"  # 进入答案生成。
    if state["attempt"] + 1 >= len(state["retrieval_queries"]):  # 如果已经没有下一种 query 可试。
        return "answer"  # 进入答案生成，但会走拒答逻辑。
    return "retry"  # 否则继续下一轮检索。


def advance_attempt(state: RagState) -> RagState:  # 定义进入下一轮检索的节点。
    state["attempt"] += 1  # 当前尝试次数加一。
    return state  # 返回更新后的状态。


def format_citation(index: int, doc: Document, question: str) -> str:  # 定义把一个检索片段格式化成引用文本的函数。
    metadata = doc.metadata  # 取出元数据。
    source = metadata.get("relative_path", metadata.get("source_docx", ""))  # 优先展示相对路径，没有就展示绝对路径。
    chapter = metadata.get("chapter", "")  # 取出章名。
    section = metadata.get("section", "")  # 取出小节名。
    snippet = extract_relevant_window(doc.page_content, extract_terms(question + chapter + section), MAX_CONTEXT_CHARS)  # 围绕用户问题关键词截取短窗口，降低 token 成本。
    chunk_id = metadata.get("chunk_id", "")  # 取出 chunk_id，便于之后精确追踪原文块。
    locator = format_source_locator(doc)  # 生成章节、小节、源文件、片段序号和字符范围组成的稳定定位。
    return f"[{index}] {chapter} / {section} / {source} / chunk_id={chunk_id}\n原文位置：{locator}\n原文片段：{snippet}..."  # 返回带稳定位置和原文窗口的引用字符串。


def append_context_citations(answer: str, context_count: int) -> str:  # 定义无新增模型调用的引用格式修复函数。
    available = "".join(f"[{index}]" for index in range(1, context_count + 1))  # 只使用当前回答上下文中真实存在的引用编号。
    repaired_lines: list[str] = []  # 准备保存补齐引用后的回答行。
    for line in answer.splitlines():  # 按行处理模型原始回答，避免重写事实内容。
        stripped = line.strip()  # 清理首尾空白，便于判断是否为空行或结构行。
        if not stripped or re.match(r"^(引用依据|检索日志|校验结果|当前问题|结合会话后的检索问题)\s*[:：]", stripped):  # 跳过空行和报告结构提示。
            repaired_lines.append(line)  # 原样保留非事实结构行，后续校验会继续按规则过滤。
            continue  # 继续处理下一行。
        if re.search(r"\[\d+\]", stripped) or stripped.startswith(("#", "**", "__")):  # 已有引用或明显标题不再重复处理。
            repaired_lines.append(line)  # 原样保留当前行。
            continue  # 继续处理下一行。
        repaired_lines.append(f"{stripped} {available}".strip())  # 给原事实行附加所有真实上下文编号，由严格校验决定是否确实被原文支持。
    return "\n".join(repaired_lines).strip()  # 返回只改变引用格式、不改变事实内容的候选答案。


def generate_answer(state: RagState) -> RagState:  # 定义答案生成节点。
    ensure_request_budget("generate_answer")  # 在调用 DeepSeek 或生成保守回答前检查总预算。
    citations = "\n\n".join(format_citation(i + 1, doc, state["resolved_question"]) for i, doc in enumerate(state["contexts"]))  # 把检索片段格式化成引用上下文。
    template_instructions = build_template_instructions(state["question_type"])  # 根据问题类型选择稳定的备考回答结构。
    if not state["evidence_enough"]:  # 如果证据分低于阈值。
        state["answer"] = build_insufficient_answer(state, citations)  # 生成保守拒答，不调用大模型硬答。
        return state  # 直接返回状态。
    prompt = f"""你是信息系统项目管理师考试辅导老师。请只根据下面的教材原文回答用户问题。
如果原文不足以回答，请明确说“当前知识库没有找到足够依据”。
回答要先给结论，再解释原因；每一行只写一个事实、定义、步骤、公式或判断，并在该行末尾紧跟对应的引用编号，例如 [1] 或 [1][2]。
不要使用教材原文之外的知识，不要补充没有引用支持的内容。
如果只能回答一部分，请明确说明哪一部分有依据、哪一部分没有依据。
除区别对比模板要求的 Markdown 对比表外，不要使用额外的 Markdown 标题、粗体、列表符号或表格；对于“有哪些、包括哪些、公式、异同”问题，要把教材原文中当前上下文明确支持的项目逐项写出，不要因为表达不完整就笼统拒答。

用户问题：
{state['question']}

结合会话上下文补全后的问题：
{state['resolved_question']}

必要的历史记忆：
{state['memory_context'] or '无'}

问题类型：
{state['question_type']}

回答模板约束：
{template_instructions}

教材原文：
{citations}
"""  # 构造最终提示词。
    try:  # 保护回答模型请求，超时或服务异常时降级为原文证据。
        answer = call_deepseek(prompt)  # 调用 DeepSeek，让它基于教材原文组织最终回答。
    except RequestGovernanceError as exc:  # 单独处理总截止时间或熔断，避免把治理阻断误报成普通模型故障。
        logger.warning("Answer generation_governance_degraded error_type=%s", type(exc).__name__)  # 记录稳定治理异常类型。
        state["answer"] = build_request_failure_answer(state, exc)  # 返回明确的治理降级答案并保留已有教材证据。
        return state  # 结束当前生成节点，交给报告节点保存诊断。
    except Exception as exc:  # 捕获 DeepSeek 超时、网络、鉴权和响应异常。
        logger.exception("Answer generation_failed evidence_score=%d error_type=%s", state["evidence_score"], type(exc).__name__)  # 记录回答阶段异常堆栈。
        state["citation_validation"] = {"passed": False, "reason": "回答模型不可用，已降级为原文证据", "error_type": type(exc).__name__}  # 保存可审计的降级原因。
        state["answer"] = build_model_failure_answer(state, citations)  # 不让模型异常导致系统胡乱补答。
        return state  # 结束当前生成节点，交给报告节点保存诊断结果。
    validation = validate_answer(answer, citations)  # 对模型回答做逐句引用校验，防止只要模型说得通顺就直接放行。
    template_validation = validate_template_structure(answer, state["question_type"])  # 在引用校验之外检查回答是否符合备考模板。
    state["template_validation"] = template_validation  # 保存模板结构校验结果，便于报告和评估复盘。
    state["citation_validation"] = validation  # 保存校验明细，方便报告复盘。
    if not validation["passed"] or not template_validation["passed"]:  # 任一安全门或结构门失败，都进入一次修复。
        repair_prompt = f"""上一版回答没有通过严格答案校验。
引用校验原因：{validation.get('reason', '通过')}。
模板校验原因：{template_validation.get('reason', '通过')}。
请根据同一份教材原文重新回答，不要扩展教材之外的知识。
要求：
1. 只输出回答正文，不要输出“引用依据”标题、原文片段、粗体小标题或 Markdown 列表符号。
2. 每行只写一个事实、定义、步骤、公式或判断，并在该行末尾紧跟有效引用编号，例如 [1] 或 [1][2]。
3. 不要输出没有引用编号的解释句；不要把引用编号单独放到下一行或整段末尾。
4. 如果原文不足，只输出“当前知识库没有找到足够依据”，并在同一行末尾引用最相关片段。
5. {build_template_repair_instructions(state['question_type'])}

原用户问题：
{state['question']}

问题类型：
{state['question_type']}

教材原文：
{citations}
"""  # 构造一次性的引用格式修复提示，避免在失败时直接丢弃全部可用答案。
        try:  # 保护一次性格式修复请求，修复失败仍然走保守降级。
            repaired_answer = call_deepseek(repair_prompt)  # 只在首次校验失败时追加一次模型请求。
            repaired_answer = normalize_repaired_template(repaired_answer, state["question_type"])  # 如果修复只丢失对比区块标签，先用零 token 方式补回固定结构。
            repaired_validation = validate_answer(repaired_answer, citations)  # 对修复后的回答再次执行同一套严格校验。
            repaired_template_validation = validate_template_structure(repaired_answer, state["question_type"])  # 对修复回答再次执行模板结构校验。
            state["template_validation"] = repaired_template_validation  # 保存修复回答的模板结果。
            repaired_validation["repair_attempted"] = True  # 记录本次回答经过了引用格式修复。
            repaired_validation["initial_reason"] = validation.get("reason", "")  # 保存首次失败原因，便于诊断是否需要继续改提示词。
            if repaired_validation["passed"] and repaired_template_validation["passed"]:  # 只有修复后的回答同时通过引用和模板校验才允许返回。
                state["citation_validation"] = repaired_validation  # 保存修复后的校验结果。
                state["answer"] = repaired_answer + "\n\n引用依据：\n" + citations  # 返回经过二次校验的答案和系统引用。
                logger.info("Answer citation_repair_succeeded question_type=%s", state["question_type"])  # 记录修复成功，不记录问题正文。
                return state  # 修复成功后结束回答节点。
            validation["repair_attempted"] = True  # 修复仍失败时也记录尝试过修复。
            validation["repair_reason"] = repaired_validation.get("reason", "修复回答未通过")  # 保存二次失败原因。
            auto_repair_sources = [answer] if template_validation["passed"] else []  # 如果主草稿结构完整，优先保留它，避免引用修复模型删掉模板区块。
            auto_repair_sources.append(repaired_answer.strip() or answer)  # 主草稿结构不完整时，再使用修复草稿作为候选。
            for auto_repair_source in auto_repair_sources:  # 依次尝试保结构的主草稿和修复草稿。
                auto_repaired_answer = append_context_citations(auto_repair_source, len(state["contexts"]))  # 模型仍未遵守格式时，尝试零新增 token 的引用编号补齐。
                auto_repaired_validation = validate_answer(auto_repaired_answer, citations)  # 自动补齐后必须重新通过同一套逐句原文覆盖校验。
                auto_repaired_template_validation = validate_template_structure(auto_repaired_answer, state["question_type"])  # 零 token 修复也必须满足模板结构。
                state["template_validation"] = auto_repaired_template_validation  # 保存当前零 token 候选的模板结果。
                auto_repaired_validation["format_repair_attempted"] = True  # 记录本次无模型调用的格式修复。
                auto_repaired_validation["initial_reason"] = validation.get("reason", "")  # 保存首次校验失败原因，便于追踪修复链路。
                if auto_repaired_validation["passed"] and auto_repaired_template_validation["passed"]:  # 只有两套校验都通过才允许使用自动补齐结果。
                    state["citation_validation"] = auto_repaired_validation  # 保存自动修复后的校验结果。
                    state["answer"] = auto_repaired_answer + "\n\n引用依据：\n" + citations  # 返回原事实和系统引用，不新增模型内容。
                    logger.info("Answer citation_format_repair_succeeded question_type=%s", state["question_type"])  # 记录零 token 格式修复成功。
                    return state  # 自动修复成功后结束回答节点。
        except RequestGovernanceError as exc:  # 引用修复阶段也必须识别总截止时间和熔断。
            validation["repair_attempted"] = True  # 记录修复请求曾经发起。
            validation["repair_reason"] = f"引用修复被请求治理阻断：{type(exc).__name__}"  # 保存稳定治理异常类型。
            logger.warning("Answer citation_repair_governance_degraded error_type=%s", type(exc).__name__)  # 记录治理阻断，不记录提示词。
            state["answer"] = build_request_failure_answer(state, exc)  # 返回可核对证据，不继续请求模型。
            return state  # 结束回答节点并保存诊断报告。
        except Exception as exc:  # 捕获修复请求鉴权或响应格式异常。
            validation["repair_attempted"] = True  # 记录修复请求曾经发起。
            validation["repair_reason"] = f"引用修复请求失败：{type(exc).__name__}"  # 记录安全的异常类型，不泄露密钥和提示词。
            logger.exception("Answer citation_repair_failed error_type=%s", type(exc).__name__)  # 记录完整异常堆栈到统一日志。
        state["answer"] = build_citation_failure_answer(state, citations, validation)  # 放弃未经证明的模型回答，只返回可核对原文。
        return state  # 结束当前生成节点。
    state["answer"] = answer + "\n\n引用依据：\n" + citations  # 把通过校验的回答和引用片段合并保存。
    return state  # 返回更新后的状态。


def build_insufficient_answer(state: RagState, citations: str) -> str:  # 定义证据不足时的保守回答函数。
    logs = "\n".join(f"- {line}" for line in state["retrieval_log"])  # 把检索日志整理成 Markdown 列表。
    return f"""当前知识库没有找到足够依据，因此我不直接回答这个问题。

我已经尝试了 {len(state['retrieval_log'])} 次检索，但证据分最高只有 {state['evidence_score']}，低于阈值 {MIN_EVIDENCE_SCORE}。

检索过程：
{logs}

最接近的原文片段如下，供你人工判断是否换一种问法：

{citations}
"""  # 返回保守拒答内容。


def build_citation_failure_answer(state: RagState, citations: str, validation: dict) -> str:  # 定义引用校验失败时的保守回答函数。
    return f"""模型草稿没有通过严格引用校验，因此我不直接采用它，避免把没有教材依据的内容返回给你。

校验结果：{validation.get('reason', '未通过引用校验')}

当前问题：{state['question']}
结合会话后的检索问题：{state['resolved_question']}

下面只保留检索到的教材原文，供你核对：

{citations}
"""  # 返回只含可核对证据的保守结果。


def build_model_failure_answer(state: RagState, citations: str) -> str:  # 定义模型服务异常时的保守降级回答函数。
    return f"""回答模型本次不可用，因此我不直接生成未经确认的答案。

当前问题：{state['question']}
检索证据分：{state['evidence_score']}

系统已经记录本次异常日志。下面保留检索到的教材原文，供你核对：

{citations}
"""  # 返回安全的原文证据，不使用常识补答。


def build_request_failure_answer(state: RagState, error: RequestGovernanceError) -> str:  # 定义总截止时间或熔断时的安全回答函数。
    state["request_error"] = type(error).__name__  # 保存稳定异常类型，避免报告中出现底层密钥或完整错误信息。
    citations = "\n\n".join(format_citation(i + 1, doc, state["resolved_question"]) for i, doc in enumerate(state["contexts"]))  # 如果超时前已经得到证据，就保留可核对原文。
    evidence_text = f"超时前已经取得的教材原文如下：\n\n{citations}" if citations else "本次没有取得可以安全展示的教材片段。"  # 组织已有证据或明确说明没有证据。
    return f"""本次查询因请求保护机制未在总时限内完成，因此不生成未经确认的答案。

治理状态：{type(error).__name__}
检索证据分：{state["evidence_score"]}

{evidence_text}
"""  # 返回明确原因和已有证据，不让模型在预算耗尽后继续猜测。


def save_answer_report(state: RagState) -> RagState:  # 定义保存报告节点，把每次问答结果落盘。
    try:  # 保护输出目录创建，目录权限异常也要进入统一日志治理。
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保 outputs 目录存在，不存在就创建。
    except OSError as exc:  # 捕获输出目录权限或磁盘异常。
        logger.exception("Answer report_directory_failed path=%s error_type=%s", OUTPUT_ROOT, type(exc).__name__)  # 记录单题报告目录创建失败。
        state["report_path"] = ""  # 清空路径，避免把不存在的报告告诉用户。
        return state  # 允许主流程继续输出已经得到的回答。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成微秒级时间戳，避免多问题同秒生成报告时互相覆盖。
    report_path = OUTPUT_ROOT / f"full_rag_answer_{timestamp}.md"  # 拼出本次完整 RAG 回答报告路径。
    binding_report = build_binding_report(state["citation_validation"])  # 在 f-string 外生成逐结论绑定区块，避免表达式中的换行转义触发语法限制。
    trace_report = build_trace_report(state["contexts"], state["resolved_question"]) if state["contexts"] else "## 原文追溯\n\n本次没有可追溯的检索片段。"  # 在 f-string 外生成追溯区块，证据不足时保留明确边界说明。
    request_report = current_request_summary()  # 读取当前完整查询的实时治理摘要，记录耗时、失败和 token。
    report = f"""# 完整 RAG 问答报告

## 用户问题

{state['question']}

## 结合会话上下文后的问题

{state['resolved_question']}

## 章节过滤

{state['chapter_filter'] or '未指定，查询全部章节'}

## 会话记忆

{state['memory_context'] or '本题未使用历史记忆'}

## 问题类型

{state['question_type']}

## 分类原因

{state['understanding_reason']}

## 增强后的检索查询

{state['query']}

## 回答与引用

{state['answer']}

## 引用校验结果

{state['citation_validation'] or '证据不足时未调用回答模型'}

## 回答模板校验结果

{state['template_validation'] or '证据不足时未调用回答模型'}

## 检索日志

{chr(10).join(state['retrieval_log'])}

{binding_report}

{trace_report}

## 请求治理统计

{request_report or '本次没有进入完整请求治理上下文。'}
"""  # 组织 Markdown 报告内容，既方便阅读，也方便以后批量分析。
    try:  # 保护问答报告写入，磁盘异常不能覆盖已经生成的回答。
        report_path.write_text(report, encoding="utf-8")  # 把报告写入本地文件。
        state["report_path"] = str(report_path)  # 把报告路径写回状态，方便主函数打印给用户。
    except OSError as exc:  # 捕获输出目录权限、磁盘空间和文件系统异常。
        logger.exception("Answer report_write_failed path=%s error_type=%s", report_path, type(exc).__name__)  # 记录报告写入失败堆栈。
        state["report_path"] = ""  # 清空路径，避免把不存在的报告告诉用户。
    return state  # 返回更新后的状态。


@lru_cache(maxsize=1)  # 一个常驻进程只编译一次图结构，单次查询仍使用各自独立的 RagState。
def build_graph():  # 定义创建 LangGraph 流程图的函数。
    graph = StateGraph(RagState)  # 创建一个以 RagState 为状态的图。
    graph.add_node("rewrite_query", rewrite_query)  # 添加查询改写节点。
    graph.add_node("retrieve_vector", retrieve_vector)  # 添加向量检索节点。
    graph.add_node("retrieve_keyword", retrieve_keyword)  # 添加关键词检索节点。
    graph.add_node("merge_and_score", merge_and_score)  # 添加合并重排节点。
    graph.add_node("advance_attempt", advance_attempt)  # 添加进入下一轮检索的节点。
    graph.add_node("generate_answer", generate_answer)  # 添加答案生成节点。
    graph.add_node("save_answer_report", save_answer_report)  # 添加保存报告节点。
    graph.set_entry_point("rewrite_query")  # 设置图的入口节点。
    graph.add_edge("rewrite_query", "retrieve_vector")  # 设置改写后先做向量检索。
    graph.add_edge("retrieve_vector", "retrieve_keyword")  # 设置向量检索后再做关键词检索。
    graph.add_edge("retrieve_keyword", "merge_and_score")  # 设置关键词检索后合并结果。
    graph.add_conditional_edges("merge_and_score", should_retry, {"retry": "advance_attempt", "answer": "generate_answer"})  # 根据证据分决定重试还是回答。
    graph.add_edge("advance_attempt", "retrieve_vector")  # 下一轮检索从向量检索重新开始。
    graph.add_edge("generate_answer", "save_answer_report")  # 设置答案生成后保存 Markdown 报告。
    graph.add_edge("save_answer_report", END)  # 设置报告保存后流程结束。
    return graph.compile()  # 编译 LangGraph，得到可运行对象。


def answer_one_question(app, question: str, resolved_question: str | None = None, memory_context: str = "", chapter_filter: str = "") -> RagState:  # 定义回答单个子问题的函数。
    initial_state = make_initial_state(question, resolved_question, memory_context, chapter_filter)  # 通过状态工厂创建字段完整的初始状态。
    try:  # 保护单个子问题，保证多问题中的一个超时不会让已经完成的其他问题丢失。
        return app.invoke(initial_state)  # 执行整条 LangGraph RAG 流程，并返回结果。
    except RequestGovernanceError as exc:  # 捕获总截止时间或熔断器主动中断。
        logger.warning("Question governance_degraded error_type=%s", type(exc).__name__)  # 记录安全异常类型，不记录问题正文。
        initial_state["answer"] = build_request_failure_answer(initial_state, exc)  # 返回不猜测的治理降级答案。
        save_answer_report(initial_state)  # 即使查询被治理层阻断，也保存诊断报告，便于复盘请求耗时和失败阶段。
        return initial_state  # 将当前子问题作为失败结果交给统一汇总。


def build_combined_answer(results: list[RagState]) -> str:  # 定义多问题统一汇总函数。
    if len(results) == 1:  # 如果只有一个子问题。
        return results[0]["answer"]  # 直接返回单题答案。
    blocks = []  # 准备保存每个子问题的回答块。
    for index, result in enumerate(results, start=1):  # 遍历每个子问题结果。
        status = "证据足够" if result["evidence_enough"] else "证据不足"  # 根据证据状态生成标签。
        block = f"""## 问题 {index}：{result['question']}

状态：{status}，证据分：{result['evidence_score']}

{result['answer']}
"""  # 组织一个子问题的回答块。
        blocks.append(block)  # 加入汇总列表。
    return "# 多问题统一回答\n\n" + "\n\n".join(blocks)  # 返回完整汇总回答。


def save_combined_report(question: str, answer: str, results: list[RagState]) -> str:  # 定义保存多问题总报告的函数。
    try:  # 保护输出目录创建，目录权限异常也要进入统一日志治理。
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保 outputs 目录存在。
    except OSError as exc:  # 捕获输出目录权限或磁盘异常。
        logger.exception("Combined report_directory_failed path=%s error_type=%s", OUTPUT_ROOT, type(exc).__name__)  # 记录总报告目录创建失败。
        return ""  # 返回空路径，避免继续构造一个必然无法写入的报告。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成微秒级时间戳，避免同一秒内多个子问题互相覆盖报告。
    report_path = OUTPUT_ROOT / f"multi_rag_answer_{timestamp}.md"  # 拼出多问题报告路径。
    sub_question_meta = [  # 准备子问题元数据。
        {  # 每个子问题保存核心诊断字段。
            "question": result["question"],  # 保存子问题文本。
            "question_type": result["question_type"],  # 保存问题类型。
            "evidence_score": result["evidence_score"],  # 保存证据分。
            "evidence_enough": result["evidence_enough"],  # 保存证据是否足够。
            "report_path": result["report_path"],  # 保存单题报告路径。
        }  # 子问题元数据结束。
        for result in results  # 遍历所有子问题结果。
    ]  # 子问题元数据列表结束。
    report = f"""# 多问题 RAG 总报告

## 原始输入

{question}

## 子问题元数据

{sub_question_meta}

## 统一回答

{answer}
"""  # 组织总报告内容。
    try:  # 保护多问题总报告写入。
        report_path.write_text(report, encoding="utf-8")  # 写入 Markdown 文件。
        return str(report_path)  # 返回总报告路径。
    except OSError as exc:  # 捕获多问题报告的文件系统异常。
        logger.exception("Combined report_write_failed path=%s error_type=%s", report_path, type(exc).__name__)  # 记录总报告写入失败堆栈。
        return ""  # 返回空路径，避免打印不存在的报告。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="信息系统项目管理师完整 RAG 问答")  # 创建参数解析器。
    parser.add_argument("question", nargs="*", help="用户问题，可以一次输入一个或多个问题")  # 读取位置参数里的问题文本。
    parser.add_argument("--session", default=None, help="会话名称，不填则使用当天默认会话")  # 支持用户指定会话名。
    parser.add_argument("--back", action="store_true", help="回退当前会话的上一轮问答")  # 支持回退上一轮。
    parser.add_argument("--last", action="store_true", help="查看当前会话的上一轮问答")  # 支持查看上一轮。
    parser.add_argument("--chapter", default=None, help="限定检索章节，例如：第17章、项目整体管理")  # 支持按章节缩小检索范围。
    return parser.parse_args()  # 返回解析结果。


def safe_append_turn(session_name: str | None, turn: dict) -> str:  # 定义带异常治理的会话保存函数。
    try:  # 保护会话 JSON、Markdown 和长期记忆写入。
        return str(append_turn(session_name, turn))  # 保存会话并返回本轮 Markdown 路径。
    except OSError as exc:  # 捕获目录权限、磁盘空间和文件系统异常。
        logger.exception("Session save_failed error_type=%s", type(exc).__name__)  # 记录会话保存失败堆栈。
        return ""  # 会话保存失败时仍允许把回答返回给用户。
    except Exception as exc:  # 捕获 JSON 损坏等其他会话异常。
        logger.exception("Session update_failed error_type=%s", type(exc).__name__)  # 记录会话更新失败堆栈。
        return ""  # 返回空路径，避免把失败误报为成功。


def execute_query(args: argparse.Namespace, question: str) -> None:  # 定义带总截止时间的完整查询执行函数。
    with request_scope(question, args.session):  # 让拆题、检索自救、答案生成和报告保存共享同一个总预算。
        sub_questions = split_questions(question)  # 把用户输入拆成一个或多个子问题。
        app = build_graph()  # 创建并编译 LangGraph RAG 流程。
        memory_context = build_memory_context(args.session, question)  # 只在当前输入依赖上文时读取最近会话记忆。
        results = [  # 对每个子问题独立运行完整 RAG，同时传入必要的记忆和章节范围。
            answer_one_question(  # 调用单题 LangGraph 流程。
                app,  # 传入已经编译好的 LangGraph。
                item,  # 传入当前子问题。
                resolved_question=resolve_follow_up_question(args.session, item),  # 将“这个/刚才/它”等追问补成完整检索问题。
                memory_context=memory_context,  # 传入最小历史记忆。
                chapter_filter=args.chapter_filter or "",  # 传入标准化章节过滤条件。
            )  # 单题调用结束。
            for item in sub_questions  # 遍历所有子问题。
        ]  # 多问题处理结束。
        combined_answer = build_combined_answer(results)  # 把所有子问题答案合并成统一回复。
        combined_report_path = save_combined_report(question, combined_answer, results) if len(results) > 1 else results[0]["report_path"]  # 保存总报告或沿用单题报告。
        chapters = sorted({doc.metadata.get("chapter", "") for result in results for doc in result["contexts"] if doc.metadata.get("chapter", "")})  # 汇总本轮引用到的章节。
        question_types = sorted({result["question_type"] for result in results if result["question_type"]})  # 汇总本轮问题类型。
        session_turn_path = safe_append_turn(args.session, {  # 把本轮问答写入会话历史和长期记忆。
            "question": question,  # 保存原始问题。
            "sub_questions": sub_questions,  # 保存拆分后的子问题。
            "answer": combined_answer,  # 保存统一回答。
            "question_types": question_types,  # 保存问题类型集合。
            "chapters": chapters,  # 保存引用章节集合。
            "report_path": combined_report_path,  # 保存报告路径。
            "chapter_filter": args.chapter_filter or "",  # 保存本轮使用的章节过滤条件。
            "memory_used": bool(memory_context),  # 保存本轮是否使用了会话记忆。
            "resolved_questions": [result["resolved_question"] for result in results],  # 保存每个子问题的记忆补全结果。
            "session_name": args.session or "",  # 保存来源会话名，为未来回退重算长期记忆提供来源。
        })  # 会话追加结束。
        print(combined_answer)  # 把最终答案打印到终端。
        print(f"\n报告已保存：{combined_report_path}" if combined_report_path else "\n报告保存失败，详情请查看 logs/rag.log。")  # 打印报告位置或明确提示报告保存失败。
        print(f"会话已保存：{session_turn_path}" if session_turn_path else "会话保存失败，详情请查看 logs/rag.log。")  # 打印会话位置或明确提示会话保存失败。


def _run_cli() -> None:  # 定义实际 CLI 主流程，外层 main 负责最后一道异常兜底。
    args = parse_args()  # 解析命令行参数。
    if args.back:  # 如果用户要求回退。
        removed = rollback_last_turn(args.session)  # 删除当前会话最后一轮。
        print("已回退上一轮问答。" if removed else "当前会话没有可回退的问题。")  # 输出回退结果。
        return  # 回退操作完成后直接结束。
    if args.last:  # 如果用户要求查看上一轮。
        turn = latest_turn(args.session)  # 读取当前会话最后一轮。
        print(turn_to_text(turn) if turn else "当前会话还没有历史问题。")  # 输出上一轮内容或空提示。
        return  # 查看操作完成后直接结束。
    chapter_filter = resolve_chapter_filter(args.chapter)  # 把用户输入的章节短名解析成标准章节名。
    if args.chapter and not chapter_filter:  # 如果用户指定了章节但没有唯一匹配。
        print(f"没有找到唯一章节：{args.chapter}。可使用完整章节名，例如：第17章 项目整体管理。")  # 输出明确提示，避免静默查询全书。
        return  # 章节条件不明确时停止本次查询。
    require_embedding_config()  # 先检查 embedding 配置，避免跑到中间才失败。
    question = " ".join(args.question).strip()  # 从命令行位置参数合并出用户问题。
    if not question:  # 如果用户没有输入问题。
        question = input("请输入你的问题：").strip()  # 就在终端里让用户输入问题。
    args.chapter_filter = chapter_filter  # 把解析后的标准章节名传给带请求治理的执行函数。
    execute_query(args, question)  # 在总截止时间上下文中执行完整查询。


def main() -> None:  # 定义带全局异常兜底的 CLI 入口。
    try:  # 保护未被局部降级逻辑覆盖的启动和编排异常。
        _run_cli()  # 执行实际查询流程。
    except Exception as exc:  # 捕获最后一级未处理异常，避免终端只显示长堆栈。
        logger.exception("CLI unhandled_failure error_type=%s", type(exc).__name__)  # 记录完整异常堆栈供后续排查。
        print(f"本次查询未完成，详情请查看 logs/rag.log。错误类型：{type(exc).__name__}")  # 向用户返回安全、明确的失败提示。


def turn_to_text(turn: dict) -> str:  # 定义把上一轮会话转成终端文本的函数。
    return f"""上一轮问题：
{turn.get('question', '')}

上一轮答案：
{turn.get('answer', '')}
"""  # 返回可直接打印的文本。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 如果是直接运行，就执行主函数。
