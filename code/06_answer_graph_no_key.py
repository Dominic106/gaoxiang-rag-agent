from datetime import datetime  # 导入 datetime，用来给回答报告生成时间。
import pickle  # 导入 pickle，用来加载 BM25 检索器。
import re  # 导入 re，用来切句和清理文件名。
import sys  # 导入 sys，用来读取命令行问题。
from typing import TypedDict  # 导入 TypedDict，用来定义 LangGraph 状态类型。

from langchain_core.documents import Document  # 导入 LangChain Document，用来标注检索结果类型。
from langgraph.graph import END  # 导入 END，用来表示 LangGraph 流程结束。
from langgraph.graph import StateGraph  # 导入 StateGraph，用来搭建无 Key 回答图。

from config import BM25_PICKLE  # 从配置文件读取 BM25 索引路径。
from config import FINAL_CONTEXTS  # 从配置文件读取最终引用片段数量。
from config import OUTPUT_ROOT  # 从配置文件读取输出目录。
from query_understanding import understand_query  # 导入问题理解函数，用来分类和增强 query。
from rag_tokenizers import tokenize_for_bm25  # 导入中文分词函数，确保 BM25 pickle 可以正常加载。


class AnswerState(TypedDict):  # 定义无 Key 回答图的状态结构。
    question: str  # 保存用户原始问题。
    question_type: str  # 保存问题类型。
    understanding_reason: str  # 保存问题分类原因。
    query: str  # 保存增强后的检索 query。
    docs: list[Document]  # 保存检索到的原文片段。
    evidence_docs: list[Document]  # 保存真正用于回答的证据来源。
    answer: str  # 保存抽取式回答。
    report_path: str  # 保存回答报告文件路径。


def prepare_query(state: AnswerState) -> AnswerState:  # 定义准备查询节点。
    understanding = understand_query(state["question"])  # 调用问题理解器。
    state["question_type"] = understanding.question_type  # 写入问题类型。
    state["understanding_reason"] = understanding.reason  # 写入分类原因。
    state["query"] = understanding.expanded_query  # 写入增强查询。
    return state  # 返回状态。


def retrieve_bm25(state: AnswerState) -> AnswerState:  # 定义 BM25 检索节点。
    with BM25_PICKLE.open("rb") as file:  # 打开 BM25 索引文件。
        retriever = pickle.load(file)  # 加载 BM25 检索器。
    retriever.k = 10  # 取 10 条候选，后面只选前几条引用。
    state["docs"] = retriever.invoke(state["query"])  # 执行检索并写入状态。
    return state  # 返回状态。


def split_sentences(text: str) -> list[str]:  # 定义中文句子切分函数。
    text = re.sub(r"\s+", " ", text)  # 合并多余空白。
    pieces = re.split(r"(?<=[。！？；])", text)  # 按中文句号、问号、叹号、分号切句。
    return [piece.strip() for piece in pieces if len(piece.strip()) >= 12]  # 返回长度足够的句子，过滤噪音。


def score_sentence(sentence: str, query: str) -> int:  # 定义句子打分函数。
    query_tokens = set(tokenize_for_bm25(query))  # 把查询分词，转成集合。
    sentence_tokens = set(tokenize_for_bm25(sentence))  # 把候选句分词，转成集合。
    return len(query_tokens & sentence_tokens)  # 用重叠词数量作为简单相关性分数。


def choose_evidence_sentences(state: AnswerState) -> list[tuple[str, Document]]:  # 定义从检索结果中挑选证据句的函数。
    candidates: list[tuple[int, str, Document]] = []  # 准备候选句列表，每项是分数、句子和来源文档。
    candidate_count = 1 if state["question_type"] == "定义解释" else 3  # 定义题优先只抽 Top1，避免其他定义段落混入。
    candidate_docs = state["docs"][:candidate_count]  # 根据问题类型选择候选 chunk 数量。
    for doc in candidate_docs:  # 遍历高相关 chunk。
        for sentence in split_sentences(doc.page_content):  # 遍历 chunk 内的句子。
            score = score_sentence(sentence, state["query"])  # 计算句子与查询的相关度。
            if score > 0:  # 只保留有命中的句子。
                candidates.append((score, sentence, doc))  # 保存分数、句子和来源文档。
    candidates.sort(key=lambda item: item[0], reverse=True)  # 按分数从高到低排序。
    seen: set[str] = set()  # 准备集合，用来去重。
    selected: list[tuple[str, Document]] = []  # 准备最终证据句列表。
    for _, sentence, doc in candidates:  # 遍历排序后的候选句。
        normalized = sentence[:80]  # 用前 80 字作为近似去重键。
        if normalized in seen:  # 如果已经见过类似句子。
            continue  # 跳过重复句。
        seen.add(normalized)  # 记录这个句子。
        selected.append((sentence, doc))  # 保存到最终证据句。
        if len(selected) >= 3:  # 最多选 3 句，避免回答太散。
            break  # 达到数量后停止。
    return selected  # 返回证据句。


def citation_lines(docs: list[Document]) -> list[str]:  # 定义生成引用来源列表的函数。
    lines: list[str] = []  # 准备引用行列表。
    for index, doc in enumerate(docs[:FINAL_CONTEXTS], start=1):  # 遍历前几个引用片段。
        metadata = doc.metadata  # 取出元数据。
        chapter = metadata.get("chapter", "")  # 取出章名。
        section = metadata.get("section", "")  # 取出小节名。
        source = metadata.get("relative_path", "")  # 取出相对来源路径。
        chunk_id = metadata.get("chunk_id", "")  # 取出 chunk_id。
        lines.append(f"[{index}] {chapter} / {section} / {source} / {chunk_id}")  # 拼成引用行。
    return lines  # 返回引用行列表。


def generate_extractive_answer(state: AnswerState) -> AnswerState:  # 定义抽取式回答节点。
    evidence = choose_evidence_sentences(state)  # 从检索结果中挑选证据句。
    state["evidence_docs"] = []  # 初始化实际证据来源列表。
    lines: list[str] = []  # 准备回答行列表。
    lines.append(f"# 回答：{state['question']}")  # 写入标题。
    lines.append("")  # 写入空行。
    lines.append(f"- 问题类型：{state['question_type']}")  # 写入问题类型。
    lines.append(f"- 分类原因：{state['understanding_reason']}")  # 写入分类原因。
    lines.append(f"- 增强查询：{state['query']}")  # 写入增强查询。
    lines.append("")  # 写入空行。
    lines.append("## 抽取式答案")  # 写入答案小标题。
    lines.append("")  # 写入空行。
    if evidence:  # 如果找到了证据句。
        for sentence, doc in evidence:  # 遍历证据句和来源文档。
            lines.append(f"- {sentence}")  # 直接用教材原文句子组成回答。
            if doc not in state["evidence_docs"]:  # 如果该来源还没记录。
                state["evidence_docs"].append(doc)  # 保存该证据来源。
    else:  # 如果没有找到证据句。
        lines.append("- 当前 BM25 知识库没有找到足够明确的原文依据。")  # 给出不足提示。
    lines.append("")  # 写入空行。
    lines.append("## 引用来源")  # 写入引用小标题。
    lines.append("")  # 写入空行。
    lines.extend(citation_lines(state["evidence_docs"] or state["docs"]))  # 加入真正用于回答的引用来源。
    lines.append("")  # 写入空行。
    lines.append("说明：这是无 API Key 版本，只做检索和原文抽取，不调用大模型改写。")  # 写入边界说明。
    state["answer"] = "\n".join(lines)  # 把回答行合并成完整 Markdown。
    return state  # 返回状态。


def save_answer(state: AnswerState) -> AnswerState:  # 定义保存回答报告节点。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成带微秒的时间戳。
    safe_question = re.sub(r"[^\w\u4e00-\u9fff]+", "_", state["question"])[:24].strip("_")  # 清理问题为文件名片段。
    path = OUTPUT_ROOT / f"answer_no_key_{safe_time}_{safe_question}.md"  # 拼出回答报告路径。
    path.write_text(state["answer"], encoding="utf-8")  # 写入回答报告。
    state["report_path"] = str(path)  # 保存报告路径。
    return state  # 返回状态。


def build_graph():  # 定义构建 LangGraph 的函数。
    graph = StateGraph(AnswerState)  # 创建状态图。
    graph.add_node("prepare_query", prepare_query)  # 添加查询准备节点。
    graph.add_node("retrieve_bm25", retrieve_bm25)  # 添加 BM25 检索节点。
    graph.add_node("generate_extractive_answer", generate_extractive_answer)  # 添加抽取式回答节点。
    graph.add_node("save_answer", save_answer)  # 添加保存回答节点。
    graph.set_entry_point("prepare_query")  # 设置入口节点。
    graph.add_edge("prepare_query", "retrieve_bm25")  # 设置准备查询后进入检索。
    graph.add_edge("retrieve_bm25", "generate_extractive_answer")  # 设置检索后生成抽取式回答。
    graph.add_edge("generate_extractive_answer", "save_answer")  # 设置生成回答后保存。
    graph.add_edge("save_answer", END)  # 设置保存后结束。
    return graph.compile()  # 编译并返回图。


def main() -> None:  # 定义主函数。
    question = " ".join(sys.argv[1:]).strip()  # 读取命令行问题。
    if not question:  # 如果命令行没有问题。
        question = input("请输入你的问题：").strip()  # 提示用户输入。
    app = build_graph()  # 构建图。
    state: AnswerState = {  # 准备初始状态。
        "question": question,  # 写入问题。
        "question_type": "",  # 初始化问题类型。
        "understanding_reason": "",  # 初始化分类原因。
        "query": "",  # 初始化查询。
        "docs": [],  # 初始化检索结果。
        "evidence_docs": [],  # 初始化证据来源。
        "answer": "",  # 初始化回答。
        "report_path": "",  # 初始化报告路径。
    }  # 初始状态结束。
    result = app.invoke(state)  # 执行图。
    print(result["answer"])  # 打印回答。
    print(f"\n报告文件：{result['report_path']}")  # 打印报告路径。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行主函数。
