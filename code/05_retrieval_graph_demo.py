from datetime import datetime  # 导入 datetime，用来给检索报告生成时间戳。
import pickle  # 导入 pickle，用来加载已经保存好的 BM25 检索器。
import re  # 导入 re，用来把用户问题清理成安全的文件名片段。
import sys  # 导入 sys，用来从命令行读取用户问题。
from typing import TypedDict  # 导入 TypedDict，用来定义 LangGraph 状态结构。

from langchain_core.documents import Document  # 导入 LangChain Document，用来标注检索结果类型。
from langgraph.graph import END  # 导入 END，用来表示 LangGraph 流程结束。
from langgraph.graph import StateGraph  # 导入 StateGraph，用来搭建检索流程图。

from config import BM25_PICKLE  # 从配置文件读取 BM25 索引文件路径。
from config import OUTPUT_ROOT  # 从配置文件读取输出目录。
from query_understanding import understand_query  # 导入问题理解函数，用来做规则分类和 query 增强。
from rag_tokenizers import tokenize_for_bm25  # 导入中文 BM25 分词函数，确保 pickle 反序列化时能找到它。  # noqa: F401


class RetrievalState(TypedDict):  # 定义这个检索图在节点之间传递的状态。
    question: str  # 保存用户输入的原始问题。
    question_type: str  # 保存规则判断出来的问题类型。
    understanding_reason: str  # 保存问题分类的原因，方便教学和调试。
    query: str  # 保存用于检索的查询文本。
    docs: list[Document]  # 保存 BM25 检索返回的候选原文片段。
    report: str  # 保存最终整理好的 Markdown 检索报告。


def prepare_query(state: RetrievalState) -> RetrievalState:  # 定义第一个节点：准备检索查询。
    question = state["question"].strip()  # 取出用户问题，并去掉首尾空白。
    understanding = understand_query(question)  # 调用规则版问题理解器，得到类型和增强 query。
    state["question_type"] = understanding.question_type  # 把问题类型写入状态。
    state["understanding_reason"] = understanding.reason  # 把分类原因写入状态。
    state["query"] = understanding.expanded_query  # 把增强后的 query 写入状态，供后续检索使用。
    return state  # 返回更新后的状态。


def retrieve_bm25(state: RetrievalState) -> RetrievalState:  # 定义第二个节点：使用 BM25 检索教材原文。
    with BM25_PICKLE.open("rb") as file:  # 打开已经生成好的 BM25 索引文件。
        retriever = pickle.load(file)  # 把 BM25 检索器从 pickle 文件里加载回来。
    retriever.k = 8  # 设置返回 8 条结果，便于观察更多候选原文。
    state["docs"] = retriever.invoke(state["query"])  # 使用 query 进行关键词检索，并把结果写入状态。
    return state  # 返回更新后的状态。


def clean_snippet(text: str, limit: int = 700) -> str:  # 定义清理引用片段的函数。
    text = " ".join(text.split())  # 把多余空白、换行合并成普通空格，方便报告阅读。
    return text[:limit] + ("..." if len(text) > limit else "")  # 截取指定长度，太长就加省略号。


def build_report(state: RetrievalState) -> RetrievalState:  # 定义第三个节点：把检索结果整理成 Markdown 报告。
    lines: list[str] = []  # 准备一个字符串列表，用来一行一行拼报告。
    lines.append(f"# 检索报告：{state['question']}")  # 写入报告标题。
    lines.append("")  # 写入空行，让 Markdown 更清楚。
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")  # 写入生成时间。
    lines.append("- 检索方式：LangGraph + BM25 + jieba 中文分词 + 2-4 字 n-gram")  # 写入检索方式。
    lines.append(f"- 问题类型：{state['question_type']}")  # 写入问题类型，方便观察分类是否合理。
    lines.append(f"- 分类原因：{state['understanding_reason']}")  # 写入分类原因，方便知道命中了哪个规则。
    lines.append(f"- 增强查询：{state['query']}")  # 写入增强后的 query，方便复盘检索为什么命中这些结果。
    lines.append(f"- 返回结果数：{len(state['docs'])}")  # 写入结果数量。
    lines.append("")  # 写入空行。
    for index, doc in enumerate(state["docs"], start=1):  # 遍历每条检索结果。
        metadata = doc.metadata  # 取出当前 chunk 的元数据。
        lines.append(f"## 结果 {index}")  # 写入结果编号。
        lines.append("")  # 写入空行。
        lines.append(f"- 章节：{metadata.get('chapter', '')}")  # 写入章名。
        lines.append(f"- 小节：{metadata.get('section', '')}")  # 写入小节名。
        lines.append(f"- 来源：{metadata.get('relative_path', '')}")  # 写入源 Word 相对路径。
        lines.append(f"- chunk_id：{metadata.get('chunk_id', '')}")  # 写入 chunk_id，方便追踪。
        lines.append("")  # 写入空行。
        lines.append("原文片段：")  # 写入原文片段标签。
        lines.append("")  # 写入空行。
        lines.append(f"> {clean_snippet(doc.page_content)}")  # 用 Markdown 引用格式写入原文片段。
        lines.append("")  # 写入空行。
    state["report"] = "\n".join(lines)  # 把所有行合并成完整 Markdown 报告。
    return state  # 返回更新后的状态。


def save_report(state: RetrievalState) -> RetrievalState:  # 定义第四个节点：保存报告到 outputs 文件夹。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    safe_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 生成包含微秒的时间戳，避免并行运行时文件名冲突。
    safe_question = re.sub(r"[^\w\u4e00-\u9fff]+", "_", state["question"])[:24].strip("_")  # 把问题清理成可放进文件名的短标识。
    path = OUTPUT_ROOT / f"retrieval_report_{safe_time}_{safe_question}.md"  # 拼出报告文件路径。
    path.write_text(state["report"], encoding="utf-8")  # 把 Markdown 报告写入文件。
    state["report"] += f"\n\n报告文件：{path}\n"  # 把报告路径追加到终端输出里。
    return state  # 返回更新后的状态。


def build_graph():  # 定义创建 LangGraph 检索流程的函数。
    graph = StateGraph(RetrievalState)  # 创建一个以 RetrievalState 为状态的数据流图。
    graph.add_node("prepare_query", prepare_query)  # 添加准备查询节点。
    graph.add_node("retrieve_bm25", retrieve_bm25)  # 添加 BM25 检索节点。
    graph.add_node("build_report", build_report)  # 添加报告生成节点。
    graph.add_node("save_report", save_report)  # 添加报告保存节点。
    graph.set_entry_point("prepare_query")  # 设置图的入口为 prepare_query。
    graph.add_edge("prepare_query", "retrieve_bm25")  # 设置准备查询后进入 BM25 检索。
    graph.add_edge("retrieve_bm25", "build_report")  # 设置检索完成后生成报告。
    graph.add_edge("build_report", "save_report")  # 设置报告生成后保存到文件。
    graph.add_edge("save_report", END)  # 设置保存报告后结束流程。
    return graph.compile()  # 编译图，返回可执行的 LangGraph 应用。


def main() -> None:  # 定义主函数。
    question = " ".join(sys.argv[1:]).strip()  # 从命令行参数读取用户问题。
    if not question:  # 如果命令行没有输入问题。
        question = input("请输入你的检索问题：").strip()  # 就在终端里提示用户输入。
    app = build_graph()  # 创建并编译 LangGraph 检索流程。
    initial_state: RetrievalState = {  # 准备流程初始状态。
        "question": question,  # 写入用户问题。
        "question_type": "",  # 问题类型先留空，交给 prepare_query 节点填写。
        "understanding_reason": "",  # 分类原因先留空，交给 prepare_query 节点填写。
        "query": "",  # query 先留空，交给 prepare_query 节点填写。
        "docs": [],  # 检索结果先留空，交给 retrieve_bm25 节点填写。
        "report": "",  # 报告先留空，交给 build_report 节点填写。
    }  # 初始状态定义结束。
    result = app.invoke(initial_state)  # 运行 LangGraph 流程。
    print(result["report"])  # 把最终报告打印到终端。


if __name__ == "__main__":  # 判断当前脚本是否被直接运行。
    main()  # 如果是直接运行，就执行主函数。
