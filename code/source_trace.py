"""提供引用定位和前后文追溯能力。"""  # 用模块注释说明本文件只处理原文定位，不参与检索排序。

from langchain_core.documents import Document  # 导入 Document，用于标注追溯对象。

from rag_evidence import extract_relevant_window  # 导入已有窗口裁剪函数，避免报告展示过长原文。
from rag_evidence import extract_terms  # 导入查询词抽取函数，让前后文窗口更贴近用户问题。
from retrieval_resources import get_section_documents  # 导入缓存的小节 chunk，支持按原文顺序找前后段。


def format_source_locator(document: Document) -> str:  # 定义稳定的源文档定位字符串。
    metadata = document.metadata  # 取出当前 chunk 的元数据。
    section_index = int(metadata.get("section_chunk_index", 0)) + 1  # 把内部从 0 开始的序号转换成人类可读序号。
    section_count = int(metadata.get("section_chunk_count", 0))  # 读取当前小节 chunk 总数。
    char_start = metadata.get("source_char_start", -1)  # 读取源正文字符起点。
    char_end = metadata.get("source_char_end", -1)  # 读取源正文字符终点。
    range_text = f"字符 {char_start}-{char_end}" if int(char_start) >= 0 and int(char_end) >= 0 else "字符范围未建立"  # 对可定位和旧索引两种情况给出明确提示。
    return f"{metadata.get('relative_path', metadata.get('source_docx', ''))}；{metadata.get('section', '')}第 {section_index}/{section_count} 个片段；{range_text}；chunk_id={metadata.get('chunk_id', '')}"  # 拼出可复制、可搜索的定位信息。


def get_adjacent_documents(document: Document, before: int = 1, after: int = 1) -> dict[str, tuple[Document, ...]]:  # 定义按同一小节顺序获取前后文。
    chapter = document.metadata.get("chapter", "")  # 读取章节，避免跨章串联上下文。
    section = document.metadata.get("section", "")  # 读取小节，优先保持同一原文文件内追溯。
    documents = list(get_section_documents(chapter, section))  # 获取同一小节按 chunk_index 排序的完整片段。
    current_id = document.metadata.get("chunk_id", "")  # 读取当前引用的稳定 chunk ID。
    current_index = next((index for index, item in enumerate(documents) if item.metadata.get("chunk_id") == current_id), -1)  # 找到当前片段在小节中的位置。
    if current_index < 0:  # 兼容旧索引或向量库元数据尚未同步的情况。
        return {"before": (), "current": (document,), "after": ()}  # 找不到邻居时仍返回当前引用，不伪造前后文。
    start = max(0, current_index - before)  # 计算前文切片起点。
    end = min(len(documents), current_index + after + 1)  # 计算后文切片终点。
    return {"before": tuple(documents[start:current_index]), "current": (document,), "after": tuple(documents[current_index + 1:end])}  # 返回前、当前、后三组文档。


def build_trace_report(documents: list[Document], question: str, window_chars: int = 280) -> str:  # 定义生成报告用的前后文追溯区块。
    lines = ["## 原文追溯", "", "以下内容只用于人工复核，不会额外发送给回答模型。", ""]  # 初始化追溯区块并明确 token 边界。
    for index, document in enumerate(documents, start=1):  # 遍历本次回答实际引用的片段。
        neighbors = get_adjacent_documents(document)  # 获取同一小节的前后片段。
        lines.extend([f"### 引用 [{index}]", "", f"定位：{format_source_locator(document)}", ""])  # 写入当前片段稳定定位。
        question_terms = extract_terms(question)  # 为当前引用和邻居统一抽取稳定查询词。
        current_text = extract_relevant_window(document.page_content, question_terms, window_chars)  # 裁剪当前片段，保留问题相关原文。
        lines.extend([f"当前片段：{current_text}", ""])  # 写入当前引用正文窗口。
        if neighbors["before"]:  # 如果存在前文。
            before_text = extract_relevant_window(neighbors["before"][-1].page_content, question_terms, window_chars)  # 只展示紧邻前一个片段的短窗口。
            lines.extend([f"前一片段：{format_source_locator(neighbors['before'][-1])}", f"前文窗口：{before_text}", ""])  # 写入前文定位和文本。
        else:  # 如果当前是小节第一段。
            lines.extend(["前一片段：无，这是该小节首个片段。", ""])  # 明确边界，避免用户以为追溯失败。
        if neighbors["after"]:  # 如果存在后文。
            after_text = extract_relevant_window(neighbors["after"][0].page_content, question_terms, window_chars)  # 只展示紧邻后一个片段的短窗口。
            lines.extend([f"后一片段：{format_source_locator(neighbors['after'][0])}", f"后文窗口：{after_text}", ""])  # 写入后文定位和文本。
        else:  # 如果当前是小节最后一段。
            lines.extend(["后一片段：无，这是该小节末个片段。", ""])  # 明确边界，避免用户以为追溯失败。
    return "\n".join(lines).rstrip()  # 返回完整追溯区块。


def build_binding_report(validation: dict) -> str:  # 定义生成逐结论引用绑定区块的函数。
    if not validation or not validation.get("sentence_results"):  # 证据不足或模型未生成事实句时没有逐结论绑定。
        return "## 逐结论引用绑定\n\n本次没有可供绑定的模型事实句。"  # 返回明确的空结果说明。
    lines = ["## 逐结论引用绑定", "", "| 结论 | 引用编号 | 原文覆盖率 | 是否支持 |", "| --- | --- | ---: | --- |"]  # 初始化可审计表格。
    for index, item in enumerate(validation["sentence_results"], start=1):  # 遍历引用校验器产生的逐句结果。
        sentence = item["sentence"].replace("|", "\\|").replace("\n", " ")  # 清理表格特殊字符但保留结论原文。
        citations = ", ".join(f"[{number}]" for number in item.get("citations", [])) or "无"  # 格式化绑定编号。
        supported = "是" if item.get("supported") else "否"  # 把布尔值转成人类可读状态。
        lines.append(f"| {index}. {sentence} | {citations} | {item.get('coverage', 0):.1%} | {supported} |")  # 写入每条结论的绑定结果。
    return "\n".join(lines)  # 返回逐结论绑定报告。
