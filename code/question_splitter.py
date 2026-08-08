import re  # 导入 re，用正则识别编号、分隔符和中文问句边界。


def split_questions(user_input: str) -> list[str]:  # 定义多问题拆分函数，把用户一次输入拆成多个子问题。
    text = user_input.strip()  # 去掉首尾空白，避免产生空问题。
    if not text:  # 如果用户输入为空。
        return []  # 返回空列表，交给调用方处理。
    numbered = re.split(r"(?:^|\n|\s)(?:\d+[\.\、)]|[一二三四五六七八九十]+[、.])", text)  # 按 1. 1、 一、 这类编号拆分。
    numbered = [item.strip(" \n\t；;") for item in numbered if item.strip(" \n\t；;")]  # 清理拆出来的片段并去掉空片段。
    if len(numbered) > 1:  # 如果识别到了多个编号问题。
        return numbered  # 直接返回编号拆分结果，因为这种结构最可靠。
    parts = re.split(r"[；;]\s*|\n+", text)  # 如果没有编号，就按分号或换行拆分。
    parts = [item.strip() for item in parts if item.strip()]  # 清理每个片段。
    if len(parts) > 1:  # 如果分号或换行能拆出多个问题。
        return parts  # 返回这些子问题。
    question_marks = re.findall(r"[^？?]+[？?]", text)  # 尝试按问号提取完整问句。
    if len(question_marks) > 1:  # 如果有多个问号句。
        merged: list[str] = []  # 准备保存合并后的问句。
        continuation_heads = ("包括", "通常", "有哪些", "有什么", "为什么", "怎么", "如何", "分别", "它", "这个", "这些", "上述")  # 定义依赖前文的追问开头，覆盖“是什么？通常包括哪些……”这类同一问题。
        for item in question_marks:  # 遍历每个问号句。
            stripped = item.strip()  # 清理当前问句。
            if merged and stripped.startswith(continuation_heads):  # 如果当前句像是上一题的追补问题。
                merged[-1] = merged[-1] + stripped  # 就并回上一题，保留上下文完整。
            else:  # 如果当前句有独立主题。
                merged.append(stripped)  # 作为新问题加入。
        return merged  # 返回合并后的问题列表。
    return [text]  # 默认认为这是一个单问题。


def is_multi_question(user_input: str) -> bool:  # 定义判断是否多问题的辅助函数。
    return len(split_questions(user_input)) > 1  # 如果拆分后超过一个子问题，就认为是多问题。
