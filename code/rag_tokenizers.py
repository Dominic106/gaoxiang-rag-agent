import re  # 导入 re，用正则提取英文缩写、数字编号和技术术语。

import jieba  # 导入 jieba，用来把中文句子切成适合 BM25 的词。

jieba.setLogLevel(30)  # 把 jieba 的日志级别调高到 warning，避免每次查询都打印加载词典的提示。


def tokenize_for_bm25(text: str) -> list[str]:  # 定义 BM25 专用分词函数，输入一段文本，输出词列表。
    words = [word.strip() for word in jieba.lcut(text) if word.strip()]  # 用 jieba 对中文正文分词，并去掉空词。
    extra = re.findall(r"[A-Za-z0-9_+.%-]+", text)  # 额外提取 WBS、PMBOK、PERT、1.2.3 等英文数字术语。
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", text))  # 把所有中文字符连起来，准备生成短语 n-gram。
    grams: list[str] = []  # 准备一个列表，用来保存 2 到 4 字的中文短语。
    for size in (2, 3, 4):  # 依次生成 2 字、3 字、4 字短语。
        grams.extend(chinese[index : index + size] for index in range(max(0, len(chinese) - size + 1)))  # 滑动窗口生成短语。
    return words + extra + grams  # 返回中文分词、英文数字术语和中文短语的合并列表。
