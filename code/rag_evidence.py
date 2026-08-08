import math  # 导入 math，用来计算核心术语最低命中数量。
import re  # 导入 re，用来抽取英文术语和裁剪引用窗口。

import jieba  # 导入 jieba，用来抽取中文核心术语。
from langchain_core.documents import Document  # 导入 Document，用来标注证据片段类型。

from config import FINAL_CONTEXTS  # 从配置读取最终上下文数量。
from config import MAX_CONTEXT_CHARS  # 从配置读取引用窗口最大长度。
from config import MAX_RETRIEVAL_ATTEMPTS  # 从配置读取最多检索自救次数。
from config import MEDIUM_CONTEXTS  # 从配置读取中等问题上下文数量。
from config import MIN_EVIDENCE_SCORE  # 从配置读取最低证据分。
from config import SIMPLE_CONTEXTS  # 从配置读取简单问题上下文数量。
from rag_tokenizers import tokenize_for_bm25  # 导入 BM25 分词函数，复用统一中文术语切分规则。


def extract_terms(text: str) -> list[str]:  # 定义关键词抽取函数，用来做证据评分和引用窗口裁剪。
    raw_terms = tokenize_for_bm25(text)  # 复用 BM25 的中文分词、n-gram 和英文术语抽取。
    useful_terms = []  # 准备保存更有辨识度的关键词。
    stop_words = {"什么", "怎么", "如何", "包括", "哪些", "区别", "不同", "比较", "一下", "的", "和", "与"}  # 定义常见停用词。
    for term in raw_terms:  # 遍历所有候选词。
        if len(term) < 2:  # 过滤单字词，单字太容易误命中。
            continue  # 跳过当前词。
        if term in stop_words:  # 如果是停用词。
            continue  # 跳过当前词。
        if term not in useful_terms:  # 如果这个词还没加入。
            useful_terms.append(term)  # 加入关键词列表。
    return useful_terms[:24]  # 控制关键词数量，避免后续评分过宽。


def extract_core_terms(text: str) -> list[str]:  # 定义核心术语抽取函数，用来判断证据是否真的覆盖用户问题。
    raw_words = jieba.lcut(text) + re.findall(r"[A-Za-z0-9_+.%-]{2,}", text)  # 用 jieba 词和英文术语做核心词，避免 n-gram 太多造成误判。
    generic_words = {"什么", "怎么", "如何", "包括", "哪些", "内容", "区别", "不同", "比较", "项目", "管理", "分析", "基准", "进行", "当前"}  # 定义会抬高假阳性的泛词。
    core_terms = []  # 准备保存核心术语。
    for word in raw_words:  # 遍历候选词。
        if len(word) < 2:  # 过滤单字。
            continue  # 跳过单字。
        if word in generic_words:  # 过滤泛词。
            continue  # 跳过泛词。
        if re.fullmatch(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_+.%-]{2,}", word) and word not in core_terms:  # 只保留中文短语或英文数字术语。
            core_terms.append(word)  # 加入核心术语列表。
    return core_terms[:12]  # 控制核心术语数量，避免超长问题影响判断。


def build_retrieval_queries(question: str, expanded_query: str) -> list[str]:  # 定义构造多次检索 query 的函数。
    terms = extract_terms(question)  # 从原问题里抽取高辨识度关键词。
    compact_query = " ".join(terms[:10])  # 把前 10 个关键词拼成更短的 query。
    queries = [expanded_query, question, compact_query]  # 第一轮增强查，第二轮原句查，第三轮关键词查。
    cleaned = []  # 准备保存去重后的 query。
    for query in queries:  # 遍历候选 query。
        query = query.strip()  # 清理空白。
        if query and query not in cleaned:  # 如果非空且没出现过。
            cleaned.append(query)  # 加入最终 query 列表。
    return cleaned[:MAX_RETRIEVAL_ATTEMPTS]  # 最多返回配置规定的尝试次数。


def score_evidence(question: str, docs: list[Document]) -> int:  # 定义证据评分函数，用来判断是否可以回答。
    terms = extract_terms(question)  # 从原问题抽取关键词。
    if not docs or not terms:  # 如果没有文档或没有关键词。
        return 0  # 直接返回 0 分。
    score = 0  # 初始化证据分。
    top_docs = docs[:3]  # 只看前三个片段，避免无关片段把分数抬高。
    for doc_index, doc in enumerate(top_docs):  # 遍历前三个片段。
        text = f"{doc.metadata.get('chapter', '')} {doc.metadata.get('section', '')} {doc.page_content}"  # 拼出标题和正文。
        hit_count = sum(1 for term in terms if term in text)  # 统计该片段命中了多少关键词。
        score += min(hit_count, 5) * (3 - doc_index)  # 越靠前的片段权重越高，每个片段最多贡献 5 个词。
    joined_text = " ".join(f"{doc.metadata.get('chapter', '')} {doc.metadata.get('section', '')} {doc.page_content}" for doc in top_docs)  # 把前三个片段合并，用来做核心术语覆盖率。
    core_terms = extract_core_terms(question)  # 抽取用户问题里的核心术语。
    if core_terms:  # 如果存在核心术语。
        core_hits = sum(1 for term in core_terms if term in joined_text)  # 统计核心术语命中数量。
        core_coverage = core_hits / len(core_terms)  # 计算核心术语覆盖率。
        required_hits = 1 if len(core_terms) <= 2 else max(2, math.ceil(len(core_terms) * 0.5))  # 短问题至少命中 1 个核心词，长问题至少命中一半核心词。
        if core_hits < required_hits or core_coverage < 0.34:  # 如果核心术语命中数或覆盖率太低。
            return min(score, MIN_EVIDENCE_SCORE - 1)  # 强制压到阈值以下，防止模型硬答。
    return score  # 返回最终证据分。


def choose_context_limit(question_type: str, evidence_score: int) -> int:  # 定义动态上下文数量函数，用来节省 token。
    if evidence_score >= 18 and question_type in {"定义解释", "考点记忆"}:  # 如果证据很强且问题相对简单。
        return SIMPLE_CONTEXTS  # 只给 3 段上下文。
    if evidence_score >= MIN_EVIDENCE_SCORE:  # 如果证据够但不算特别强。
        return MEDIUM_CONTEXTS  # 给 5 段上下文。
    return FINAL_CONTEXTS  # 如果证据不足，保留更多片段用于报告“最接近内容”。


def extract_relevant_window(text: str, terms: list[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:  # 定义引用窗口裁剪函数。
    flat_text = text.replace("\n", " ")  # 把换行替换为空格，方便模型阅读。
    first_hit = None  # 准备记录第一个命中关键词的位置。
    for term in terms:  # 遍历关键词。
        index = flat_text.find(term)  # 查找关键词位置。
        if index >= 0 and (first_hit is None or index < first_hit):  # 如果命中且位置更靠前。
            first_hit = index  # 更新第一个命中位置。
    if first_hit is None:  # 如果没有关键词命中。
        return flat_text[:max_chars]  # 返回开头窗口。
    start = max(0, first_hit - max_chars // 3)  # 从命中点前面一点开始截，保留上下文。
    end = min(len(flat_text), start + max_chars)  # 计算结束位置。
    return flat_text[start:end]  # 返回相关窗口。
