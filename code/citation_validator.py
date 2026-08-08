import re  # 导入 re，用来拆分句子、读取引用编号和提取英文术语。

import jieba  # 导入 jieba，用来抽取中文回答中的重要词语。


GENERIC_WORDS = {"当前", "教材", "项目", "管理", "内容", "主要", "包括", "可以", "通常", "进行", "相关", "部分", "问题", "因此", "根据", "依据", "如下", "作用", "结论"}  # 定义不适合用来判断原文支撑关系的泛词。


def claim_terms(sentence: str) -> list[str]:  # 定义提取回答句重要词语的函数。
    chinese_words = jieba.lcut(sentence)  # 使用 jieba 拆分中文词语。
    english_words = re.findall(r"[A-Za-z0-9_+.%-]{2,}", sentence)  # 提取英文、数字、缩写和带符号术语。
    terms: list[str] = []  # 准备保存去重后的有效词语。
    for term in chinese_words + english_words:  # 遍历中文词和英文术语。
        term = term.strip()  # 清理词语两端空白。
        if len(term) < 2 or term in GENERIC_WORDS:  # 过滤单字和泛化词。
            continue  # 跳过无辨识度词语。
        if term not in terms:  # 只保留第一次出现的词语。
            terms.append(term)  # 加入有效词语。
    return terms[:16]  # 控制最多检查 16 个词，避免单句过长影响校验。


def split_claims(answer: str) -> list[str]:  # 定义把模型回答拆成待校验句子的函数。
    return [item["sentence"] for item in extract_claim_records(answer)]  # 保留原有函数接口，只返回事实句文本。


def extract_claim_records(answer: str) -> list[dict]:  # 定义提取事实句及其引用编号的函数。
    raw_sentences = re.split(r"[。！？!?\n]+", answer)  # 按句末标点和换行拆分；同一回答行末尾的引用绑定整行事实，避免分号把一条带引用的模板事实误拆成多条。
    claims: list[dict] = []  # 准备保存事实句和对应的引用编号。
    for sentence in raw_sentences:  # 遍历拆出的句子。
        cited_numbers = sorted({int(value) for value in re.findall(r"\[(\d+)\]", sentence)})  # 读取当前片段里的引用编号。
        cleaned = re.sub(r"^\s*[-*#>\d.、)]+\s*", "", sentence).strip()  # 去掉 Markdown 符号和列表编号。
        cleaned = re.sub(r"\[\d+\]", "", cleaned).strip()  # 从事实文本中移除引用标记，避免引用编号参与术语校验。
        if not cleaned:  # 忽略空句子。
            if cited_numbers and claims:  # 如果引用编号被标点拆成了独立片段，就把它绑定到上一句。
                claims[-1]["citations"].extend(cited_numbers)  # 把独立引用追加到上一条事实句。
            continue  # 继续检查下一句。
        if cited_numbers and claims and ("引用" in cleaned or "依据" in cleaned):  # 模型有时把“引用编号：[1]”单独放在事实句之后，需要绑定到上一条事实句。
            claims[-1]["citations"].extend(cited_numbers)  # 把引用编号归属到最近一条事实句，避免格式差异导致安全降级。
            continue  # 引用说明本身不是新的事实句。
        if cleaned.endswith((":", "：")):  # 以冒号结尾通常只是引出后续内容的结构提示。
            continue  # 不把“例如：”这类提示当作事实句。
        if cleaned.startswith("|") and ("对比维度" in cleaned or re.fullmatch(r"\|?[\s|:-]+\|?", cleaned)):  # 对比模板的表头和分隔线是结构，不应被当成无引用事实。
            continue  # 跳过表格结构行，表格事实行仍然必须带引用并接受覆盖校验。
        if (cleaned.startswith("**") and cleaned.endswith("**")) or (cleaned.startswith("__") and cleaned.endswith("__")):  # 纯粗体或下划线文本通常是模型生成的小标题，不是需要逐句引用的事实。
            continue  # 忽略结构性小标题，避免格式本身造成误拒答。
        placeholder = re.sub(r"^[^：:]{1,12}[：:]\s*", "", cleaned).strip()  # 去掉“计算步骤：”“易错点：”等模板标签，识别无事实占位提示。
        if placeholder in {"当前教材片段未明确列出", "当前片段未明确给出易错提示", "当前片段未明确给出", "本题未提供数值，暂不进行代入计算"}:  # 这些提示只是诚实说明证据范围，不是教材事实。
            continue  # 不要求占位提示再绑定虚假的引用编号。
        if "引用" in cleaned or "原文" in cleaned or "知识库没有找到足够依据" in cleaned or "均可在" in cleaned or "直接依据" in cleaned or "明确依据" in cleaned or cleaned.startswith("本题未提供数值，暂不进行代入计算"):  # 这些是引用说明或计算流程提示，不属于模型知识陈述。
            continue  # 不把引用说明当作需要校验的事实句。
        if not claim_terms(cleaned):  # 如果句子只剩引用编号或结构性提示词。
            continue  # 不把“依据如下”和“[1]”这类结构内容当作事实句。
        claims.append({"sentence": cleaned, "citations": cited_numbers})  # 保存事实句和它同一片段里的引用编号。
    for claim in claims:  # 遍历所有事实句，清理可能重复绑定的引用编号。
        claim["citations"] = sorted(set(claim["citations"]))  # 去重并排序，便于审计和比较。
    return claims  # 返回待检查句子。


def validate_answer(answer: str, citation_text: str) -> dict:  # 定义严格引用校验函数，判断模型回答是否能被原文支撑。
    cited_numbers = sorted({int(value) for value in re.findall(r"\[(\d+)\]", answer)})  # 读取模型显式写出的引用编号。
    available_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", citation_text)}  # 读取系统实际提供的引用编号。
    invalid_numbers = [number for number in cited_numbers if number not in available_numbers]  # 找出模型引用了但系统没有提供的编号。
    source_blocks = {  # 按编号拆分系统提供的原文，确保每句话只核对自己引用的片段。
        int(number): block.strip()  # 把编号转为整数，并清理原文块空白。
        for number, block in re.findall(r"(?ms)^\[(\d+)\]\s+(.*?)(?=^\[\d+\]\s+|\Z)", citation_text)  # 读取每个引用编号对应的完整原文块。
    }  # 原文块字典构造结束。
    claims = extract_claim_records(answer)  # 拆出事实句、对应引用编号和引用关系。
    sentence_results: list[dict] = []  # 准备保存逐句校验结果。
    for claim in claims:  # 逐句检查回答是否被其标注的原文支撑。
        sentence = claim["sentence"]  # 取出当前事实句。
        claim_citations = claim["citations"]  # 取出当前事实句绑定的引用编号。
        terms = claim_terms(sentence)  # 抽取当前句的核心词语。
        cited_source_text = " ".join(source_blocks[number] for number in claim_citations if number in source_blocks)  # 只拼接当前句实际引用且有效的原文。
        hits = [term for term in terms if term in cited_source_text]  # 统计核心词语在对应引用原文中的出现情况。
        coverage = len(hits) / len(terms) if terms else 1.0  # 计算当前句的原文覆盖率。
        supported = bool(claim_citations) and not any(number not in available_numbers for number in claim_citations) and (coverage >= 0.45 or len(hits) >= 2)  # 必须有本句引用，且对应原文达到覆盖阈值。
        sentence_results.append({"sentence": sentence, "citations": claim_citations, "coverage": round(coverage, 3), "supported": supported})  # 保存可审计的校验结果。
    unsupported = [item for item in sentence_results if not item["supported"]]  # 汇总没有达到最低覆盖要求的句子。
    passed = bool(cited_numbers) and not invalid_numbers and not unsupported  # 必须有引用编号且每句都能被原文支撑才算通过。
    reason = "通过" if passed else "；".join(["缺少有效引用编号" if not cited_numbers else "", f"存在无效引用编号：{invalid_numbers}" if invalid_numbers else "", f"有 {len(unsupported)} 句未达到原文覆盖阈值" if unsupported else ""]).strip("；")  # 组织失败原因。
    return {"passed": passed, "cited_numbers": cited_numbers, "invalid_numbers": invalid_numbers, "sentence_results": sentence_results, "reason": reason or "未通过引用校验"}  # 返回完整校验结果。
