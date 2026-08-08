"""正式评估集的数据质量检查。"""  # 说明本脚本只检查评估数据结构，不调用模型 API。

import json  # 导入 json，用来读取正式评估集。
from collections import Counter  # 导入 Counter，用来统计章节和问题类型覆盖。

from config import CHUNKS_JSONL  # 从配置读取 chunk 明细，验证期望章节确实存在于索引。
from config import EVAL_SAMPLES_PATH  # 从配置读取正式评估集路径。


ALLOWED_TYPES = {"定义解释", "区别对比", "流程步骤", "输入输出工具技术", "公式计算", "考点记忆", "泛化查询"}  # 固定当前问题理解层支持的问题类型。
MIN_TOTAL = 100  # 定义评估集最低总题数。
MIN_POSITIVE = 80  # 定义正向检索题最低数量，避免大量拒答题掩盖检索覆盖不足。
MIN_NEGATIVE = 5  # 定义拒答题最低数量，确保有基本的范围边界测试。


def load_chunks_chapters() -> set[str]:  # 定义读取索引章节集合的函数。
    chapters: set[str] = set()  # 准备保存索引中的标准章节名。
    for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines():  # 逐行读取 chunk 明细。
        if not line.strip():  # 如果是空行。
            continue  # 跳过空行。
        item = json.loads(line)  # 解析一条 chunk 记录。
        chapters.add(item["metadata"]["chapter"])  # 把 chunk 的标准章节加入集合。
    return chapters  # 返回索引章节集合。


def validate_samples() -> dict:  # 定义评估集质量检查函数。
    samples = json.loads(EVAL_SAMPLES_PATH.read_text(encoding="utf-8"))  # 读取正式评估集。
    assert isinstance(samples, list), "评估集顶层必须是数组"  # 确认数据结构是题目列表。
    assert len(samples) >= MIN_TOTAL, f"评估集数量不足：{len(samples)} < {MIN_TOTAL}"  # 确认总题数达到阶段目标。
    questions = [sample.get("question", "").strip() for sample in samples]  # 收集所有问题文本。
    assert all(questions), "评估集不能存在空问题"  # 拒绝空题目。
    assert len(questions) == len(set(questions)), "评估集存在重复问题"  # 拒绝完全重复题目，避免虚高样本量。
    chapters = load_chunks_chapters()  # 读取当前索引中的真实章节。
    positive = [sample for sample in samples if not sample.get("should_refuse", False)]  # 筛出正向检索题。
    negative = [sample for sample in samples if sample.get("should_refuse", False)]  # 筛出应该拒答题。
    assert len(positive) >= MIN_POSITIVE, f"正向检索题数量不足：{len(positive)} < {MIN_POSITIVE}"  # 确认正向题足够覆盖检索。
    assert len(negative) >= MIN_NEGATIVE, f"拒答题数量不足：{len(negative)} < {MIN_NEGATIVE}"  # 确认有基本范围边界。
    for sample in samples:  # 遍历每道题检查字段。
        assert sample.get("question_type") in ALLOWED_TYPES, f"问题类型非法：{sample}"  # 确认类型来自统一枚举。
        assert sample.get("key_terms"), f"题目缺少 key_terms：{sample.get('question')}"  # 确认后续答案评估有术语依据。
        if sample.get("should_refuse", False):  # 如果是拒答题。
            assert not sample.get("expected_contains"), f"拒答题不应配置期望章节：{sample.get('question')}"  # 拒答题不应伪造章节命中目标。
        else:  # 如果是正向题。
            expected = sample.get("expected_contains_any", [sample.get("expected_contains", "")])  # 读取一个或多个合法章节。
            assert all(value in chapters for value in expected), f"期望章节不在当前索引：{sample.get('question')} -> {expected}"  # 确认期望章节确实存在。
    chapter_counter: Counter[str] = Counter()  # 准备统计正向题章节覆盖。
    for sample in positive:  # 遍历正向题。
        expected = sample.get("expected_contains_any", [sample["expected_contains"]])  # 取正向题合法章节。
        for chapter in expected:  # 遍历合法章节。
            chapter_counter[chapter] += 1  # 累加章节覆盖题数。
    assert len(chapter_counter) >= 35, f"章节覆盖不足：{len(chapter_counter)} < 35"  # 确认全书 35 章都有正向评估题。
    return {  # 返回结构化质量统计。
        "total": len(samples),  # 保存总题数。
        "positive": len(positive),  # 保存正向题数。
        "negative": len(negative),  # 保存拒答题数。
        "chapters": len(chapter_counter),  # 保存覆盖章节数。
        "types": dict(Counter(sample["question_type"] for sample in samples)),  # 保存问题类型分布。
        "minimums": {"total": MIN_TOTAL, "positive": MIN_POSITIVE, "negative": MIN_NEGATIVE},  # 保存当前阶段最低标准。
    }  # 质量统计结束。


def main() -> None:  # 定义评估集质量检查入口。
    result = validate_samples()  # 执行全部数据质量检查。
    print(f"评估集质量通过：总题数={result['total']}，正向题={result['positive']}，拒答题={result['negative']}，章节={result['chapters']}。")  # 输出可读的阶段验收结果。
    print(f"问题类型分布：{result['types']}")  # 输出类型分布，方便发现题型偏斜。


if __name__ == "__main__":  # 判断脚本是否直接运行。
    main()  # 直接运行时执行质量检查。
