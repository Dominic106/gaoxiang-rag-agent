"""从已经保存的公开章节题库页面提取考点树和题量。"""

import json  # 导入 json，用来保存结构化考点树。
import re  # 导入 re，用来解析页面中的父级和子级 HTML 片段。
from pathlib import Path  # 导入 Path，用来定位临时资料目录。


TEST_ROOT = Path(__file__).resolve().parent  # 获取 test 目录。
RAW_PATH = TEST_ROOT / "raw_pages" / "cnitpm_chapter_index.html"  # 定义已保存章节题库页面路径。
DATASET_ROOT = TEST_ROOT / "datasets"  # 定义结构化测试数据目录。


def parse_tree(html_text: str) -> list[dict]:  # 定义章节树解析函数。
    parent_pattern = re.compile(r'<div class="shitgolf" data-id="(?P<id>\d+)"><p><i></i>(?P<name>.*?)</p><p>题量：(?P<count>\d+)</p></div>(?P<tail>.*?)(?=<div class="shitgolf"|\Z)', re.DOTALL)  # 匹配一级主题和其直到下一个一级主题前的子主题区域。
    child_pattern = re.compile(r"categoryId2=(?P<id>\d+)[^>]*>.*?<div><p>(?P<name>.*?)</p><p>题量：(?P<count>\d+)</p>", re.DOTALL)  # 匹配一级主题下的子主题编号、名称和题量。
    tree: list[dict] = []  # 准备保存全部一级主题。
    for parent in parent_pattern.finditer(html_text):  # 遍历所有一级主题。
        children = [  # 构造当前一级主题的子主题列表。
            {"id": child.group("id"), "name": re.sub(r"\s+", " ", child.group("name")).strip(), "question_count": int(child.group("count"))}  # 保存子主题最小字段。
            for child in child_pattern.finditer(parent.group("tail"))  # 从当前一级主题的尾部区域提取子主题。
        ]  # 子主题列表构造结束。
        tree.append({"id": parent.group("id"), "name": parent.group("name").strip(), "question_count": int(parent.group("count")), "children": children})  # 保存一级主题和子主题。
    return tree  # 返回完整考点树。


def main() -> None:  # 定义主函数。
    html_text = RAW_PATH.read_text(encoding="utf-8", errors="ignore")  # 读取已经保存的原始页面。
    tree = parse_tree(html_text)  # 解析一级主题和子主题。
    if not tree:  # 检查页面结构是否发生变化。
        raise RuntimeError("没有解析到章节题库，可能是公开页面结构发生变化。")  # 用明确错误阻止生成空考点树。
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)  # 确保数据目录存在。
    output_path = DATASET_ROOT / "external_knowledge_tree.json"  # 定义考点树输出路径。
    output_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存结构化考点树。
    child_count = sum(len(item["children"]) for item in tree)  # 统计子主题数量。
    question_count = sum(item["question_count"] for item in tree)  # 统计一级主题题量合计。
    print(json.dumps({"parent_topics": len(tree), "child_topics": child_count, "question_count_sum": question_count, "output": str(output_path)}, ensure_ascii=False))  # 输出解析摘要。


if __name__ == "__main__":  # 判断是否由命令行直接运行。
    main()  # 执行主函数。
