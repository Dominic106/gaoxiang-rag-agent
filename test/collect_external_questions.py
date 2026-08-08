"""从已经保存的公开网页中提取外部真题摘要，生成可审计压力测试集。"""

import html  # 导入 html，用来还原网页 title 属性里的 HTML 实体。
import json  # 导入 json，用来保存结构化题目和来源元数据。
import re  # 导入 re，用来提取题干、题号和网页链接。
from pathlib import Path  # 导入 Path，用来定位 test 临时资料目录。


TEST_ROOT = Path(__file__).resolve().parent  # 获取 test 目录。
RAW_ROOT = TEST_ROOT / "raw_pages"  # 获取已经下载的网页目录。
DATASET_ROOT = TEST_ROOT / "datasets"  # 获取压力测试数据目录。


TOPIC_RULES = (  # 按题干关键词给外部题目预标注可能的教材章节，后续仍以检索结果和人工抽查为准。
    (("范围", "WBS"), ("第18章 项目范围管理",)),  # 范围、WBS 归入范围管理。
    (("成本", "预算", "挣值", "利润", "投资回收"), ("第20章 项目成本管理", "第13章 信息系统项目管理基础")),  # 成本和计算题允许两个相关章节。
    (("风险", "威胁", "机会"), ("第24章 项目风险管理",)),  # 风险题归入风险管理。
    (("变更", "整体管理", "项目知识", "项目章程"), ("第17章 项目整体管理", "第26章 文档和配置管理")),  # 整合、变更和知识管理允许多个章节。
    (("团队", "人力", "资源"), ("第22章 项目人力资源管理",)),  # 团队与人力资源题归入人力资源章节。
    (("数据", "数据库", "数据表", "人工智能", "大模型"), ("第1章 信息系统基础知识", "第11章 信息化基础知识", "第34章 信息安全知识")),  # 数据和 AI 题覆盖基础、信息化和安全章节。
    (("安全", "隐私", "认证", "完整性", "密码"), ("第34章 信息安全知识",)),  # 安全题归入信息安全章节。
    (("流程", "组织", "质量"), ("第21章 项目质量管理", "第31章 用户业务流程管理")),  # 流程质量题允许质量和业务流程章节。
    (("网络", "通信"), ("第8章 计算机网络知识",)),  # 网络题归入网络章节。
    (("云", "容器"), ("第9章 云计算",)),  # 云题归入云计算章节。
    (("软件", "开发", "构件", "中间件"), ("第2章 软件工程基础知识", "第3章 软件构件与中间件")),  # 软件工程题允许相邻软件章节。
)  # 关键词与章节映射结束。


def infer_expected_chapters(question: str) -> list[str]:  # 定义外部题目章节预标注函数。
    chapters: list[str] = []  # 准备保存所有命中的可能章节。
    for keywords, candidates in TOPIC_RULES:  # 遍历关键词规则。
        if any(keyword in question for keyword in keywords):  # 当前题目命中任意规则关键词时。
            for chapter in candidates:  # 遍历该规则允许的章节。
                if chapter not in chapters:  # 避免重复章节。
                    chapters.append(chapter)  # 加入候选章节。
    return chapters or ["第13章 信息系统项目管理基础"]  # 没有明显关键词时归入项目管理基础，标记为需要人工复核。


def parse_page(path: Path) -> list[dict]:  # 定义解析一张真题网页的函数。
    page_suffix = path.stem.rsplit("_", 1)[-1] if path.stem != "cnitpm_2025_batch1" else ""  # 从保存文件名提取分页编号，第一页使用根路径。
    source_url = f"https://www.cnitpm.com/examst/13145674/{page_suffix}.html" if page_suffix else "https://www.cnitpm.com/examst/13145674/"  # 记录当前分页的公开来源 URL。
    text = path.read_text(encoding="utf-8", errors="ignore")  # 读取网页原始 HTML。
    pattern = r'<a href="(?P<href>//www\.cnitpm\.com/st/[^" ]+)" title="(?P<title>[^"]*)"[^>]*>.*?\[第(?P<number>\d+)道试题\]'  # 匹配题目详情链接、title 题干和页面题号。
    rows: list[dict] = []  # 准备保存当前网页的题目摘要。
    for match in re.finditer(pattern, text, flags=re.DOTALL):  # 遍历所有匹配到的公开题目卡片。
        question = re.sub(r"\s+", " ", html.unescape(match.group("title"))).strip()  # 还原 HTML 实体并压缩空白。
        if not question:  # 如果 title 没有题干。
            continue  # 跳过坏数据。
        rows.append({  # 保存最小可审计题目结构，不复制整套试卷正文。
            "question_id": f"cnitpm-2025-batch1-{match.group('number')}-{match.group('href').rsplit('/', 1)[-1]}",  # 组合稳定题目标识。
            "question": question,  # 保存网页公开题干摘要。
            "source_url": "https:" + match.group("href"),  # 保存题目详情页地址。
            "paper_url": source_url,  # 保存试卷分页地址。
            "year": 2025,  # 保存试卷年份。
            "batch": "第一批次",  # 保存试卷批次。
            "subject": "综合知识",  # 保存科目。
            "source_type": "public_true_question_excerpt",  # 标记为公开真题摘要，不是完整试卷复制。
            "expected_chapters": infer_expected_chapters(question),  # 保存待验证的章节候选。
        })  # 单题记录结束。
    return rows  # 返回当前网页的所有题目。


def main() -> None:  # 定义采集主函数。
    pages = sorted(RAW_ROOT.glob("cnitpm_2025_batch1*.html"))  # 按稳定文件名读取 2025 第一批次的全部分页。
    rows: list[dict] = []  # 准备保存去重后的外部题目。
    seen_questions: set[str] = set()  # 用题干去重，避免分页或网页重复导致统计膨胀。
    for page in pages:  # 遍历保存好的网页。
        for row in parse_page(page):  # 解析当前网页题目。
            if row["question"] in seen_questions:  # 如果题干已经出现。
                continue  # 跳过重复题。
            seen_questions.add(row["question"])  # 记录新题干。
            rows.append(row)  # 保存外部题目。
    rows.sort(key=lambda item: (item["year"], item["batch"], item["question_id"]))  # 按稳定字段排序，保证每次生成结果一致。
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    output_path = DATASET_ROOT / "external_true_questions_2025_batch1.json"  # 定义外部真题摘要文件路径。
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存结构化题目摘要。
    print(f"外部真题摘要：{output_path}")  # 输出生成文件路径。
    print(f"去重题数：{len(rows)}")  # 输出题目数量。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行采集主函数。
