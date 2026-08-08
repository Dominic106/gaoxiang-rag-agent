from chapter_filter import list_chapters  # 导入章节清单函数，验证教材章节可以被过滤模块读取。
from chapter_filter import resolve_chapter_filter  # 导入章节解析函数，验证短章节名可以映射到标准名称。
from citation_validator import validate_answer  # 导入引用校验函数，验证通过和失败两条路径。
from langchain_core.documents import Document  # 导入 Document，用于构造稳定定位字段的最小回归样例。
from query_understanding import understand_query  # 导入问题理解函数，验证关键问题类型没有回归。
from question_splitter import split_questions  # 导入多问题拆分函数，验证多个知识点不会互相污染。
from source_trace import format_source_locator  # 导入原文定位格式化函数，验证引用位置可读且稳定。
from study_memory import is_follow_up_question  # 导入追问判断函数，验证会话记忆触发条件。


def test_query_understanding() -> None:  # 定义问题分类回归测试。
    cases = {"关键路径法怎么计算": "公式计算", "项目章程的输入输出工具技术是什么": "输入输出工具技术", "风险分析有什么区别": "区别对比", "项目采购的主要过程是什么": "流程步骤", "云计算有哪些服务模式": "考点记忆"}  # 准备覆盖五类核心问题的样例。
    for question, expected_type in cases.items():  # 遍历每个分类样例。
        actual_type = understand_query(question).question_type  # 调用规则分类器获取实际类型。
        assert actual_type == expected_type, f"问题分类错误：{question} -> {actual_type}，期望 {expected_type}"  # 分类不一致时立即报错。


def test_chapter_filter() -> None:  # 定义章节过滤回归测试。
    chapters = list_chapters()  # 读取教材章节清单。
    assert len(chapters) >= 35, "章节清单数量异常，可能没有正确读取 chunks.jsonl"  # 确认整本教材章节都可读取。
    assert resolve_chapter_filter("第17章") == "第17章 项目整体管理", "短章节名没有解析到标准章节"  # 验证用户常用的短输入。
    assert resolve_chapter_filter("项目整体管理") == "第17章 项目整体管理", "章节标题过滤解析失败"  # 验证用户输入标题关键字。


def test_memory_trigger() -> None:  # 定义会话记忆触发回归测试。
    assert is_follow_up_question("它和项目管理计划有什么区别"), "没有识别出代词追问"  # 代词追问必须读取上文。
    assert is_follow_up_question("继续刚才那个问题"), "没有识别出刚才追问"  # “刚才”追问必须读取上文。
    assert not is_follow_up_question("什么是范围基准"), "独立问题不应该无条件读取会话记忆"  # 独立问题要节省上下文 token。


def test_question_splitter() -> None:  # 定义多问题拆分回归测试。
    numbered = split_questions("1. 什么是项目章程？ 2. 项目管理计划有什么特点？")  # 构造用户一次提出两个编号问题的输入。
    assert len(numbered) == 2, f"编号问题拆分数量错误：{numbered}"  # 两个编号必须得到两个独立子问题。
    assert numbered[0].startswith("什么是项目章程"), "第一个子问题内容被编号污染"  # 编号不应该留在子问题正文中。
    continued = split_questions("什么是范围基准？包括哪些内容？")  # 构造后一句依赖前一句主题的连续问法。
    assert len(continued) == 1, f"连续追问不应该被拆成两题：{continued}"  # 追补内容应保留在同一个检索任务里。
    usual_continued = split_questions("什么是信息系统的生命周期？通常包括哪些阶段？")  # 覆盖“是什么？通常包括哪些……”这类教材学习场景常见问法。
    assert len(usual_continued) == 1, f"通常包括式连续问句不应该被拆成两题：{usual_continued}"  # 后一句仍然依赖前一句主题。


def test_citation_validator() -> None:  # 定义严格引用校验回归测试。
    citations = "[1] 第17章 / 原文片段：项目章程正式授权项目并任命项目经理。"  # 构造一条最小可核对原文。
    supported = validate_answer("项目章程正式授权项目并任命项目经理。[1]", citations)  # 构造有引用且有原文依据的回答。
    missing_citation = validate_answer("项目章程正式授权项目并任命项目经理。", citations)  # 构造事实存在但故意漏掉引用编号的回答。
    separated_citation = validate_answer("项目章程正式授权项目并任命项目经理。\n引用编号：[1]", citations)  # 模拟模型把引用编号单独放到下一行的常见格式。
    unsupported = validate_answer("项目章程一定会降低项目成本。[1]", citations)  # 构造引用编号存在但原文不支持的回答。
    assert supported["passed"], "有原文支持的回答没有通过校验"  # 正确回答必须放行。
    assert not missing_citation["passed"], "没有引用编号的事实不应该放行"  # 漏引必须被拦截。
    assert separated_citation["passed"], "单独成行的引用编号应该绑定到上一条事实句"  # 引用格式换行不应导致安全降级。
    assert not unsupported["passed"], "无原文支持的回答不应该放行"  # 无依据回答必须拦截。
    isolated_sources = "[1] 项目章程正式授权项目并任命项目经理。\n\n[2] 风险登记册记录已识别风险。"  # 构造两个主题不同的原文块，检查引用不能跨块借词。
    cross_block = validate_answer("风险登记册记录风险。[1]", isolated_sources)  # 故意引用错误的原文编号。
    assert not cross_block["passed"], "回答不应该借用未标注的其他引用块通过校验"  # 引用绑定错误必须被拦截。


def test_source_locator() -> None:  # 定义稳定原文定位格式回归测试。
    document = Document(page_content="正文", metadata={"relative_path": "第17章 项目整体管理/17.1 项目章程.docx", "section": "17.1 项目章程", "section_chunk_index": 2, "section_chunk_count": 5, "source_char_start": 1800, "source_char_end": 2600, "chunk_id": "xg-000123"})  # 构造一个带完整定位元数据的最小文档。
    locator = format_source_locator(document)  # 生成用户可读的定位信息。
    assert "17.1 项目章程第 3/5 个片段" in locator, "小节片段序号没有正确展示"  # 确认序号从 1 开始且包含总数。
    assert "字符 1800-2600" in locator, "源文档字符范围没有正确展示"  # 确认字符范围可追溯。
    assert "chunk_id=xg-000123" in locator, "稳定 chunk_id 没有展示"  # 确认原有稳定 ID 仍保留。


def main() -> None:  # 定义回归测试入口。
    test_query_understanding()  # 执行问题理解测试。
    test_chapter_filter()  # 执行章节过滤测试。
    test_memory_trigger()  # 执行记忆触发测试。
    test_question_splitter()  # 执行多问题拆分测试。
    test_citation_validator()  # 执行引用校验测试。
    test_source_locator()  # 执行原文定位格式测试。
    print("回归测试通过：问题理解、章节过滤、会话记忆触发、严格引用校验、原文定位。")  # 输出测试成功信息。


if __name__ == "__main__":  # 判断脚本是否被直接运行。
    main()  # 直接运行时执行全部回归测试。
