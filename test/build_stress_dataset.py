"""合并内部评估集、外部真题摘要和章节知识点，生成压力测试输入。"""

import json  # 导入 json，用来读取和写入测试集。
from pathlib import Path  # 导入 Path，用来处理项目路径。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 获取 RAG 项目根目录。
TEST_ROOT = PROJECT_ROOT / "test"  # 获取临时测试目录。
DATASET_ROOT = TEST_ROOT / "datasets"  # 获取数据集输出目录。


KNOWLEDGE_POINT_QUESTIONS = (  # 定义覆盖新版考纲和章节树的知识点问题。
    ("什么是信息系统的生命周期？通常包括哪些阶段？", "第1章 信息系统基础知识"),  # 信息系统基础。
    ("软件工程中的瀑布模型、原型模型和迭代模型有什么区别？", "第2章 软件工程基础知识"),  # 软件工程模型。
    ("什么是中间件？中间件主要解决什么问题？", "第3章 软件构件与中间件"),  # 构件与中间件。
    ("面向对象方法中的封装、继承和多态分别是什么意思？", "第4章 面向对象方法"),  # 面向对象基础。
    ("J2EE和.NET平台有哪些异同？", "第5章 J2EE 与.NET平台"),  # 平台对比。
    ("Web Service的基本使用流程是什么？", "第6章 Web Service 技术"),  # Web Service。
    ("工作流管理系统的基本概念和组成是什么？", "第7章 工作流"),  # 工作流。
    ("TCP和UDP有什么区别？", "第8章 计算机网络知识"),  # 网络协议。
    ("云计算的IaaS、PaaS和SaaS分别是什么？", "第9章 云计算"),  # 云服务模式。
    ("物联网的感知层、网络层和应用层分别承担什么作用？", "第10章 物联网"),  # 物联网。
    ("什么是数字化转型？信息化和数字化转型有什么关系？", "第11章 信息化基础知识"),  # 信息化。
    ("信息系统服务管理包括哪些主要内容？", "第12章 信息系统服务管理"),  # 服务管理。
    ("项目和项目管理有什么区别？", "第13章 信息系统项目管理基础"),  # 项目管理基础。
    ("项目生命周期和产品生命周期有什么区别？", "第14章 项目生命周期和组织"),  # 生命周期。
    ("项目管理五大过程组分别是什么？", "第15章 项目管理过程"),  # 过程组。
    ("项目立项管理包括哪些过程？", "第16章 项目立项与招投标管理"),  # 立项。
    ("项目章程的作用是什么？整体变更控制如何进行？", "第17章 项目整体管理"),  # 整体管理。
    ("什么是范围基准？范围蔓延时应该如何控制？", "第18章 项目范围管理"),  # 范围管理。
    ("关键路径法如何计算？总时差和自由时差有什么区别？", "第19章 进度控制"),  # 进度控制。
    ("挣值管理中的PV、EV、AC、CPI和SPI分别是什么？", "第20章 项目成本管理"),  # 成本和挣值。
    ("规划质量、管理质量和控制质量有什么区别？", "第21章 项目质量管理"),  # 质量管理。
    ("建设团队和管理团队分别关注什么？", "第22章 项目人力资源管理"),  # 人力资源。
    ("沟通模型中的编码、传递、噪声和解码分别是什么意思？", "第23章 项目沟通管理"),  # 沟通管理。
    ("定性风险分析和定量风险分析有什么区别？风险应对策略有哪些？", "第24章 项目风险管理"),  # 风险。
    ("采购管理中的自制或外购分析如何进行？合同类型有哪些？", "第25章 项目采购和合同管理"),  # 采购。
    ("配置管理和变更管理分别解决什么问题？", "第26章 文档和配置管理"),  # 配置。
    ("需求获取、需求分析、需求确认和需求跟踪之间是什么关系？", "第27章 需求管理"),  # 需求。
    ("外包管理的主要风险和控制措施有哪些？", "第28章 外包管理"),  # 外包。
    ("项目集、项目组合和组织级项目管理有什么区别？", "第29章 大型、复杂项目和多项目管理"),  # 高级项目管理。
    ("组织战略、项目管理和组织绩效之间有什么关系？", "第30章 战略管理"),  # 战略。
    ("业务流程管理的基本步骤是什么？", "第31章 用户业务流程管理"),  # 流程。
    ("知识管理中的显性知识和隐性知识有什么区别？", "第32章 知识管理"),  # 知识管理。
    ("项目绩效考核应关注哪些指标？", "第33章 项目绩效考核与绩效管理"),  # 绩效。
    ("密码技术如何保护信息的机密性、完整性和真实性？", "第34章 信息安全知识"),  # 信息安全。
    ("信息系统工程监理的主要工作内容是什么？", "第35章 信息系统工程监理"),  # 工程监理。
)  # 知识点问题结束。


def load_json(path: Path) -> list[dict]:  # 定义读取 JSON 数组的函数。
    value = json.loads(path.read_text(encoding="utf-8"))  # 读取并解析 JSON。
    if not isinstance(value, list):  # 确保数据结构是数组。
        raise TypeError(f"测试数据必须是数组：{path}")  # 阻止坏数据进入压力测试。
    return value  # 返回数据数组。


def main() -> None:  # 定义压力测试集构建函数。
    external = load_json(DATASET_ROOT / "external_true_questions_2025_batch1.json")  # 读取外部真题摘要。
    internal = load_json(PROJECT_ROOT / "notes" / "answer_eval_v1.json")  # 读取已有 31 题答案可靠性集。
    rows: list[dict] = []  # 准备保存合并后的压力样本。
    for item in external:  # 遍历外部真题。
        rows.append({**item, "test_group": "external_true_question", "should_answer": True, "question_type": "考点记忆"})  # 保留外部来源和章节候选。
    for index, item in enumerate(internal, start=1):  # 遍历内部答案评估集。
        rows.append({**item, "question_id": f"internal-answer-eval-{index:03d}", "test_group": "internal_answer_eval", "should_answer": not item.get("should_refuse", False), "source_type": "existing_curated_eval"})  # 保留内部标准答案评估字段。
    for index, (question, chapter) in enumerate(KNOWLEDGE_POINT_QUESTIONS, start=1):  # 遍历 35 个章节知识点问题。
        rows.append({"question_id": f"knowledge-point-{index:03d}", "question": question, "expected_chapters": [chapter], "test_group": "knowledge_point", "source_type": "chapter_tree_probe", "should_answer": True, "question_type": "考点记忆"})  # 生成章节覆盖探针。
    rows.extend([  # 加入用于拒答边界和用户需求判断的负向问题。
        {"question_id": "negative-001", "question": "请预测下个月股票价格并给出买入建议", "test_group": "negative_boundary", "source_type": "out_of_scope", "should_answer": False},  # 金融预测不属于教材知识。
        {"question_id": "negative-002", "question": "请帮我规划一次旅游行程", "test_group": "negative_boundary", "source_type": "out_of_scope", "should_answer": False},  # 旅游规划不属于教材知识。
        {"question_id": "negative-003", "question": "请告诉我2027年世界杯冠军是谁", "test_group": "negative_boundary", "source_type": "out_of_scope", "should_answer": False},  # 未来事实不属于教材知识。
        {"question_id": "negative-004", "question": "请写一首与项目管理无关的现代诗", "test_group": "negative_boundary", "source_type": "out_of_scope", "should_answer": False},  # 创作任务不属于教材检索。
    ])  # 负向问题结束。
    output_path = DATASET_ROOT / "rag_stress_questions.json"  # 定义合并压力测试集路径。
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存完整压力样本。
    summary = {"external_true_question": len(external), "internal_answer_eval": len(internal), "knowledge_point": len(KNOWLEDGE_POINT_QUESTIONS), "negative_boundary": 4, "total": len(rows)}  # 组织数量摘要。
    (DATASET_ROOT / "rag_stress_questions_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存数量摘要。
    print(json.dumps(summary, ensure_ascii=False))  # 输出数量摘要供自动化记录。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行压力测试集构建。
