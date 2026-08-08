"""备考场景回答模板和结构校验。"""  # 说明本模块只约束回答结构，不放宽原文引用安全门。

import re  # 导入 re，用来识别模板标签、表格表头和结构完整性。
from dataclasses import dataclass  # 导入 dataclass，用来定义不可变的模板说明对象。


@dataclass(frozen=True)
class AnswerTemplate:  # 定义一类学习回答模板的结构。
    key: str  # 保存模板稳定键。
    name: str  # 保存用户可读的模板名称。
    required_sections: tuple[str, ...]  # 保存回答必须出现的结构标签。
    instructions: str  # 保存发送给回答模型的格式和内容要求。


TEMPLATES = {  # 定义五类备考回答模板。
    "definition": AnswerTemplate(  # 定义解释模板。
        key="definition",  # 保存定义模板键。
        name="定义解释",  # 保存模板名称。
        required_sections=("定义结论", "核心要点", "教材依据"),  # 固定定义、要点和依据三个学习区块。
        instructions="输出顺序固定为：定义结论、核心要点、教材依据。定义结论先给教材中的一句话定义；核心要点逐项解释组成、作用或边界；教材依据只概括本次引用支持的原文位置。每个事实句末尾都必须有对应引用。",  # 规定定义题的教学顺序。
    ),  # 定义解释模板结束。
    "comparison": AnswerTemplate(  # 定义区别对比模板。
        key="comparison",  # 保存对比模板键。
        name="区别对比",  # 保存模板名称。
        required_sections=("概念A", "概念B", "差异表", "结论提醒"),  # 固定两个概念、对比表和考试提醒四个区块。
        instructions="输出顺序固定为：概念A、概念B、差异表、结论提醒。先分别说明两个概念的教材含义，再使用 Markdown 对比表，表头固定为“| 对比维度 | 概念A | 概念B |”；涉及风险分析时，差异表必须优先比较分析顺序、可能性与影响、分析目的、分析重点、输入、输出和操作难度，但只保留当前原文明确支持的维度；表格中的每一条事实行末尾都必须带引用。最后只总结教材明确支持的判断，不添加常识推断。",  # 规定对比题必须能并排复习，并覆盖风险分析的高频辨析点。
    ),  # 区别对比模板结束。
    "process": AnswerTemplate(  # 定义流程和 ITTO 模板。
        key="process",  # 保存流程模板键。
        name="流程与输入输出",  # 保存模板名称。
        required_sections=("过程顺序", "输入", "输出", "注意事项"),  # 固定过程、输入、输出和注意事项四个区块。
        instructions="输出顺序固定为：过程顺序、输入、输出、注意事项。过程顺序按教材原文的先后逐项列出；如果本题是输入输出工具技术题，输入和输出必须分别列出，工具与技术放在过程顺序或注意事项中；没有原文支持的区块写“当前教材片段未明确列出”，不要自行补充。每个事实句末尾都必须带引用。",  # 规定流程题和 ITTO 题的共同结构。
    ),  # 流程模板结束。
    "formula": AnswerTemplate(  # 定义公式计算模板。
        key="formula",  # 保存公式模板键。
        name="公式计算",  # 保存模板名称。
        required_sections=("公式或指标", "变量含义", "计算步骤", "结果判读"),  # 固定公式、变量、步骤和解释四个区块。
        instructions="输出顺序固定为：公式或指标、变量含义、计算步骤、结果判读。先写教材原文支持的核心公式或指标，再解释变量；没有用户给出的数值时写“本题未提供数值，暂不进行代入计算”，不要自行展开教材例题；结果判读每项只保留一条教材明确支持的含义。对于关键路径或关键路线法（CPM）问题，必须在当前原文支持范围内覆盖网络图、ES/EF/LS/LF、正向计算、反向计算、总时差或关键活动、关键路径及其工期含义；对于“有哪些公式”问题，优先覆盖 BCWS、ACWP、BCWP、SV、CV、CPI、SPI 和原文明确出现的 EAC，控制在必要的核心项目内，不复述无关段落。公式、变量和判断都必须有引用。",  # 规定公式题先覆盖高频核心考点，再避免例题和扩展内容挤占结构预算。
    ),  # 公式模板结束。
    "review": AnswerTemplate(  # 定义章节复习模板。
        key="review",  # 保存章节复习模板键。
        name="章节复习",  # 保存模板名称。
        required_sections=("本章重点", "考点清单", "易错点", "原文依据"),  # 固定重点、考点、易错点和原文依据四个区块。
        instructions="输出顺序固定为：本章重点、考点清单、易错点、原文依据。本章重点只概括当前证据覆盖的知识范围；考点清单逐项列出教材明确内容；易错点只指出原文中容易混淆的边界或相邻概念，没有依据时写“当前片段未明确给出易错提示”；原文依据说明对应引用。每个事实句末尾都必须有引用。",  # 规定复习题适合考前扫描。
    ),  # 章节复习模板结束。
}  # 模板字典结束。


QUESTION_TYPE_TO_TEMPLATE = {  # 定义现有问题类型到五类模板的映射。
    "定义解释": "definition",  # 定义解释直接使用定义模板。
    "区别对比": "comparison",  # 区别对比直接使用对比模板。
    "流程步骤": "process",  # 流程步骤使用流程模板。
    "输入输出工具技术": "process",  # ITTO 题沿用流程模板的输入输出区块。
    "公式计算": "formula",  # 公式计算使用公式模板。
    "考点记忆": "review",  # 考点列表使用章节复习模板的扫描结构。
    "章节复习": "review",  # 明确章节复习使用章节复习模板。
}  # 类型映射结束。


def get_answer_template(question_type: str) -> AnswerTemplate:  # 根据问题类型返回对应模板。
    template_key = QUESTION_TYPE_TO_TEMPLATE.get(question_type, "definition")  # 泛化查询使用最保守的定义解释结构。
    return TEMPLATES[template_key]  # 返回完整模板对象。


def build_template_instructions(question_type: str) -> str:  # 构造主回答提示词中的模板说明。
    template = get_answer_template(question_type)  # 读取当前问题类型模板。
    sections = "、".join(template.required_sections)  # 把必需区块拼成可读顺序。
    return f"当前回答模板：{template.name}。必须按以下顺序组织区块：{sections}。\n{template.instructions}\n区块标签使用普通中文文本加冒号，不要把标签本身当作事实；除对比表表头和纯结构标签外，所有教材事实都必须紧跟引用编号。"  # 返回短而明确的模板约束。


def build_template_repair_instructions(question_type: str) -> str:  # 构造引用修复提示词中的模板说明。
    template = get_answer_template(question_type)  # 读取当前问题类型模板。
    sections = "、".join(template.required_sections)  # 把必需区块拼成可读顺序。
    return f"引用修复时仍必须使用“{template.name}”模板，区块顺序固定为：{sections}。{template.instructions}"  # 返回修复阶段的同一套结构约束。


def validate_template_structure(answer: str, question_type: str) -> dict:  # 定义答案结构校验函数。
    template = get_answer_template(question_type)  # 读取当前问题类型模板。
    found_sections = [section for section in template.required_sections if re.search(rf"(?m)^\s*(?:#{{1,6}}\s*)?(?:\d+[.、)]\s*)?(?:\*{{1,2}})?{re.escape(section)}(?:\*{{1,2}})?\s*[:：]", answer)]  # 兼容模型偶尔添加的 Markdown 标题或粗体包裹，但仍要求标签位于行首并带冒号。
    missing_sections = [section for section in template.required_sections if section not in found_sections]  # 计算缺失的必需区块。
    table_required = template.key == "comparison"  # 只有区别对比模板要求 Markdown 对比表。
    table_found = bool(re.search(r"(?m)^\s*\|\s*对比维度\s*\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|", answer)) if table_required else True  # 允许表头使用真实概念名称，只要保留对比维度和两列事实。
    passed = not missing_sections and table_found  # 必需区块和对比表都存在才通过。
    return {"passed": passed, "template": template.name, "required_sections": list(template.required_sections), "found_sections": found_sections, "missing_sections": missing_sections, "table_found": table_found, "reason": "通过" if passed else f"缺少区块：{'、'.join(missing_sections)}" + ("；缺少对比表" if table_required and not table_found else "")}  # 返回结构化校验结果。


def normalize_repaired_template(answer: str, question_type: str) -> str:  # 定义只补结构标签、不改写事实内容的修复函数。
    template_validation = validate_template_structure(answer, question_type)  # 先确认修复草稿是否只是缺少模板标签。
    if template_validation["passed"]:  # 如果结构已经完整。
        return answer  # 已经完整或不是本函数负责的模板就原样返回。
    if question_type != "区别对比" and question_type in {"定义解释", "流程步骤", "输入输出工具技术", "公式计算", "考点记忆", "章节复习", "泛化查询"}:  # 对其他模板也做只补结构的确定性兜底。
        sections = template_validation["required_sections"]  # 读取当前模板要求的区块顺序。
        normalized = answer.strip()  # 保留模型原始事实内容，只清理首尾空白。
        if not template_validation["found_sections"] and normalized:  # 模型完全没有输出标签时。
            normalized = f"{sections[0]}：\n{normalized}"  # 把全部事实放入第一个区块，不改写事实或引用。
        placeholders = {  # 为缺失区块准备不带事实推断的诚实占位提示。
            "计算步骤": "本题未提供数值，暂不进行代入计算",  # 没有数值时公式题的标准说明。
        }  # 占位提示映射结束。
        found_sections = set(template_validation["found_sections"])  # 记录已经存在的标签。
        for section in sections:  # 按模板顺序补齐缺失结构。
            if section not in found_sections:  # 只处理当前缺失的区块。
                placeholder = placeholders.get(section, "当前教材片段未明确列出")  # 没有证据时明确说明当前片段没有单独列出。
                normalized = f"{normalized}\n\n{section}：\n{placeholder}"  # 追加结构和中性占位提示，不编造教材内容。
        return normalized.strip()  # 返回只增加标签和诚实占位语句的候选答案。
    if question_type != "区别对比":  # 其他未知模板不做猜测式结构改写。
        return answer  # 保留原稿交给严格安全门处理，避免误改写。
    if not template_validation["table_found"]:  # 没有对比表时无法安全推断区块边界。
        return answer  # 保留原稿交给严格安全门处理，避免猜测结构。
    lines = answer.splitlines()  # 按模型原始换行读取事实行。
    table_start = next((index for index, line in enumerate(lines) if line.strip().startswith("|")), None)  # 找到对比表第一行。
    if table_start is None:  # 理论上由上面的表格校验保证不会发生。
        return answer  # 没有表格锚点时不做任何改动。
    table_end = table_start  # 初始化表格结束位置。
    while table_end < len(lines) and (not lines[table_end].strip() or lines[table_end].strip().startswith("|")):  # 连续读取表格行和表格内部空行。
        table_end += 1  # 移动到表格后的第一条非表格内容。
    before_table = [line for line in lines[:table_start] if line.strip()]  # 收集表格前的概念事实行。
    after_table = [line for line in lines[table_end:] if line.strip()]  # 收集表格后的结论事实行。
    before_text = "\n".join(before_table)  # 合并可能被模型压成一行的概念说明。
    before_claims = [claim.strip() for claim in re.split(r"(?<=。)", before_text) if claim.strip()]  # 按句号恢复概念 A、概念 B 的最小边界。
    if len(before_claims) < 2 or not after_table:  # 至少要有两个概念说明和一条结论才能安全补标签。
        return answer  # 信息不足时保留原稿，继续走保守降级。
    normalized_lines: list[str] = ["概念A：", before_claims[0], "", "概念B：", "\n".join(before_claims[1:]), "", "差异表："]  # 按固定顺序补回两个概念和差异表标签。
    normalized_lines.extend(lines[table_start:table_end])  # 原样保留表格及其引用。
    normalized_lines.extend(["", "结论提醒：", "\n".join(after_table)])  # 原样保留结论并补回最后一个标签。
    return "\n".join(normalized_lines).strip()  # 返回只改变结构标签的候选答案。
