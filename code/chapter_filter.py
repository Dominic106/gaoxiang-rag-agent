import json  # 导入 json，用来读取可审计的 chunk 元数据。
from functools import lru_cache  # 导入 lru_cache，避免每次查询都重复读取章节清单。

from config import CHUNKS_JSONL  # 从配置读取 chunk 明细文件路径。


@lru_cache(maxsize=1)  # 缓存章节列表，因为教材章节在运行期间不会变化。
def list_chapters() -> tuple[str, ...]:  # 定义读取全部章节名称的函数。
    chapters: set[str] = set()  # 使用集合去重保存章节名。
    with CHUNKS_JSONL.open("r", encoding="utf-8") as file:  # 打开可审计 chunk 文件。
        for line in file:  # 逐行读取 chunk 记录。
            record = json.loads(line)  # 将当前行解析为字典。
            chapter = record.get("metadata", {}).get("chapter", "").strip()  # 取出章节元数据。
            if chapter:  # 只有非空章节才加入结果。
                chapters.add(chapter)  # 保存章节名称。
    return tuple(sorted(chapters))  # 返回稳定排序后的章节元组。


def resolve_chapter_filter(user_value: str | None) -> str | None:  # 定义把用户输入解析成标准章节名的函数。
    if not user_value or not user_value.strip():  # 如果用户没有指定章节。
        return None  # None 表示不启用章节过滤。
    value = user_value.strip().lower()  # 清理输入并转成小写，便于英文平台名匹配。
    chapters = list_chapters()  # 读取全部标准章节名。
    exact = [chapter for chapter in chapters if chapter.lower() == value]  # 先尝试完整名称精确匹配。
    if exact:  # 如果找到完整名称。
        return exact[0]  # 返回唯一标准章节名。
    matched = [chapter for chapter in chapters if value in chapter.lower() or chapter.lower() in value]  # 再支持输入“第17章”或“整体管理”等短名称。
    if len(matched) == 1:  # 只有唯一匹配时才自动采用，避免章节歧义。
        return matched[0]  # 返回唯一匹配章节。
    return None  # 没有匹配或匹配多个时返回 None，由主流程提示用户修正。
