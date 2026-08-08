import json  # 导入 json，用来把会话和记忆写成可读可改的 JSON 文件。
from datetime import datetime  # 导入 datetime，用来记录每次学习行为发生的时间。
from pathlib import Path  # 导入 Path，用来安全拼接本地目录。

from app_logging import get_logger  # 导入统一日志器，记录损坏会话和记忆文件但不阻断查询。
from config import MEMORY_ROOT  # 从配置读取长期记忆目录。
from config import SESSION_ROOT  # 从配置读取会话目录。
from file_locks import locked_file  # 导入跨进程文件锁，保护会话和长期记忆的读改写操作。


FOLLOW_UP_MARKERS = ("刚才", "上一轮", "上一个", "这个", "那个", "它", "这部分", "上述", "继续", "再解释", "再说说", "换一种")  # 定义常见的上下文依赖表达。
logger = get_logger(__name__)  # 创建当前模块日志器。


def safe_session_name(name: str | None) -> str:  # 定义会话名清理函数，避免文件夹名包含奇怪字符。
    if name and name.strip():  # 如果用户传入了会话名。
        cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in name.strip())  # 只保留安全字符。
        return cleaned[:80] or today_session_name()  # 限制长度，清理后为空就回退到今天的默认会话名。
    return today_session_name()  # 如果用户没传会话名，就使用今天的默认会话名。


def today_session_name() -> str:  # 定义默认会话名函数。
    return datetime.now().strftime("%Y%m%d_高项学习")  # 用日期命名，适合长期学习按天归档。


def session_dir(session_name: str | None) -> Path:  # 定义获取会话目录的函数。
    directory = SESSION_ROOT / safe_session_name(session_name)  # 拼出会话目录路径。
    directory.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    return directory  # 返回会话目录。


def session_json_path(session_name: str | None) -> Path:  # 定义获取 session.json 路径的函数。
    return session_dir(session_name) / "session.json"  # 每个会话统一保存到 session.json。


def _empty_session(session_name: str | None) -> dict:  # 构造一个新的空会话对象。
    return {"session_name": safe_session_name(session_name), "turns": []}  # 保持会话结构完整。


def _load_session_unlocked(session_name: str | None) -> dict:  # 在调用方已经持锁时读取会话文件。
    path = session_json_path(session_name)  # 获取会话文件路径。
    if not path.exists():  # 如果会话文件还不存在。
        return _empty_session(session_name)  # 返回一个新会话结构。
    try:  # 保护 JSON 文件损坏和磁盘读取异常。
        session = json.loads(path.read_text(encoding="utf-8"))  # 读取并解析已有会话 JSON。
        if not isinstance(session, dict) or not isinstance(session.get("turns", []), list):  # 检查最小会话结构。
            raise TypeError("会话文件结构不是对象或 turns 不是列表。")  # 把结构类型错误统一转成可诊断异常。
        return session  # 返回合法会话对象。
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:  # 损坏会话不能阻断新的学习查询。
        logger.exception("Session load_failed path=%s error_type=%s", path, type(exc).__name__)  # 记录文件路径和异常类型，便于修复历史文件。
        return _empty_session(session_name)  # 使用空会话继续本次查询。


def load_session(session_name: str | None) -> dict:  # 定义加载会话的函数。
    directory = session_dir(session_name)  # 获取会话目录并确保目录存在。
    with locked_file(directory / ".session.lock"):  # 读取时也加锁，避免读到其他进程的半成品写入。
        return _load_session_unlocked(session_name)  # 在锁内读取合法会话或安全返回空会话。


def _save_session_unlocked(session_name: str | None, session: dict) -> None:  # 在调用方已经持锁时保存会话文件。
    path = session_json_path(session_name)  # 获取会话文件路径。
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入格式化 JSON，方便人读。


def save_session(session_name: str | None, session: dict) -> None:  # 定义保存会话的函数。
    directory = session_dir(session_name)  # 获取会话目录并确保目录存在。
    with locked_file(directory / ".session.lock"):  # 保护整个会话文件写入。
        _save_session_unlocked(session_name, session)  # 在锁内写入会话 JSON。


def append_turn(session_name: str | None, turn: dict) -> Path:  # 定义追加一轮问答的函数。
    directory = session_dir(session_name)  # 获取会话目录并确保目录存在。
    with locked_file(directory / ".session.lock"):  # 锁住会话的读、分配轮次和写入全过程。
        session = _load_session_unlocked(session_name)  # 在锁内加载当前会话。
        turn_number = len(session["turns"]) + 1  # 根据已有轮数计算这次是第几轮。
        turn["turn_number"] = turn_number  # 把轮次号写入 turn。
        turn["created_at"] = datetime.now().isoformat(timespec="seconds")  # 记录创建时间。
        session["turns"].append(turn)  # 把这轮问答追加到会话列表。
        _save_session_unlocked(session_name, session)  # 在锁内保存更新后的会话 JSON。
        markdown_path = directory / f"turn_{turn_number:03d}.md"  # 为这一轮创建单独 Markdown 文件。
        markdown_path.write_text(turn_to_markdown(turn), encoding="utf-8")  # 写入 Markdown，方便用户直接阅读。
    update_long_term_memory(turn)  # 会话锁释放后再锁长期记忆，避免不同会话互相阻塞过久。
    return markdown_path  # 返回本轮 Markdown 路径。


def turn_to_markdown(turn: dict) -> str:  # 定义把一轮问答转成 Markdown 的函数。
    return f"""# 第 {turn.get('turn_number', '')} 轮问答

## 问题

{turn.get('question', '')}

## 答案

{turn.get('answer', '')}

## 子问题

{json.dumps(turn.get('sub_questions', []), ensure_ascii=False, indent=2)}

## 引用章节

{json.dumps(turn.get('chapters', []), ensure_ascii=False, indent=2)}
"""  # 返回 Markdown 字符串。


def rollback_last_turn(session_name: str | None) -> dict | None:  # 定义回退上一轮问答的函数。
    directory = session_dir(session_name)  # 获取会话目录并确保目录存在。
    with locked_file(directory / ".session.lock"):  # 锁住会话的读取、删除和保存全过程。
        session = _load_session_unlocked(session_name)  # 在锁内加载当前会话。
        if not session["turns"]:  # 如果没有任何历史轮次。
            return None  # 返回 None 表示无可回退。
        removed = session["turns"].pop()  # 弹出最后一轮。
        _save_session_unlocked(session_name, session)  # 保存删除后的会话。
        markdown_path = directory / f"turn_{removed.get('turn_number', 0):03d}.md"  # 找到对应 Markdown 文件。
        if markdown_path.exists():  # 如果文件存在。
            markdown_path.unlink()  # 删除这轮 Markdown，保持 JSON 和文件一致。
        return removed  # 返回被回退的那轮内容，方便用户确认。


def latest_turn(session_name: str | None) -> dict | None:  # 定义读取上一轮问答的函数。
    session = load_session(session_name)  # 加载会话。
    if not session["turns"]:  # 如果没有历史。
        return None  # 返回 None。
    return session["turns"][-1]  # 返回最后一轮。


def is_follow_up_question(question: str) -> bool:  # 定义判断问题是否依赖前文的函数。
    cleaned = question.strip()  # 清理问题文本。
    return any(marker in cleaned for marker in FOLLOW_UP_MARKERS)  # 只要命中上下文指代词，就认为需要读取会话记忆。


def resolve_follow_up_question(session_name: str | None, question: str) -> str:  # 定义把依赖前文的问题补全成可检索问题的函数。
    if not is_follow_up_question(question):  # 如果问题本身已经包含完整主题。
        return question  # 不额外加入历史，节省检索和模型 token。
    session = load_session(session_name)  # 读取当前会话历史。
    if not session.get("turns"):  # 如果当前会话没有历史轮次。
        return question  # 没有上下文时保持原问题，后续由证据门禁决定是否回答。
    previous = session["turns"][-1]  # 取最近一轮作为追问锚点。
    previous_questions = previous.get("sub_questions") or [previous.get("question", "")]  # 优先使用上一轮拆分后的子问题。
    anchor = previous_questions[-1].strip()  # 取上一轮最后一个子问题作为主题锚点。
    return f"{anchor}；用户追问：{question}"  # 用规则方式补全主题，不额外调用大模型。


def build_memory_context(session_name: str | None, question: str, max_turns: int = 2) -> str:  # 定义为回答模型准备的最小会话记忆函数。
    if not is_follow_up_question(question):  # 如果不是上下文依赖问题。
        return ""  # 不发送历史，避免无意义 token 消耗。
    session = load_session(session_name)  # 读取当前会话。
    turns = session.get("turns", [])[-max_turns:]  # 只保留最近几轮，防止上下文无限增长。
    if not turns:  # 如果没有可用历史。
        return ""  # 返回空记忆。
    blocks: list[str] = []  # 准备保存短记忆块。
    for turn in turns:  # 遍历最近的会话轮次。
        question_text = turn.get("question", "").strip()  # 取出历史问题。
        answer_text = turn.get("answer", "").replace("\n", " ").strip()[:600]  # 只保留历史回答前 600 个字符，避免把引用全文重复发送。
        blocks.append(f"历史问题：{question_text}\n历史回答摘要：{answer_text}")  # 组织最小可用记忆。
    return "\n\n".join(blocks)  # 返回给查询图使用的会话记忆文本。


def update_long_term_memory(turn: dict) -> None:  # 定义更新长期记忆的函数。
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)  # 确保长期记忆目录存在。
    with locked_file(MEMORY_ROOT / ".memory.lock"):  # 锁住历史追加和画像读改写，避免并发请求丢统计。
        history_path = MEMORY_ROOT / "question_history.jsonl"  # 定义问题历史流水文件。
        memory_event = dict(turn)  # 复制本轮数据，避免为了长期记忆修改会话对象本身。
        memory_event["event_type"] = "turn_added"  # 写入事件类型，为未来回退重算保留事件来源。
        memory_event["memory_source"] = {"session_name": turn.get("session_name", ""), "turn_number": turn.get("turn_number", 0)}  # 记录会话名和轮次号，未来可按来源重算画像。
        with history_path.open("a", encoding="utf-8") as history_file:  # 以追加方式保存一条完整 JSONL 事件。
            history_file.write(json.dumps(memory_event, ensure_ascii=False) + "\n")  # 追加一条带来源的长期记忆事件。
        profile_path = MEMORY_ROOT / "user_profile.json"  # 定义用户学习画像文件。
        profile = load_json(profile_path, {"question_count": 0, "question_types": {}, "chapters": {}})  # 读取已有画像，没有就新建。
        profile["question_count"] += 1  # 总提问次数加一。
        for question_type in turn.get("question_types", []):  # 遍历本轮涉及的问题类型。
            profile["question_types"][question_type] = profile["question_types"].get(question_type, 0) + 1  # 累计问题类型次数。
        for chapter in turn.get("chapters", []):  # 遍历本轮引用章节。
            profile["chapters"][chapter] = profile["chapters"].get(chapter, 0) + 1  # 累计章节出现次数，用来发现薄弱点。
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存学习画像。


def load_json(path: Path, default: dict) -> dict:  # 定义安全读取 JSON 的小工具。
    lock_path = path.with_suffix(path.suffix + ".lock")  # 为画像等普通 JSON 文件生成独立锁文件。
    with locked_file(lock_path):  # 保护读取过程，避免读到并发写入的半成品。
        if not path.exists():  # 如果文件不存在。
            return default  # 返回默认值。
        try:  # 保护 JSON 损坏和磁盘读取异常。
            value = json.loads(path.read_text(encoding="utf-8"))  # 读取 JSON 文件并解析。
            return value if isinstance(value, dict) else default  # 只接受对象结构，其他结构回退默认值。
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:  # 损坏长期记忆不能阻断新查询。
            logger.exception("Memory load_failed path=%s error_type=%s", path, type(exc).__name__)  # 记录异常以便定位历史数据问题。
            return default  # 返回安全默认结构。
