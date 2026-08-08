"""会话、长期记忆和本地文件锁回归测试。"""  # 说明本文件验证服务接口进入并发场景后的本地存储安全性。

import json  # 导入 json，用来检查最终落盘文件仍然是完整结构。
import tempfile  # 导入 tempfile，为测试创建隔离目录。
import threading  # 导入 threading，模拟同一服务进程的并发请求。
from pathlib import Path  # 导入 Path，用来构造临时目录路径。
from unittest.mock import patch  # 导入 patch，临时替换项目的会话和记忆根目录。

import study_memory  # 导入待测的会话和长期记忆模块。


def test_corrupt_session_falls_back_safely() -> None:  # 验证损坏的 session.json 不会阻断新查询。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 创建一次性测试目录。
        root = Path(temporary_directory)  # 把字符串目录转换成 Path。
        with patch.object(study_memory, "SESSION_ROOT", root / "sessions"), patch.object(study_memory, "MEMORY_ROOT", root / "memory"):  # 把测试写入隔离目录。
            session_path = study_memory.session_json_path("corrupt")  # 创建测试会话文件路径。
            session_path.write_text("{not valid json", encoding="utf-8")  # 写入损坏的 JSON。
            session = study_memory.load_session("corrupt")  # 读取损坏会话。
            assert session["turns"] == [], "损坏会话没有安全回退为空会话"  # 确认读取不会抛出异常。
            turn_path = study_memory.append_turn("corrupt", {"question": "恢复后问题", "answer": "测试答案", "sub_questions": [], "chapters": [], "question_types": []})  # 验证后续仍可以追加新轮次。
            assert turn_path.exists(), "损坏会话恢复后没有生成新的 turn 文件"  # 确认存储链路可以继续工作。
            assert len(study_memory.load_session("corrupt")["turns"]) == 1, "恢复后的会话轮次数量不正确"  # 确认新会话结构完整。


def test_concurrent_append_keeps_turns_and_profile() -> None:  # 验证并发追加不会重复轮次或丢失长期画像计数。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 创建一次性测试目录。
        root = Path(temporary_directory)  # 把字符串目录转换成 Path。
        errors: list[BaseException] = []  # 收集线程中的异常，避免测试静默失败。

        def append(index: int) -> None:  # 定义一个并发追加任务。
            try:  # 捕获当前线程的任何异常。
                study_memory.append_turn("concurrent", {"question": f"问题{index}", "answer": "测试答案", "sub_questions": [], "chapters": ["第1章"], "question_types": ["定义解释"]})  # 追加一轮独立学习记录。
            except Exception as exc:  # 把业务异常交给主线程统一断言。  # noqa: BLE001
                errors.append(exc)  # 保存当前线程异常。

        with patch.object(study_memory, "SESSION_ROOT", root / "sessions"), patch.object(study_memory, "MEMORY_ROOT", root / "memory"):  # 把并发测试写入隔离目录。
            threads = [threading.Thread(target=append, args=(index,)) for index in range(8)]  # 创建八个并发写入线程。
            for thread in threads:  # 遍历所有线程。
                thread.start()  # 启动当前写入线程。
            for thread in threads:  # 再次遍历所有线程。
                thread.join()  # 等待当前写入线程结束。
            assert not errors, f"并发追加出现异常：{errors}"  # 确认文件锁没有导致写入异常。
            session = study_memory.load_session("concurrent")  # 读取最终会话。
            turn_numbers = sorted(turn["turn_number"] for turn in session["turns"])  # 收集所有分配到的轮次号。
            assert turn_numbers == list(range(1, 9)), "并发追加产生了重复或缺失轮次"  # 确认轮次严格连续。
            history_path = root / "memory" / "question_history.jsonl"  # 定位长期问题历史文件。
            history_records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]  # 解析所有历史事件。
            assert len(history_records) == 8, "并发追加丢失长期历史事件"  # 确认 JSONL 事件数量完整。
            profile = json.loads((root / "memory" / "user_profile.json").read_text(encoding="utf-8"))  # 读取长期学习画像。
            assert profile["question_count"] == 8, "并发追加丢失学习画像计数"  # 确认读改写没有覆盖彼此结果。


def main() -> None:  # 定义存储回归测试入口。
    test_corrupt_session_falls_back_safely()  # 执行损坏会话恢复测试。
    test_concurrent_append_keeps_turns_and_profile()  # 执行并发写入一致性测试。
    print("本地存储回归通过：损坏会话可恢复，并发会话轮次和长期画像不丢失。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 直接运行时执行全部存储回归测试。
