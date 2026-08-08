"""用 pytest 收口现有离线回归脚本，避免项目只能依靠人工逐个执行脚本。"""  # 说明这个文件只负责测试入口编排，不复制业务断言。

import subprocess  # 导入 subprocess，用独立进程执行每个教学版回归脚本。
import sys  # 导入 sys，确保 pytest 使用当前项目虚拟环境的解释器。
from pathlib import Path  # 导入 Path，用来定位当前 code 目录和脚本文件。

import pytest  # 导入 pytest，提供参数化测试和失败报告。


CODE_ROOT = Path(__file__).resolve().parent  # 获取 code 目录，所有回归脚本都从这里运行。
OFFLINE_COMMANDS: tuple[tuple[str, ...], ...] = (  # 定义不需要真实模型 API 的核心回归命令集合。
    ("11_regression_tests.py",),  # 问题理解、章节过滤、记忆、引用和原文定位。
    ("12_core_regression.py",),  # 核心总结测试和离线检索回归。
    ("13_performance_benchmark.py",),  # BM25 性能和向量跳过率基准。
    ("14_exception_regression.py",),  # 超时参数、异常日志和模型失败降级。
    ("15_eval_data_quality.py",),  # 正式评估集结构和章节覆盖。
    ("17_index_health.py", "check"),  # 索引健康检查命令。
    ("18_index_health_regression.py",),  # 索引版本和健康治理断言。
    ("19_request_governance_regression.py",),  # 总截止时间、异常分类和熔断。
    ("20_answer_template_regression.py",),  # 五类学习模板和引用安全门。
    ("21_service_interface_regression.py",),  # 稳定服务接口契约。
    ("22_storage_regression.py",),  # 损坏文件恢复和并发持久化。
    ("23_index_rebuild_regression.py",),  # 索引失败保留旧版本、成功切换新版本。
    ("24_answer_eval_parser_regression.py",),  # 答案评估报告解析和关键词别名。
)  # 离线命令集合结束。


@pytest.mark.parametrize("command", OFFLINE_COMMANDS)  # 为每个回归脚本生成独立 pytest 用例。
def test_offline_regression_script(command: tuple[str, ...]) -> None:  # 定义脚本级回归测试。
    completed = subprocess.run(  # 在独立进程中执行，隔离模块缓存和测试替身状态。
        [sys.executable, *command],  # 使用当前虚拟环境解释器运行目标脚本及其显式参数。
        cwd=CODE_ROOT,  # 统一从 code 目录启动，保持原有脚本导入方式。
        capture_output=True,  # 捕获输出，失败时把诊断内容带入 pytest 报告。
        text=True,  # 让 stdout 和 stderr 直接返回字符串。
        check=False,  # 显式保留退出码，下面统一生成带输出的断言失败。
    )  # 子进程执行结束。
    assert completed.returncode == 0, f"{' '.join(command)} 失败：\n{completed.stdout}\n{completed.stderr}"  # 任一既有回归失败都必须阻断 pytest。
