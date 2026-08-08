"""请求超时、异常降级和日志治理的离线回归测试。"""  # 说明本脚本只测试治理逻辑，不调用真实 API。

import importlib  # 导入 importlib，用来加载文件名以数字开头的主流程模块。
import os  # 导入 os，用来临时设置模拟客户端需要的环境变量。
from pathlib import Path  # 导入 Path，用来读取日志文件。
from typing import ClassVar  # 导入 ClassVar，标注测试替身中的类级共享状态。

from langchain_core.documents import Document  # 导入 Document，用来构造最小证据片段。

import deepseek_llm  # 导入 DeepSeek 适配器，测试客户端参数和异常路径。
import doubao_embeddings  # 导入豆包适配器，测试 embedding 客户端参数。
from app_logging import get_logger  # 导入统一日志器，确保测试开始前日志文件已经创建。
from config import DEEPSEEK_MAX_RETRIES  # 读取 DeepSeek 重试配置，核对客户端收到的值。
from config import DEEPSEEK_TIMEOUT_SECONDS  # 读取 DeepSeek 超时配置，核对客户端收到的值。
from config import DEEPSEEK_THINKING  # 读取 DeepSeek 思考模式配置，核对请求显式传递的值。
from config import DOUBAO_MAX_RETRIES  # 读取豆包重试配置，核对客户端收到的值。
from config import DOUBAO_TIMEOUT_SECONDS  # 读取豆包超时配置，核对客户端收到的值。
from config import LOG_FILE  # 读取统一日志文件路径，验证异常确实落盘。


class FakeCompletionResponse:  # 定义最小的成功响应对象，模拟 OpenAI SDK 的返回结构。
    class Choice:  # 定义 choices 列表中的单条选择对象。
        class Message:  # 定义 message 对象。
            content = "模拟回答。[1]"  # 提供带引用标记的最小回答内容。

        message = Message()  # 创建消息对象。

    choices: ClassVar[list[Choice]] = [Choice()]  # 创建 choices 列表。


class FakeCompletions:  # 定义 chat.completions 入口。
    last_kwargs: ClassVar[dict] = {}  # 保存最近一次调用参数，便于断言请求治理选项确实传入 SDK。

    @staticmethod  # 声明静态方法，不需要实例状态。
    def create(**kwargs):  # 接收真实调用会传入的参数。
        FakeCompletions.last_kwargs = kwargs  # 记录本次调用参数。
        return FakeCompletionResponse()  # 返回模拟成功响应。


class FakeChat:  # 定义 chat 入口。
    def __init__(self) -> None:  # 定义模拟 chat 初始化函数。
        self.completions = FakeCompletions()  # 挂载模拟补全接口。


class FakeDeepSeekClient:  # 定义不会访问网络的 DeepSeek 模拟客户端。
    def __init__(self) -> None:  # 定义模拟客户端初始化函数。
        self.chat = FakeChat()  # 挂载模拟聊天接口。


def test_deepseek_client_limits() -> None:  # 验证 DeepSeek 客户端收到超时和重试配置。
    captured: dict = {}  # 准备保存模拟构造器收到的参数。
    original_openai = deepseek_llm.OpenAI  # 保存真实 OpenAI 构造器，测试后恢复。

    def fake_openai(**kwargs):  # 定义模拟 OpenAI 构造器，避免建立网络客户端。
        captured.update(kwargs)  # 保存构造参数供断言使用。
        return FakeDeepSeekClient()  # 返回不访问网络的客户端。

    deepseek_llm.OpenAI = fake_openai  # 临时替换 DeepSeek 模块中的构造器。
    deepseek_llm.make_deepseek_client.cache_clear()  # 清理客户端缓存，确保本次一定重新构造。
    try:  # 保护测试替换过程。
        answer = deepseek_llm.call_deepseek("离线超时参数测试")  # 调用真实适配器逻辑，但网络部分由模拟客户端完成。
        assert answer == "模拟回答。[1]", "模拟 DeepSeek 返回内容不符合预期"  # 确认成功路径没有被治理代码破坏。
        assert captured["timeout"] == DEEPSEEK_TIMEOUT_SECONDS, "DeepSeek 超时参数没有传入客户端"  # 确认请求具备显式超时。
        assert captured["max_retries"] == DEEPSEEK_MAX_RETRIES, "DeepSeek 重试参数没有传入客户端"  # 确认重试次数受配置控制。
        assert FakeCompletions.last_kwargs["extra_body"] == {"thinking": {"type": DEEPSEEK_THINKING}}, "DeepSeek 思考模式没有显式传入"  # 确认长回答不会被隐藏思考 token 挤占。
    finally:  # 无论断言是否成功都恢复模块状态。
        deepseek_llm.OpenAI = original_openai  # 恢复真实构造器。
        deepseek_llm.make_deepseek_client.cache_clear()  # 清空模拟客户端缓存，避免影响后续测试。


def test_deepseek_failure_is_logged() -> None:  # 验证 DeepSeek 异常会转成安全错误并写入日志。
    original_factory = deepseek_llm.make_deepseek_client  # 保存真实客户端工厂，测试后恢复。

    class FailingCompletions:  # 定义会模拟超时的补全接口。
        @staticmethod  # 声明静态方法，不需要实例状态。
        def create(**kwargs):  # 接收真实调用参数。
            raise TimeoutError("simulated timeout")  # 模拟网络请求超时。

    class FailingChat:  # 定义失败客户端的 chat 入口。
        completions = FailingCompletions()  # 挂载失败补全接口。

    class FailingClient:  # 定义失败客户端。
        def __init__(self) -> None:  # 定义失败客户端初始化函数。
            self.chat = FailingChat()  # 挂载失败聊天接口。

    deepseek_llm.make_deepseek_client = lambda: FailingClient()  # 临时让调用路径返回失败客户端。
    try:  # 保护异常断言。
        try:  # 进入预期失败调用。
            deepseek_llm.call_deepseek("离线异常日志测试")  # 调用真实适配器异常路径。
        except RuntimeError as exc:  # 捕获适配器暴露给上层的安全异常。
            assert "请求失败或超时" in str(exc), "DeepSeek 失败没有转换成安全提示"  # 确认用户不会看到底层长堆栈。
        else:  # 如果没有抛出预期异常。
            raise AssertionError("DeepSeek 模拟超时没有被捕获")  # 明确报告治理失效。
    finally:  # 无论测试结果如何恢复工厂。
        deepseek_llm.make_deepseek_client = original_factory  # 恢复真实客户端工厂。
    log_text = Path(LOG_FILE).read_text(encoding="utf-8")  # 读取本次测试产生的日志。
    assert "DeepSeek request_failed" in log_text, "DeepSeek 异常没有写入统一日志"  # 确认日志包含稳定事件名。
    assert "离线异常日志测试" not in log_text, "日志不应写入完整用户提示词"  # 确认日志治理没有泄露提示词正文。


def test_doubao_client_limits() -> None:  # 验证普通豆包 embedding 客户端收到超时和重试配置。
    captured: dict = {}  # 准备保存模拟构造器收到的参数。
    original_openai = doubao_embeddings.OpenAI  # 保存真实 OpenAI 构造器。
    original_values = {  # 保存测试前可能存在的环境变量。
        name: os.environ.get(name)  # 读取当前配置值。
        for name in ("DOUBAO_API_KEY", "DOUBAO_BASE_URL", "DOUBAO_EMBEDDING_MODEL")  # 遍历本测试需要临时覆盖的配置。
    }  # 环境变量快照结束。

    def fake_openai(**kwargs):  # 定义模拟 OpenAI 构造器。
        captured.update(kwargs)  # 保存构造参数供断言。
        return object()  # 初始化阶段只需要一个占位客户端对象。

    doubao_embeddings.OpenAI = fake_openai  # 临时替换豆包模块中的构造器。
    os.environ["DOUBAO_API_KEY"] = "offline-test-key"  # 设置不会被打印的模拟 Key。
    os.environ["DOUBAO_BASE_URL"] = "https://offline.example/v1"  # 使用普通 OpenAI 兼容分支，避免真实网络。
    os.environ["DOUBAO_EMBEDDING_MODEL"] = "offline-test-model"  # 设置模拟模型名。
    try:  # 保护测试替换过程。
        adapter = doubao_embeddings.DoubaoEmbeddings()  # 初始化真实豆包适配器逻辑。
        assert adapter.client is not None, "普通豆包模式没有创建客户端"  # 确认普通模式初始化成功。
        assert captured["timeout"] == DOUBAO_TIMEOUT_SECONDS, "豆包超时参数没有传入客户端"  # 确认请求具备显式超时。
        assert captured["max_retries"] == DOUBAO_MAX_RETRIES, "豆包重试参数没有传入客户端"  # 确认重试次数受配置控制。
    finally:  # 无论断言是否成功都恢复模块和环境。
        doubao_embeddings.OpenAI = original_openai  # 恢复真实构造器。
        for name, value in original_values.items():  # 遍历测试前的环境快照。
            if value is None:  # 如果测试前变量不存在。
                os.environ.pop(name, None)  # 删除测试临时变量。
            else:  # 如果测试前变量存在。
                os.environ[name] = value  # 恢复原始配置值。


def test_model_failure_fallback() -> None:  # 验证回答模型失败时系统只返回证据，不生成猜测答案。
    query_graph = importlib.import_module("03_query_graph")  # 加载完整 LangGraph 主流程模块。
    original_call = query_graph.call_deepseek  # 保存真实模型调用函数。

    def fail_call(prompt: str) -> str:  # 定义永远失败的模型调用替身。
        raise RuntimeError("simulated model outage")  # 模拟上游服务不可用。

    query_graph.__dict__["call_deepseek"] = fail_call  # 临时替换主流程中的模型调用，直接写模块字典以兼容动态导入模块的类型检查。
    state = query_graph.make_initial_state("什么是范围基准？")  # 创建最小完整状态。
    state["resolved_question"] = "什么是范围基准？"  # 填入已经补全的查询问题。
    state["question_type"] = "定义解释"  # 填入问题类型，满足报告和提示逻辑。
    state["evidence_enough"] = True  # 模拟检索已经找到足够证据，确保会进入模型调用分支。
    state["evidence_score"] = 10  # 填入足够高的证据分。
    state["contexts"] = [Document(page_content="范围基准是经过批准的范围说明书、WBS 和 WBS 词典。", metadata={"chapter": "第17章 项目整体管理", "section": "范围基准", "chunk_id": "offline-001"})]  # 构造一条可引用教材证据。
    try:  # 保护主流程函数替换。
        result = query_graph.generate_answer(state)  # 执行真实回答节点。
    finally:  # 无论测试结果如何恢复模型调用函数。
        query_graph.__dict__["call_deepseek"] = original_call  # 恢复真实模型调用函数。
    assert "不可用" in result["answer"], "模型失败时没有返回不可用提示"  # 确认用户知道模型服务没有成功。
    assert "范围基准是经过批准" in result["answer"], "模型失败时没有保留检索到的原文证据"  # 确认降级仍然有学习价值。
    assert result["citation_validation"]["passed"] is False, "模型失败时不能把引用校验标记为成功"  # 确认异常状态可审计。


def main() -> None:  # 定义异常治理回归测试入口。
    get_logger("exception_regression")  # 确保统一日志器已经初始化。
    test_deepseek_client_limits()  # 执行 DeepSeek 超时和重试参数测试。
    test_deepseek_failure_is_logged()  # 执行 DeepSeek 异常日志测试。
    test_doubao_client_limits()  # 执行豆包超时和重试参数测试。
    test_model_failure_fallback()  # 执行模型失败保守降级测试。
    print("异常治理回归通过：DeepSeek/豆包超时参数、异常日志、模型失败降级。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 直接运行时执行全部异常治理测试。
