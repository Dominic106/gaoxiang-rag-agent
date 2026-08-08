"""全链路请求治理的离线故障注入测试。"""  # 说明本文件验证截止时间、异常分类、熔断和统计，不调用真实 API。

import time  # 导入 time，用来制造总预算耗尽场景。

import request_governance  # 导入请求治理模块，测试其运行时和熔断器。
from request_governance import CircuitBreaker  # 导入独立熔断器类。
from request_governance import CircuitOpenError  # 导入熔断阻断异常。
from request_governance import RequestDeadlineExceeded  # 导入总截止时间异常。
from request_governance import classify_exception  # 导入异常分类函数。
from request_governance import current_request_summary  # 导入当前请求汇总函数。
from request_governance import ensure_request_budget  # 导入剩余预算检查函数。
from request_governance import request_scope  # 导入完整请求上下文。
from request_governance import tracked_service_call  # 导入外部调用追踪上下文。


class AuthenticationError(Exception):  # 定义一个只用于测试的鉴权异常类型。
    pass  # 测试异常不需要额外逻辑。


def test_exception_classification() -> None:  # 验证可重试和不可重试异常被区分。
    assert classify_exception(TimeoutError())[1] is True, "超时应该被标记为可重试"  # 网络超时通常可以有限重试。
    assert classify_exception(AuthenticationError())[1] is False, "鉴权异常不应该重试"  # 鉴权失败重试只会浪费时间和成本。
    assert classify_exception(ValueError())[1] is False, "参数或响应格式异常不应该重试"  # 结构错误不属于瞬时故障。


def test_deadline_and_metrics() -> None:  # 验证总截止时间能够阻断新工作并留下统计。
    original_deadline = request_governance.REQUEST_DEADLINE_SECONDS  # 保存正式配置，测试后恢复。
    request_governance.REQUEST_DEADLINE_SECONDS = 0.02  # 把测试预算缩短到几十毫秒，避免真正等待 180 秒。
    try:  # 保护临时配置。
        with request_scope("离线截止时间测试") as runtime:  # 创建一条真实请求上下文。
            with tracked_service_call("offline", "success", 12):  # 记录一条成功外部调用。
                pass  # 不访问网络，只验证统计钩子。
            time.sleep(0.03)  # 主动耗尽整条请求预算。
            try:  # 进入预期超时检查。
                ensure_request_budget("offline_after_sleep")  # 尝试在截止时间后继续工作。
            except RequestDeadlineExceeded:  # 捕获预期的治理异常。
                pass  # 超时被阻断即为通过。
            else:  # 如果没有抛出截止时间异常。
                raise AssertionError("总截止时间耗尽后仍允许继续工作")  # 明确报告治理失效。
            summary = runtime.summary()  # 读取当前请求的统计快照。
            assert summary["calls"] == 1, "请求统计没有记录成功外部调用"  # 确认调用数量被记录。
            assert summary["estimated_input_tokens"] == 12, "请求统计没有记录输入 token 估算"  # 确认成本字段被记录。
        assert current_request_summary() == {}, "请求上下文退出后没有正确清理"  # 确认不会污染后续独立请求。
    finally:  # 无论测试结果如何恢复配置。
        request_governance.REQUEST_DEADLINE_SECONDS = original_deadline  # 恢复正式总预算。


def test_circuit_breaker() -> None:  # 验证连续可重试失败会打开熔断并阻止请求。
    original_threshold = request_governance.CIRCUIT_FAILURE_THRESHOLD  # 保存正式熔断阈值。
    original_recovery = request_governance.CIRCUIT_RECOVERY_SECONDS  # 保存正式恢复时间。
    request_governance.CIRCUIT_FAILURE_THRESHOLD = 2  # 测试中两次失败就打开熔断。
    request_governance.CIRCUIT_RECOVERY_SECONDS = 10  # 保证测试期间不会自动恢复。
    breaker = CircuitBreaker("offline")  # 创建独立测试熔断器。
    try:  # 保护临时配置。
        breaker.record_failure(True)  # 注入第一次可重试失败。
        breaker.record_failure(True)  # 注入第二次可重试失败。
        try:  # 进入预期熔断检查。
            breaker.before_call()  # 尝试发起第三次调用。
        except CircuitOpenError:  # 捕获预期熔断异常。
            pass  # 熔断成功即为通过。
        else:  # 如果没有阻断第三次调用。
            raise AssertionError("连续可重试失败后没有打开熔断")  # 明确报告熔断失效。
        breaker.record_success()  # 模拟恢复探测成功。
        breaker.before_call()  # 成功后应该允许下一次调用。
    finally:  # 无论测试结果如何恢复配置。
        request_governance.CIRCUIT_FAILURE_THRESHOLD = original_threshold  # 恢复正式失败阈值。
        request_governance.CIRCUIT_RECOVERY_SECONDS = original_recovery  # 恢复正式熔断恢复时间。


def main() -> None:  # 定义回归测试入口。
    test_exception_classification()  # 执行异常分类测试。
    test_deadline_and_metrics()  # 执行总截止时间和统计测试。
    test_circuit_breaker()  # 执行熔断器测试。
    print("全链路请求治理回归通过：总截止时间、异常分类、熔断和耗时/token 统计。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断脚本是否直接运行。
    main()  # 直接运行时执行全部治理回归。
