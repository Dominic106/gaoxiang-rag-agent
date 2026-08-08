"""全链路请求截止时间、异常分类、熔断和成本统计。"""  # 说明本模块只负责请求治理，不负责教材检索或回答内容。

import contextvars  # 导入 contextvars，让同一条查询链路中的各模块共享请求上下文。
import hashlib  # 导入 hashlib，用来生成不包含问题正文的请求指纹。
import json  # 导入 json，用来持久化机器可读的请求统计。
import math  # 导入 math，用来向上估算输入 token 数。
import threading  # 导入 threading，让进程级熔断器在未来并发服务中保持一致。
import time  # 导入 time，用单调时钟计算请求剩余时间，避免系统时间回拨影响截止时间。
import uuid  # 导入 uuid，为每一次请求生成不会碰撞的追踪编号。
from collections.abc import Iterator  # 导入 Iterator，标注上下文管理器类型。
from contextlib import contextmanager  # 导入 contextmanager，提供简洁的请求和外部调用上下文。
from dataclasses import dataclass, field  # 导入 dataclass，定义结构化的请求统计对象。
from datetime import datetime  # 导入 datetime，用来记录可读的事件时间。
from typing import Any  # 导入 Any，标注统计字典类型。

from config import CIRCUIT_FAILURE_THRESHOLD  # 从配置读取熔断前连续失败阈值。
from config import CIRCUIT_RECOVERY_SECONDS  # 从配置读取熔断恢复等待时间。
from config import OUTPUT_ROOT  # 从配置读取请求统计输出目录。
from config import REQUEST_DEADLINE_SECONDS  # 从配置读取整条查询的总截止时间。
from app_logging import get_logger  # 导入统一日志器，记录治理事件但不记录密钥和完整提示词。
from file_locks import locked_file  # 导入文件锁，保护多个服务进程追加请求统计时的 JSONL 完整性。


logger = get_logger(__name__)  # 创建当前模块日志器。


class RequestGovernanceError(RuntimeError):  # 定义所有请求治理主动中断的基类。
    """请求被治理层主动阻断，而不是回答内容本身失败。"""  # 让上层可以区分治理失败和普通模型失败。


class RequestDeadlineExceeded(RequestGovernanceError):  # 定义整条查询超过截止时间的异常。
    """当前查询没有足够时间继续发起新的工作。"""  # 该异常不能通过重试解决。


class CircuitOpenError(RequestGovernanceError):  # 定义外部服务熔断期间的异常。
    """外部服务近期连续失败，当前调用被保护性拒绝。"""  # 该异常不能在熔断窗口内继续撞击服务。


def estimate_tokens(text: str | None) -> int:  # 定义保守的 token 估算函数，统计中明确标记为 estimated。
    normalized = text or ""  # 把 None 统一成空字符串，避免统计阶段再次抛异常。
    return max(1, math.ceil(len(normalized) / 4))  # 用每 4 个字符约 1 token 的保守估算，适合跨中英文内容做趋势比较。


def request_id_for(question: str, session_name: str | None = None) -> str:  # 定义一次查询的短追踪编号。
    fingerprint = hashlib.sha256(f"{session_name or ''}\n{question}".encode()).hexdigest()[:8]  # 用问题和会话生成不可逆短指纹，方便同类请求聚合但不暴露正文。
    nonce = uuid.uuid4().hex[:8]  # 为每次实际执行增加随机短编号，避免重复问题覆盖日志关联。
    return f"{fingerprint}{nonce}"  # 组合成 16 位追踪编号。


def classify_exception(exc: BaseException) -> tuple[str, bool]:  # 定义异常分类函数，返回稳定类别和是否可重试。
    if isinstance(exc, RequestDeadlineExceeded):  # 总截止时间到达时。
        return "deadline_exceeded", False  # 截止时间不能通过继续重试解决。
    if isinstance(exc, CircuitOpenError):  # 熔断器已经打开时。
        return "circuit_open", False  # 熔断期间继续请求只会放大故障。
    name = type(exc).__name__.lower()  # 读取异常类名，用于兼容不同 SDK 版本的异常类型。
    status_code = getattr(exc, "status_code", None)  # 尝试读取 OpenAI 兼容异常上的 HTTP 状态码。
    response = getattr(exc, "response", None)  # 尝试读取 requests 或 SDK 异常上的响应对象。
    if status_code is None and response is not None:  # 如果异常自身没有状态码但响应对象有状态码。
        status_code = getattr(response, "status_code", None)  # 使用响应状态码辅助分类。
    if status_code in {408, 409, 425, 429} or (isinstance(status_code, int) and status_code >= 500):  # 超时、限流、冲突和服务端错误通常可以重试。
        return "http_retryable", True  # 标记为可重试 HTTP 异常。
    if status_code in {400, 401, 403, 404, 422} or any(token in name for token in ("authentication", "permission", "unauthorized", "forbidden", "badrequest", "invalid", "notfound")):  # 鉴权、权限、参数和资源错误重试没有意义。
        return "http_non_retryable", False  # 标记为不可重试 HTTP 异常。
    if any(token in name for token in ("timeout", "connection", "connect", "ratelimit", "internalserver", "serviceunavailable", "temporarily")):  # 网络断开、超时、限流和临时服务异常可重试。
        return "network_retryable", True  # 标记为可重试网络异常。
    if any(token in name for token in ("jsondecode", "response", "validation", "keyerror", "typeerror", "valueerror")):  # 响应格式和参数结构错误通常不是瞬时网络问题。
        return "response_non_retryable", False  # 避免无效响应导致重复扣费和重复失败。
    return "unknown_non_retryable", False  # 未知异常默认保守地不重试，避免把未知故障放大。


@dataclass
class CircuitBreaker:  # 定义单个外部服务的进程级熔断器。
    service: str  # 保存服务名，例如 deepseek 或 doubao。
    failure_count: int = 0  # 保存连续可重试失败次数。
    opened_at: float | None = None  # 保存熔断打开的单调时间。
    half_open: bool = False  # 标记是否正在允许一次恢复探测请求。
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # 用锁保护并发调用下的计数和状态。

    def before_call(self) -> None:  # 定义外部调用前的熔断检查。
        with self.lock:  # 在检查期间锁住状态。
            if self.opened_at is None:  # 如果当前没有熔断。
                return  # 允许正常调用。
            elapsed = time.monotonic() - self.opened_at  # 计算熔断已经持续多久。
            if elapsed < CIRCUIT_RECOVERY_SECONDS:  # 如果还没有达到恢复等待时间。
                raise CircuitOpenError(f"{self.service} circuit is open")  # 阻断本次调用。
            if self.half_open:  # 如果已经有一个恢复探测请求在执行。
                raise CircuitOpenError(f"{self.service} circuit half-open probe is busy")  # 防止并发恢复请求同时撞击服务。
            self.half_open = True  # 只放行当前这一条恢复探测请求。

    def record_success(self) -> None:  # 定义成功调用后的熔断恢复操作。
        with self.lock:  # 在修改状态期间锁住熔断器。
            self.failure_count = 0  # 清空连续失败计数。
            self.opened_at = None  # 关闭熔断。
            self.half_open = False  # 结束恢复探测状态。

    def record_failure(self, retryable: bool) -> None:  # 定义失败调用后的熔断计数操作。
        with self.lock:  # 在修改状态期间锁住熔断器。
            if not retryable:  # 不可重试错误通常是参数或鉴权问题。
                self.half_open = False  # 释放可能占用的恢复探测状态，但不累计熔断失败。
                return  # 不因为非瞬时错误自动打开熔断。
            self.failure_count += 1  # 累计一次可重试失败。
            self.half_open = False  # 当前探测请求结束。
            if self.failure_count >= CIRCUIT_FAILURE_THRESHOLD:  # 连续失败达到阈值时。
                self.opened_at = time.monotonic()  # 打开熔断并记录开始时间。


@dataclass
class ServiceCall:  # 定义一次外部服务调用的统计载体。
    service: str  # 保存服务名称。
    stage: str  # 保存调用所在阶段。
    input_tokens_estimated: int = 0  # 保存输入 token 估算值。
    output_tokens: int = 0  # 保存服务返回的实际或估算输出 token 数。

    def set_usage(self, input_tokens: int | None = None, output_tokens: int | None = None) -> None:  # 定义让调用方补充真实 usage 的函数。
        if input_tokens is not None:  # 如果调用方拿到了真实输入 token。
            self.input_tokens_estimated = int(input_tokens)  # 用真实值覆盖估算值。
        if output_tokens is not None:  # 如果调用方拿到了真实输出 token。
            self.output_tokens = int(output_tokens)  # 保存真实输出 token。


@dataclass
class RequestRuntime:  # 定义一条完整用户查询的共享运行时。
    request_id: str  # 保存整条查询指纹。
    deadline_seconds: float  # 保存本次查询允许的总秒数。
    started_monotonic: float = field(default_factory=time.monotonic)  # 保存单调时钟开始点。
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))  # 保存可读开始时间。
    events: list[dict] = field(default_factory=list)  # 保存每次外部调用的结构化事件。
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # 保护事件列表和统计更新。

    @property
    def deadline_monotonic(self) -> float:  # 定义本次查询的单调截止点。
        return self.started_monotonic + self.deadline_seconds  # 用单调时间计算，避免系统时间变化影响预算。

    def remaining_seconds(self, stage: str = "") -> float:  # 返回当前阶段还剩多少秒。
        remaining = self.deadline_monotonic - time.monotonic()  # 计算截止点和当前时间的差值。
        if remaining <= 0:  # 如果已经没有剩余预算。
            raise RequestDeadlineExceeded(f"request deadline exceeded at {stage or 'unknown'}")  # 阻断新工作。
        return remaining  # 返回正的剩余秒数。

    def record(self, event: dict) -> None:  # 记录一次结构化外部调用事件。
        with self.lock:  # 保护当前请求的事件列表。
            self.events.append(event)  # 追加事件供最终统计使用。

    def summary(self) -> dict:  # 汇总当前请求的耗时、失败和 token 指标。
        elapsed_ms = round((time.monotonic() - self.started_monotonic) * 1000, 2)  # 计算整条查询耗时。
        summary: dict[str, Any] = {"request_id": self.request_id, "started_at": self.started_at, "elapsed_ms": elapsed_ms, "deadline_seconds": self.deadline_seconds, "deadline_exceeded": elapsed_ms / 1000 > self.deadline_seconds, "calls": len(self.events), "retryable_failures": 0, "non_retryable_failures": 0, "estimated_input_tokens": 0, "output_tokens": 0, "cache_hits": 0, "events": list(self.events)}  # 初始化可审计的汇总字段。
        for event in self.events:  # 遍历所有外部调用事件。
            if event.get("retryable") is True:  # 如果事件是可重试失败。
                summary["retryable_failures"] += 1  # 增加可重试失败计数。
            if event.get("retryable") is False and event.get("status") == "failed":  # 如果事件是不可重试失败。
                summary["non_retryable_failures"] += 1  # 增加不可重试失败计数。
            summary["estimated_input_tokens"] += int(event.get("input_tokens_estimated", 0))  # 累加输入 token 估算。
            summary["output_tokens"] += int(event.get("output_tokens", 0))  # 累加输出 token。
            if event.get("status") == "cache_hit":  # 如果本次调用被本地缓存短路。
                summary["cache_hits"] += 1  # 增加缓存命中次数。
        return summary  # 返回完整汇总。

    def finish(self) -> dict:  # 保存当前请求统计并返回汇总结果。
        summary = self.summary()  # 先计算最终统计。
        try:  # 统计文件写入失败不能反过来影响用户已经得到的答案。
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
            path = OUTPUT_ROOT / "request_metrics.jsonl"  # 定义统一请求统计文件。
            with locked_file(path.with_suffix(path.suffix + ".lock")), path.open("a", encoding="utf-8") as file:  # 锁住统计文件追加操作，避免并发请求交错写入。
                file.write(json.dumps(summary, ensure_ascii=False) + "\n")  # 写入机器可读 JSONL。
            summary["metrics_path"] = str(path)  # 把统计文件路径返回给报告和调试调用方。
        except OSError as exc:  # 捕获磁盘权限或空间异常。
            logger.exception("Request metrics_write_failed request_id=%s error_type=%s", self.request_id, type(exc).__name__)  # 记录统计写入异常。
        logger.info("Request finished request_id=%s elapsed_ms=%s calls=%d retryable_failures=%d non_retryable_failures=%d estimated_input_tokens=%d output_tokens=%d cache_hits=%d", self.request_id, summary["elapsed_ms"], summary["calls"], summary["retryable_failures"], summary["non_retryable_failures"], summary["estimated_input_tokens"], summary["output_tokens"], summary["cache_hits"])  # 写入不含问题正文的请求汇总日志。
        return summary  # 返回最终统计。


CURRENT_REQUEST: contextvars.ContextVar[RequestRuntime | None] = contextvars.ContextVar("current_request", default=None)  # 定义当前请求上下文变量。
BREAKERS = {"deepseek": CircuitBreaker("deepseek"), "doubao": CircuitBreaker("doubao")}  # 为两个外部模型服务建立进程级熔断器。


def current_request() -> RequestRuntime | None:  # 定义读取当前请求上下文的函数。
    return CURRENT_REQUEST.get()  # 返回当前线程或异步上下文绑定的运行时。


def current_request_summary() -> dict:  # 定义读取当前请求摘要的函数。
    runtime = current_request()  # 读取当前上下文。
    return runtime.summary() if runtime else {}  # 没有请求上下文时返回空字典，兼容离线模块调用。


def ensure_request_budget(stage: str) -> float:  # 定义统一的截止时间检查入口。
    runtime = current_request()  # 读取当前请求上下文。
    return runtime.remaining_seconds(stage) if runtime else float("inf")  # 离线评估没有总请求上下文时不阻断流程。


@contextmanager
def request_scope(question: str, session_name: str | None = None, deadline_seconds: float | None = None, request_id: str | None = None) -> Iterator[RequestRuntime]:  # 定义一条完整用户查询的上下文管理器，并允许服务接口传入已经生成的追踪编号。
    effective_deadline = REQUEST_DEADLINE_SECONDS if deadline_seconds is None else float(deadline_seconds)  # 没有覆盖值时沿用项目默认配置。
    if not math.isfinite(effective_deadline) or effective_deadline <= 0:  # 检查接口或内部调用传入的覆盖值。
        raise ValueError("deadline_seconds 必须是有限正数。")  # 拒绝会立即失效或无限等待的请求预算。
    runtime = RequestRuntime(request_id or request_id_for(question, session_name), effective_deadline)  # 创建带本次总预算和唯一追踪编号的请求运行时。
    token = CURRENT_REQUEST.set(runtime)  # 把运行时绑定到当前执行上下文。
    try:  # 保护完整的用户查询过程。
        yield runtime  # 把运行时交给调用方执行多问题、检索和回答。
    finally:  # 无论成功、超时还是异常都要落盘统计。
        CURRENT_REQUEST.reset(token)  # 恢复进入本次请求前的上下文。
        runtime.finish()  # 保存耗时、失败和 token 统计。


@contextmanager
def tracked_service_call(service: str, stage: str, input_tokens: int = 0) -> Iterator[ServiceCall]:  # 定义外部服务调用追踪上下文。
    runtime = current_request()  # 读取当前完整查询运行时。
    started = time.perf_counter()  # 记录单次外部调用开始时间。
    tracker = ServiceCall(service, stage, input_tokens)  # 创建本次调用统计对象。
    breaker = BREAKERS.get(service)  # 读取对应服务的进程级熔断器。
    try:  # 保护调用前检查、实际调用和调用后统计。
        ensure_request_budget(stage)  # 在发起外部调用前先检查整条查询的剩余预算。
        if breaker:  # 如果当前服务有熔断器。
            breaker.before_call()  # 在网络请求前执行熔断保护。
        yield tracker  # 交给调用方执行真实网络请求。
    except Exception as exc:  # 捕获调用前阻断或实际网络异常。
        failure_kind, retryable = classify_exception(exc)  # 将底层异常归类成稳定治理信号。
        if breaker:  # 如果当前服务有熔断器。
            breaker.record_failure(retryable)  # 只有可重试失败才累计熔断计数。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)  # 计算本次失败调用耗时。
        event = {"service": service, "stage": stage, "status": "blocked" if isinstance(exc, (RequestDeadlineExceeded, CircuitOpenError)) else "failed", "failure_kind": failure_kind, "retryable": retryable, "elapsed_ms": elapsed_ms, "input_tokens_estimated": tracker.input_tokens_estimated, "output_tokens": tracker.output_tokens}  # 组织失败或阻断事件。
        if runtime:  # 只有完整用户查询才写入请求事件。
            runtime.record(event)  # 把事件加入本次请求统计。
        logger.warning("Request service_failed service=%s stage=%s failure_kind=%s retryable=%s elapsed_ms=%s", service, stage, failure_kind, retryable, elapsed_ms)  # 记录稳定异常类别，便于后续聚合分析。
        raise  # 保留原始异常类型，让上层可以安全降级或终止重试。
    else:  # 如果外部服务调用成功。
        if breaker:  # 如果当前服务有熔断器。
            breaker.record_success()  # 成功调用会清空连续失败计数并关闭熔断。
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)  # 计算成功调用耗时。
        success_event: dict[str, Any] = {"service": service, "stage": stage, "status": "success", "failure_kind": "", "retryable": None, "elapsed_ms": elapsed_ms, "input_tokens_estimated": tracker.input_tokens_estimated, "output_tokens": tracker.output_tokens}  # 组织成功事件。
        if runtime:  # 只有完整用户查询才写入请求事件。
            runtime.record(success_event)  # 把成功事件加入本次请求统计。
        logger.info("Request service_success service=%s stage=%s elapsed_ms=%s input_tokens_estimated=%s output_tokens=%s", service, stage, elapsed_ms, tracker.input_tokens_estimated, tracker.output_tokens)  # 记录成功调用的耗时和 token 指标。


def record_cache_hit(service: str, stage: str, input_tokens: int = 0) -> None:  # 定义不发起网络调用时的缓存统计函数。
    runtime = current_request()  # 读取当前请求上下文。
    if runtime:  # 只有在完整查询期间才保存缓存事件。
        runtime.record({"service": service, "stage": stage, "status": "cache_hit", "failure_kind": "", "retryable": None, "elapsed_ms": 0.0, "input_tokens_estimated": input_tokens, "output_tokens": 0})  # 记录一次省掉外部调用的缓存命中。
    logger.info("Request cache_hit service=%s stage=%s", service, stage)  # 在统一日志中记录缓存命中。
