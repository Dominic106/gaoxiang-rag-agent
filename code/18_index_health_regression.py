"""索引治理模块的离线回归测试。"""  # 说明本文件验证版本和指标比较逻辑，不调用 embedding API。

from index_health import build_index_version  # 导入确定性索引版本函数。
from index_health import compare_metric_summaries  # 导入前后指标比较函数。
from index_health import embedding_snapshot  # 导入 embedding 配置快照函数。


def main() -> None:  # 定义测试主函数。
    first = build_index_version("source-hash")  # 用相同输入计算第一次索引版本。
    second = build_index_version("source-hash")  # 用相同输入计算第二次索引版本。
    assert first == second and first.startswith("idx-")  # 确认版本指纹稳定且带有可识别前缀。
    snapshot = embedding_snapshot()  # 读取 embedding 配置快照。
    assert "api_key" not in snapshot and "key" not in snapshot  # 确认健康治理结果不会泄露 API Key。
    passed = compare_metric_summaries({"hybrid_top1": 10, "hybrid_top5": 12}, {"hybrid_top1": 10, "hybrid_top5": 12})  # 比较相同指标，应该通过。
    assert passed["passed"] and not passed["regressions"]  # 确认无回归时通过。
    failed = compare_metric_summaries({"hybrid_top1": 10, "hybrid_top5": 12}, {"hybrid_top1": 9, "hybrid_top5": 12})  # 构造 Top1 下降的回归样例。
    assert not failed["passed"] and failed["regressions"] == ["hybrid_top1"]  # 确认检索质量下降会被拦截。
    print("索引治理离线回归通过：版本稳定、密钥不落盘、指标回退可识别。")  # 输出测试结论。


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    main()  # 执行离线回归测试。
