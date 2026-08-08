"""索引版本和健康治理命令行入口。"""  # 说明本文件只负责把治理能力暴露给开发者和部署脚本。

import argparse  # 导入 argparse，用来解析 check 和 compare 子命令。
from pathlib import Path  # 导入 Path，用来接收评估报告路径。

from index_health import run_health_check  # 导入索引健康检查函数。
from index_health import write_health_report  # 导入健康报告写入函数。
from index_health import write_metric_comparison  # 导入前后评估指标比较函数。


def build_parser() -> argparse.ArgumentParser:  # 定义命令行参数解析器。
    parser = argparse.ArgumentParser(description="检查信息系统项目管理师 RAG 索引版本和健康状态")  # 创建主解析器。
    subparsers = parser.add_subparsers(dest="command", required=True)  # 创建必须指定子命令的解析器集合。
    subparsers.add_parser("check", help="检查源文件、manifest、chunks、BM25 和 Chroma")  # 注册健康检查命令。
    compare = subparsers.add_parser("compare", help="比较重建前后的检索评估报告")  # 注册评估比较命令。
    compare.add_argument("--before", required=True, type=Path, help="重建前的 JSON、CSV 或 Markdown 评估报告")  # 接收重建前报告路径。
    compare.add_argument("--after", required=True, type=Path, help="重建后的 JSON、CSV 或 Markdown 评估报告")  # 接收重建后报告路径。
    return parser  # 返回配置完成的解析器。


def main() -> int:  # 定义命令行主函数，返回适合 shell 判断的退出码。
    args = build_parser().parse_args()  # 解析用户输入的子命令和参数。
    if args.command == "check":  # 如果用户请求健康检查。
        result = run_health_check()  # 执行完整索引健康检查。
        json_path, md_path = write_health_report(result)  # 保存 JSON 和 Markdown 两份报告。
        print(f"健康检查：{'通过' if result['passed'] else '失败'}")  # 打印人类可读状态。
        print(f"索引版本：{result.get('index_version', '未知')}")  # 打印当前索引版本。
        print(f"JSON 报告：{json_path}")  # 打印机器报告路径。
        print(f"Markdown 报告：{md_path}")  # 打印人工报告路径。
        for item in result.get("checks", []):  # 遍历检查项。
            print(f"- {'通过' if item['passed'] else '失败'} {item['name']}：{item['detail']}")  # 输出每项检查细节，便于快速定位问题。
        return 0 if result["passed"] else 1  # 健康检查失败时让自动化脚本阻断后续发布。
    if not args.before.exists() or not args.after.exists():  # 比较前先检查两个报告是否存在。
        print("比较失败：before 或 after 报告不存在。")  # 输出明确错误。
        return 2  # 返回参数或文件错误码。
    json_path, md_path = write_metric_comparison(args.before, args.after)  # 执行并保存前后指标比较。
    print(f"指标比较 JSON：{json_path}")  # 打印机器报告路径。
    print(f"指标比较 Markdown：{md_path}")  # 打印人工报告路径。
    return 0 if "回归" not in Path(md_path).read_text(encoding="utf-8") else 1  # 把发现回归转成非零退出码，支持 CI 或发布脚本拦截。


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    raise SystemExit(main())  # 执行主函数并把结果交给操作系统。
