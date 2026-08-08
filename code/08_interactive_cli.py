import importlib  # 导入 importlib，用来动态加载文件名以数字开头的教学脚本模块。

answer_module = importlib.import_module("06_answer_graph_no_key")  # 动态导入无 Key 回答图模块。


def main() -> None:  # 定义交互式主函数。
    app = answer_module.build_graph()  # 构建一次 LangGraph 应用，后续循环复用。
    print("信息系统项目管理师 RAG 交互模式")  # 打印欢迎语。
    print("输入 exit 或 quit 退出。")  # 打印退出提示。
    while True:  # 开始循环。
        question = input("\n问题> ").strip()  # 读取用户问题。
        if question.lower() in {"exit", "quit"}:  # 如果用户输入退出命令。
            break  # 跳出循环。
        if not question:  # 如果用户输入空问题。
            continue  # 继续下一轮。
        state = {  # 准备图的初始状态。
            "question": question,  # 写入问题。
            "question_type": "",  # 初始化问题类型。
            "understanding_reason": "",  # 初始化分类原因。
            "query": "",  # 初始化 query。
            "docs": [],  # 初始化文档列表。
            "evidence_docs": [],  # 初始化证据来源列表。
            "answer": "",  # 初始化答案。
            "report_path": "",  # 初始化报告路径。
        }  # 初始状态结束。
        result = app.invoke(state)  # 执行无 Key 回答图。
        print(result["answer"])  # 打印回答。
        print(f"\n报告已保存：{result['report_path']}")  # 打印报告路径。


if __name__ == "__main__":  # 判断是否直接运行。
    main()  # 执行主函数。
