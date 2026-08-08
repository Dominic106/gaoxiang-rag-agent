import pickle  # 导入 pickle，用来加载已经保存好的 BM25 检索器。
import sys  # 导入 sys，用来从命令行读取用户输入的问题。

from config import BM25_PICKLE  # 从配置文件读取 BM25 索引文件路径。
from rag_tokenizers import tokenize_for_bm25  # 导入 BM25 分词函数，确保加载 pickle 时函数定义存在。  # noqa: F401


def main() -> None:  # 定义主函数，脚本从这里开始执行。
    query = " ".join(sys.argv[1:]).strip()  # 把命令行参数拼成一个查询问题。
    if not query:  # 如果命令行里没有传问题。
        query = input("请输入要检索的关键词或问题：").strip()  # 就在终端里提示用户输入。
    with BM25_PICKLE.open("rb") as file:  # 打开 BM25 索引文件。
        retriever = pickle.load(file)  # 把保存好的 BM25 检索器加载回来。
    retriever.k = 5  # 设置只返回前 5 条，方便人工检查是否命中得准。
    docs = retriever.invoke(query)  # 用 BM25 对用户查询做关键词检索。
    for index, doc in enumerate(docs, start=1):  # 遍历检索结果，并从 1 开始编号。
        metadata = doc.metadata  # 取出当前 chunk 的元数据。
        print("=" * 80)  # 打印分隔线，让每条结果更清楚。
        print(f"结果 {index}")  # 打印当前结果编号。
        print(f"章节：{metadata.get('chapter', '')}")  # 打印章名。
        print(f"小节：{metadata.get('section', '')}")  # 打印小节名。
        print(f"来源：{metadata.get('relative_path', '')}")  # 打印源 Word 相对路径。
        print(f"chunk_id：{metadata.get('chunk_id', '')}")  # 打印 chunk_id，方便以后追踪。
        print("-" * 80)  # 打印正文前的分隔线。
        print(doc.page_content[:800])  # 打印前 800 个字符的原文片段，避免终端输出太长。


if __name__ == "__main__":  # 判断当前文件是否直接运行。
    main()  # 如果是直接运行，就执行主函数。
