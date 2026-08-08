from doubao_embeddings import make_doubao_embeddings  # 导入豆包 embedding 工厂函数。


def main() -> None:  # 定义主函数。
    embeddings = make_doubao_embeddings()  # 根据 .env 创建豆包 embedding 客户端。
    text = "项目的范围基准包括项目范围说明书、工作分解结构和工作分解结构词汇表。"  # 准备一条测试文本。
    vector = embeddings.embed_query(text)  # 调用豆包 embedding，把测试文本转成向量。
    print("模型调用成功。")  # 打印成功提示。
    print(f"向量维度：{len(vector)}")  # 打印向量维度，用来确认模型输出规格。
    print(f"前 8 个数值：{vector[:8]}")  # 打印前几个数值，确认返回的是 float 向量。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 执行主函数。
