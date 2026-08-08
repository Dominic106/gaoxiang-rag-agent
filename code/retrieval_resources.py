import pickle  # 导入 pickle，用来从磁盘恢复已经建立好的 BM25 检索器。
import os  # 导入 os，用来检查 OpenAI embedding 模式的配置。
import warnings  # 导入 warnings，用来屏蔽旧版 LangChain 反序列化提示。
from functools import lru_cache  # 导入 lru_cache，用来在同一进程内复用检索资源。

from langchain_chroma import Chroma  # 导入 Chroma，用来加载持久化向量索引。
from langchain_community.retrievers import BM25Retriever  # 导入 BM25Retriever，用来建立章节限定的关键词检索器。
from langchain_openai import OpenAIEmbeddings  # 导入 OpenAIEmbeddings，支持可替换的 embedding 提供方。

from config import BM25_PICKLE  # 从配置读取 BM25 索引路径。
from config import CHROMA_COLLECTION_NAME  # 从配置读取 Chroma collection 名称。
from config import CHROMA_DIR  # 从配置读取 Chroma 持久化目录。
from config import EMBEDDING_PROVIDER  # 从配置读取 embedding 提供方。
from doubao_embeddings import make_doubao_embeddings  # 导入豆包 embedding 工厂函数。
from rag_tokenizers import tokenize_for_bm25  # 导入中文 BM25 分词函数。


@lru_cache(maxsize=1)  # 一个进程只创建一个 embedding 对象，避免每个子问题重复初始化客户端和读取缓存。
def get_embedding_model():  # 定义获取查询 embedding 模型的函数。
    if EMBEDDING_PROVIDER == "doubao":  # 如果项目配置使用豆包 embedding。
        return make_doubao_embeddings()  # 创建并缓存豆包 embedding 适配器。
    if not os.environ.get("OPENAI_API_KEY"):  # 如果使用 OpenAI 但没有配置 API Key。
        raise RuntimeError("请先设置 OPENAI_API_KEY，否则无法使用 OpenAI embedding 做向量检索。")  # 抛出清晰错误。
    return OpenAIEmbeddings(model="text-embedding-3-large")  # 创建并缓存 OpenAI embedding 适配器。


@lru_cache(maxsize=1)  # 一个进程只加载一次 Chroma，多个子问题共享同一个向量库对象。
def get_vector_store() -> Chroma:  # 定义获取向量数据库对象的函数。
    return Chroma(  # 创建本地 Chroma 连接。
        collection_name=CHROMA_COLLECTION_NAME,  # 指定内部 collection 名称。
        embedding_function=get_embedding_model(),  # 使用缓存的 embedding 对象生成查询向量。
        persist_directory=str(CHROMA_DIR),  # 指定本地持久化目录。
    )  # 返回可复用的 Chroma 对象。


@lru_cache(maxsize=1)  # 一个进程只从磁盘反序列化一次全量 BM25 检索器。
def get_bm25_retriever() -> BM25Retriever:  # 定义获取全量 BM25 检索器的函数。
    with warnings.catch_warnings():  # 创建局部 warning 管理区域。
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # 忽略旧版 LangChain 反序列化提示。
        with BM25_PICKLE.open("rb") as file:  # 打开 BM25 索引文件。
            return pickle.load(file)  # 反序列化并返回全量检索器。


@lru_cache(maxsize=64)  # 缓存常用章节检索器，避免每次章节查询都重新分词建 BM25。
def get_chapter_bm25_retriever(chapter: str) -> BM25Retriever:  # 定义获取单章节 BM25 检索器的函数。
    base_retriever = get_bm25_retriever()  # 复用已经加载的全量检索器。
    filtered_docs = [doc for doc in base_retriever.docs if doc.metadata.get("chapter") == chapter]  # 从全量文档中筛出目标章节。
    return BM25Retriever.from_documents(filtered_docs, preprocess_func=tokenize_for_bm25)  # 建立并缓存章节限定检索器。


@lru_cache(maxsize=128)  # 缓存常用小节文档列表，避免答案阶段重复扫描整本教材。
def get_section_documents(chapter: str, section: str) -> tuple:  # 定义获取同一小节全部 chunk 的函数，用于补回定义或流程起始段。
    retriever = get_keyword_retriever(chapter)  # 复用全量或章节限定的 BM25 文档集合。
    documents = [doc for doc in retriever.docs if doc.metadata.get("section", "") == section]  # 按小节元数据筛选相关 chunk。
    return tuple(sorted(documents, key=lambda doc: int(doc.metadata.get("chunk_index", 0))))  # 按原文顺序返回不可变元组，方便缓存复用。


def get_keyword_retriever(chapter: str = "") -> BM25Retriever:  # 定义统一的关键词检索器入口，避免主流程重复判断章节条件。
    return get_chapter_bm25_retriever(chapter) if chapter else get_bm25_retriever()  # 根据章节条件返回缓存的全量或章节检索器。


def clear_runtime_caches() -> None:  # 定义清理运行时缓存的函数，方便测试或索引更新后的进程内重载。
    get_embedding_model.cache_clear()  # 清理 embedding 对象缓存。
    get_vector_store.cache_clear()  # 清理 Chroma 对象缓存。
    get_bm25_retriever.cache_clear()  # 清理全量 BM25 缓存。
    get_chapter_bm25_retriever.cache_clear()  # 清理章节 BM25 缓存。
    get_section_documents.cache_clear()  # 清理小节文档缓存。
