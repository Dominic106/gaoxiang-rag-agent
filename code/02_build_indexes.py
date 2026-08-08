import json  # 导入 json，用来读取第一步生成的 chunks.jsonl。
import os  # 导入 os，用来检查环境变量里有没有 OPENAI_API_KEY。
import pickle  # 导入 pickle，用来把 BM25 检索器保存到本地。
import shutil  # 导入 shutil，用来清理失败的临时向量索引目录。
import tempfile  # 导入 tempfile，用来在正式索引目录旁边创建临时构建目录。
import uuid  # 导入 uuid，用来生成本次临时构建目录的唯一名称。
from pathlib import Path  # 导入 Path，用来处理临时目录和正式索引目录。

from langchain_chroma import Chroma  # 从 LangChain Chroma 集成导入 Chroma 向量数据库。
from langchain_community.retrievers import BM25Retriever  # 从 LangChain 社区包导入 BM25 关键词检索器。
from langchain_core.documents import Document  # 从 LangChain 导入 Document，承载 chunk 正文和元数据。
from langchain_openai import OpenAIEmbeddings  # 从 LangChain OpenAI 集成导入 OpenAI embedding 模型。

from config import BM25_PICKLE  # 从配置文件读取 BM25 保存路径。
from config import CHROMA_COLLECTION_NAME  # 从配置文件读取 Chroma 内部集合名。
from config import CHROMA_DIR  # 从配置文件读取 Chroma 保存目录。
from config import CHUNKS_JSONL  # 从配置文件读取 chunk 明细文件路径。
from config import EMBEDDING_PROVIDER  # 从配置文件读取当前 embedding provider。
from config import INDEX_ROOT  # 从配置文件读取索引目录。
from doubao_embeddings import make_doubao_embeddings  # 导入豆包 embedding 工厂函数。
from rag_tokenizers import tokenize_for_bm25  # 从独立模块导入 BM25 分词函数，确保 pickle 加载时也能找到它。
from index_health import finalize_manifest  # 导入清单收口函数，在 BM25 和 Chroma 完成后记录最终版本状态。
from app_logging import get_logger  # 导入统一日志器，记录旧索引备份清理等非核心异常。


logger = get_logger(__name__)  # 创建当前模块日志器。


def load_chunks() -> list[Document]:  # 定义读取 chunks.jsonl 并恢复成 LangChain Document 的函数。
    docs: list[Document] = []  # 准备列表，用来保存恢复出来的 Document。
    with CHUNKS_JSONL.open("r", encoding="utf-8") as file:  # 打开第一步生成的 chunk 文件。
        for line in file:  # 逐行读取，每一行都是一个 chunk。
            record = json.loads(line)  # 把 JSON 字符串解析成 Python 字典。
            text = record["text"]  # 取出 chunk 正文。
            metadata = record["metadata"]  # 取出 chunk 元数据。
            docs.append(Document(page_content=text, metadata=metadata))  # 恢复成 LangChain Document。
    return docs  # 返回所有 chunk 文档。


def build_embedding_model():  # 定义创建 embedding 模型的函数，根据配置返回 OpenAI 或豆包 embedding。
    if EMBEDDING_PROVIDER == "doubao":  # 如果配置为豆包。
        return make_doubao_embeddings()  # 返回豆包 embedding 适配器。
    if not os.environ.get("OPENAI_API_KEY"):  # 检查环境变量里有没有 OpenAI API Key。
        print("没有检测到 OPENAI_API_KEY：本次只建立 BM25 关键词索引，暂时跳过 Chroma 向量索引。")  # 给出清晰提示。
        return None  # 返回 None，表示暂时没有可用 embedding 模型。
    return OpenAIEmbeddings(model="text-embedding-3-large")  # 返回 OpenAI 的高质量 embedding 模型。


def build_vector_index(docs: list[Document]) -> bool:  # 定义建立向量索引的函数，返回是否真的建立成功。
    embeddings = build_embedding_model()  # 创建 embedding 模型，用来把每个 chunk 变成向量。
    if embeddings is None:  # 如果没有 embedding 模型，说明缺少 API Key。
        return False  # 返回 False，告诉主函数向量索引被跳过。
    ids = [doc.metadata["chunk_id"] for doc in docs]  # 准备每个 chunk 的唯一 ID，保证向量库记录可追溯。
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{CHROMA_DIR.name}.building-{uuid.uuid4().hex[:8]}-", dir=str(INDEX_ROOT)))  # 在索引目录旁创建临时目录，避免中断时破坏当前可用索引。
    backup_dir = INDEX_ROOT / f".{CHROMA_DIR.name}.backup-{uuid.uuid4().hex[:8]}"  # 为旧索引准备临时备份目录，支持切换失败时恢复。
    try:  # 保护临时构建和最终目录切换。
        Chroma.from_documents(  # 先把完整向量索引写入临时目录。
            documents=docs,  # 传入所有 chunk 文档。
            embedding=embeddings,  # 传入 embedding 模型。
            ids=ids,  # 传入稳定 ID。
            collection_name=CHROMA_COLLECTION_NAME,  # 指定 Chroma collection 名称。
            persist_directory=str(temp_dir),  # 指定临时向量数据库目录。
        )  # Chroma 会自动计算向量并保存到临时目录。
        if CHROMA_DIR.exists():  # 如果当前存在旧的向量索引。
            CHROMA_DIR.rename(backup_dir)  # 先把旧索引移到备份位置，避免正式目录出现半成品。
        try:  # 保护临时目录到正式目录的切换。
            temp_dir.rename(CHROMA_DIR)  # 在同一文件系统内切换目录名称。
        except Exception:  # 如果正式切换失败。
            if backup_dir.exists() and not CHROMA_DIR.exists():  # 只有旧目录已备份且正式目录不存在时才执行恢复。
                backup_dir.rename(CHROMA_DIR)  # 恢复上一版可用索引。
            raise  # 把切换失败交给外层统一处理。
        if backup_dir.exists():  # 新索引切换成功后。
            try:  # 旧备份清理失败不应把已经成功切换的新索引误报为构建失败。
                shutil.rmtree(backup_dir)  # 删除旧版本备份，避免长期占用磁盘。
            except OSError as exc:  # 捕获权限或磁盘异常，保留备份供后续人工清理。
                logger.warning("Vector index backup_cleanup_failed path=%s error_type=%s", backup_dir, type(exc).__name__)  # 记录稳定诊断信息，不影响新索引可用性。
        return True  # 返回 True，表示新向量索引已经完整切换。
    except Exception:  # 构建或切换失败时保留原索引并向上报告失败。
        if temp_dir.exists():  # 如果临时目录还存在。
            shutil.rmtree(temp_dir, ignore_errors=True)  # 清理半成品临时目录。
        if backup_dir.exists() and not CHROMA_DIR.exists():  # 如果旧索引已备份但正式目录尚未恢复。
            backup_dir.rename(CHROMA_DIR)  # 尽力恢复旧索引。
        raise  # 让主函数把本次建库标记为失败。


def build_keyword_index(docs: list[Document]) -> None:  # 定义建立 BM25 关键词索引的函数。
    retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize_for_bm25)  # 用中文分词后的 chunk 创建 BM25 检索器。
    retriever.k = 12  # 设置默认返回数量，和配置里的 TOP_K_KEYWORD 保持一致。
    with BM25_PICKLE.open("wb") as file:  # 打开 BM25 保存文件。
        pickle.dump(retriever, file)  # 把 BM25 检索器序列化保存，查询时可直接加载。


def main() -> None:  # 定义主函数。
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)  # 确保索引目录存在。
    docs = load_chunks()  # 第一步：读取第一阶段生成的 chunk。
    build_keyword_index(docs)  # 第二步：先建立 BM25 关键词索引，因为它不需要 API Key。
    vector_ok = build_vector_index(docs)  # 第三步：尝试建立 Chroma 向量索引，这一步需要 OPENAI_API_KEY。
    manifest = finalize_manifest(vector_ok)  # 第四步：写入 BM25/Chroma artifact 指纹，并确定索引是否整体可用。
    print(f"读取文本块：{len(docs)} 个")  # 打印参与建索引的 chunk 数量。
    print(f"BM25 索引：{BM25_PICKLE}")  # 打印 BM25 索引位置。
    print(f"Chroma 索引：{CHROMA_DIR if vector_ok else '未生成，等待 OPENAI_API_KEY'}")  # 根据实际情况打印向量索引状态。
    print(f"索引版本：{manifest.get('index_version', '未知')}")  # 打印本次构建的确定性索引版本。
    print(f"索引状态：{'就绪' if manifest.get('index_ready') else '未就绪'}")  # 明确告诉使用者是否可以进入查询阶段。


if __name__ == "__main__":  # 判断当前脚本是否被直接运行。
    main()  # 如果是直接运行，就执行主函数。
