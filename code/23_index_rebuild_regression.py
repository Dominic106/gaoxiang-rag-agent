"""向量索引原子重建回归测试。"""  # 说明本文件验证重建中断时不会破坏上一版 Chroma 索引。

import importlib  # 导入 importlib，用来加载数字开头的建库脚本。
import tempfile  # 导入 tempfile，创建隔离的索引目录。
from pathlib import Path  # 导入 Path，用来检查临时目录和正式目录。
from types import SimpleNamespace  # 导入 SimpleNamespace，构造不访问网络的 embedding 替身。
from unittest.mock import patch  # 导入 patch，替换真实向量建库调用。

from langchain_core.documents import Document  # 导入 Document，构造最小 chunk 输入。


build_indexes = importlib.import_module("02_build_indexes")  # 加载待测的向量索引构建模块。


def test_failed_rebuild_keeps_previous_index() -> None:  # 验证新索引构建失败时旧目录仍然可用。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 创建隔离测试目录。
        root = Path(temporary_directory)  # 把字符串目录转换成 Path。
        old_index = root / "chroma"  # 定义模拟的正式向量索引目录。
        old_index.mkdir()  # 创建旧索引目录。
        (old_index / "sentinel.txt").write_text("old", encoding="utf-8")  # 写入旧版本标记。
        docs = [Document(page_content="测试", metadata={"chunk_id": "chunk-1"})]  # 构造一个最小 chunk。
        with patch.object(build_indexes, "INDEX_ROOT", root), patch.object(build_indexes, "CHROMA_DIR", old_index), patch.object(build_indexes, "build_embedding_model", return_value=SimpleNamespace()), patch.object(build_indexes.Chroma, "from_documents", side_effect=RuntimeError("simulated build failure")):  # 模拟向量建库中断。
            try:  # 捕获预期的构建异常。
                build_indexes.build_vector_index(docs)  # 执行原子重建逻辑。
            except RuntimeError as exc:  # 确认异常仍然向上报告。
                assert str(exc) == "simulated build failure", "重建失败没有保留原始诊断异常"  # 确认调用方仍能知道失败原因。
            else:  # 如果没有抛出异常则测试失败。
                raise AssertionError("模拟失败的向量重建没有抛出异常")  # 明确报告错误行为。
        assert (old_index / "sentinel.txt").read_text(encoding="utf-8") == "old", "向量重建失败破坏了旧索引"  # 确认旧目录和标记仍然存在。
        assert not list(root.glob(".chroma.building-*")), "失败重建留下了临时半成品目录"  # 确认临时目录已清理。


def test_successful_rebuild_swaps_index() -> None:  # 验证成功构建后新目录替换旧目录。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 创建隔离测试目录。
        root = Path(temporary_directory)  # 把字符串目录转换成 Path。
        old_index = root / "chroma"  # 定义模拟的正式向量索引目录。
        old_index.mkdir()  # 创建旧索引目录。
        (old_index / "sentinel.txt").write_text("old", encoding="utf-8")  # 写入旧版本标记。
        docs = [Document(page_content="测试", metadata={"chunk_id": "chunk-1"})]  # 构造一个最小 chunk。

        def fake_from_documents(**kwargs):  # 定义不访问网络的 Chroma 建库替身。
            persist_directory = Path(kwargs["persist_directory"])  # 读取临时持久化目录。
            (persist_directory / "new-index.txt").write_text("new", encoding="utf-8")  # 写入新版本标记模拟建库成功。
            return object()  # 返回一个占位向量库对象。

        with patch.object(build_indexes, "INDEX_ROOT", root), patch.object(build_indexes, "CHROMA_DIR", old_index), patch.object(build_indexes, "build_embedding_model", return_value=SimpleNamespace()), patch.object(build_indexes.Chroma, "from_documents", side_effect=fake_from_documents):  # 模拟向量建库成功。
            assert build_indexes.build_vector_index(docs) is True, "成功的向量重建没有返回 True"  # 确认构建结果状态正确。
        assert (old_index / "new-index.txt").read_text(encoding="utf-8") == "new", "成功重建没有切换到新索引"  # 确认正式目录已换成新版本。
        assert not (old_index / "sentinel.txt").exists(), "成功重建仍然保留旧索引文件"  # 确认旧索引没有和新索引混合。
        assert not list(root.glob(".chroma.backup-*")), "成功重建没有清理旧索引备份"  # 确认不会长期堆积备份目录。


def main() -> None:  # 定义索引重建回归测试入口。
    test_failed_rebuild_keeps_previous_index()  # 执行失败恢复测试。
    test_successful_rebuild_swaps_index()  # 执行成功切换测试。
    print("索引重建回归通过：失败保留旧索引，成功原子切换新索引。")  # 输出统一成功结论。


if __name__ == "__main__":  # 判断当前脚本是否直接运行。
    main()  # 直接运行时执行全部索引重建测试。
