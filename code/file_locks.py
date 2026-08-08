"""提供跨线程、跨进程的本地文件锁。"""  # 说明这个小模块只负责保护本地文件读写边界。

from contextlib import contextmanager  # 导入 contextmanager，提供 with 风格的锁生命周期。
from collections.abc import Iterator  # 导入 Iterator，标注上下文管理器类型。
from pathlib import Path  # 导入 Path，用来处理锁文件路径。

import fcntl  # macOS 和 Unix 提供的文件锁实现，保护多个服务进程之间的写入。


@contextmanager
def locked_file(lock_path: Path) -> Iterator[None]:  # 定义一个独立锁文件，避免锁住正在替换的目标文件。
    lock_path.parent.mkdir(parents=True, exist_ok=True)  # 确保锁文件所在目录存在。
    with lock_path.open("a+", encoding="utf-8") as lock_file:  # 以追加模式打开锁文件，锁文件本身不保存业务数据。
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # 获取排他锁，阻塞等待其他请求完成。
        try:  # 保护调用方的整个读写临界区。
            yield  # 把锁交给 with 代码块使用。
        finally:  # 无论业务成功还是异常都释放文件锁。
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # 释放排他锁。
