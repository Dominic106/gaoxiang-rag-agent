import logging  # 导入 logging，用来记录运行过程、异常和性能诊断信息。
from logging.handlers import RotatingFileHandler  # 导入轮转文件处理器，防止日志文件无限增长。

from config import LOG_BACKUP_COUNT  # 从配置读取日志备份数量。
from config import LOG_FILE  # 从配置读取日志文件路径。
from config import LOG_MAX_BYTES  # 从配置读取单个日志文件最大大小。


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"  # 定义统一日志格式，不记录 API Key 和完整提示词。


def get_logger(name: str) -> logging.Logger:  # 定义获取项目日志器的函数，让所有模块使用同一套输出规则。
    logger = logging.getLogger(name)  # 按模块名称获取日志器。
    if logger.handlers:  # 如果当前日志器已经配置过处理器。
        return logger  # 直接复用，避免重复写入同一条日志。
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在。
    handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")  # 创建按大小轮转的日志文件处理器。
    handler.setFormatter(logging.Formatter(LOG_FORMAT))  # 设置统一的时间、级别、模块和消息格式。
    logger.addHandler(handler)  # 把文件处理器挂到模块日志器上。
    logger.setLevel(logging.INFO)  # 默认记录 INFO 及以上级别，便于诊断请求耗时和异常。
    logger.propagate = False  # 禁止向根日志器传播，避免 CLI 重复打印或重复写文件。
    return logger  # 返回已经配置好的模块日志器。
