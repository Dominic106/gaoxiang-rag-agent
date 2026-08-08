from pathlib import Path  # 导入 Path，用来定位当前代码目录。

from dotenv import load_dotenv  # 导入 load_dotenv，用来读取 .env 配置文件。


CODE_DIR = Path(__file__).resolve().parent  # 计算当前 Python 文件所在的 code 目录。

ENV_PATH = CODE_DIR / ".env"  # 拼出 .env 配置文件路径。

load_dotenv(ENV_PATH)  # 加载 .env 文件，让其中的配置进入环境变量。
