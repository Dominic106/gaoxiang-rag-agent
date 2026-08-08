#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" # 计算当前安装脚本所在目录，避免依赖本机绝对路径。
cd "$SCRIPT_DIR" # 进入代码目录，后续虚拟环境和依赖都放在这里。

python3 -m venv .venv # 创建一个独立 Python 虚拟环境，避免污染你系统里的 Python 包。

source .venv/bin/activate # 激活虚拟环境，后面安装的包都会进入 .venv。

python -m pip install --upgrade pip # 先升级 pip，减少安装新包时的兼容问题。

python -m pip install -r requirements.txt # 按 requirements.txt 安装 LangChain、LangGraph、Chroma 等依赖。

echo "依赖安装完成。" # 提示依赖安装已经结束。

echo "如需完整 RAG，请复制 .env.example 为 .env，并填写豆包 Embedding 和 DeepSeek API 配置。" # 提醒用户使用本地环境变量配置，而不是把密钥写入代码。
