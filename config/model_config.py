"""
模型配置模块

管理云端（DashScope）和本地（Ollama/vLLM）两种模型的配置。
所有模型调用都使用OpenAI兼容接口，这样切换云端和本地只需要改base_url和model名称，
业务代码完全不用动。这也是TAM推荐客户使用的最佳实践。
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
load_dotenv()


def get_model_config() -> dict[str, str]:
    """
    获取当前模型配置

    根据 MODEL_MODE 环境变量决定使用云端还是本地模型。
    返回一个字典，包含所有模型调用所需的参数。

    返回字段说明：
    - mode: 运行模式，cloud或local
    - api_key: API密钥，本地模型不需要真实密钥
    - base_url: API接口地址
    - model: 文本生成模型名称
    - vl_model: 视觉语言模型名称（用于OCR）
    - embedding_model: 文本向量化模型名称（用于RAG）
    """
    mode = os.getenv("MODEL_MODE", "cloud")

    if mode == "cloud":
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key or api_key.startswith("sk-xxx"):
            raise ValueError(
                "云端模式需要有效的 DASHSCOPE_API_KEY，"
                "请在 .env 文件中设置真实的API密钥"
            )
        return {
            "mode": "cloud",
            # DashScope的API Key，从环境变量读取
            "api_key": api_key,
            # 百炼平台的OpenAI兼容接口地址
            # 注意：base_url根据账号所在地域不同而不同
            # 北京地域: https://dashscope.aliyuncs.com/compatible-mode/v1
            # 美国弗吉尼亚: https://dashscope-us.aliyuncs.com/compatible-mode/v1
            # 通过环境变量 DASHSCOPE_BASE_URL 可以覆盖默认值
            "base_url": os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            ),
            # qwen-plus 是性价比最高的选择，能力和成本平衡
            "model": os.getenv("CLOUD_MODEL", "qwen-plus"),
            # qwen-vl-plus 支持图片理解，用于合同扫描件OCR
            "vl_model": os.getenv("CLOUD_VL_MODEL", "qwen-vl-plus"),
            # text-embedding-v3 是最新的向量模型，维度1024
            "embedding_model": os.getenv("CLOUD_EMBEDDING", "text-embedding-v3"),
        }
    else:
        return {
            "mode": "local",
            # 本地模型不需要真实的API Key，但openai库要求传一个值
            "api_key": "not-needed",
            # Ollama默认监听11434端口，vLLM默认8000端口
            "base_url": os.getenv(
                "LOCAL_MODEL_SERVER", "http://localhost:11434/v1"
            ),
            "model": os.getenv("LOCAL_MODEL", "qwen3-7b"),
            # 本地VL模型，需要单独下载
            "vl_model": os.getenv("LOCAL_VL_MODEL", "qwen2.5-vl-7b"),
            # 本地Embedding模型
            "embedding_model": os.getenv(
                "LOCAL_EMBEDDING", "nomic-embed-text"
            ),
        }


def get_openai_client() -> Any:
    """
    创建OpenAI兼容的客户端实例

    无论是云端DashScope还是本地Ollama/vLLM，
    都通过这个统一接口调用，业务代码无需关心底层差异。
    """
    from openai import OpenAI

    config = get_model_config()
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


# 方便其他模块直接导入使用
if __name__ == "__main__":
    config = get_model_config()
    logger.info("当前模式: %s", config['mode'])
    logger.info("API地址: %s", config['base_url'])
    logger.info("文本模型: %s", config['model'])
    logger.info("视觉模型: %s", config['vl_model'])
    logger.info("向量模型: %s", config['embedding_model'])
