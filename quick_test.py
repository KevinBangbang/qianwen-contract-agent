"""
DashScope API 连通性测试脚本

验证三个核心能力：
1. 文本生成（qwen-plus）
2. 文本向量化（text-embedding-v3）
3. 多模态图片理解（qwen-vl-plus）

运行方式：
    先配置 .env 文件中的 DASHSCOPE_API_KEY
    然后执行: python quick_test.py
"""

import logging
import sys

from config.model_config import get_openai_client, get_model_config

logger = logging.getLogger(__name__)


def test_text_generation() -> bool:
    """
    测试1: 文本生成

    使用OpenAI兼容接口调用千问模型，验证基本的对话能力。
    这是最基础的测试，如果这个通不过，说明API Key或网络有问题。
    """
    logger.info("=" * 50)
    logger.info("测试1: 文本生成")
    logger.info("=" * 50)

    client = get_openai_client()
    config = get_model_config()

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "你是一个专业的法律助手。"},
                {"role": "user", "content": "请用一句话解释什么是合同违约责任。"},
            ],
            # temperature控制输出的随机性，0.7是比较平衡的值
            temperature=0.7,
            max_tokens=200,
        )
        answer = response.choices[0].message.content
        logger.info("模型: %s", config['model'])
        logger.info("回答: %s", answer)
        logger.info("文本生成测试通过!\n")
        return True
    except Exception as e:
        logger.info("文本生成测试失败: %s\n", e)
        return False


def test_embedding() -> bool:
    """
    测试2: 文本向量化

    调用Embedding模型将文本转换为向量，这是RAG的基础能力。
    向量维度应该是1024（text-embedding-v3的默认维度）。
    """
    logger.info("=" * 50)
    logger.info("测试2: 文本向量化 (Embedding)")
    logger.info("=" * 50)

    client = get_openai_client()
    config = get_model_config()

    try:
        response = client.embeddings.create(
            model=config["embedding_model"],
            input="合同违约责任条款",
        )
        embedding = response.data[0].embedding
        logger.info("模型: %s", config['embedding_model'])
        logger.info("向量维度: %d", len(embedding))
        logger.info("向量前5个值: %s", embedding[:5])
        logger.info("向量化测试通过!\n")
        return True
    except Exception as e:
        logger.info("向量化测试失败: %s\n", e)
        return False


def test_vision() -> bool:
    """
    测试3: 多模态图片理解

    调用VL模型识别图片内容，用于合同扫描件的OCR场景。
    这里用一个公开的测试图片URL来验证。
    """
    logger.info("=" * 50)
    logger.info("测试3: 多模态图片理解 (Vision)")
    logger.info("=" * 50)

    client = get_openai_client()
    config = get_model_config()

    try:
        response = client.chat.completions.create(
            model=config["vl_model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请描述这张图片的内容。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                # 使用千问官方示例图片进行测试
                                "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
                            },
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        answer = response.choices[0].message.content
        logger.info("模型: %s", config['vl_model'])
        logger.info("图片描述: %s", answer)
        logger.info("多模态测试通过!\n")
        return True
    except Exception as e:
        logger.info("多模态测试失败: %s\n", e)
        return False


def test_streaming() -> bool:
    """
    测试4: 流式输出

    验证流式输出能力，这对前端实时展示Agent推理过程很重要。
    """
    logger.info("=" * 50)
    logger.info("测试4: 流式输出 (Streaming)")
    logger.info("=" * 50)

    client = get_openai_client()
    config = get_model_config()

    try:
        stream = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "user", "content": "用三句话介绍中国合同法的核心原则。"},
            ],
            stream=True,
            max_tokens=300,
        )
        logger.info("模型: %s", config['model'])
        sys.stdout.write("流式输出: ")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                sys.stdout.write(chunk.choices[0].delta.content)
                sys.stdout.flush()
        logger.info("\n流式输出测试通过!\n")
        return True
    except Exception as e:
        logger.info("流式输出测试失败: %s\n", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger.info("\n智能合同审查Agent - DashScope API 连通性测试\n")

    config = get_model_config()
    logger.info("当前模式: %s", config['mode'])
    logger.info("API地址: %s", config['base_url'])
    logger.info("")

    results = {
        "文本生成": test_text_generation(),
        "文本向量化": test_embedding(),
        "多模态理解": test_vision(),
        "流式输出": test_streaming(),
    }

    # 汇总测试结果
    logger.info("=" * 50)
    logger.info("测试结果汇总")
    logger.info("=" * 50)
    all_passed = True
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        logger.info("  %s: %s", name, status)
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n所有测试通过! 可以开始 Phase 2 的开发。")
    else:
        logger.info("\n部分测试失败，请检查：")
        logger.info("  1. .env 文件中的 DASHSCOPE_API_KEY 是否正确")
        logger.info("  2. 网络是否能访问阿里云 DashScope")
        logger.info("  3. 如在海外，尝试使用国际版endpoint:")
        logger.info("     https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        sys.exit(1)
