"""
知识库构建脚本

将 legal_docs/ 和 risk_templates/ 下的文档进行分块、向量化，
构建本地向量知识库，供 risk_checker 工具进行RAG检索。

RAG流程：文档加载 -> 文本分块 -> 向量化 -> 存储到本地

运行方式：python knowledge/build_kb.py
"""

import json
import logging
import os
import sys

# 确保项目根目录在导入路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_openai_client, get_model_config

logger = logging.getLogger(__name__)

# 知识库存储路径
KB_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(KB_DIR, 'vector_store')
LEGAL_DOCS_DIR = os.path.join(KB_DIR, 'legal_docs')
RISK_TEMPLATES_DIR = os.path.join(KB_DIR, 'risk_templates')

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 10


def load_documents() -> list[dict]:
    """
    加载所有文档

    遍历 legal_docs/ 和 risk_templates/ 目录，读取所有 .txt 文件。
    返回文档列表，每个文档包含内容和来源信息。
    """
    documents = []
    for directory in [LEGAL_DOCS_DIR, RISK_TEMPLATES_DIR]:
        if not os.path.exists(directory):
            logger.info("目录不存在，跳过: %s", directory)
            continue
        for filename in os.listdir(directory):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append({
                'content': content,
                'source': filename,
                'directory': os.path.basename(directory),
            })
            logger.info("已加载: %s (%d 字)", filename, len(content))

    logger.info("共加载 %d 个文档", len(documents))
    return documents


def split_into_chunks(documents: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    将文档分块

    分块策略：
    1. 先按段落分割（双换行符）
    2. 如果段落超过chunk_size，再按chunk_size切分
    3. 相邻块之间有overlap个字符的重叠，防止切块边界处信息丢失

    为什么这样做：
    - 按段落分块保持了语义完整性
    - overlap确保即使关键信息恰好在切块边界也不会丢失
    - chunk_size=500 是中文法律文档的经验值，太大检索不精确，太小语义不完整
    """
    chunks = []
    for doc in documents:
        # 先按段落分割
        paragraphs = doc['content'].split('\n\n')

        current_chunk = ''
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落不超过chunk_size，合并
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += ('\n\n' + para) if current_chunk else para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'source': doc['source'],
                        'directory': doc['directory'],
                    })

                # 如果单个段落就超过chunk_size，需要再切分
                if len(para) > chunk_size:
                    for i in range(0, len(para), chunk_size - overlap):
                        sub_chunk = para[i:i + chunk_size]
                        if sub_chunk.strip():
                            chunks.append({
                                'text': sub_chunk,
                                'source': doc['source'],
                                'directory': doc['directory'],
                            })
                    current_chunk = ''
                else:
                    current_chunk = para

        # 保存最后一个块
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'source': doc['source'],
                'directory': doc['directory'],
            })

    logger.info("文档分块完成: %d 个块", len(chunks))
    return chunks


def generate_embeddings(chunks: list[dict]) -> list[list[float]]:
    """
    为每个文本块生成向量

    调用DashScope的 text-embedding-v3 模型，将文本转换为1024维的向量。
    DashScope的embedding API支持批量调用，每次最多25条，这里分批处理。
    """
    client = get_openai_client()
    config = get_model_config()

    embeddings = []
    batch_size = BATCH_SIZE  # DashScope限制每批最多10条

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]

        logger.info("正在向量化: %d-%d/%d", i + 1, min(i + batch_size, len(chunks)), len(chunks))

        response = client.embeddings.create(
            model=config['embedding_model'],
            input=texts,
        )

        for item in response.data:
            embeddings.append(item.embedding)

    logger.info("向量化完成: %d 个向量", len(embeddings))
    return embeddings


def save_vector_store(chunks: list[dict], embeddings: list[list[float]]) -> str:
    """
    保存向量知识库到本地文件

    存储格式：JSON文件，包含文本块和对应的向量。
    生产环境会用专门的向量数据库（如Milvus、Faiss），
    学习阶段用JSON文件存储，简单直观。
    """
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    store = {
        'chunks': chunks,
        'embeddings': embeddings,
        'metadata': {
            'total_chunks': len(chunks),
            'embedding_dim': len(embeddings[0]) if embeddings else 0,
            'embedding_model': get_model_config()['embedding_model'],
        }
    }

    store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
    with open(store_path, 'w', encoding='utf-8') as f:
        json.dump(store, f, ensure_ascii=False)

    logger.info("知识库已保存到: %s", store_path)
    logger.info("  文本块数量: %d", len(chunks))
    logger.info("  向量维度: %d", store['metadata']['embedding_dim'])

    return store_path


def search_similar(query: str, top_k: int = 3) -> list[dict]:
    """
    检索与查询最相关的文本块

    使用余弦相似度计算查询向量和知识库中所有向量的相似度，
    返回最相似的top_k个文本块。

    这个函数供 risk_checker 工具调用，实现RAG中的"R"（Retrieval）。
    """
    import numpy as np

    # 加载知识库
    store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
    if not os.path.exists(store_path):
        return []

    with open(store_path, 'r', encoding='utf-8') as f:
        store = json.load(f)

    # 为查询文本生成向量
    client = get_openai_client()
    config = get_model_config()
    response = client.embeddings.create(
        model=config['embedding_model'],
        input=query,
    )
    query_embedding = np.array(response.data[0].embedding)

    # 计算余弦相似度
    # 公式：cos(A, B) = (A·B) / (|A| * |B|)
    kb_embeddings = np.array(store['embeddings'])
    # 归一化
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    kb_norms = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
    # 点积计算相似度
    similarities = np.dot(kb_norms, query_norm)

    # 获取top_k个最相似的索引
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'text': store['chunks'][idx]['text'],
            'source': store['chunks'][idx]['source'],
            'similarity': float(similarities[idx]),
        })

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logger.info("=" * 50)
    logger.info("知识库构建开始")
    logger.info("=" * 50)

    # 步骤1: 加载文档
    logger.info("步骤1: 加载文档")
    documents = load_documents()

    # 步骤2: 文本分块
    logger.info("步骤2: 文本分块")
    chunks = split_into_chunks(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # 步骤3: 生成向量
    logger.info("步骤3: 生成向量（调用DashScope Embedding API）")
    embeddings = generate_embeddings(chunks)

    # 步骤4: 保存知识库
    logger.info("步骤4: 保存知识库")
    save_vector_store(chunks, embeddings)

    # 步骤5: 验证检索效果
    logger.info("步骤5: 验证检索效果")
    test_queries = [
        "违约金过高怎么处理",
        "合同自动续约的风险",
        "知识产权归属如何约定",
    ]
    for query in test_queries:
        logger.info("查询: %s", query)
        results = search_similar(query, top_k=2)
        for i, r in enumerate(results):
            logger.info("  结果%d [相似度: %.4f] 来源: %s",
                        i + 1, r['similarity'], r['source'])
            # 只显示前100个字
            logger.info("  内容: %s...", r['text'][:100])

    logger.info("知识库构建完成!")
