"""
知识库构建脚本

将 legal_docs/ 和 risk_templates/ 下的文档进行分块、向量化，
构建本地向量知识库，供 risk_checker 工具进行RAG检索。

支持两种检索模式：
1. Dense Retrieval: 纯向量余弦相似度（原始方案）
2. Hybrid Retrieval: BM25 稀疏检索 + Dense 稠密检索 + RRF 融合（升级方案）

RAG流程：文档加载 -> 文本分块 -> 向量化 + BM25索引 -> 存储到本地

运行方式：python knowledge/build_kb.py
"""

import json
import logging
import math
import os
import re
import sys
from collections import Counter
from typing import Optional

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
RRF_K = 60  # Reciprocal Rank Fusion 常数，标准值为 60


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


def _tokenize_chinese(text: str) -> list[str]:
    """中文分词（基于字符和标点切分，无需外部依赖）。

    对中文文本按单字切分，对英文和数字按空格/标点切分。
    这是一个轻量级分词方案，适合 BM25 等统计检索模型。

    Args:
        text: 输入文本。

    Returns:
        分词后的 token 列表。
    """
    # 去除标点和特殊字符，保留中文字符、英文字母和数字
    tokens = []
    current_word = ''
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            # 中文字符：先保存累积的英文词，然后单字切分
            if current_word:
                tokens.append(current_word.lower())
                current_word = ''
            tokens.append(char)
        elif char.isalnum():
            current_word += char
        else:
            if current_word:
                tokens.append(current_word.lower())
                current_word = ''
    if current_word:
        tokens.append(current_word.lower())
    return tokens


class BM25Index:
    """BM25 稀疏检索索引。

    实现 Okapi BM25 算法，支持中文法律文档的关键词匹配检索。
    与 Dense Retrieval 互补：BM25 擅长精确的关键词/法条编号匹配，
    Dense Retrieval 擅长语义相似度匹配。

    Attributes:
        k1: 词频饱和参数，控制 TF 的增长速度。默认 1.5。
        b: 文档长度归一化参数。默认 0.75。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}  # 每个 token 出现在几个文档中
        self.tf_cache: list[Counter] = []     # 每个文档的词频

    def build(self, texts: list[str]) -> None:
        """构建 BM25 索引。

        Args:
            texts: 文档文本列表。
        """
        self.doc_count = len(texts)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.tf_cache = []

        total_len = 0
        for text in texts:
            tokens = _tokenize_chinese(text)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            tf = Counter(tokens)
            self.tf_cache.append(tf)

            # 统计文档频率
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_len / max(self.doc_count, 1)
        logger.info("BM25索引构建完成: %d 个文档, 词表大小: %d",
                     self.doc_count, len(self.doc_freqs))

    def score(self, query: str) -> list[float]:
        """计算查询与所有文档的 BM25 分数。

        Args:
            query: 查询文本。

        Returns:
            每个文档的 BM25 分数列表。
        """
        query_tokens = _tokenize_chinese(query)
        scores = [0.0] * self.doc_count

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            df = self.doc_freqs[token]
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self.doc_count):
                tf = self.tf_cache[i].get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[i]
                # BM25 TF 计算
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )
                scores[i] += idf * tf_norm

        return scores

    def to_dict(self) -> dict:
        """序列化为字典，用于持久化存储。"""
        return {
            'k1': self.k1,
            'b': self.b,
            'doc_count': self.doc_count,
            'avg_doc_len': self.avg_doc_len,
            'doc_lengths': self.doc_lengths,
            'doc_freqs': self.doc_freqs,
            'tf_cache': [dict(tf) for tf in self.tf_cache],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BM25Index':
        """从字典反序列化。"""
        index = cls(k1=data['k1'], b=data['b'])
        index.doc_count = data['doc_count']
        index.avg_doc_len = data['avg_doc_len']
        index.doc_lengths = data['doc_lengths']
        index.doc_freqs = data['doc_freqs']
        index.tf_cache = [Counter(tf) for tf in data['tf_cache']]
        return index


def _reciprocal_rank_fusion(
    dense_ranking: list[int],
    sparse_ranking: list[int],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion 融合两路检索结果。

    RRF 公式：score(d) = sum(1 / (k + rank_i(d)))
    其中 k 是常数（通常为 60），rank_i 是文档在第 i 路排名中的位置。

    Args:
        dense_ranking: Dense 检索的文档索引列表（按相似度降序）。
        sparse_ranking: BM25 检索的文档索引列表（按分数降序）。
        k: RRF 常数，默认 60。

    Returns:
        融合后的 (文档索引, RRF分数) 列表，按分数降序排列。
    """
    rrf_scores: dict[int, float] = {}

    for rank, doc_idx in enumerate(dense_ranking):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_idx in enumerate(sparse_ranking):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    # 按 RRF 分数降序排列
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def search_similar(query: str, top_k: int = 3) -> list[dict]:
    """检索与查询最相关的文本块（纯向量检索，向后兼容）。

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
    kb_embeddings = np.array(store['embeddings'])
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    kb_norms = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
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


def search_hybrid(query: str, top_k: int = 3, dense_weight: float = 0.5) -> list[dict]:
    """混合检索：BM25 + Dense Retrieval + RRF 融合。

    结合两种检索方法的优势：
    - BM25 擅长精确的关键词/法条编号匹配（如"第三百八十六条"、"30%"）
    - Dense Retrieval 擅长语义相似度匹配（如"违约金过高"匹配"约定的违约金超过损失"）

    使用 Reciprocal Rank Fusion 融合两路排名结果。

    Args:
        query: 查询文本。
        top_k: 返回的最相似文档块数量。
        dense_weight: Dense 检索的权重（0-1），用于调整两路的侧重。

    Returns:
        融合排序后的检索结果列表。
    """
    import numpy as np

    # 加载知识库
    store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
    if not os.path.exists(store_path):
        return []

    with open(store_path, 'r', encoding='utf-8') as f:
        store = json.load(f)

    chunks = store['chunks']
    retrieve_top = min(len(chunks), top_k * 4)  # 两路各取更多候选再融合

    # ===== Dense Retrieval =====
    client = get_openai_client()
    config = get_model_config()
    response = client.embeddings.create(
        model=config['embedding_model'],
        input=query,
    )
    query_embedding = np.array(response.data[0].embedding)

    kb_embeddings = np.array(store['embeddings'])
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    kb_norms = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
    dense_similarities = np.dot(kb_norms, query_norm)

    dense_ranking = np.argsort(dense_similarities)[::-1][:retrieve_top].tolist()

    # ===== BM25 Sparse Retrieval =====
    bm25_path = os.path.join(VECTOR_STORE_PATH, 'bm25_index.json')
    if os.path.exists(bm25_path):
        with open(bm25_path, 'r', encoding='utf-8') as f:
            bm25_index = BM25Index.from_dict(json.load(f))
    else:
        # 如果 BM25 索引不存在，动态构建
        logger.info("BM25索引不存在，动态构建中...")
        bm25_index = BM25Index()
        bm25_index.build([c['text'] for c in chunks])

    bm25_scores = bm25_index.score(query)
    sparse_ranking = sorted(range(len(bm25_scores)),
                            key=lambda i: bm25_scores[i],
                            reverse=True)[:retrieve_top]

    # ===== RRF Fusion =====
    fused_results = _reciprocal_rank_fusion(dense_ranking, sparse_ranking)

    # 取 top_k 个结果
    results = []
    for doc_idx, rrf_score in fused_results[:top_k]:
        results.append({
            'text': chunks[doc_idx]['text'],
            'source': chunks[doc_idx]['source'],
            'similarity': float(dense_similarities[doc_idx]),
            'bm25_score': float(bm25_scores[doc_idx]),
            'rrf_score': float(rrf_score),
        })

    return results


def _check_retrieval_quality(query: str, results: list[dict], threshold: float = 0.35) -> bool:
    """评估检索结果质量是否足够回答查询。

    Corrective RAG 的核心：不盲目信任检索结果。
    通过相似度阈值判断检索到的文档是否真正相关。

    Args:
        query: 原始查询文本。
        results: 检索结果列表。
        threshold: 相似度阈值，低于此值视为低质量。

    Returns:
        True 表示质量合格，False 表示需要改写查询重试。
    """
    if not results:
        return False

    # 检查最高相似度是否达标
    max_sim = max(r.get('similarity', 0) for r in results)
    if max_sim < threshold:
        logger.info("[CRAG] 检索质量不足: 最高相似度 %.4f < 阈值 %.2f", max_sim, threshold)
        return False

    # 检查是否有 BM25 分数为正的结果（至少有关键词匹配）
    bm25_matches = sum(1 for r in results if r.get('bm25_score', 0) > 0)
    if bm25_matches == 0:
        logger.info("[CRAG] 无 BM25 关键词匹配，可能存在词汇鸿沟")
        return False

    return True


def _rewrite_query(query: str) -> list[str]:
    """用 LLM 改写查询，生成多个检索变体（Multi-Query）。

    融合了 HyDE-Query-Rewrite 笔记中的 Multi-Query 技术：
    从不同角度生成查询变体，提升检索召回率。

    Args:
        query: 原始查询文本。

    Returns:
        改写后的查询列表（不含原始查询）。
    """
    client = get_openai_client()
    config = get_model_config()

    prompt = (
        f"你是法律检索查询优化专家。请将以下法律相关查询改写为3个不同角度的检索查询，"
        f"以提升在法律文档知识库中的检索召回率。\n\n"
        f"原始查询: {query}\n\n"
        f"要求:\n"
        f"1. 每个查询从不同角度表述（同义词、上位概念、具体法条等）\n"
        f"2. 适合中文法律文档检索\n"
        f"3. 每行一个查询，不要编号\n"
        f"4. 只输出查询，不要其他内容"
    )

    try:
        response = client.chat.completions.create(
            model=config['model'],
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        queries = [
            q.strip() for q in response.choices[0].message.content.strip().split('\n')
            if q.strip()
        ]
        logger.info("[CRAG] 查询改写: %s -> %s", query, queries)
        return queries[:3]
    except Exception as e:
        logger.warning("[CRAG] 查询改写失败: %s", e)
        return []


def search_corrective(
    query: str,
    top_k: int = 3,
    max_retries: int = 1,
    quality_threshold: float = 0.35,
) -> list[dict]:
    """Corrective RAG：检索 + 质量自检 + 查询改写重试。

    流程（参考 MathGraph/50-AI-Agent/RAG/Corrective-RAG.md）：
    1. 使用混合检索获取初始结果
    2. 评估检索质量（相似度阈值 + BM25匹配度）
    3. 如果质量不足，用 LLM 改写查询生成多个变体
    4. 用改写后的查询重新检索，合并去重取最优

    Args:
        query: 查询文本。
        top_k: 返回的最终结果数量。
        max_retries: 最大重试次数。
        quality_threshold: 检索质量阈值。

    Returns:
        经过质量校验的检索结果列表。
    """
    # 第一次检索
    results = search_hybrid(query, top_k=top_k)

    if _check_retrieval_quality(query, results, threshold=quality_threshold):
        logger.info("[CRAG] 检索质量合格，直接返回")
        return results

    # 质量不足，进入纠正流程
    for retry in range(max_retries):
        logger.info("[CRAG] 第 %d 次纠正检索...", retry + 1)

        rewritten_queries = _rewrite_query(query)
        if not rewritten_queries:
            break

        # 用改写后的查询分别检索，合并结果
        all_results: dict[str, dict] = {}  # text -> result，用于去重
        for r in results:
            all_results[r['text']] = r

        for rq in rewritten_queries:
            new_results = search_hybrid(rq, top_k=top_k)
            for r in new_results:
                key = r['text']
                if key not in all_results or r.get('rrf_score', 0) > all_results[key].get('rrf_score', 0):
                    all_results[key] = r

        # 按 RRF 分数重新排序
        results = sorted(
            all_results.values(),
            key=lambda x: x.get('rrf_score', 0),
            reverse=True,
        )[:top_k]

        if _check_retrieval_quality(query, results, threshold=quality_threshold):
            logger.info("[CRAG] 纠正后检索质量合格")
            break

    return results


def search_with_rerank(query: str, top_k: int = 3, rerank_candidates: int = 10) -> list[dict]:
    """完整 RAG 检索管线：Hybrid Retrieval + Reranking。

    三阶段检索：
    1. 混合检索（BM25 + Dense + RRF）取 rerank_candidates 个候选
    2. Reranker 对候选文档精排
    3. 返回 top_k 个最终结果

    这是项目中最高质量的检索方案，适合对精度要求高的场景。

    Args:
        query: 查询文本。
        top_k: 最终返回的文档数量。
        rerank_candidates: 送入 Reranker 的候选文档数量。

    Returns:
        经过 Hybrid + Rerank 的最终检索结果。
    """
    # 阶段1: Corrective RAG 混合检索（含质量自检和查询改写）
    candidates = search_corrective(query, top_k=rerank_candidates)
    if not candidates:
        return []

    # 阶段2: Reranking 精排
    try:
        from knowledge.reranker import LLMReranker
        reranker = LLMReranker()
        reranked = reranker.rerank(query, candidates, top_n=top_k)
        logger.info("Reranking 完成: %d -> %d", len(candidates), len(reranked))
        return reranked
    except Exception as e:
        logger.warning("Reranking 失败，回退到混合检索结果: %s", e)
        return candidates[:top_k]


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
    logger.info("步骤4: 保存向量知识库")
    save_vector_store(chunks, embeddings)

    # 步骤5: 构建BM25索引
    logger.info("步骤5: 构建BM25稀疏检索索引")
    bm25_index = BM25Index()
    bm25_index.build([c['text'] for c in chunks])
    bm25_path = os.path.join(VECTOR_STORE_PATH, 'bm25_index.json')
    with open(bm25_path, 'w', encoding='utf-8') as f:
        json.dump(bm25_index.to_dict(), f, ensure_ascii=False)
    logger.info("BM25索引已保存到: %s", bm25_path)

    # 步骤6: 验证检索效果（对比纯向量 vs 混合检索）
    logger.info("步骤6: 验证检索效果")
    test_queries = [
        "违约金过高怎么处理",
        "合同自动续约的风险",
        "知识产权归属如何约定",
    ]
    for query in test_queries:
        logger.info("查询: %s", query)

        logger.info("  [纯向量检索]")
        results = search_similar(query, top_k=2)
        for i, r in enumerate(results):
            logger.info("    结果%d [相似度: %.4f] 来源: %s",
                        i + 1, r['similarity'], r['source'])

        logger.info("  [混合检索 BM25+Dense+RRF]")
        results = search_hybrid(query, top_k=2)
        for i, r in enumerate(results):
            logger.info("    结果%d [RRF: %.6f, Dense: %.4f, BM25: %.4f] 来源: %s",
                        i + 1, r['rrf_score'], r['similarity'],
                        r['bm25_score'], r['source'])

    logger.info("知识库构建完成!")
