"""
Reranker 重排序模块

对 RAG 初步检索的候选文档进行精细重排，提升检索精度。
初步检索追求召回率（取 top-20），Reranker 追求精确率（取 top-3）。

支持两种 Reranker 实现：
1. LLM-as-Reranker: 使用千问 API 对 query-doc 对打相关性分数（零额外依赖）
2. CrossEncoder Reranker: 使用本地 BGE/Cross-encoder 模型（需安装 sentence-transformers）

参考：MathGraph/50-AI-Agent/RAG/Reranking-Guide.md
"""

import json
import logging
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Reranker 协议接口，支持不同实现的统一调用。"""

    def rerank(self, query: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        """对候选文档重排序。

        Args:
            query: 用户查询文本。
            documents: 初步检索的候选文档列表，每个文档至少包含 'text' 字段。
            top_n: 返回的最终文档数量。

        Returns:
            重排序后的 top_n 个文档列表，新增 'rerank_score' 字段。
        """
        ...


class LLMReranker:
    """使用 LLM 进行重排序。

    将 query 和每个候选文档拼接后，让 LLM 打相关性分数（0-10）。
    优势：零额外依赖，直接复用已有的千问 API。
    劣势：每个文档需要一次 LLM 调用，候选数不宜太多（建议 10-20 个）。
    """

    def __init__(self, client: Any = None, model: str = ''):
        if client is None:
            from config.model_config import get_openai_client, get_model_config
            self.client = get_openai_client()
            self.model = get_model_config()['model']
        else:
            self.client = client
            self.model = model

    def rerank(self, query: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        """使用 LLM 对候选文档评分并重排序。

        Args:
            query: 用户查询。
            documents: 候选文档列表。
            top_n: 返回数量。

        Returns:
            重排序后的文档列表。
        """
        if not documents:
            return []

        scored_docs = []
        for i, doc in enumerate(documents):
            score = self._score_relevance(query, doc['text'])
            scored_doc = dict(doc)
            scored_doc['rerank_score'] = score
            scored_docs.append(scored_doc)
            logger.info("  Rerank [%d/%d] score=%.1f source=%s",
                        i + 1, len(documents), score, doc.get('source', ''))

        # 按 rerank_score 降序排列
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_docs[:top_n]

    def _score_relevance(self, query: str, document: str) -> float:
        """让 LLM 对 query-document 对打相关性分数。

        Args:
            query: 查询文本。
            document: 文档文本。

        Returns:
            相关性分数（0-10）。
        """
        # 截断过长的文档（reranker 通常限制 512 tokens）
        if len(document) > 800:
            document = document[:800]

        prompt = (
            f"请评估以下查询和文档的相关性。\n\n"
            f"查询: {query}\n\n"
            f"文档: {document}\n\n"
            f"请给出一个0到10之间的相关性分数，"
            f"10表示完全相关，0表示完全无关。只回复数字。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(10.0, score))
        except (ValueError, TypeError) as e:
            logger.warning("Rerank 评分解析失败: %s", e)
            return 5.0  # 解析失败给中间分
        except Exception as e:
            logger.warning("Rerank API 调用失败: %s", e)
            return 5.0


class CrossEncoderReranker:
    """使用本地 Cross-encoder 模型进行重排序。

    需要安装: pip install sentence-transformers
    推荐模型: BAAI/bge-reranker-v2-m3（中文效果好，免费）

    参考：MathGraph/50-AI-Agent/RAG/Reranking-Guide.md
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info("CrossEncoder Reranker 已加载: %s", model_name)
        except ImportError:
            raise ImportError(
                "使用 CrossEncoderReranker 需要安装 sentence-transformers:\n"
                "  pip install sentence-transformers"
            )

    def rerank(self, query: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        """使用 Cross-encoder 对候选文档评分并重排序。

        Args:
            query: 用户查询。
            documents: 候选文档列表。
            top_n: 返回数量。

        Returns:
            重排序后的文档列表。
        """
        if not documents:
            return []

        pairs = [[query, doc['text'][:512]] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = dict(doc)
            scored_doc['rerank_score'] = float(score)
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_docs[:top_n]


def get_reranker(mode: str = 'llm', **kwargs) -> LLMReranker | CrossEncoderReranker:
    """获取 Reranker 实例。

    Args:
        mode: 'llm' 使用 LLM-as-Reranker，'cross_encoder' 使用本地模型。
        **kwargs: 传递给具体 Reranker 的参数。

    Returns:
        Reranker 实例。
    """
    if mode == 'cross_encoder':
        return CrossEncoderReranker(**kwargs)
    return LLMReranker(**kwargs)
