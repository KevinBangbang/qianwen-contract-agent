"""
RAG 进阶功能测试

测试 Hybrid Retrieval、BM25、RRF Fusion、Reranker、Reflexion 等新增模块。
分为两类：
1. 离线测试：不调用 API，测试 BM25、RRF、分词等纯逻辑功能
2. 在线测试：需要 DashScope API，测试完整检索管线

运行方式：
    pytest tests/test_rag_advanced.py -v                # 运行所有测试
    pytest tests/test_rag_advanced.py -v -k "offline"    # 只运行离线测试
    pytest tests/test_rag_advanced.py -v -k "online"     # 只运行在线测试
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 离线测试：BM25、RRF、分词
# ============================================================

class TestTokenizerOffline:
    """中文分词器测试"""

    def test_chinese_tokenization(self):
        """测试纯中文文本分词"""
        from knowledge.build_kb import _tokenize_chinese
        tokens = _tokenize_chinese("违约金过高")
        assert len(tokens) == 5, f"期望5个token，实际得到{len(tokens)}: {tokens}"
        assert tokens == ["违", "约", "金", "过", "高"]

    def test_mixed_tokenization(self):
        """测试中英混合文本分词"""
        from knowledge.build_kb import _tokenize_chinese
        tokens = _tokenize_chinese("合同金额350万元")
        assert "350" in tokens, f"期望包含'350'，实际: {tokens}"
        assert "万" in tokens, f"期望包含'万'，实际: {tokens}"

    def test_empty_text(self):
        """测试空文本分词"""
        from knowledge.build_kb import _tokenize_chinese
        tokens = _tokenize_chinese("")
        assert tokens == [], f"空文本应返回空列表，实际: {tokens}"

    def test_punctuation_removal(self):
        """测试标点符号被正确过滤"""
        from knowledge.build_kb import _tokenize_chinese
        tokens = _tokenize_chinese("合同（甲方）签订。")
        assert "（" not in tokens, f"标点不应出现在token中: {tokens}"
        assert "）" not in tokens, f"标点不应出现在token中: {tokens}"


class TestBM25Offline:
    """BM25 索引测试"""

    def setup_method(self):
        from knowledge.build_kb import BM25Index
        self.bm25 = BM25Index()
        self.docs = [
            "违约金超过合同标的额百分之三十的，一般可以认定为过高",
            "合同到期后自动续约一年，除非提前通知终止",
            "知识产权归属应当在合同中明确约定",
            "甲方应当在签署合同后十个工作日内支付首期款项",
        ]
        self.bm25.build(self.docs)

    def test_build_index(self):
        """测试索引构建"""
        assert self.bm25.doc_count == 4, f"期望4个文档，实际{self.bm25.doc_count}"
        assert self.bm25.avg_doc_len > 0, "平均文档长度应大于0"
        assert len(self.bm25.doc_freqs) > 0, "词表不应为空"

    def test_score_relevant_query(self):
        """测试相关查询的BM25分数"""
        scores = self.bm25.score("违约金过高")
        assert scores[0] > scores[1], (
            f"'违约金过高'应该与第一个文档最相关，"
            f"但得分: doc0={scores[0]:.4f}, doc1={scores[1]:.4f}"
        )

    def test_score_irrelevant_query(self):
        """测试不相关查询的分数较低"""
        scores = self.bm25.score("天气预报")
        assert all(s == 0.0 for s in scores), (
            f"不相关查询的分数应全为0，实际: {scores}"
        )

    def test_serialization(self):
        """测试BM25索引序列化和反序列化"""
        from knowledge.build_kb import BM25Index
        data = self.bm25.to_dict()
        restored = BM25Index.from_dict(data)
        assert restored.doc_count == self.bm25.doc_count, "反序列化后文档数不一致"

        # 验证分数一致
        original_scores = self.bm25.score("违约金")
        restored_scores = restored.score("违约金")
        for i in range(len(original_scores)):
            assert abs(original_scores[i] - restored_scores[i]) < 1e-6, (
                f"位置{i}分数不一致: {original_scores[i]} vs {restored_scores[i]}"
            )

    def test_keyword_matching(self):
        """测试BM25对精确关键词的匹配能力"""
        scores = self.bm25.score("知识产权归属")
        max_idx = scores.index(max(scores))
        assert max_idx == 2, (
            f"'知识产权归属'应该匹配第3个文档(idx=2)，实际匹配idx={max_idx}"
        )


class TestRRFOffline:
    """Reciprocal Rank Fusion 测试"""

    def test_rrf_basic(self):
        """测试基本RRF融合"""
        from knowledge.build_kb import _reciprocal_rank_fusion
        dense_ranking = [0, 1, 2, 3]
        sparse_ranking = [2, 0, 3, 1]

        results = _reciprocal_rank_fusion(dense_ranking, sparse_ranking, k=60)

        # 验证返回格式
        assert len(results) > 0, "RRF结果不应为空"
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results), (
            "RRF结果应为(doc_idx, score)元组列表"
        )

        # doc 0 在两路都排前列，应该分数最高
        doc_scores = {idx: score for idx, score in results}
        assert doc_scores[0] > doc_scores[3], (
            f"doc0应该比doc3分高，但doc0={doc_scores[0]:.6f}, doc3={doc_scores[3]:.6f}"
        )

    def test_rrf_identical_rankings(self):
        """测试两路排名完全一致的情况"""
        from knowledge.build_kb import _reciprocal_rank_fusion
        ranking = [0, 1, 2]
        results = _reciprocal_rank_fusion(ranking, ranking, k=60)

        # 排名应保持不变
        result_indices = [idx for idx, _ in results]
        assert result_indices == [0, 1, 2], (
            f"相同排名融合后应保持原顺序，实际: {result_indices}"
        )

    def test_rrf_complementary_rankings(self):
        """测试两路排名互补的情况"""
        from knowledge.build_kb import _reciprocal_rank_fusion
        dense = [0, 1, 2]
        sparse = [3, 4, 5]

        results = _reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(results) == 6, f"应包含6个文档，实际{len(results)}"


class TestRerankerOffline:
    """Reranker 模块离线测试"""

    def test_reranker_import(self):
        """测试 Reranker 模块可以正确导入"""
        from knowledge.reranker import LLMReranker, get_reranker
        assert LLMReranker is not None
        assert get_reranker is not None

    def test_get_reranker_llm_mode(self):
        """测试 get_reranker 工厂函数 LLM 模式"""
        from knowledge.reranker import get_reranker, LLMReranker
        # LLM 模式不需要额外依赖，但需要 API client
        # 这里只验证函数存在和接口正确
        assert callable(get_reranker)


class TestReflexionOffline:
    """Reflexion 模块离线测试"""

    def test_experience_save_load(self):
        """测试经验保存和加载"""
        import tempfile
        from agents import reflexion

        # 使用临时文件
        original_path = reflexion.EXPERIENCE_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = f.name
            reflexion.EXPERIENCE_PATH = temp_path

            # 保存经验
            reflexion.save_experience("审查时要注意检查自动续约条款")
            reflexion.save_experience("违约金条款需要对比30%阈值")

            # 加载经验
            experiences = reflexion.load_experiences()
            assert len(experiences) == 2, f"期望2条经验，实际{len(experiences)}"
            assert "自动续约" in experiences[0], f"第一条经验内容不正确: {experiences[0]}"
        finally:
            reflexion.EXPERIENCE_PATH = original_path
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_experience_context_format(self):
        """测试经验上下文格式化"""
        import tempfile
        from agents import reflexion

        original_path = reflexion.EXPERIENCE_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = f.name
            reflexion.EXPERIENCE_PATH = temp_path

            reflexion.save_experience("测试经验1")
            context = reflexion.get_experience_context()
            assert "经验教训" in context, f"上下文应包含'经验教训': {context}"
            assert "测试经验1" in context, f"上下文应包含经验内容: {context}"
        finally:
            reflexion.EXPERIENCE_PATH = original_path
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_empty_experience(self):
        """测试无历史经验的情况"""
        import tempfile
        from agents import reflexion

        original_path = reflexion.EXPERIENCE_PATH
        try:
            reflexion.EXPERIENCE_PATH = "/nonexistent/path.json"
            context = reflexion.get_experience_context()
            assert context == "", f"无经验时应返回空字符串，实际: {context}"
        finally:
            reflexion.EXPERIENCE_PATH = original_path

    def test_experience_max_limit(self):
        """测试经验数量上限"""
        import tempfile
        from agents import reflexion

        original_path = reflexion.EXPERIENCE_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_path = f.name
            reflexion.EXPERIENCE_PATH = temp_path

            # 保存超过上限的经验
            for i in range(15):
                reflexion.save_experience(f"经验{i}")

            experiences = reflexion.load_experiences()
            assert len(experiences) <= 10, (
                f"经验数量应不超过10条，实际{len(experiences)}"
            )
        finally:
            reflexion.EXPERIENCE_PATH = original_path
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestGoldenDatasetOffline:
    """黄金测试集格式验证"""

    def test_dataset_format(self):
        """验证黄金测试集格式正确"""
        dataset_path = os.path.join(
            os.path.dirname(__file__), 'rag_golden_dataset.json'
        )
        assert os.path.exists(dataset_path), f"黄金测试集不存在: {dataset_path}"

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'dataset' in data, "数据集应包含'dataset'字段"
        dataset = data['dataset']
        assert len(dataset) >= 10, f"至少需要10个评估样本，实际{len(dataset)}"

        for i, sample in enumerate(dataset):
            assert 'question' in sample, f"样本{i}缺少'question'字段"
            assert 'ground_truth_answer' in sample, f"样本{i}缺少'ground_truth_answer'字段"
            assert 'relevant_doc_sources' in sample, f"样本{i}缺少'relevant_doc_sources'字段"
            assert len(sample['relevant_doc_sources']) > 0, (
                f"样本{i}的relevant_doc_sources不应为空"
            )

    def test_dataset_sources_valid(self):
        """验证数据集引用的文档来源确实存在"""
        dataset_path = os.path.join(
            os.path.dirname(__file__), 'rag_golden_dataset.json'
        )
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取实际存在的文档文件名
        legal_docs_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'knowledge', 'legal_docs'
        )
        risk_templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'knowledge', 'risk_templates'
        )
        existing_files = set()
        for d in [legal_docs_dir, risk_templates_dir]:
            if os.path.exists(d):
                existing_files.update(os.listdir(d))

        # 验证每个引用的来源都存在
        for sample in data['dataset']:
            for source in sample['relevant_doc_sources']:
                assert source in existing_files, (
                    f"数据集引用了不存在的文档: {source}，"
                    f"可用文档: {existing_files}"
                )


# ============================================================
# 在线测试：需要 DashScope API
# ============================================================

has_api_key = bool(os.getenv('DASHSCOPE_API_KEY'))
online_skip_reason = '未配置 DASHSCOPE_API_KEY，跳过在线测试'


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestHybridRetrievalOnline:
    """混合检索在线测试"""

    def test_search_hybrid_returns_results(self):
        """测试混合检索能返回结果"""
        from knowledge.build_kb import search_hybrid, VECTOR_STORE_PATH

        store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
        if not os.path.exists(store_path):
            pytest.skip("知识库未构建，请先运行 python knowledge/build_kb.py")

        results = search_hybrid("违约金过高", top_k=3)
        assert len(results) > 0, "混合检索应返回结果"
        assert 'rrf_score' in results[0], "混合检索结果应包含rrf_score字段"
        assert 'bm25_score' in results[0], "混合检索结果应包含bm25_score字段"

    def test_hybrid_vs_dense(self):
        """测试混合检索的结果与纯向量检索有差异"""
        from knowledge.build_kb import search_similar, search_hybrid, VECTOR_STORE_PATH

        store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
        if not os.path.exists(store_path):
            pytest.skip("知识库未构建")

        dense_results = search_similar("第三百八十六条", top_k=3)
        hybrid_results = search_hybrid("第三百八十六条", top_k=3)

        # 对于精确法条编号，混合检索应该比纯向量更准
        # 这里只验证两种方法确实都能返回结果
        assert len(dense_results) > 0, "纯向量检索应返回结果"
        assert len(hybrid_results) > 0, "混合检索应返回结果"


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestReflexionOnline:
    """Reflexion 在线测试"""

    def test_evaluate_quality(self):
        """测试审查质量评估"""
        from agents.reflexion import evaluate_review_quality

        contract = "甲方：A公司，乙方：B公司。合同金额100万元。"
        review = "审查报告：合同主体明确，金额清晰。未发现重大风险。"

        result = evaluate_review_quality(contract, review)
        assert 'overall_score' in result, f"评估结果应包含overall_score: {result}"
        assert 'passed' in result, f"评估结果应包含passed: {result}"
        assert 0 <= result['overall_score'] <= 10, (
            f"分数应在0-10之间，实际: {result['overall_score']}"
        )


class TestCorrectiveRAGOffline:
    """Corrective RAG 离线测试"""

    def test_check_retrieval_quality_pass(self):
        """检索质量合格的情况"""
        from knowledge.build_kb import _check_retrieval_quality
        results = [
            {'text': 'doc1', 'similarity': 0.8, 'bm25_score': 2.5},
            {'text': 'doc2', 'similarity': 0.5, 'bm25_score': 1.0},
        ]
        assert _check_retrieval_quality("test", results, threshold=0.35) is True

    def test_check_retrieval_quality_low_sim(self):
        """相似度过低的情况"""
        from knowledge.build_kb import _check_retrieval_quality
        results = [
            {'text': 'doc1', 'similarity': 0.2, 'bm25_score': 1.0},
        ]
        assert _check_retrieval_quality("test", results, threshold=0.35) is False

    def test_check_retrieval_quality_no_bm25(self):
        """无 BM25 匹配的情况"""
        from knowledge.build_kb import _check_retrieval_quality
        results = [
            {'text': 'doc1', 'similarity': 0.8, 'bm25_score': 0},
        ]
        assert _check_retrieval_quality("test", results, threshold=0.35) is False

    def test_check_retrieval_quality_empty(self):
        """空结果"""
        from knowledge.build_kb import _check_retrieval_quality
        assert _check_retrieval_quality("test", [], threshold=0.35) is False

    def test_rewrite_query_import(self):
        """测试查询改写函数可导入"""
        from knowledge.build_kb import _rewrite_query
        assert callable(_rewrite_query)

    def test_search_corrective_import(self):
        """测试 Corrective RAG 检索函数可导入"""
        from knowledge.build_kb import search_corrective
        assert callable(search_corrective)


class TestGuardrailsOffline:
    """Guardrails 护栏离线测试"""

    def test_input_guardrail_valid(self):
        """有效输入通过"""
        from agents.guardrails import InputGuardrail
        guard = InputGuardrail()
        ok, msg = guard.check("这是一份合同文本，甲方为A公司，乙方为B公司。")
        assert ok is True

    def test_input_guardrail_empty(self):
        """空输入被拦截"""
        from agents.guardrails import InputGuardrail
        guard = InputGuardrail()
        ok, msg = guard.check("")
        assert ok is False
        assert "为空" in msg

    def test_input_guardrail_too_short(self):
        """过短输入被拦截"""
        from agents.guardrails import InputGuardrail
        guard = InputGuardrail()
        ok, msg = guard.check("短")
        assert ok is False
        assert "过短" in msg

    def test_input_guardrail_too_long(self):
        """过长输入被拦截"""
        from agents.guardrails import InputGuardrailConfig, InputGuardrail
        config = InputGuardrailConfig(max_length=100)
        guard = InputGuardrail(config)
        ok, msg = guard.check("a" * 200)
        assert ok is False
        assert "过长" in msg

    def test_cost_guardrail_within_limits(self):
        """成本在限制内"""
        from agents.guardrails import CostGuardrail
        guard = CostGuardrail()
        guard.record_llm_call()
        guard.record_tool_call()
        ok, msg = guard.check()
        assert ok is True

    def test_cost_guardrail_llm_exceeded(self):
        """LLM 调用次数超限"""
        from agents.guardrails import CostGuardrailConfig, CostGuardrail
        config = CostGuardrailConfig(max_llm_calls=2)
        guard = CostGuardrail(config)
        guard.record_llm_call()
        guard.record_llm_call()
        ok, msg = guard.check()
        assert ok is False
        assert "LLM" in msg

    def test_cost_guardrail_tool_exceeded(self):
        """工具调用次数超限"""
        from agents.guardrails import CostGuardrailConfig, CostGuardrail
        config = CostGuardrailConfig(max_tool_calls=1)
        guard = CostGuardrail(config)
        guard.record_tool_call()
        ok, msg = guard.check()
        assert ok is False
        assert "工具" in msg

    def test_cost_guardrail_reset(self):
        """重置后计数器清零"""
        from agents.guardrails import CostGuardrailConfig, CostGuardrail
        config = CostGuardrailConfig(max_llm_calls=1)
        guard = CostGuardrail(config)
        guard.record_llm_call()
        guard.reset()
        ok, msg = guard.check()
        assert ok is True

    def test_cost_guardrail_usage_report(self):
        """使用报告格式正确"""
        from agents.guardrails import CostGuardrail
        guard = CostGuardrail()
        guard.record_llm_call(input_tokens=500)
        guard.record_tool_call()
        report = guard.get_usage_report()
        assert report['llm_calls'] == 1
        assert report['tool_calls'] == 1
        assert report['estimated_tokens'] == 500
        assert 'elapsed_seconds' in report

    def test_output_guardrail_valid(self):
        """有效输出通过"""
        from agents.guardrails import OutputGuardrail
        guard = OutputGuardrail()
        output = "合同审查报告\n\n经审查，本合同存在以下风险点：\n1. 违约金条款风险较高" + "x" * 100
        ok, msg, details = guard.check(output)
        assert ok is True

    def test_output_guardrail_too_short(self):
        """输出过短"""
        from agents.guardrails import OutputGuardrail
        guard = OutputGuardrail()
        ok, msg, details = guard.check("短")
        assert ok is False

    def test_output_guardrail_missing_sections(self):
        """缺少必要章节"""
        from agents.guardrails import OutputGuardrail
        guard = OutputGuardrail()
        ok, msg, details = guard.check("这是一份很长的报告内容但是没有包含必要的关键词" * 10)
        assert ok is False
        assert "缺少" in msg

    def test_guardrail_chain(self):
        """护栏链完整流程"""
        from agents.guardrails import GuardrailChain
        chain = GuardrailChain()

        # 输入检查
        ok, msg = chain.check_input("这是一份测试合同文本，甲方乙方签订。")
        assert ok is True

        # 成本检查
        chain.cost.record_llm_call()
        ok, msg = chain.check_cost()
        assert ok is True

        # 输出检查
        output = "审查结果：发现以下风险点和问题" + "x" * 100
        ok, msg, details = chain.check_output(output)
        assert ok is True

        # 报告
        report = chain.get_report()
        assert 'cost_usage' in report


class TestStructuredOutputOffline:
    """Structured Output 离线测试"""

    def test_risk_level_enum(self):
        """风险等级枚举"""
        from config.schemas import RiskLevel
        assert RiskLevel.HIGH == "高"
        assert RiskLevel.MEDIUM == "中"
        assert RiskLevel.LOW == "低"

    def test_risk_item_creation(self):
        """风险项创建"""
        from config.schemas import RiskItem, RiskLevel
        item = RiskItem(
            category="违约金过高",
            level=RiskLevel.MEDIUM,
            description="违约金超过30%",
            suggestion="建议降低违约金比例",
        )
        assert item.level == RiskLevel.MEDIUM
        assert item.category == "违约金过高"

    def test_risk_assessment_validation(self):
        """风险评估结果验证"""
        from config.schemas import validate_risk_assessment
        data = {
            'overall_risk_level': '中',
            'risk_items': [{
                'category': '自动续约',
                'level': '中',
                'description': '合同自动续约无终止条件',
                'suggestion': '建议添加明确的终止条款',
            }],
            'risk_summary': '整体风险中等',
            'total_high_risks': 0,
            'total_medium_risks': 1,
            'total_low_risks': 0,
        }
        ok, result, err = validate_risk_assessment(data)
        assert ok is True
        assert result is not None
        assert len(result.risk_items) == 1

    def test_risk_assessment_invalid(self):
        """无效风险评估被拒绝"""
        from config.schemas import validate_risk_assessment
        data = {'invalid': 'data'}
        ok, result, err = validate_risk_assessment(data)
        assert ok is False
        assert err is not None

    def test_quality_evaluation_validation(self):
        """质量评估验证"""
        from config.schemas import validate_quality_evaluation
        data = {
            'completeness': {'score': 8.0, 'reason': '覆盖全面'},
            'risk_identification': {'score': 7.5, 'reason': '识别准确'},
            'legal_basis': {'score': 7.0, 'reason': '有法律依据'},
            'actionability': {'score': 8.0, 'reason': '建议具体'},
            'clarity': {'score': 8.5, 'reason': '结构清晰'},
            'overall_score': 7.8,
            'passed': True,
            'major_issues': [],
        }
        ok, result, err = validate_quality_evaluation(data)
        assert ok is True
        assert result.overall_score == 7.8

    def test_quality_evaluation_score_bounds(self):
        """评分边界检查"""
        from config.schemas import validate_quality_evaluation
        data = {
            'completeness': {'score': 11.0, 'reason': '超限'},
            'risk_identification': {'score': 5.0, 'reason': 'ok'},
            'legal_basis': {'score': 5.0, 'reason': 'ok'},
            'actionability': {'score': 5.0, 'reason': 'ok'},
            'clarity': {'score': 5.0, 'reason': 'ok'},
            'overall_score': 5.0,
            'passed': False,
            'major_issues': [],
        }
        ok, result, err = validate_quality_evaluation(data)
        assert ok is False  # score 11.0 超出 0-10 范围

    def test_review_report_schema(self):
        """审查报告 Schema 可生成"""
        from config.schemas import get_review_report_schema
        schema = get_review_report_schema()
        assert 'properties' in schema
        assert 'risk_assessment' in schema['properties']

    def test_clause_extraction_model(self):
        """条款提取模型"""
        from config.schemas import ClauseExtraction, ContractParty, ContractType
        clause = ClauseExtraction(
            contract_parties=[
                ContractParty(name="A公司", role="甲方"),
                ContractParty(name="B公司", role="乙方"),
            ],
            contract_type=ContractType.SERVICE,
            subject_matter="软件开发服务",
        )
        assert len(clause.contract_parties) == 2
        assert clause.contract_type == ContractType.SERVICE


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestCorrectiveRAGOnline:
    """Corrective RAG 在线测试"""

    def test_search_corrective_returns_results(self):
        """测试 Corrective RAG 检索能返回结果"""
        from knowledge.build_kb import search_corrective, VECTOR_STORE_PATH

        store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
        if not os.path.exists(store_path):
            pytest.skip("知识库未构建")

        results = search_corrective("违约金过高怎么处理", top_k=3)
        assert len(results) > 0, "Corrective RAG 应返回结果"

    def test_rewrite_query_generates_variants(self):
        """测试查询改写能生成变体"""
        from knowledge.build_kb import _rewrite_query
        variants = _rewrite_query("合同自动续约的风险")
        assert len(variants) > 0, "应生成至少1个查询变体"
        assert all(isinstance(v, str) for v in variants)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
