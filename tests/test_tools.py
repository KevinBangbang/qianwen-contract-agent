"""
工具模块单元测试

测试5个自定义工具的核心功能。
分为两类：
1. 离线测试：不调用API，测试纯逻辑功能（amount_calculator、report_generator）
2. 在线测试：需要调用DashScope API（contract_parser、clause_extractor、risk_checker）

运行方式：
    pytest tests/test_tools.py -v               # 运行所有测试
    pytest tests/test_tools.py -v -k "offline"   # 只运行离线测试
    pytest tests/test_tools.py -v -k "online"    # 只运行在线测试
"""

import json
import os
import sys
import pytest

# 确保项目根目录在导入路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具模块（导入时会自动注册到全局工具表）
from tools.contract_parser import ContractParser
from tools.clause_extractor import ClauseExtractor
from tools.risk_checker import RiskChecker
from tools.amount_calculator import AmountCalculator
from tools.report_generator import ReportGenerator

# 测试用合同样本路径
SAMPLE_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), 'sample_contracts', 'sample_contract.txt'
)


# ============================================================
# 离线测试：不需要API调用
# ============================================================

class TestAmountCalculatorOffline:
    """金额计算工具测试（纯离线，不调用API）"""

    def setup_method(self):
        self.tool = AmountCalculator()

    def test_penalty_fixed_rate(self):
        """测试固定比例违约金计算"""
        params = json.dumps({
            'calculation_type': 'penalty',
            'params': {
                'contract_amount': 1000000,
                'penalty_rate': 0.1,
                'method': 'fixed',
            }
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert result['result']['penalty_amount'] == 100000.0, f"Expected 100000.0 but got: {result['result']['penalty_amount']}"

    def test_penalty_daily_rate(self):
        """测试按日计算违约金"""
        params = json.dumps({
            'calculation_type': 'penalty',
            'params': {
                'contract_amount': 3500000,
                'penalty_rate': 0.05,
                'overdue_days': 30,
                'method': 'daily',
            }
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert result['result']['penalty_amount'] > 0, f"Expected positive penalty but got: {result['result']['penalty_amount']}"
        assert result['result']['overdue_days'] == 30, f"Expected 30 overdue days but got: {result['result']['overdue_days']}"

    def test_penalty_high_warning(self):
        """测试违约金过高警告"""
        params = json.dumps({
            'calculation_type': 'penalty',
            'params': {
                'contract_amount': 100000,
                'penalty_rate': 0.5,
                'method': 'fixed',
            }
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        # 50%违约金超过30%，应该有警告
        assert result['result']['warning'] is not None, "Expected high penalty warning but got None"

    def test_payment_schedule(self):
        """测试付款计划生成"""
        params = json.dumps({
            'calculation_type': 'payment_schedule',
            'params': {
                'total_amount': 3500000,
                'installments': 3,
                'start_date': '2026-04-01',
            }
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert len(result['result']['schedule']) == 3, f"Expected 3 installments but got: {len(result['result']['schedule'])}"
        # 验证总金额正确
        total = sum(s['amount'] for s in result['result']['schedule'])
        assert abs(total - 3500000) < 0.01, f"Total amount mismatch: {total} != 3500000"

    def test_date_diff(self):
        """测试日期间隔计算"""
        params = json.dumps({
            'calculation_type': 'date_diff',
            'params': {
                'start_date': '2026-04-01',
                'end_date': '2027-03-31',
            }
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert result['result']['days'] == 364, f"Expected 364 days but got: {result['result']['days']}"

    def test_amount_verify(self):
        """测试金额大写转换"""
        params = json.dumps({
            'calculation_type': 'amount_verify',
            'params': {
                'amount': 3500000,
                'currency': 'CNY',
            }
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        # 验证生成了大写金额
        assert '万' in result['result']['chinese_amount'], f"Expected '万' in chinese_amount but got: {result['result']['chinese_amount']}"

    def test_invalid_calculation_type(self):
        """测试无效的计算类型"""
        params = json.dumps({
            'calculation_type': 'invalid_type',
            'params': {}
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is False, f"Expected failure for invalid type but got: {result}"


class TestReportGeneratorOffline:
    """报告生成工具测试（纯离线）"""

    def setup_method(self):
        self.tool = ReportGenerator()

    def test_generate_basic_report(self):
        """测试基础报告生成"""
        params = json.dumps({
            'contract_info': {
                '合同名称': '技术服务合同',
                '合同编号': 'QW-2026-0042',
            },
            'clauses': {
                '合同主体': '甲方：杭州星辰科技 / 乙方：北京智云数据',
                '合同金额': '350万元',
                '合同期限': '2026年4月至2027年3月',
            },
            'risk_assessment': {
                'risks': [
                    {
                        'level': '中',
                        'description': '自动续约条款未设置明确终止条件',
                        'suggestion': '建议增加明确的终止条件和通知期限',
                    },
                    {
                        'level': '低',
                        'description': '管辖法院约定在甲方所在地',
                        'suggestion': '双方可协商选择中立地点的仲裁机构',
                    },
                ]
            },
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        report = result['report']
        # 验证报告包含关键内容
        assert '合同审查报告' in report, "Expected '合同审查报告' in report content"
        assert '技术服务合同' in report, "Expected '技术服务合同' in report content"
        assert '风险评估' in report, "Expected '风险评估' in report content"
        assert '中风险' in report, "Expected '中风险' in report content"

    def test_empty_risk_report(self):
        """测试无风险的报告"""
        params = json.dumps({
            'contract_info': {'合同名称': '测试合同'},
            'clauses': {'合同主体': '甲方和乙方'},
            'risk_assessment': {'risks': []},
        })
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert '风险较低' in result['report'], f"Expected '风险较低' in report but got: {result['report'][:100]}"


# ============================================================
# 在线测试：需要调用DashScope API
# ============================================================

# 检查API Key是否配置，未配置则跳过在线测试
has_api_key = bool(os.getenv('DASHSCOPE_API_KEY'))
online_skip_reason = '未配置 DASHSCOPE_API_KEY，跳过在线测试'


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestContractParserOnline:
    """合同解析工具在线测试"""

    def setup_method(self):
        self.tool = ContractParser()

    def test_parse_nonexistent_file(self):
        """测试解析不存在的文件"""
        params = json.dumps({'file_path': '/not/exist/file.pdf'})
        result = json.loads(self.tool.call(params))
        assert result['success'] is False, f"Expected failure for nonexistent file but got: {result}"
        assert '不存在' in result['error'], f"Expected '不存在' in error but got: {result['error']}"

    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        # 创建一个临时的 .doc 文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as f:
            f.write(b'test')
            temp_path = f.name

        try:
            params = json.dumps({'file_path': temp_path})
            result = json.loads(self.tool.call(params))
            assert result['success'] is False, f"Expected failure for unsupported format but got: {result}"
            assert '不支持' in result['error'], f"Expected '不支持' in error but got: {result['error']}"
        finally:
            os.unlink(temp_path)


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestClauseExtractorOnline:
    """条款提取工具在线测试"""

    def setup_method(self):
        self.tool = ClauseExtractor()

    def test_extract_clauses(self):
        """测试从合同文本中提取条款"""
        with open(SAMPLE_CONTRACT_PATH, 'r', encoding='utf-8') as f:
            contract_text = f.read()

        params = json.dumps({'contract_text': contract_text})
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert 'clauses' in result, f"Expected 'clauses' key in result but got keys: {list(result.keys())}"

    def test_empty_text(self):
        """测试空文本"""
        params = json.dumps({'contract_text': ''})
        result = json.loads(self.tool.call(params))
        assert result['success'] is False, f"Expected failure for empty text but got: {result}"


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestRiskCheckerOnline:
    """风险检查工具在线测试"""

    def setup_method(self):
        self.tool = RiskChecker()

    def test_check_risks(self):
        """测试风险检查"""
        clauses = json.dumps({
            'contract_parties': '甲方：杭州星辰科技 / 乙方：北京智云数据',
            'amount': '350万元',
            'breach_liability': '逾期一日支付万分之五违约金',
            'special_clauses': '合同到期后自动续约一年',
        })
        params = json.dumps({'clauses': clauses})
        result = json.loads(self.tool.call(params))
        assert result['success'] is True, f"Expected success but got: {result}"
        assert 'risk_assessment' in result, f"Expected 'risk_assessment' key in result but got keys: {list(result.keys())}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
