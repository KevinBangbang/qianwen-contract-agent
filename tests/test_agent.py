"""
Agent模块端到端测试

测试Agent编排的完整流程：
1. 编排器的输入类型检测
2. review_agent的合同审查流程
3. 编排器的文件路由

运行方式：
    pytest tests/test_agent.py -v               # 运行所有测试
    pytest tests/test_agent.py -v -k "offline"   # 只运行离线测试
    pytest tests/test_agent.py -v -k "online"    # 只运行在线测试（需要API Key）
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), 'sample_contracts', 'sample_contract.txt'
)

has_api_key = bool(os.getenv('DASHSCOPE_API_KEY'))
online_skip_reason = '未配置 DASHSCOPE_API_KEY，跳过在线测试'


class TestOrchestratorOffline:
    """编排器离线测试"""

    def test_detect_text_file(self):
        """测试文本文件类型检测"""
        from agents.orchestrator import detect_input_type
        assert detect_input_type(SAMPLE_CONTRACT_PATH) == 'text', f"Expected 'text' but got: {detect_input_type(SAMPLE_CONTRACT_PATH)}"

    def test_detect_image_file(self):
        """测试图片文件类型检测"""
        import tempfile
        from agents.orchestrator import detect_input_type
        for ext in ['.png', '.jpg']:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                temp_path = f.name
                f.write(b'fake')
            assert detect_input_type(temp_path) == 'image', f"Expected 'image' for {ext} but got: {detect_input_type(temp_path)}"
            os.unlink(temp_path)

    def test_detect_pdf_file(self):
        """测试PDF文件类型检测"""
        import tempfile
        from agents.orchestrator import detect_input_type
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
            f.write(b'fake')
        assert detect_input_type(temp_path) == 'pdf', f"Expected 'pdf' but got: {detect_input_type(temp_path)}"
        os.unlink(temp_path)

    def test_detect_unknown_file(self):
        """测试未知文件类型"""
        from agents.orchestrator import detect_input_type
        assert detect_input_type('contract.doc') == 'unknown', f"Expected 'unknown' for .doc but got: {detect_input_type('contract.doc')}"
        assert detect_input_type('') == 'unknown', f"Expected 'unknown' for empty string but got: {detect_input_type('')}"
        assert detect_input_type(None) == 'unknown', f"Expected 'unknown' for None but got: {detect_input_type(None)}"


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestReviewAgentOnline:
    """审查Agent在线测试"""

    def test_review_agent_creation(self):
        """测试Agent能否正常创建"""
        from agents.review_agent import create_review_agent
        agent = create_review_agent()
        assert agent is not None, "Expected agent to be created but got None"
        assert agent.name == '合同审查助手', f"Expected agent name '合同审查助手' but got: {agent.name}"

    def test_review_short_contract(self):
        """测试对简短合同的审查"""
        from agents.review_agent import create_review_agent

        agent = create_review_agent()
        messages = [{
            'role': 'user',
            'content': '请简单分析以下合同条款的风险：\n'
                       '违约金为合同总额的50%，合同到期后自动续约。'
        }]

        response = []
        for chunk in agent.run(messages=messages):
            response = chunk

        # 验证Agent返回了有效响应
        assert len(response) > 0, "Expected non-empty response from agent"
        # 检查最后一条消息是assistant的回复
        has_assistant_reply = any(
            msg.get('role') == 'assistant' and msg.get('content')
            for msg in response
        )
        assert has_assistant_reply, "Agent did not return an assistant reply"


@pytest.mark.skipif(not has_api_key, reason=online_skip_reason)
class TestOrchestratorOnline:
    """编排器在线测试"""

    def test_orchestrate_text_file(self):
        """测试编排器处理文本文件"""
        from agents.orchestrator import process_contract

        response = []
        for chunk in process_contract(file_path=SAMPLE_CONTRACT_PATH):
            response = chunk

        assert len(response) > 0, "Expected non-empty response from orchestrator"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
