"""
Gradio 前端界面

提供可交互的合同审查Demo界面，包含：
1. 文件上传区（支持PDF/图片/文本）
2. 文本直接输入区
3. 模型配置切换
4. 审查结果展示（Markdown渲染）

运行方式：python app/gradio_app.py
"""

import logging
import os
import sys
import json
import tempfile
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

logger = logging.getLogger(__name__)

from config.model_config import get_model_config


def review_contract_file(file: Any, model_choice: str, progress: Any = gr.Progress()) -> tuple[str, str]:
    """
    处理上传的合同文件

    Args:
        file: Gradio上传的文件对象
        model_choice: 用户选择的模型
    """
    if file is None:
        return "请先上传合同文件", ""

    # 动态设置模型
    _set_model(model_choice)

    progress(0.1, desc="正在加载文件...")
    file_path = file.name

    progress(0.2, desc="正在初始化审查Agent...")
    from agents.orchestrator import process_contract

    progress(0.3, desc="正在审查合同（Agent思考中）...")
    response = []
    for chunk in process_contract(file_path=file_path):
        response = chunk

    progress(0.9, desc="正在整理结果...")

    # 提取审查结果
    result_text = _extract_response(response)
    # 提取工具调用过程
    process_log = _extract_tool_calls(response)

    progress(1.0, desc="审查完成!")
    return result_text, process_log


def review_contract_text(text: str, model_choice: str, progress: Any = gr.Progress()) -> tuple[str, str]:
    """
    处理直接输入的合同文本
    """
    if not text or not text.strip():
        return "请输入合同文本", ""

    _set_model(model_choice)

    progress(0.2, desc="正在初始化审查Agent...")
    from agents.orchestrator import process_contract

    progress(0.3, desc="正在审查合同...")
    response = []
    for chunk in process_contract(text=text):
        response = chunk

    progress(1.0, desc="审查完成!")

    result_text = _extract_response(response)
    process_log = _extract_tool_calls(response)

    return result_text, process_log


def _set_model(model_choice: str) -> None:
    """根据用户选择设置模型"""
    model_map = {
        'qwen-plus（推荐）': 'qwen-plus',
        'qwen-turbo（快速）': 'qwen-turbo',
        'qwen-max（最强）': 'qwen-max',
    }
    model = model_map.get(model_choice, 'qwen-plus')
    os.environ['CLOUD_MODEL'] = model


def _extract_response(response: list[dict]) -> str:
    """从Agent响应中提取最终回复文本"""
    for msg in reversed(response):
        if msg.get('role') == 'assistant' and msg.get('content'):
            content = msg['content']
            # 如果内容是列表格式（Qwen-Agent的ContentItem）
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
                return '\n'.join(texts)
            return str(content)
    return "审查完成，但未生成结果文本。"


def _extract_tool_calls(response: list[dict]) -> str:
    """从Agent响应中提取工具调用过程"""
    logs = []
    for msg in response:
        role = msg.get('role', '')
        if role == 'assistant' and msg.get('function_call'):
            fc = msg['function_call']
            tool_name = fc.get('name', '未知工具')
            logs.append(f"**调用工具**: `{tool_name}`")
        elif role == 'function':
            fn_name = msg.get('name', '')
            content = msg.get('content', '')
            # 截取结果摘要
            if len(str(content)) > 200:
                content = str(content)[:200] + '...'
            logs.append(f"**{fn_name} 返回**: {content}\n")

    if logs:
        return "## Agent推理过程\n\n" + '\n'.join(logs)
    return "Agent直接生成了回复（未调用工具）"


def create_app() -> gr.Blocks:
    """创建Gradio界面"""
    config = get_model_config()

    with gr.Blocks(
        title="智能合同审查系统",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 智能合同审查系统\n"
            "基于阿里千问生态构建的AI合同审查工具，支持PDF、图片、文本格式\n"
        )

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=[
                    'qwen-plus（推荐）',
                    'qwen-turbo（快速）',
                    'qwen-max（最强）',
                ],
                value='qwen-plus（推荐）',
                label='选择模型',
                info='qwen-plus性价比最高，qwen-max适合复杂合同',
            )
            status = gr.Markdown(
                f"**当前模式**: {config['mode']} | "
                f"**API**: {config['base_url'][:40]}..."
            )

        with gr.Tabs():
            # 文件上传Tab
            with gr.TabItem("上传文件"):
                file_input = gr.File(
                    label="上传合同文件",
                    file_types=['.pdf', '.txt', '.png', '.jpg', '.jpeg'],
                    type='filepath',
                )
                file_btn = gr.Button("开始审查", variant="primary")

            # 文本输入Tab
            with gr.TabItem("输入文本"):
                text_input = gr.Textbox(
                    label="合同文本",
                    placeholder="在此粘贴合同文本内容...",
                    lines=10,
                )
                text_btn = gr.Button("开始审查", variant="primary")

        gr.Markdown("## 审查结果")

        with gr.Row():
            with gr.Column(scale=3):
                result_output = gr.Markdown(
                    label="审查报告",
                    value="等待审查...",
                )
            with gr.Column(scale=2):
                process_output = gr.Markdown(
                    label="推理过程",
                    value="Agent推理过程将在此显示",
                )

        # 绑定事件
        file_btn.click(
            fn=review_contract_file,
            inputs=[file_input, model_choice],
            outputs=[result_output, process_output],
        )
        text_btn.click(
            fn=review_contract_text,
            inputs=[text_input, model_choice],
            outputs=[result_output, process_output],
        )

        gr.Markdown(
            "---\n"
            "*本系统基于千问大模型（Qwen）生态构建，审查结果仅供参考，不构成法律意见。*"
        )

    return app


if __name__ == '__main__':
    app = create_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
    )
