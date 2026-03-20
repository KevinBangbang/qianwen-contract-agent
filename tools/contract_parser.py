"""
合同解析工具

负责将各种格式的合同文件转换为纯文本。
支持两种输入：
1. PDF文件：使用PyPDF2提取文本
2. 扫描件图片：调用Qwen-VL进行OCR识别

这是审查流程的第一步，后续所有工具都依赖它的输出。
"""

import json
import logging
import os
import base64

from qwen_agent.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool('contract_parser')
class ContractParser(BaseTool):
    # 工具描述会被注入到prompt中，模型根据这个描述决定何时调用此工具
    description = '合同文件解析工具，支持PDF文本提取和扫描件OCR识别，将合同文件转换为纯文本'
    # 参数定义，使用列表格式（Qwen-Agent支持列表和字典两种格式）
    parameters = [{
        'name': 'file_path',
        'type': 'string',
        'description': '合同文件的路径，支持.pdf和图片格式(.png/.jpg/.jpeg)',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        解析合同文件，返回提取的文本内容

        处理流程：
        1. 判断文件类型（PDF还是图片）
        2. PDF用PyPDF2提取文本
        3. 图片用Qwen-VL做OCR
        4. 返回JSON格式的结果
        """
        # 解析参数，_verify_json_format_args是BaseTool提供的工具方法
        params = self._verify_json_format_args(params)
        file_path = params['file_path']

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return json.dumps(
                {'success': False, 'error': f'文件不存在: {file_path}'},
                ensure_ascii=False
            )

        # 根据文件扩展名选择解析方式
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self._parse_image(file_path)
        else:
            return json.dumps(
                {'success': False, 'error': f'不支持的文件格式: {ext}'},
                ensure_ascii=False
            )

    def _parse_pdf(self, file_path: str) -> str:
        """
        使用PyPDF2提取PDF中的文本

        PyPDF2只能提取文本层的PDF，如果PDF是扫描件（纯图片），
        提取结果会是空的，这时候需要走OCR流程。
        """
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append({
                        'page': i + 1,
                        'content': text.strip()
                    })

            # 如果所有页面都没有文本，说明可能是扫描件
            if not pages_text:
                return json.dumps({
                    'success': True,
                    'format': 'pdf_scanned',
                    'text': '',
                    'message': 'PDF中未提取到文本，可能是扫描件，建议使用OCR处理',
                    'total_pages': len(reader.pages)
                }, ensure_ascii=False)

            # 合并所有页面的文本
            full_text = '\n\n'.join(p['content'] for p in pages_text)

            return json.dumps({
                'success': True,
                'format': 'pdf_text',
                'text': full_text,
                'total_pages': len(reader.pages),
                'pages_with_text': len(pages_text)
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {'success': False, 'error': f'PDF解析失败: {str(e)}'},
                ensure_ascii=False
            )

    def _parse_image(self, file_path: str) -> str:
        """
        使用Qwen-VL模型对图片进行OCR识别

        将图片转为base64编码后发送给VL模型，
        让模型识别图片中的文字并返回结构化文本。
        """
        try:
            # 读取图片并转为base64
            with open(file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # 根据文件扩展名确定MIME类型
            ext = os.path.splitext(file_path)[1].lower()
            mime_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
            }
            mime_type = mime_map.get(ext, 'image/png')

            # 调用DashScope VL模型进行OCR
            from config.model_config import get_openai_client, get_model_config
            client = get_openai_client()
            config = get_model_config()

            response = client.chat.completions.create(
                model=config['vl_model'],
                messages=[{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': '请仔细识别这张合同图片中的所有文字内容，'
                                    '保持原文的段落结构和格式。'
                                    '如果有表格，请用Markdown表格格式输出。'
                                    '如果某些文字模糊无法识别，请标注[无法识别]。'
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{mime_type};base64,{image_data}'
                            }
                        }
                    ]
                }],
                max_tokens=4096,
            )

            ocr_text = response.choices[0].message.content

            return json.dumps({
                'success': True,
                'format': 'image_ocr',
                'text': ocr_text,
                'source_file': os.path.basename(file_path)
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {'success': False, 'error': f'OCR识别失败: {str(e)}'},
                ensure_ascii=False
            )
