# 智能合同审查Agent

[English](./README_EN.md) | **中文**

基于阿里千问（Qwen）生态构建的智能合同审查系统，整合 **Agent编排**、**RAG检索增强**、**多模态OCR**、**端侧部署** 四个方向。

## 系统架构

```mermaid
graph TB
    subgraph 输入层
        A[PDF文件] --> D[编排器]
        B[扫描件图片] --> D
        C[文本输入] --> D
    end

    subgraph 护栏层
        D --> G1[输入护栏<br/>长度/格式校验]
        G1 --> E{类型判断}
    end

    subgraph Agent层
        E -->|图片| F[OCR Agent<br/>Qwen-VL]
        E -->|PDF/文本| G[审查Agent<br/>Qwen-Plus]
        F --> G
        G --> REF[Reflexion<br/>自我反思]
        REF -->|质量不达标| G
    end

    subgraph 工具层
        G --> H[合同解析器]
        G --> I[条款提取器]
        G --> J[风险检查器]
        G --> K[金额计算器]
        G --> L[报告生成器]
    end

    subgraph 知识层
        J --> CRAG[Corrective RAG]
        CRAG --> HYB[Hybrid Retrieval<br/>BM25 + Dense + RRF]
        HYB --> RR[Reranker<br/>LLM精排]
        HYB --> M[向量知识库<br/>法律法规+风险模板]
        M --> N[DashScope<br/>Embedding]
    end

    subgraph 输出层
        L --> VAL[结构化验证<br/>Pydantic Schema]
        VAL --> G2[输出护栏]
        G2 --> O[审查报告<br/>Markdown格式]
        G --> P[Gradio界面]
    end
```

## 核心特性

- **多格式支持** — PDF文本、扫描件图片、纯文本输入
- **ReAct多步推理** — Agent自主规划并链式调用工具，完成完整审查流程
- **Hybrid RAG** — BM25稀疏检索 + Dense稠密检索 + RRF融合 + LLM Reranker精排
- **Corrective RAG** — 检索质量自检，低质量时自动改写查询（Multi-Query）重试
- **Reflexion自我反思** — 审查质量5维度评估，不达标自动反思改进并累积经验
- **Guardrails护栏** — 输入验证 + 成本控制 + 输出结构验证，三层纵深防御
- **Structured Output** — Pydantic Schema约束审查报告、风险评估等输出格式
- **多模态OCR** — 使用Qwen-VL识别合同扫描件
- **灵活部署** — 支持云端API（DashScope）和本地Ollama / vLLM两种模式

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 构建知识库

```bash
python knowledge/build_kb.py
```

### 4. 验证API连通性

```bash
python quick_test.py
```

### 5. 启动前端

```bash
python app/gradio_app.py
# 访问 http://localhost:7860
```

## 项目结构

```
├── config/                  # 配置模块
│   ├── model_config.py      # 模型配置（云端/本地切换）
│   ├── prompts.py           # 系统提示词模板
│   └── schemas.py           # Pydantic结构化输出Schema
├── tools/                   # 自定义工具（5个BaseTool）
│   ├── contract_parser.py   # 合同解析（PDF提取 + OCR）
│   ├── clause_extractor.py  # LLM驱动的条款提取
│   ├── risk_checker.py      # Corrective RAG增强的风险检查
│   ├── amount_calculator.py # 确定性金额与日期计算
│   └── report_generator.py  # Markdown报告生成
├── knowledge/               # RAG知识库
│   ├── build_kb.py          # 构建流程（分块 → 向量化 → BM25 → 存储）
│   ├── reranker.py          # Reranker模块（LLM / CrossEncoder）
│   ├── legal_docs/          # 8篇法律法规文档
│   └── risk_templates/      # 常见风险条款模板
├── agents/                  # Agent编排
│   ├── review_agent.py      # 主审查Agent（ReAct循环）
│   ├── ocr_agent.py         # 多模态OCR Agent
│   ├── orchestrator.py      # 输入路由 + Reflexion编排
│   ├── reflexion.py         # 自我反思与经验累积
│   └── guardrails.py        # 护栏（输入/成本/输出三层）
├── app/                     # 前端应用
│   └── gradio_app.py        # Gradio交互式Demo
├── deploy/                  # 部署与评估
│   ├── cloud_deploy.md      # DashScope云端部署指南
│   ├── edge_deploy.md       # Ollama / vLLM本地部署指南
│   ├── benchmark.py         # 性能评测脚本
│   └── rag_eval.py          # RAGAS风格RAG评估框架
├── tests/                   # 测试（共68+个）
│   ├── test_tools.py        # 14个工具单元测试
│   ├── test_agent.py        # 7个Agent端到端测试
│   ├── test_rag_advanced.py # 47个进阶功能测试
│   └── rag_golden_dataset.json  # 18题RAG黄金测试集
└── docs/
    ├── architecture.md
    ├── tam_solution.md
    └── performance_report.md
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 框架 | [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) | 阿里官方Agent框架，ReAct + 工具注册 |
| 模型 | qwen-plus / qwen-vl-plus | DashScope API 文本生成与视觉理解 |
| 向量化 | text-embedding-v3 | 1024维向量，用于RAG检索 |
| 稀疏检索 | BM25 (Okapi) | 关键词匹配，与向量检索互补 |
| 结构化输出 | Pydantic v2 | Schema约束LLM输出格式 |
| 前端 | Gradio | 交互式Demo界面 |
| 本地部署 | Ollama / vLLM | 端侧模型推理服务 |

## 设计原则

1. **确定性逻辑用代码，语义理解用模型** — 路由和计算在Python中实现（`if-else`、纯数学），只有需要语言理解的任务交给模型。
2. **全链路OpenAI兼容接口** — 云端DashScope和本地Ollama切换只需改`base_url`和`model`，业务代码零改动。
3. **工具即边界** — 每个`BaseTool`有清晰的输入输出契约。Agent决定"何时"调用工具，工具决定"如何"执行。
4. **纵深防御** — 输入护栏 → 成本控制 → 输出验证，三层护栏保障Agent安全可控。
5. **检索不盲信** — Corrective RAG在使用检索结果前先评估质量，低质量时改写查询重试。

## 性能对比

| 模型 | TTFT | 生成速度 | 提取质量 | 推荐场景 |
|------|------|---------|---------|---------|
| qwen-turbo | ~1.2秒 | 33.5 tok/s | 83% | 开发调试 |
| qwen-plus | ~1.1秒 | 10.8 tok/s | 83% | 生产推荐 |

运行评测：

```bash
python deploy/benchmark.py
```

## 测试

```bash
# 仅离线测试（无需API Key）
pytest tests/ -v -k "offline"

# 仅运行进阶功能离线测试（47个）
pytest tests/test_rag_advanced.py -v -k "offline"

# 全部测试（需要 DASHSCOPE_API_KEY）
DASHSCOPE_API_KEY=sk-xxx pytest tests/ -v
```

## RAG评估

内置 RAGAS 风格的 RAG 评估框架，支持 Context Precision、Recall、MRR 等指标：

```bash
# 仅检索指标评估（无需LLM调用）
python deploy/rag_eval.py

# 完整评估（含LLM-as-Judge）
python deploy/rag_eval.py --full
```

## 许可证

MIT
