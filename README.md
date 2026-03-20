# Qwen Contract Agent

An intelligent contract review system built on the Alibaba Qwen ecosystem, integrating **Agent orchestration**, **RAG**, **multimodal OCR**, and **edge deployment**.

## Architecture

```mermaid
graph TB
    subgraph Input
        A[PDF] --> D[Orchestrator]
        B[Scanned Image] --> D
        C[Plain Text] --> D
    end

    subgraph Agent Layer
        D --> E{Type Detection}
        E -->|Image| F[OCR Agent<br/>Qwen-VL]
        E -->|PDF / Text| G[Review Agent<br/>Qwen-Plus]
        F --> G
    end

    subgraph Tool Layer
        G --> H[Contract Parser]
        G --> I[Clause Extractor]
        G --> J[Risk Checker]
        G --> K[Amount Calculator]
        G --> L[Report Generator]
    end

    subgraph Knowledge Layer
        J --> M[Vector KB<br/>Legal Docs + Risk Templates]
        M --> N[DashScope<br/>Embedding]
    end

    subgraph Output
        L --> O[Review Report<br/>Markdown]
        G --> P[Gradio UI]
    end
```

## Key Features

- **Multi-format input** — PDF, scanned images, plain text
- **ReAct reasoning** — Agent autonomously plans and chains tool calls to complete a full review
- **RAG knowledge base** — Retrieves relevant legal provisions and risk templates to enhance review accuracy
- **Multimodal OCR** — Uses Qwen-VL to recognize scanned contract documents
- **Flexible deployment** — Cloud API (DashScope) or local Ollama / vLLM

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your DASHSCOPE_API_KEY
```

### 3. Build the knowledge base

```bash
python knowledge/build_kb.py
```

### 4. Verify API connectivity

```bash
python quick_test.py
```

### 5. Launch the UI

```bash
python app/gradio_app.py
# Open http://localhost:7860
```

## Project Structure

```
├── config/                  # Configuration
│   ├── model_config.py      # Model config (cloud / local switch)
│   └── prompts.py           # System prompt templates
├── tools/                   # Custom tools (5 × BaseTool)
│   ├── contract_parser.py   # PDF extraction + OCR
│   ├── clause_extractor.py  # LLM-driven clause extraction
│   ├── risk_checker.py      # Risk analysis with RAG retrieval
│   ├── amount_calculator.py # Deterministic amount & date math
│   └── report_generator.py  # Markdown report assembly
├── knowledge/               # RAG knowledge base
│   ├── build_kb.py          # Build pipeline (chunk → embed → store)
│   ├── legal_docs/          # 8 legal reference documents
│   └── risk_templates/      # Common risk clause templates
├── agents/                  # Agent orchestration
│   ├── review_agent.py      # Main review agent (ReAct loop)
│   ├── ocr_agent.py         # Multimodal OCR agent
│   └── orchestrator.py      # Input routing (code-based, not LLM)
├── app/                     # Frontend
│   └── gradio_app.py        # Gradio interactive demo
├── deploy/                  # Deployment
│   ├── cloud_deploy.md      # DashScope cloud deployment guide
│   ├── edge_deploy.md       # Ollama / vLLM local deployment guide
│   └── benchmark.py         # Performance benchmark script
├── tests/                   # Tests (21 total)
│   ├── test_tools.py        # 14 tool unit tests (9 offline + 5 online)
│   └── test_agent.py        # 7 agent E2E tests (4 offline + 3 online)
└── docs/
    ├── architecture.md
    ├── tam_solution.md
    └── performance_report.md
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) | Agent framework with ReAct, tool registration |
| LLM | qwen-plus / qwen-vl-plus | Text generation & vision via DashScope API |
| Embedding | text-embedding-v3 | 1024-dim vectors for RAG retrieval |
| Frontend | Gradio | Interactive demo UI |
| Local deploy | Ollama / vLLM | On-premise model serving |

## Design Principles

1. **Deterministic logic in code, semantic understanding in LLM** — Routing and calculations are done in Python (`if-else`, pure math); only tasks requiring language understanding go to the model.
2. **OpenAI-compatible interface everywhere** — Switching between DashScope cloud and local Ollama only requires changing `base_url` and `model`; zero business code changes.
3. **Tools are the boundary** — Each `BaseTool` has a clear input/output contract. The agent decides *when* to call tools; the tools decide *how* to execute.

## Benchmark

| Model | TTFT | Generation Speed | Extraction Quality | Recommended For |
|-------|------|-----------------|-------------------|----------------|
| qwen-turbo | ~1.2 s | 33.5 tok/s | 83% | Development & debugging |
| qwen-plus | ~1.1 s | 10.8 tok/s | 83% | Production (best cost/quality) |

Run your own benchmark:

```bash
python deploy/benchmark.py
```

## Testing

```bash
# Offline tests only (no API key needed)
pytest tests/ -v -k "offline"

# All tests (requires DASHSCOPE_API_KEY)
DASHSCOPE_API_KEY=sk-xxx pytest tests/ -v
```

## License

MIT
