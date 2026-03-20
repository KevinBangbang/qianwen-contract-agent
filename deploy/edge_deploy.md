# 端侧部署方案

## 方案一：Ollama 部署（推荐入门）

Ollama 是最简单的本地模型部署工具，一条命令即可运行。

### 安装 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 从 https://ollama.com/download 下载安装包
```

### 下载千问模型

```bash
# Qwen3 7B（约4.7GB，推荐笔记本电脑使用）
ollama pull qwen3:7b

# Qwen3 14B（约9GB，需要16GB以上内存）
ollama pull qwen3:14b

# 查看已下载的模型
ollama list
```

### 启动服务

```bash
# 启动 Ollama 服务（默认监听 http://localhost:11434）
ollama serve

# 测试是否正常
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3:7b", "messages": [{"role": "user", "content": "你好"}]}'
```

### 集成到项目

修改 `.env` 文件：
```bash
MODEL_MODE=local
LOCAL_MODEL_SERVER=http://localhost:11434/v1
LOCAL_MODEL=qwen3:7b
```

项目代码无需任何修改，因为 Ollama 提供 OpenAI 兼容接口。

### Ollama 适用场景

- 个人开发和学习
- 快速原型验证
- 数据敏感场景（数据不出本机）
- CPU或低端GPU环境

---

## 方案二：vLLM 部署（生产环境推荐）

vLLM 是高性能的LLM推理引擎，使用 PagedAttention 技术优化显存利用。

### 安装 vLLM

```bash
pip install vllm
```

### 启动服务

```bash
# 部署 Qwen3-7B
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-7B \
  --served-model-name qwen3-7b \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# 部署 Qwen3-14B（需要更多显存）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B \
  --served-model-name qwen3-14b \
  --port 8000 \
  --tensor-parallel-size 2  # 双GPU并行
```

### 集成到项目

```bash
MODEL_MODE=local
LOCAL_MODEL_SERVER=http://localhost:8000/v1
LOCAL_MODEL=qwen3-7b
```

### vLLM 适用场景

- 生产环境部署
- 高并发请求处理
- GPU服务器（至少1张A10/A100）
- 需要高吞吐量的场景

---

## 量化方案对比

量化可以减小模型体积、加速推理，代价是精度略有下降。

| 方案 | 压缩比 | 速度提升 | 精度影响 | 适用工具 |
|------|--------|---------|---------|---------|
| AWQ | 4bit，约75% | 2到3倍 | 小 | vLLM原生支持 |
| GPTQ | 4bit，约75% | 2到3倍 | 小 | vLLM原生支持 |
| GGUF | 多种精度 | 1.5到3倍 | 取决于精度 | Ollama/llama.cpp |

### 推荐策略

1. **开发调试**：用云端 qwen-plus，不需要本地部署
2. **Demo演示**：Ollama + qwen3:7b，笔记本即可运行
3. **生产部署**：vLLM + Qwen3-14B-AWQ，A10 GPU即可
4. **私有化部署**：vLLM + 量化模型，根据客户硬件条件选型

---

## 部署选型建议（TAM视角）

| 客户类型 | 推荐方案 | 理由 |
|---------|---------|------|
| 中小企业 | 云端API按量付费 | 成本低，无需运维 |
| 金融/政务 | 私有化 vLLM | 数据敏感，必须私有部署 |
| 边缘场景 | Ollama + 量化模型 | 网络受限，需要离线运行 |
| 大型企业 | vLLM集群 + 负载均衡 | 高并发，多业务线共享 |
