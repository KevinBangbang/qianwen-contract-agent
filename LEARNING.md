# 学习笔记：智能合同审查Agent

本文档记录每个Phase的代码讲解和学习要点。配合代码阅读，从实践中理解原理。

---

## Phase 1：DashScope API 和模型生态

### 代码讲解

#### config/model_config.py 逐行解析

这个文件是整个项目的配置中心，解决的核心问题是：**如何让同一套代码既能调用云端API，又能调用本地模型？**

```python
from openai import OpenAI
client = OpenAI(
    api_key=config["api_key"],
    base_url=config["base_url"],  # 关键：通过切换base_url实现云端/本地切换
)
```

**为什么用OpenAI兼容接口？**

DashScope提供了两套API：
1. 原生DashScope SDK（`dashscope`包）：阿里自己的接口格式
2. OpenAI兼容接口（`openai`包）：跟OpenAI的API格式一模一样

我们选择第2种，原因：
- OpenAI接口是行业事实标准，所有开发者都熟悉
- vLLM、Ollama等本地部署工具也提供OpenAI兼容接口
- 切换模型供应商只需改`base_url`，业务代码零修改
- 这也是TAM给客户的推荐做法

#### quick_test.py 逐行解析

这个脚本验证了四个核心API能力：

1. **文本生成** `chat.completions.create`
   - 最基础的能力，发送消息列表，返回模型回复
   - `temperature=0.7` 控制随机性，0是确定性输出，1是最随机

2. **文本向量化** `embeddings.create`
   - 把文本转成1024维的数字向量
   - 语义相近的文本，向量的余弦相似度高
   - 这是RAG检索的基础

3. **多模态理解** 发送图片URL给VL模型
   - content字段支持混合类型：text和image_url
   - 用于合同扫描件的识别

4. **流式输出** `stream=True`
   - 模型一边生成一边返回，不用等全部生成完
   - 前端实时展示"正在思考"的效果

### 模型选型指南

| 模型 | 适用场景 | 成本 | 推荐度 |
|------|---------|------|--------|
| qwen-turbo | 简单任务、高并发 | 最低 | 开发调试用 |
| qwen-plus | 通用任务、性价比 | 中等 | TAM最常推荐 |
| qwen-max | 复杂推理、高质量 | 最高 | 关键场景用 |
| qwen-vl-plus | 图片理解 | 中等 | OCR必备 |
| text-embedding-v3 | 文本向量化 | 很低 | RAG必备 |

### 关键概念

**OpenAI兼容模式的工作原理**

DashScope在服务端做了一层协议转换：
- 接收OpenAI格式的请求
- 内部转换成千问模型的原生格式
- 调用千问模型推理
- 把结果转换回OpenAI格式返回

这意味着任何支持OpenAI API的工具、框架、客户端都可以直接接入千问。

### 动手练习

1. 复制 `.env.example` 为 `.env`，填入你的API Key
2. 运行 `python quick_test.py` 验证四个测试全部通过
3. 修改 `quick_test.py`，尝试：
   - 切换到 `qwen-turbo` 和 `qwen-max`，对比回答质量
   - 把 `temperature` 改为 0 和 1，对比输出差异
   - 在文本生成测试中加入多轮对话（多条消息）

---

## Phase 2：Qwen-Agent 工具系统

### 代码讲解

#### 工具开发的标准模式

每个工具都遵循相同的模式，以 `clause_extractor.py` 为例：

```python
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('clause_extractor')
class ClauseExtractor(BaseTool):
    # description 会被注入到Agent的prompt中
    # 模型根据这个描述决定"什么时候该调用这个工具"
    description = '从合同全文中提取关键条款...'

    # parameters 定义工具的输入参数，JSON Schema格式
    # 模型会根据这个schema生成正确格式的参数
    parameters = [{
        'name': 'contract_text',
        'type': 'string',
        'description': '合同全文文本内容',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        # params 是模型传来的JSON字符串
        # _verify_json_format_args 帮我们解析并校验参数
        params = self._verify_json_format_args(params)
        # 业务逻辑在这里实现...
```

**关键理解：`@register_tool` 装饰器做了什么？**

查看 `qwen_agent/tools/base.py` 源码，注册逻辑很简单：
```python
TOOL_REGISTRY = {}  # 全局工具注册表

def register_tool(name):
    def decorator(cls):
        TOOL_REGISTRY[name] = cls  # 把类对象存入字典
        cls.name = name
        return cls
    return decorator
```
这样Agent初始化时只需要传工具名字符串，框架就能从注册表中找到对应的类并实例化。

#### 5个工具的设计思路

| 工具 | 核心能力来源 | 说明 |
|------|------------|------|
| contract_parser | PyPDF2 + Qwen-VL | PDF提取用代码，扫描件用VL模型OCR |
| clause_extractor | LLM prompt | 纯靠prompt驱动模型做结构化提取 |
| risk_checker | LLM + 规则库 | 内置风险规则 + LLM推理分析 |
| amount_calculator | 纯Python | 数值计算不用LLM，代码更准确 |
| report_generator | 纯Python | 模板化报告生成，不依赖LLM |

**设计原则：能用代码解决的不用LLM。** LLM擅长理解语义、生成文本，但数值计算容易出错。所以金额计算和报告拼装都用确定性的Python代码实现。

#### amount_calculator.py 的特殊之处

这是唯一不调用LLM的工具。亮点：
- 违约金计算支持固定比例和按日两种方式
- 自动检测违约金是否超过30%（司法实践中的判断标准）
- 金额大写转换（壹万贰仟...）用于核对合同中的大小写金额是否一致

#### contract_parser.py 的OCR设计

图片OCR使用base64编码方式发送给VL模型，而不是URL方式。原因：
- URL方式要求图片在公网可访问，本地文件做不到
- base64编码直接把图片数据发送给模型，无网络依赖
- 这也是生产环境中处理敏感合同文件的更安全做法

### Function Calling 的底层原理

工具调用并不是魔法，本质是prompt engineering：

1. Agent初始化时，读取每个工具的 `name`、`description`、`parameters`
2. 这些信息被格式化后拼接到system message中
3. 模型被训练成：当需要外部能力时，输出特定格式的JSON
4. Qwen-Agent框架解析这个JSON，找到对应的工具类，调用 `call` 方法
5. 执行结果作为新的消息再喂给模型，继续推理

整个过程是一个循环：**模型思考 -> 选择工具 -> 执行工具 -> 观察结果 -> 继续思考**

### 动手练习

1. 给 `clause_extractor.py` 添加一个 `contract_type` 参数，让模型能指定合同类型
2. 修改 `risk_checker.py` 的 `DEFAULT_RISK_RULES`，添加你认为重要的风险规则
3. 在 `amount_calculator.py` 中添加一个新的计算类型（比如利息计算）
4. 故意修改某个工具的 `description`，改成模糊的描述，观察Agent是否还能正确调用

---

## Phase 3：RAG 检索增强生成

### 代码讲解

#### build_kb.py 知识库构建流程

整个RAG流程分四步：**加载 -> 分块 -> 向量化 -> 存储**

```
法律文档(.txt) -> 按段落分块(500字/块, overlap=50) -> DashScope Embedding -> 本地JSON存储
```

**分块策略的选择**

```python
def split_into_chunks(documents, chunk_size=500, overlap=50):
    # 先按段落分割（双换行符 \n\n）
    paragraphs = doc['content'].split('\n\n')
    # 短段落合并，长段落再切分
```

为什么用500字的chunk_size：
- 太大（1000+）：检索时会匹配到大段无关内容，噪声多
- 太小（200以下）：一个法律条文被截断，语义不完整
- 500字是中文法律文档的经验平衡点

overlap=50的作用：相邻块有50字重叠，防止关键信息恰好在切块边界丢失

#### 向量检索的原理

```python
# 余弦相似度计算
query_norm = query_embedding / np.linalg.norm(query_embedding)
kb_norms = kb_embeddings / np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
similarities = np.dot(kb_norms, query_norm)
```

核心思想：把文本变成数字向量后，用数学方法衡量"语义距离"。
- "违约金过高" 和 "违约金超过30%" 的向量方向接近，余弦相似度高
- "违约金过高" 和 "合同签署日期" 的向量方向差异大，相似度低

#### risk_checker.py 的RAG集成

集成后的风险检查流程：

1. 接收合同条款
2. 将条款作为query，从知识库检索相关法律条文（top 3）
3. 将检索到的法条 + 内置风险规则合并为参考依据
4. 连同合同条款一起发送给LLM
5. LLM基于法律依据分析风险，输出评估结果

这就是RAG的完整流程：**Retrieval（检索）-> Augmentation（增强prompt）-> Generation（生成回答）**

### RAG效果调优的TAM经验

客户最常问的问题："为什么RAG回答不准确？"

排查顺序：
1. **检查分块质量**：分块太大太小都会影响检索精度
2. **检查Embedding匹配度**：用 `search_similar` 测试查询和结果的相似度分数
3. **检查prompt模板**：检索到了但模型没用好，通常是prompt设计问题
4. **检查知识库覆盖度**：知识库缺少相关文档，检索不到自然不准

80%的RAG问题出在前两步，而不是模型能力。

### 动手练习

1. 修改 `chunk_size` 为 200 和 1000，运行 `build_kb.py` 后对比检索效果
2. 在 `legal_docs/` 中添加一些无关文档（如菜谱），观察检索结果中的噪声
3. 修改 `search_similar` 的 `top_k` 参数，从1到5对比，观察召回率和精确率的变化

---

## Phase 4：Agent编排和ReAct

### 代码讲解

#### review_agent.py 主审查Agent

核心就是一行代码创建Agent：

```python
agent = Assistant(
    llm=llm_cfg,                    # 模型配置
    function_list=tool_list,        # 可用工具列表
    system_message=REVIEW_AGENT_SYSTEM_PROMPT,  # 系统提示词
)
```

Agent初始化后，调用 `agent.run(messages)` 开始工作。内部会进入ReAct循环：

```
用户输入 -> 模型思考 -> 选择工具 -> 执行工具 -> 观察结果 -> 继续思考 -> ... -> 最终回复
```

Agent不需要我们硬编码调用顺序。给了它5个工具和system prompt后，它会自主决定：
- 先用 contract_parser 解析文件
- 再用 clause_extractor 提取条款
- 然后用 risk_checker 检查风险
- 用 amount_calculator 验证金额
- 最后用 report_generator 生成报告

#### orchestrator.py 编排器设计

编排器使用简单的if-else路由：

```python
if input_type == 'image':
    ocr_text = run_ocr(file_path)     # 先OCR
    yield from _run_review(ocr_text)   # 再审查
elif input_type == 'pdf':
    yield from _run_review_with_prompt(prompt)  # 让Agent自己调用parser
elif input_type == 'text':
    yield from _run_review(contract_text)  # 直接审查
```

为什么不用另一个Agent做路由？因为文件类型判断是确定性的逻辑，用代码比用LLM更可靠。
只在需要语义理解的地方用LLM，确定性的逻辑用代码。这是Agent设计的核心原则。

#### System Prompt 的设计

`config/prompts.py` 中的 `REVIEW_AGENT_SYSTEM_PROMPT` 决定了Agent 80%的行为：

1. **角色设定**：告诉模型它是"专业的合同审查助手"
2. **职责范围**：明确审查的步骤和内容
3. **审查标准**：列出具体的检查维度
4. **输出格式**：要求中文回答、给出风险等级和修改建议

好的prompt = 明确的角色 + 清晰的任务 + 具体的标准 + 固定的格式

### ReAct范式详解

ReAct = Reasoning + Acting，是当前Agent最主流的工作模式。

一个完整的ReAct循环：
1. **Thought**：模型思考当前应该做什么
2. **Action**：选择一个工具并传入参数
3. **Observation**：获取工具执行结果
4. **Thought**：基于结果继续思考
5. 重复2-4直到任务完成
6. **Final Answer**：输出最终回答

Qwen3原生支持ReAct，框架自动处理了思考/工具调用/结果回传的循环。

### 动手练习

1. 修改 `REVIEW_AGENT_SYSTEM_PROMPT`，把角色改为"严格的法律审查员"，对比审查风格变化
2. 从 `function_list` 中移除 `amount_calculator`，观察Agent如何应对缺少工具的情况
3. 在 `prompts.py` 中添加few-shot示例，给一个"输入条款 -> 期望输出"的例子，观察质量变化

---

## Phase 5：前端和产品化

### 代码讲解

#### gradio_app.py 界面设计

界面结构：

```
[模型选择下拉框]  [状态信息]
[Tabs: 上传文件 | 输入文本]
    [上传区/文本框]
    [开始审查按钮]
[审查报告（Markdown）] [Agent推理过程]
```

关键设计决策：

1. **两种输入方式**：文件上传和文本粘贴，适配不同使用场景
2. **模型切换**：让用户在速度和质量之间自主选择
3. **推理过程展示**：右侧显示Agent的工具调用链，让用户看到"AI在思考"
4. **进度条**：`gr.Progress()` 提供实时反馈

#### 为什么选择Gradio而不是React/Vue

- Gradio：10分钟搭建Demo，Python一把梭，适合原型验证和内部展示
- React/Vue：生产级前端，需要前端工程师，适合面向终端用户
- TAM场景：客户看Demo用Gradio足够，生产环境再换专业前端

### 动手练习

1. 在Gradio界面中添加"审查历史"功能（用 `gr.State` 保存历史记录）
2. 添加一个"导出PDF报告"按钮
3. 尝试用 `gr.ChatInterface` 替代当前界面，做成对话式审查

---

## Phase 6：模型部署和性能优化

### 代码讲解

#### benchmark.py 性能测试设计

测试四个维度：

1. **TTFT（首token延迟）**：从发出请求到收到第一个token的时间
   - 影响用户体验："点击后多久开始看到输出"
   - 云端通常0.5到2秒，本地取决于模型大小和GPU

2. **生成速度（tokens/秒）**：每秒生成的token数
   - qwen-turbo约33 tok/s，qwen-plus约11 tok/s
   - 速度差异主要来自模型大小

3. **端到端时间**：完成一次完整审查的总耗时

4. **提取质量**：通过关键词匹配评估结构化提取的准确度

#### Ollama vs vLLM 选型

| 维度 | Ollama | vLLM |
|------|--------|------|
| 安装难度 | 一条命令 | 需要配置CUDA |
| 适用硬件 | CPU/低端GPU | 需要GPU |
| 并发性能 | 一般 | 优秀（PagedAttention） |
| 量化支持 | GGUF格式 | AWQ/GPTQ |
| 适用场景 | 开发/Demo | 生产环境 |

#### 量化的本质

量化 = 用更少的bit表示模型权重。

- FP16（原始）：每个权重16bit -> 7B模型约14GB显存
- INT4（量化）：每个权重4bit -> 7B模型约4GB显存
- 代价：部分精度损失，对简单任务影响小，对复杂推理有影响

### 动手练习

1. 安装Ollama，下载qwen3:7b，对比本地和云端的审查结果差异
2. 修改benchmark.py，添加qwen-max的测试
3. 尝试调整vLLM的 `max-model-len` 参数，观察对性能的影响

---

## 完成所有Phase后的综合回顾

### 你应该能回答的TAM面试题

1. "千问有哪些模型，分别适合什么场景？" -> 参考Phase 1的模型选型表
2. "如何为客户搭建一个RAG知识库？常见的坑有哪些？" -> 参考Phase 3
3. "Agent和普通的API调用有什么区别？" -> Agent有ReAct循环和工具调用，API是单次请求
4. "客户说RAG效果不好，你怎么排查？" -> 分块质量 > Embedding匹配度 > prompt > 知识库覆盖度
5. "客户想要私有化部署，你推荐什么方案？" -> 参考Phase 6的部署选型表
6. "千问和竞品比有什么优势？" -> 开源生态完善、工具链齐全、性价比高

### 项目技术栈总结

| 层次 | 技术 | 用途 |
|------|------|------|
| 模型层 | DashScope API / Ollama | 千问模型调用 |
| 框架层 | Qwen-Agent | Agent编排和工具管理 |
| 工具层 | 5个自定义BaseTool | 合同审查业务逻辑 |
| 知识层 | DashScope Embedding + numpy | RAG向量检索 |
| 前端层 | Gradio | 交互界面 |
| 部署层 | Ollama / vLLM | 本地模型部署 |
