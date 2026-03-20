# 云端部署方案

## 使用DashScope API（推荐）

最简单的部署方式，适合大多数场景。

### 优势

1. **零运维**：无需管理GPU服务器
2. **按量付费**：用多少付多少，无前期投入
3. **自动扩容**：阿里云自动处理并发扩展
4. **模型自动升级**：qwen-plus会自动路由到最新版本

### 部署步骤

1. 注册阿里云百炼平台 https://bailian.console.aliyun.com
2. 创建API Key
3. 配置 `.env` 文件
4. 部署Gradio应用到云服务器

### 生产环境建议

```bash
# 使用 gunicorn 部署（Linux）
pip install gunicorn
gunicorn app.gradio_app:create_app() -w 4 -b 0.0.0.0:7860

# 或者用 Docker
docker build -t contract-review .
docker run -p 7860:7860 --env-file .env contract-review
```

### 成本估算

以合同审查场景为例（每份合同约3000字）：

| 模型 | 每次审查Token | 单价 | 每份合同成本 |
|------|------------|------|------------|
| qwen-turbo | 约5000 | 0.3元/百万token | 约0.0015元 |
| qwen-plus | 约5000 | 2元/百万token | 约0.01元 |
| qwen-max | 约5000 | 20元/百万token | 约0.1元 |

按qwen-plus计算，审查1万份合同的API成本约100元。
