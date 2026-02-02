# MatriXNest (矩巢 / MxN)

一套基于 **RAG (Retrieval-Augmented Generation)** 架构的智能文档存储与精准检索工具，旨在将专业文档（工程预算、技术规范、行业标准）转化为 AI 生产力引擎。

![MatiXNest](images/figure1.png)

## 核心特性

- **语义级精准检索** — 突破传统关键词限制，理解条款、表格与章节间的逻辑关联
- **表格感知智能切分** — 自动检测并合并跨页长表格，切块时注入表头与章节层级信息
- **可信溯源 (Trustworthy Sourcing)** — 每个回答精准标注文档出处（精确到页码和段落），减少 AI 幻觉
- **全链路自动化 Pipeline** — 支持 GB 级长文档的稳定解析，内置断点续传与批量 Embedding 机制
- **可视化运维监控** — Streamlit 界面支持实时审查 OCR 识别精度与切块逻辑合理性

## 技术架构

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Streamlit │────▶│  Orchestrator   │────▶│   Mistral   │
│   Frontend  │     │     Agent       │     │    APIs     │
└─────────────┘     └────────┬────────┘     └─────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Ingestion Agent │ │   Query Agent   │ │ Vector Database │
│  - OCR Extract  │ │  - Retrieval    │ │    (ChromaDB)   │
│  - Chunking     │ │  - Reranker     │ │                 │
│  - Embedding    │ │  - Generation   │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘b
```

## 技术栈

| 组件 | 技术 |
|------|------|
| RAG 架构 | Mistral OCR + Embedding + Cross-encoder Reranker |
| 向量数据库 | ChromaDB |
| 前端框架 | Streamlit |
| 文档解析 | PyMuPDF + TableAwareChunker |
| 核心模型 | mistral-ocr-latest, mistral-embed, mistral-large-latest |

## 快速开始

### 前提条件

- Python 3.10+
- Mistral AI API Key ([获取地址](https://console.mistral.ai/api-keys))

### 1. 克隆仓库

```bash
git clone <repository-url>
cd MatriXNest
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# MISTRAL_API_KEY=your_api_key_here
```

### 4. 准备文档

将你的 PDF 文档放入 `data/` 目录：

```bash
# 示例：放置一份市政工程预算定额 PDF
cp your_document.pdf data/Tunnel\ budget.pdf
```

### 5. 构建向量索引

```bash
python ingest.py
```

首次运行会自动：
- OCR 提取文本、表格、图片
- 智能切块（表格感知）
- 批量 Embedding 向量化
- 存入 ChromaDB

### 6. 启动应用

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`

## 项目结构

```
MatriXNest/
├── app.py              # Streamlit 前端应用
├── ingest.py           # 文档摄入与向量化 Pipeline
├── rag.py              # RAG 查询逻辑（检索 + 生成）
├── chunker.py          # 表格感知智能切块器
├── config.py           # 配置管理
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
├── data/               # 文档目录（PDF）
├── vectorstore/        # 向量数据库（自动生成，不提交 Git）
└── introduction/       # 项目介绍文档
```

## 配置说明

### 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `MISTRAL_API_KEY` | Mistral AI API Key | ✅ |

### 可调参数（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_CHUNK_SIZE` | 1500 | 文本块最大字符数 |
| `CHUNK_OVERLAP` | 200 | 块间重叠字符数 |
| `TOP_K_RESULTS` | 5 | 检索返回的文档块数量 |
| `EMBEDDING_MODEL` | mistral-embed | 向量化模型 |
| `CHAT_MODEL` | mistral-large-latest | 问答生成模型 |

## 应用场景

本项目特别适用于：
- 📋 **工程造价查询** — 快速检索预算定额、单价分析
- 📘 **技术规范查阅** — 语义理解行业标准条款
- 📊 **报告检索** — 跨页表格数据关联查询
- 🔍 **合规审查** — 精准定位政策条款出处

## 示例查询

```
"在路面标线中，纵向标线的工程预算定额是多少？"
"在现浇混凝土工程中，承台的工程预算定额是多少？"
"在隧道爆破开挖中，平洞钻爆开挖工程预算定额是多少？当断面面积在100平方米以内？"
```

## 设计亮点

### 高扩展解耦设计

- **OCR 层**、**Embedding 层**、**Chat 层** 完全解耦
- 通过环境变量即可无缝切换模型提供商

### 表格感知技术

```python
# 智能检测跨页表格并合并
TableAwareChunker:
  - 识别表格边界
  - 合并跨页连续表格
  - 切块时保留表头上下文
```

## 贡献指南

欢迎提交 Issue 和 PR！

## License

[Apache-2.0](LICENSE) © 2025 MatriXNest Contributors
