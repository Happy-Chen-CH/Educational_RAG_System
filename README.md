<p align="center">
  <h1 align="center">🎓 EduRAG — 教育领域智能问答系统</h1>
  <p align="center">
    一个面向教育场景的 <b>RAG (检索增强生成)</b> 智能问答系统，融合关键词匹配与语义检索双引擎，<br>
    支持多种文档格式、自动查询分类、智能策略选择与流式 LLM 答案生成。
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Milvus-2.5+-green.svg" alt="Milvus">
  <img src="https://img.shields.io/badge/MySQL-8.0+-orange.svg" alt="MySQL">
  <img src="https://img.shields.io/badge/Redis-7.0+-red.svg" alt="Redis">
  <img src="https://img.shields.io/badge/LLM-DashScope%20(qwen)-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

---

## 📖 目录

- [系统架构](#-系统架构)
- [核心特性](#-核心特性)
- [项目结构](#-项目结构)
- [查询流程](#-查询流程)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 安装依赖](#2-安装依赖)
  - [3. 配置服务](#3-配置服务)
  - [4. 导入数据](#4-导入数据)
  - [5. 运行系统](#5-运行系统)
- [配置说明](#-配置说明)
- [RAG 检索策略](#-rag-检索策略)
- [技术栈](#-技术栈)
- [常见问题](#-常见问题)

---

## 🏗 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                     main.py (顶层入口)                         │
│                   IntegratedQASystem                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐        ┌──────────────────────────┐ │
│  │     mysql_qa         │        │        rag_qa            │ │
│  │   (关键词检索引擎)     │        │     (语义检索引擎)        │ │
│  │                      │        │                          │ │
│  │  • BM25 关键词匹配    │        │  • BERT 查询意图分类      │ │
│  │  • Redis 查询缓存     │  未命中 │  • LLM 检索策略选择       │ │
│  │  • MySQL 答案存储     │ ─────→ │  • Milvus 混合向量检索    │ │
│  │  • jieba 中文分词     │        │  • BGE-Reranker 精排     │ │
│  │                      │        │  • DashScope LLM 生成     │ │
│  └─────────────────────┘        └──────────────────────────┘ │
│                                                               │
│                    ▼ 最终答案 ▼                                │
│              • 流式输出 (token-by-token)                       │
│              • 对话历史持久化 (MySQL)                           │
└──────────────────────────────────────────────────────────────┘
```

**两级检索策略：**
1. **第一级 — BM25 关键词匹配**：对已有明确问答对，直接通过关键词匹配快速返回精确答案
2. **第二级 — RAG 语义检索**：当关键词匹配置信度不足时，自动进入深度语义检索流程

---

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| 🔍 **两级检索** | BM25 关键词匹配（快速）+ Milvus 混合向量检索（深度语义） |
| 🧠 **智能查询分类** | 微调 BERT 模型自动判断「通用知识」/「专业咨询」，避免无意义检索 |
| 🎯 **四种检索策略** | 直接检索、HyDE、子查询分解、回溯简化 — LLM 自动选择最优策略 |
| 📊 **混合检索 + 精排** | BGE-M3 稠密 + 稀疏双路检索 → BGE-Reranker Cross-Encoder 重排序 |
| 📄 **多格式文档支持** | PDF、DOCX、PPT/PPTX、PNG/JPG（OCR）、TXT、Markdown |
| ✂️ **父子块切分** | 小块检索（高精度）+ 大块返回（完整上下文），兼顾精度与召回 |
| 💬 **多轮对话** | MySQL 持久化存储会话历史，支持上下文连续对话 |
| ⚡ **Redis 缓存** | 预计算分词结果缓存 + 热门查询答案缓存，加速重复查询 |
| 🔌 **模型在线下载** | BERT/BGE/文档分割模型通过 HuggingFace/ModelScope 首次自动下载 |
| 🛡️ **环境变量注入** | 支持通过环境变量覆盖敏感配置，适配容器化部署 |

---

## 📁 项目结构

```
Educational_RAG_System/
├── main.py                    # 🔝 主入口：集成问答系统
├── config.ini.example         # 📋 配置文件模板（复制为 config.ini）
├── pyproject.toml             # 📦 项目元数据与依赖声明 (uv)
├── uv.lock                    # 🔒 锁定依赖版本 (可复现安装)
├── README.md                  # 📖 项目文档
├── .gitignore                 # 🔒 Git 忽略规则
│
├── base/                      # 🔧 基础工具模块
│   ├── config.py              #   配置管理（config.ini + 环境变量覆盖）
│   └── logger.py              #   统一日志系统
│
├── mysql_qa/                  # 🔑 关键词检索子系统
│   ├── mysql_main.py          #   独立运行入口（仅 BM25）
│   ├── db/mysql_client.py     #   MySQL 连接与 CRUD
│   ├── cache/redis_client.py  #   Redis 缓存管理
│   ├── retrieval/bm25_search.py  # BM25 关键词检索引擎
│   ├── utils/preprocess.py    #   jieba 中文分词预处理
│   └── data/                  #   MySQL 知识库 CSV 数据
│
├── rag_qa/                    # 🧠 语义检索子系统
│   ├── rag_main.py            #   独立运行入口（支持数据导入/交互查询）
│   ├── core/
│   │   ├── rag_system.py      #   RAG 主控：分类→策略→检索→生成
│   │   ├── vector_store.py    #   Milvus 向量存储与混合检索
│   │   ├── query_classifier.py #  BERT 查询意图分类器（含自动训练）
│   │   ├── strategy_selector.py # LLM 检索策略选择器
│   │   ├── prompts.py         #   Prompt 模板集合
│   │   └── document_processor.py # 文档加载与父子块切分
│   ├── edu_text_spliter/      #   中文文本分割器
│   ├── edu_document_loaders/  #   多格式文档加载器（含 OCR）
│   ├── classify_data/         #   BERT 分类器训练数据
│   ├── data/                  #   RAG 知识库文档
│   └── samples/               #   测试样本文件
│
├── logs/                      # 📝 日志目录
└── tmp_trainer/               # 🔄 模型训练临时目录
```

---

## 🔄 查询流程

```
用户输入查询
    │
    ▼
┌──────────────────────────┐
│ 1. BM25 关键词匹配         │  jieba 分词 → BM25 打分 → Softmax 归一化
│    置信度 ≥ 0.85?         │
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     │ YES       │ NO
     ▼           ▼
  返回精确     ┌──────────────────────────┐
  答案        │ 2. BERT 查询分类           │  bert-base-chinese 微调模型
              │    通用知识? / 专业咨询?    │
              └──────────┬───────────────┘
                         │
                   ┌─────┴─────┐
                   │ 通用知识   │ 专业咨询
                   ▼           ▼
              LLM 直接     ┌──────────────────────────┐
              回答        │ 3. LLM 策略选择            │  分析查询特征，从
                          │    直接 / HyDE /          │  四种策略中选最优
                          │    子查询 / 回溯          │
                          └──────────┬───────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────┐
                          │ 4. Milvus 混合检索         │  BGE-M3 Dense(稠密)
                          │    稠密 + 稀疏双路检索      │  + Sparse(稀疏) 向量
                          └──────────┬───────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────┐
                          │ 5. BGE-Reranker 重排序    │  Cross-Encoder 精排
                          │    父文档去重 → 精排       │  → Top-M 文档
                          └──────────┬───────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────┐
                          │ 6. LLM 上下文生成          │  检索文档 + 原始问题
                          │    DashScope 流式输出      │  → 自然语言答案
                          └──────────────────────────┘
```

---

## 📋 环境要求

| 组件 | 版本要求 | 用途 |
|---|---|---|
| Python | 3.9+ | 运行环境 |
| MySQL | 8.0+ | 问答对存储 + 对话历史 |
| Redis | 7.0+ | 分词缓存 + 答案缓存 |
| Milvus | 2.5+ | 向量存储与混合检索 |
| DashScope API Key | — | LLM 调用（阿里云灵积） |

> **💡 模型说明：** 所有深度学习模型（BGE-M3、BGE-Reranker、BERT、文档分割模型）首次运行时通过 HuggingFace/ModelScope 自动下载，无需手动下载模型文件。

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd Educational_RAG_System_online_model
```

### 2. 安装依赖

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖，首次使用请先安装 uv：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip 安装
pip install uv
```

然后一键同步所有依赖：

```bash
uv sync
```

> ⚠️ 依赖包含 PyTorch、Transformers 等大型框架，约需 3–5 分钟。uv 自动创建隔离的虚拟环境 (`.venv/`)。

运行系统：

```bash
uv run python main.py
```

### 3. 配置服务

**创建配置文件：**

```bash
cp config.ini.example config.ini
```

**编辑 `config.ini`，填入真实值：**

```ini
[mysql]
host = localhost
user = root
password = your_real_mysql_password
database = subject_kg

[redis]
host = localhost
port = 6379
password = your_real_redis_password

[milvus]
host = localhost
port = 19530

[llm]
dashscope_api_key = your_dashscope_api_key_here
dashscope_base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
```

> 🔒 **安全提示：** 也可以使用环境变量注入敏感信息（如 `DASHSCOPE_API_KEY`），环境变量优先级高于 `config.ini`。详见 `base/config.py`。

**启动基础服务（以 Docker 为例）：**

```bash
# 启动 MySQL
docker run -d --name mysql -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=your_password mysql:8.0

# 启动 Redis
docker run -d --name redis -p 6379:6379 \
  redis:7.0 --requirepass your_redis_password

# 启动 Milvus (使用 Milvus Standalone)
# 详见: https://milvus.io/docs/install_standalone-docker.md
```

### 4. 导入数据

**MySQL 问答数据（BM25 检索用）：**

```bash
cd mysql_qa
python db/mysql_client.py   # 自动建表 + 从 CSV 导入数据
cd ..
```

**知识库文档（RAG 语义检索用）：**

```bash
# 将文档放入对应学科目录，例如：
#   rag_qa/data/ai_data/    → AI 学科文档
#   rag_qa/data/java_data/  → Java 学科文档

# 执行数据导入（文档 → 切分 → 向量化 → 存入 Milvus）
python -m rag_qa.rag_main --data_processing --data_dir rag_qa/data
```

### 5. 运行系统

**方式一：集成问答系统（推荐）**

```bash
python main.py
```

```
欢迎使用集成问答系统！
会话ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
支持的学科类别：['ai', 'java', 'test', 'ops', 'bigdata']

请输入查询: AI学科有哪些课程？
请输入学科类别 (ai/java/test/ops/bigdata) (直接回车默认不过滤): ai

答案: AI学科主要包括以下课程：机器学习基础、深度学习...
```

**方式二：仅 RAG 语义检索**

```bash
python -m rag_qa.rag_main
```

**方式三：仅 MySQL 关键词检索**

```bash
python mysql_qa/mysql_main.py
```

---

## ⚙ 配置说明

### 配置文件结构 (`config.ini`)

| 配置段 | 键 | 说明 | 默认值 |
|---|---|---|---|
| `[mysql]` | `host`, `user`, `password`, `database` | MySQL 连接参数 | `localhost`, `root` |
| `[redis]` | `host`, `port`, `password`, `db` | Redis 连接参数 | `localhost:6379` |
| `[milvus]` | `host`, `port`, `database_name`, `collection_name` | Milvus 连接参数 | `localhost:19530` |
| `[llm]` | `model_name`, `dashscope_api_key`, `dashscope_base_url` | LLM API 配置 | `qwen3.7-plus` |
| `[retrieval]` | `parent_chunk_size`, `child_chunk_size`, `chunk_overlap` | 文档切分参数 | `1200`, `300`, `50` |
| `[retrieval]` | `retrieval_k`, `candidate_m` | 检索召回与最终候选数 | `5`, `2` |
| `[app]` | `valid_source`, `customer_service_phone` | 学科列表、客服电话 | `['mysql']` |
| `[logger]` | `log_file` | 日志文件路径 | `./logs/app.log` |

### 环境变量覆盖

所有 `config.ini` 配置项均可被同名环境变量覆盖（优先级更高），无需修改配置文件即可注入敏感信息：

```bash
export DASHSCOPE_API_KEY="sk-your-key"
export MYSQL_PASSWORD="secure_password"
export REDIS_PASSWORD="redis_pass"
```

---

## 🎯 RAG 检索策略

系统内置四种检索策略，由 LLM 根据查询特征自动选择：

| 策略 | 原理 | 适用场景 | 示例 |
|---|---|---|---|
| **直接检索** | 原始查询直接检索 | 查询意图明确，需特定信息 | "AI学科学费多少？" |
| **HyDE** | LLM 生成假答案 → 用假答案检索 | 查询较抽象，原始查询与文档语义不匹配 | "人工智能在教育中的应用" |
| **子查询检索** | LLM 拆分为 N 个子查询 → 分别检索 → 合并去重 | 查询涉及多方面比较 | "比较 Java 和 Python 的优缺点" |
| **回溯问题检索** | LLM 简化为基础问题 → 用简化问题检索 | 查询过于具体、细节太多 | "我有100亿条数据存Milvus可以吗？" |

策略选择由 `StrategySelector` 通过 LLM Few-Shot Prompt 自动完成，无需用户干预。

---

## 🛠 技术栈

| 层次 | 技术选型 |
|---|---|
| **LLM** | DashScope (qwen3.7-plus) / OpenAI 兼容 API |
| **嵌入模型** | BGE-M3 (BAAI) — 1024维稠密 + 稀疏双路向量 |
| **重排序模型** | BGE-Reranker-Large (BAAI) — Cross-Encoder 精排 |
| **查询分类** | bert-base-chinese 微调 — 二分类（通用/专业） |
| **向量数据库** | Milvus 2.5 — IVF_FLAT (稠密) + SPARSE_INVERTED_INDEX (稀疏) |
| **关键词检索** | BM25 (rank-bm25) + jieba 中文分词 |
| **缓存** | Redis — JSON 序列化存储 |
| **关系数据库** | MySQL (PyMySQL) — 问答对 + 对话历史 |
| **文档处理** | PyMuPDF、python-docx、python-pptx、RapidOCR、Unstructured |
| **文本分割** | ChineseRecursiveTextSplitter、MarkdownTextSplitter、语义分割 (ModelScope) |
| **深度学习框架** | PyTorch 2.7、Transformers 4.5、Sentence-Transformers 4.1 |

---

## ❓ 常见问题

### Q: 首次运行很慢？
首次运行会自动下载多个模型（BERT ~400MB、BGE-M3 ~2.2GB、BGE-Reranker ~2.1GB、文档分割 ~400MB），并可能自动训练 BERT 分类器（~10-20 分钟 CPU）。模型下载后会缓存，后续启动秒开。

### Q: 不想用 MySQL/Redis 可以吗？
可以。仅运行 RAG 子系统即可：`python -m rag_qa.rag_main`。BM25 关键词匹配和对话历史功能需要 MySQL；性能优化和预计算缓存需要 Redis。

### Q: 如何添加新的学科类别？
1. 在 `config.ini` 的 `valid_source` 列表中添加新学科（如 `math`）
2. 创建对应数据目录 `rag_qa/data/math_data/`，放入文档
3. 重新运行数据处理 `python -m rag_qa.rag_main --data_processing`

### Q: 支持其他 LLM 吗？
支持任何 OpenAI API 兼容的后端。修改 `config.ini` 中的 `dashscope_api_key` 和 `dashscope_base_url` 即可切换到其他服务（如 vLLM、Ollama、本地模型等）。

### Q: BERT 分类器已训练好，可以跳过训练吗？
训练好的模型保存在 `rag_qa/core/bert_query_classifier/` 目录。`.gitignore` 已排除此目录的模型文件，首次 clone 后会自动从 HuggingFace 下载基座并训练。你也可以手动放入预训练好的模型文件。

---

## 👤 作者

**Happy-Chen-CH** — [@Happy-Chen-CH](https://github.com/Happy-Chen-CH)

---

