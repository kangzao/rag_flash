# RAG Flash - 财报问答系统

基于 RAG (Retrieval-Augmented Generation) 的财报智能问答系统，支持从 PDF 报告提取信息并回答用户问题。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline 主流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐          │
│  │ PDF Parser│ -> │ Text Splitter │ -> │ Vector Store │          │
│  │  (Docling)│    │  (LangChain)  │    │  (FAISS)    │          │
│  └──────────┘    └──────────────┘    └─────────────┘          │
│         │                                    │                   │
│         v                                    v                   │
│  ┌──────────────┐                    ┌─────────────┐           │
│  │ Markdown转换  │                    │  Query路由   │           │
│  │  (MinerU)    │                    │   + 检索    │           │
│  └──────────────┘                    └─────────────┘           │
│                                              │                   │
│                                              v                   │
│                                    ┌─────────────────┐           │
│                                    │  LLM 生成答案   │           │
│                                    │  (通义千问)     │           │
│                                    └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 关键链路时序图

### 1. PDF 解析流程

```
PDF文件 -> Docling解析 -> JSON中间结果 -> TextSplitter -> Chunked JSON
                                │
                                v
                          MinerU转换 -> Markdown文件
```

### 2. 向量库构建流程

```
Chunked JSON -> EmbeddingService(DashScope) -> FAISS向量库
                                        │
                                        v
                              vector_dbs/{company}.faiss
```

### 3. 问答流程

```
用户问题 -> Router(判断问题类型) -> Vector检索 -> LLM重排序(可选)
                                              │
                                              v
                                    构造Prompt + 调用通义千问
                                              │
                                              v
                                        输出答案JSON
```

## 文件夹结构

```
rag_flash/
├── src/                          # 源代码
│   ├── pipeline.py               # 主流程编排
│   ├── pdf_parsing.py            # PDF解析 (Docling)
│   ├── pdf_mineru.py             # PDF转Markdown (MinerU)
│   ├── text_splitter.py          # 文本分块 (LangChain)
│   ├── ingestion.py              # 向量库构建 (FAISS)
│   ├── retrieval.py              # 检索器
│   ├── reranking.py              # LLM重排序
│   ├── questions_processing.py    # 问答处理
│   ├── api_requests.py           # API调用封装
│   ├── prompts.py                # Prompt模板
│   ├── services/
│   │   └── embedding.py          # Embedding服务
│   └── ...
│
├── data/                         # 数据目录 (不提交)
│   └── stock_data/
│       ├── subset.csv            # 公司元数据 (file_name -> company_name)
│       ├── questions/            # 问题列表
│       │   └── questions.json
│       ├── pdf_reports/          # 原始PDF报告 (输入)
│       ├── databases/            # 向量库 (输出)
│       │   ├── vector_dbs/       # FAISS向量库
│       │   └── chunked_reports/  # 分块后的JSON
│       ├── debug_data/           # 中间产物
│       │   └── 03_reports_markdown/  # Markdown文件
│       └── answers/              # 答案输出
│
├── tests/                        # 测试
├── docs/                         # 文档
├── requirements.txt              # 依赖
└── README.md
```

## 数据文件分类

| 类型 | 路径 | 说明 | 是否提交 |
|------|------|------|---------|
| 输入 | `subset.csv` | 公司元数据映射表 | ✅ |
| 输入 | `questions/` | 问题列表 JSON | ✅ |
| 输入 | `pdf_reports/` | 原始 PDF 报告 | ❌ |
| 输出 | `databases/` | 向量库、分块文档 | ❌ |
| 输出 | `answers/` | 答案文件 | ❌ |
| 输出 | `debug_data/` | 中间处理产物 | ❌ |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key_here
JINA_API_KEY=your_jina_api_key_here  # 可选，用于Reranker
```

### 3. 准备数据

将 PDF 报告放入 `data/stock_data/pdf_reports/`，确保 `subset.csv` 包含正确的文件映射。

### 4. 运行流程

```python
from src.pipeline import Pipeline, RunConfig
from pathlib import Path

root_path = Path("data/stock_data")

# 使用默认配置
pipeline = Pipeline(root_path)

# 解析PDF
pipeline.parse_pdf_reports_parallel()

# 构建向量库
pipeline.create_vector_dbs()

# 问答
pipeline.process_questions()
```

## 配置说明

`RunConfig` 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `answering_model` | `qwen-turbo` | 通义千问模型 |
| `top_n_retrieval` | `10` | 检索返回数量 |
| `parent_document_retrieval` | `False` | 启用父文档检索 |
| `llm_reranking` | `False` | 启用 LLM 重排序 |
| `parallel_requests` | `1` | 并发请求数 |

## 技术栈

- **LangChain** - LLM 应用框架
- **FAISS** - 向量相似度检索
- **Docling** - PDF 文档解析
- **MinerU** - PDF 转 Markdown
- **DashScope** - 通义千问 API

## License

MIT
