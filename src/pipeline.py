# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import shutil
import time

from src.pdf_parsing import PDFParser
from src import pdf_mineru
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor


@dataclass
class PipelineConfig:
    """
    路径配置类：管理整个RAG流程中所有输入输出文件的目录结构

    核心职责：
    1. 根据是否使用序列化表格生成不同的数据库后缀（_ser_tab / 无后缀）
    2. 统一管理原始数据、中间产物、最终结果的存储路径
    3. 隔离不同阶段的输出目录（解析、分块、向量化）
    """

    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json",
                 pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 基础路径配置
        self.root_path = root_path
        # 根据是否使用序列化表格添加后缀，区分不同版本的向量数据库
        suffix = "_ser_tab" if serialized else ""

        # === 输入文件路径 ===
        self.subset_path = root_path / subset_name  # 公司元数据CSV（file_name -> company_name映射）
        self.questions_file_path = root_path / questions_file_name  # 问题列表JSON文件
        self.pdf_reports_dir = root_path / pdf_reports_dir_name  # 原始PDF报告目录

        # === 输出文件路径 ===
        self.answers_file_path = root_path / f"answers{config_suffix}.json"  # 答案输出文件（支持后缀区分不同配置）
        self.debug_data_path = root_path / "debug_data"  # 调试数据根目录
        self.databases_path = root_path / f"databases{suffix}"  # 数据库目录（向量库+分块文档）

        # === 数据库子目录 ===
        self.vector_db_dir = self.databases_path / "vector_dbs"  # FAISS向量数据库目录（每个公司一个.faiss文件）
        self.documents_dir = self.databases_path / "chunked_reports"  # 分块后的JSON文档目录
        self.bm25_db_path = self.databases_path / "bm25_dbs"  # BM25稀疏检索数据库路径

        # === 中间处理目录（已注释，保留用于历史参考）===
        # self.parsed_reports_dirname = "01_parsed_reports"  # Docling解析后的原始JSON
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"  # 解析调试信息
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"  # 合并后的报告
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"  # MinerU转换的Markdown文件目录

        # self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        # self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        # self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname  # Markdown文件实际存储路径


@dataclass
class RunConfig:
    """
    运行流程参数配置类：控制RAG管道的各个功能开关和超参数

    核心参数说明：
    - use_serialized_tables: 是否使用序列化表格增强检索（将表格转换为自然语言描述）
    - parent_document_retrieval: 父文档检索策略（检索小片段但返回更大上下文）
    - llm_reranking: 是否使用LLM对检索结果进行重排序
    - parallel_requests: 并发请求数（需根据API限流调整，Qwen-Turbo限制500 QPM）
    - full_context: 是否使用完整上下文（影响prompt长度和token消耗）
    """
    # 功能开关
    use_serialized_tables: bool = False  # 启用序列化表格插入到文本分块中
    parent_document_retrieval: bool = False  # 启用父文档检索（提升上下文完整性）
    use_vector_dbs: bool = True  # 使用向量数据库进行语义检索
    use_bm25_db: bool = False  # 使用BM25进行关键词检索
    llm_reranking: bool = False  # 启用LLM重排序（提升检索质量但增加延迟）

    # 超参数配置
    llm_reranking_sample_size: int = 30  # LLM重排序的候选文档数量
    top_n_retrieval: int = 10  # 最终返回给LLM的检索结果数量
    parallel_requests: int = 1  # 并行API请求数（需限制以避免超出Qwen-Turbo阈值）

    # 元数据配置
    pipeline_details: str = ""  # 管道描述信息（记录在答案文件中用于追踪实验）
    submission_file: bool = True  # 是否生成提交格式的答案文件
    full_context: bool = False  # 是否提供完整上下文给LLM

    # API配置
    api_provider: str = "dashscope"  # API提供商（dashscope=阿里云，openai=OpenAI）
    answering_model: str = "qwen-turbo-latest"  # 回答模型（可选：gpt-4o-mini, gpt-4o, qwen-turbo等）
    config_suffix: str = ""  # 配置文件后缀（用于区分不同实验的答案文件）


class Pipeline:
    """
    RAG主流程编排类：协调PDF解析、文本分块、向量化、检索、问答生成的完整流水线

    核心工作流程：
    1. 初始化阶段：加载路径配置、转换数据格式（JSON->CSV）
    2. 预处理阶段：PDF解析 -> Markdown转换 -> 文本分块 -> 向量化建库
    3. 推理阶段：问题检索 -> LLM重排序 -> 答案生成 -> 结果保存

    设计特点：
    - 支持多种配置组合（通过RunConfig灵活切换）
    - 自动处理文件名冲突（_get_next_available_filename）
    - 单问模式支持（answer_single_question用于交互式测试）
    """

    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json",
                 pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        """
        初始化主流程，加载路径和配置

        数据流向：
        root_path (数据根目录)
          ├── subset.csv (公司元数据)
          ├── questions.json (问题列表)
          └── pdf_reports/ (原始PDF)

        参数：
            root_path: 数据集根目录（包含所有输入文件）
            subset_name: 公司元数据文件名
            questions_file_name: 问题列表文件名
            pdf_reports_dir_name: PDF报告目录名
            run_config: 运行配置（控制功能开关和超参数）
        """
        self.run_config = run_config
        # 根据配置初始化所有路径（不同配置可能产生不同的数据库目录）
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        # 自动转换JSON格式的子集文件为CSV（兼容性处理）
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str,
                          pdf_reports_dir_name: str) -> PipelineConfig:
        """
        根据配置初始化所有路径

        关键逻辑：
        - 将run_config中的use_serialized_tables传递给PipelineConfig
        - 不同配置会生成不同的数据库目录后缀（避免覆盖）
        """
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,  # 决定是否添加_ser_tab后缀
            config_suffix=self.run_config.config_suffix,  # 决定答案文件后缀
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV

        数据加工逻辑：
        1. 检测JSON格式的元数据文件（旧版本格式）
        2. 使用pandas将JSON数组转换为DataFrame
        3. 导出为标准CSV格式（保持列名不变）

        目的：统一后续处理的输入格式（所有模块都期望CSV输入）
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"

        if json_path.exists() and not csv_path.exists():
            try:
                # 读取JSON数组
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 转换为DataFrame（每行代表一个公司/文件）
                df = pd.DataFrame(data)

                # 导出为CSV（不包含索引列）
                df.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models():
        """
        下载Docling所需模型，避免首次运行时自动下载

        执行流程：
        1. 创建临时PDFParser实例
        2. 解析一个dummy PDF文件（触发模型下载）
        3. 模型缓存到本地（后续运行无需重复下载）

        注意：这是一个一次性操作，通常在部署时执行
        """
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """
        多进程并行解析PDF报告，提升处理效率

        数据加工流程：
        1. 创建PDFParser实例（配置输出目录和元数据路径）
        2. 扫描pdf_reports目录下所有PDF文件
        3. 将文件列表分块（每个chunk包含chunk_size个PDF）
        4. 启动max_workers个进程并行处理
        5. 每个worker调用Docling解析PDF -> 提取文本/表格/图片 -> 输出JSON

        内部处理步骤（由PDFParser完成）：
        - Docling解析PDF结构（识别标题、段落、表格、图片）
        - 提取页面级文本内容（保留Markdown格式）
        - 序列化表格（将表格转换为结构化JSON）
        - 关联元数据（从subset.csv获取company_name）

        参数：
            chunk_size: 每个worker处理的PDF数量（影响内存占用）
            max_workers: 并发worker数（受CPU核心数限制）
        """
        logging.basicConfig(level=logging.DEBUG)

        # 创建PDF解析器（输出到parsed_reports目录）
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path,  # 用于补充company_name字段
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        # 获取所有待处理的PDF文件路径
        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        # 并行解析（内部使用multiprocessing.Pool）
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size,
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def export_reports_to_markdown(self, file_name):
        """
        使用MinerU API将指定PDF文件转换为Markdown格式

        数据加工流程：
        1. 调用MinerU云端API上传PDF并获取task_id
        2. 轮询查询任务状态直到完成
        3. 下载并解压结果包（包含full.md + 图片资源）
        4. 移动full.md到reports_markdown目录并重命名

        内部处理步骤（由MinerU完成）：
        - OCR识别扫描件文字
        - 布局分析（识别标题层级、段落边界）
        - 公式渲染（LaTeX格式）
        - 表格重建（Markdown表格语法）
        - 图片提取（单独保存为PNG）

        参数：
            file_name: PDF文件名（如'【财报】中芯国际：中芯国际2024年年度报告.pdf'）

        输出：
            debug_data/03_reports_markdown/{base_name}.md
        """
        # 调用 pdf_mineru 获取 task_id 并下载、解压
        print(f"开始处理: {file_name}")
        task_id = pdf_mineru.get_task_id()  # 上传PDF并启动异步任务
        print(f"task_id: {task_id}")
        pdf_mineru.get_result(task_id)  # 等待任务完成并下载结果

        # 解压后目录名与 task_id 相同
        extract_dir = f"{task_id}"
        md_path = os.path.join(extract_dir, "full.md")  # MinerU输出的完整Markdown文件
        if not os.path.exists(md_path):
            print(f"未找到 markdown 文件: {md_path}")
            return

        # 创建目标目录
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)

        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")

        # 移动文件到目标位置
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将Markdown报告分块，便于后续向量化和检索

        数据加工流程：
        1. 扫描reports_markdown目录下所有.md文件
        2. 对每个文件按行分割（默认30行/块，重叠5行）
        3. 为每个分块添加元数据（lines范围、text内容）
        4. 从subset.csv读取company_name并写入metainfo
        5. 输出JSON格式的分块文件到documents_dir

        内部处理步骤（由TextSplitter完成）：
        - 按固定行数切分（保证上下文连贯性）
        - 计算每个分块的起止行号（便于溯源）
        - 构建标准JSON结构：{"metainfo": {...}, "content": {"chunks": [...]}}

        输出示例：
        {
          "metainfo": {
            "sha1": "abc123...",
            "company_name": "中芯国际",
            "file_name": "【财报】中芯国际：中芯国际2024年年度报告.md"
          },
          "content": {
            "chunks": [
              {"lines": [1, 30], "text": "..."},
              {"lines": [26, 55], "text": "..."}  // 重叠5行
            ]
          }
        }

        参数：
            include_serialized_tables: （预留）是否插入序列化表格分块
        """
        text_splitter = TextSplitter()

        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")

        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,  # 输入：Markdown文件目录
            output_dir=self.paths.documents_dir,  # 输出：分块JSON目录
            subset_csv=self.paths.subset_path,  # 元数据映射表
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """
        从分块报告创建FAISS向量数据库

        数据加工流程：
        1. 扫描documents_dir下所有分块JSON文件
        2. 对每个公司的分块文本进行Embedding（使用DashScope/BGE模型）
        3. 构建FAISS索引（扁平索引或IVF索引）
        4. 保存索引文件和元数据到vector_db_dir

        内部处理步骤（由VectorDBIngestor完成）：
        - 读取分块JSON，提取所有chunk的text字段
        - 批量调用Embedding API（考虑速率限制）
        - 将向量矩阵存入FAISS索引
        - 建立向量ID到原文的映射关系

        输出：
        databases/vector_dbs/stock_{stock_id}.faiss  # 每个公司一个索引文件

        用途：
        - 支持语义相似度检索（余弦相似度/内积）
        - 加速Top-K检索（毫秒级响应）
        """
        input_dir = self.paths.documents_dir  # 分块JSON目录
        output_dir = self.paths.vector_db_dir  # FAISS索引输出目录

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def create_bm25_db(self):
        """
        从分块报告创建BM25稀疏检索数据库

        数据加工流程：
        1. 扫描documents_dir下所有分块JSON文件
        2. 构建倒排索引（词项 -> 文档列表）
        3. 计算每个词的IDF权重
        4. 序列化索引到磁盘（pickle格式）

        内部处理步骤（由BM25Ingestor完成）：
        - 中文分词（使用jieba或类似工具）
        - 过滤停用词
        - 统计词频（TF）和逆文档频率（IDF）
        - 构建BM25评分函数

        输出：
        databases/bm25_dbs/bm25_index.pkl

        用途：
        - 关键词精确匹配（适合专有名词、数字查询）
        - 与向量检索互补（混合检索策略）
        """
        input_dir = self.paths.documents_dir  # 分块JSON目录
        output_file = self.paths.bm25_db_path  # BM25索引输出路径

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        """
        解析PDF报告的入口方法（支持串行/并行模式）

        当前实现：仅支持并行模式（串行模式已废弃）

        参数：
            parallel: 是否使用并行处理（默认为True）
            chunk_size: 每个worker处理的PDF数
            max_workers: 并发worker数
        """
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库

        数据流向：
        Markdown文件 -> 文本分块(JSON) -> FAISS向量索引

        执行顺序：
        Step 1: chunk_reports()
                输入：debug_data/03_reports_markdown/*.md
                输出：databases/chunked_reports/*.json

        Step 2: create_vector_dbs()
                输入：databases/chunked_reports/*.json
                输出：databases/vector_dbs/*.faiss
        """
        print("开始处理报告流程...")

        print("步骤1：报告分块...")
        self.chunk_reports()

        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()

        print("报告处理流程已成功完成！")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀

        处理逻辑：
        1. 检查base_path是否存在
        2. 若不存在，直接返回
        3. 若存在，尝试添加_01, _02, ...后缀直到找到可用文件名

        示例：
        answers.json -> answers_01.json -> answers_02.json -> ...

        目的：避免覆盖历史实验结果
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem  # 文件名（不含扩展名）
        suffix = base_path.suffix  # 扩展名（如.json）
        parent = base_path.parent  # 父目录

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"  # 格式化编号（两位数）
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        """
        批量处理所有问题，生成答案文件

        数据加工流程：
        1. 初始化QuestionsProcessor（加载向量库、配置检索策略）
        2. 遍历questions.json中的所有问题
        3. 对每个问题执行：
           a. 路由判断（确定查询范围：单公司/多公司/全量）
           b. 向量检索（从FAISS获取Top-K相关分块）
           c. LLM重排序（可选，提升相关性）
           d. 父文档检索（可选，扩展上下文）
           e. 构造Prompt（注入检索结果 + 问题）
           f. 调用LLM生成答案
           g. 解析答案（提取结构化字段）
        4. 保存答案到JSON文件（支持自动编号避免覆盖）

        内部处理步骤（由QuestionsProcessor完成）：
        - 路由器：分析问题类型（财务指标/事件/对比等）
        - 检索器：根据路由结果选择检索策略
        - 重排序器：使用LLM对候选文档打分
        - 答案生成器：构造Self-Organizing CoT Prompt

        输出格式：
        {
          "question": "...",
          "answer": "...",
          "kind": "number|string|boolean|names",
          "metadata": {
            "retrieved_docs": [...],
            "pipeline_details": "..."
          }
        }
        """
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,  # FAISS索引目录
            documents_dir=self.paths.documents_dir,  # 分块JSON目录
            questions_file_path=self.paths.questions_file_path,  # 问题列表
            new_challenge_pipeline=True,  # 启用新版管道
            subset_path=self.paths.subset_path,  # 公司元数据
            parent_document_retrieval=self.run_config.parent_document_retrieval,  # 父文档检索开关
            llm_reranking=self.run_config.llm_reranking,  # LLM重排序开关
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,  # 重排序候选数
            top_n_retrieval=self.run_config.top_n_retrieval,  # 最终返回文档数
            parallel_requests=self.run_config.parallel_requests,  # 并发请求数
            api_provider=self.run_config.api_provider,  # API提供商
            answering_model=self.run_config.answering_model,  # 回答模型
            full_context=self.run_config.full_context,  # 完整上下文开关
        )

        # 获取可用文件名（避免覆盖）
        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        # 批量处理所有问题
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,  # 是否生成提交格式
            pipeline_details=self.run_config.pipeline_details,  # 管道描述（记录在答案中）
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）

        使用场景：
        - 交互式测试（快速验证某个问题的效果）
        - Debug调试（观察中间步骤）
        - 在线问答服务（实时响应）

        数据加工流程：
        1. 初始化QuestionsProcessor（加载向量库和索引）
        2. 调用process_single_question：
           a. 路由判断
           b. 向量检索
           c. LLM重排序（可选）
           d. 构造Prompt
           e. 调用LLM
           f. 解析答案
        3. 返回结构化答案字典

        性能监控：
        - 记录初始化耗时（加载向量库的时间）
        - 记录推理耗时（检索+LLM生成的时间）
        - 输出总耗时（端到端延迟）

        参数：
            question: 问题文本
            kind: 答案类型（'string'/'number'/'boolean'/'names'）

        返回：
            dict: 包含answer、metadata等字段的结构化答案
        """
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")

        # 初始化处理器（与批量处理相同配置）
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,  # 单问模式强制串行
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context,
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1 - t0:.2f} 秒")

        print("[计时] 开始调用 process_single_question ...")
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2 - t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2 - t0:.2f} 秒")
        return answer


# === 预定义配置模板 ===

# 预处理配置（用于表格序列化）
preprocess_configs = {
    "ser_tab": RunConfig(use_serialized_tables=True),  # 启用序列化表格
    "no_ser_tab": RunConfig(use_serialized_tables=False),  # 不启用序列化表格
}

# 基础配置：简单向量检索 + Self-Organizing CoT
base_config = RunConfig(
    parallel_requests=10,  # 10并发
    submission_file=True,  # 生成提交格式
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base",  # 答案文件后缀：answers_base.json
    answering_model="gpt-4o-mini-2024-07-18",
)

# 父文档检索配置：检索小片段但返回更大上下文
parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,  # 启用父文档检索
    parallel_requests=20,  # 提高并发到20
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",  # 使用更强的GPT-4o
    config_suffix="_pdr",  # 答案文件后缀：answers_pdr.json
)

## 最大配置：启用所有高级功能（父文档检索 + LLM重排序）
max_config = RunConfig(
    use_serialized_tables=False,  # 不启用序列化表格（可根据需要修改）
    parent_document_retrieval=True,  # 启用父文档检索
    llm_reranking=True,  # 启用LLM重排序（提升精度但增加延迟）
    parallel_requests=4,  # 降低并发（重排序增加单次请求耗时）
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen-turbo",
    answering_model="qwen-turbo-latest",  # 使用Qwen-Turbo
    config_suffix="_qwen_turbo",  # 答案文件后缀：answers_qwen_turbo.json
)

# 配置注册表（方便切换不同实验配置）
configs = {
    "base": base_config,  # 基础版
    "pdr": parent_document_retrieval_config,  # 父文档检索版
    "max": max_config,  # 完整版（最高精度）
}

# === 主执行入口 ===
# 可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    # print(type(root_path))

    # 初始化主流程，使用推荐的最佳配置（max_config）
    pipeline = Pipeline(root_path, run_config=max_config)

    print('4. 将pdf转化为纯markdown文本')
    pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf')

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports()

    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()
