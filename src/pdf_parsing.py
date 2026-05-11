import os
import time
import logging
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List

# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

_log = logging.getLogger(__name__)

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """
    在独立进程中处理一组PDF文件的辅助函数。
    
    该函数被ProcessPoolExecutor的子进程调用，每个子进程创建独立的PDFParser实例，
    避免多进程间的状态共享问题。
    
    参数:
        pdf_paths (List[Path]): 当前chunk需要处理的PDF文件路径列表
        pdf_backend: PDF解析后端类（如DoclingParseV2DocumentBackend）
        output_dir (Path): 解析后JSON文件的输出目录
        num_threads (int): 每个进程使用的线程数（控制OCR和表格识别的并行度）
        metadata_lookup (dict): 元数据查找字典，key为sha1哈希，value包含company_name等信息
        debug_data_path (Path): 调试数据的保存路径（存储Docling原始输出）
    
    返回:
        str: 处理结果描述字符串，格式为"Processed X PDFs."
    """
    # 为当前进程创建独立的parser实例（避免多进程共享状态）
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None,  # 元数据查找表直接通过赋值传递
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    """
    PDF解析器主类，负责将PDF文档转换为结构化的JSON格式。
    
    核心功能：
    1. 使用Docling库进行布局分析、OCR识别、表格提取
    2. 支持单进程和多进程并行处理模式
    3. 自动补充公司元数据（从CSV文件）
    4. 页面序列标准化（填补空白页）
    """
    
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
    ):
        """
        初始化PDF解析器。
        
        参数:
            pdf_backend: PDF解析后端类，默认为DoclingParseV2DocumentBackend（第二代解析引擎）
            output_dir (Path): 解析后JSON文件的输出目录，默认为./parsed_pdfs
            num_threads (int): 每个进程使用的线程数，设置为None时使用默认值。
                              会设置环境变量OMP_NUM_THREADS控制底层库的并行度
            csv_metadata_path (Path): CSV元数据文件路径，包含sha1到company_name的映射。
                                     如果为None，则不补充公司名称信息
        """
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None

        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)
            
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """
        解析CSV文件并创建以sha1为键的元数据查找字典。
        
        数据加工逻辑：
        1. 读取CSV文件（期望包含sha1、company_name列）
        2. 兼容旧版格式（name列代替company_name）
        3. 去除公司名称字段的双引号包裹
        4. 构建字典：{sha1_hash: {"company_name": "xxx"}}
        
        参数:
            csv_path (Path): CSV文件路径，文件格式示例：
                           sha1,company_name
                           abc123,"中芯国际"
                           def456,"比亚迪"
        
        返回:
            dict: 元数据查找字典，结构为：
                  {
                      "abc123...": {"company_name": "中芯国际"},
                      "def456...": {"company_name": "比亚迪"}
                  }
        """
        import csv
        metadata_lookup = {}
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 兼容新旧CSV格式：优先使用company_name，不存在则使用name
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name,
                }
        return metadata_lookup

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """
        创建并配置DocumentConverter实例，设置PDF解析管线的各项参数。
        
        配置项说明：
        1. OCR配置：启用EasyOCR识别扫描件，主要针对英文和数字优化
        2. 表格配置：使用TableFormer高精度模式识别复杂表格结构
        3. 后端选择：使用DoclingParseV2提升解析质量
        
        返回:
            DocumentConverter: 配置好的文档转换器实例，可用于批量转换PDF
        """
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend,
            ),
        }
        
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        """
        批量转换PDF文档为结构化数据。
        
        内部处理流程：
        1. 读取PDF文件流
        2. 逐页进行布局分析（识别标题、段落、表格、图片）
        3. 对扫描区域执行OCR识别
        4. 提取表格结构和单元格内容
        5. 建立元素间的引用关系（texts/tables/pictures数组）
        
        参数:
            input_doc_paths (List[Path]): PDF文件路径列表，如[Path("report1.pdf"), Path("report2.pdf")]
        
        返回:
            Iterable[ConversionResult]: 转换结果迭代器，每个结果包含：
                - status: 转换状态（SUCCESS/FAILURE）
                - input: 输入文件信息
                - document: 结构化文档对象（可通过export_to_dict()导出）
        """
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results
    
    def process_documents(self, conv_results: Iterable[ConversionResult]):
        """
        处理转换结果，生成标准化的JSON报告文件。
        
        数据加工步骤：
        1. 检查转换状态，统计成功/失败数量
        2. 对成功的结果：
           a. 导出为字典格式
           b. 标准化页面序列（填补空白页）
           c. 组装报告（提取元信息、内容、表格、图片）
           d. 保存为JSON文件（使用原始文件名）
        3. 记录失败的文档
        
        参数:
            conv_results (Iterable[ConversionResult]): convert_documents返回的转换结果流
        
        返回:
            tuple[int, int]: (成功数量, 失败数量)，如(8, 2)表示10个文档中8个成功
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)
                
                # 标准化文档数据以确保页面连续
                data = conv_res.document.export_to_dict()
                normalized_data = self._normalize_page_sequence(data)
                
                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """
        确保内容中的页码是连续的，通过填充空白页来填补缺失的页码。
        
        问题场景：
        某些PDF可能跳过空白页或封面，导致页码不连续，如pages = [1, 2, 5, 7]
        这会导致后续按页码索引时出现KeyError。
        
        处理逻辑：
        1. 获取所有现有页码集合和最大页码
        2. 遍历1到max_page的所有页码
        3. 对于缺失的页码，创建空页面模板（content=[], page_dimensions={}）
        4. 重新构建连续的content数组
        
        参数:
            data (dict): 从conv_res.document.export_to_dict()导出的文档字典，
                        期望包含'content'键，值为页面列表
        
        返回:
            dict: 标准化后的文档字典，保证page字段从1到max_page连续分布
        """
        if 'content' not in data:
            return data
        
        # 创建数据副本以避免修改原始数据
        normalized_data = data.copy()
        
        # 获取现有页码集合并找到最大页码
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)
        
        # 创建空白页模板
        empty_page_template = {
            "content": [],
            "page_dimensions": {},  # 或者使用默认页面尺寸
        }
        
        # 创建包含所有页面的新content数组
        new_content = []
        for page_num in range(1, max_page + 1):
            # 查找现有页面或创建空白页
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template},
            )
            new_content.append(page_content)
        
        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        """
        解析PDF文档并导出为JSON格式（单进程模式）。
        
        执行流程：
        1. 获取待处理的PDF文件列表（优先使用input_doc_paths，否则扫描doc_dir）
        2. 调用convert_documents进行批量转换
        3. 调用process_documents处理转换结果并保存JSON
        4. 统计耗时和成功率
        5. 如果有失败的文档，抛出RuntimeError异常
        
        参数:
            input_doc_paths (List[Path], optional): PDF文件路径列表。如果为None则使用doc_dir
            doc_dir (Path, optional): 包含PDF文件的目录，会自动扫描*.pdf文件
        
        异常:
            RuntimeError: 当有任何文档解析失败时抛出，包含失败文件路径列表
        """
        start_time = time.time()
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))
        
        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")
        
        conv_results = self.convert_documents(input_doc_paths)
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time

        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)

        _log.info(f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}")

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None,
    ):
        """
        使用多进程并行解析PDF文件，显著提升大批量处理效率。
        
        并行策略：
        1. 将PDF文件列表分割为多个chunks（每个chunk包含多个PDF）
        2. 使用ProcessPoolExecutor启动多个worker进程
        3. 每个worker进程独立处理一个chunk（调用_process_chunk）
        4. 主进程等待所有chunks完成，汇总统计结果
        
        性能优化：
        - chunk_size控制每个worker的任务量，避免内存溢出
        - optimal_workers根据CPU核心数和PDF总数动态调整
        - 子进程间无状态共享，避免锁竞争
        
        参数:
            input_doc_paths (List[Path], optional): PDF文件路径列表
            doc_dir (Path, optional): 包含PDF文件的目录（当input_doc_paths为None时使用）
            optimal_workers (int): worker进程数量，默认为10。如果为None则使用CPU核心数
            chunk_size (int): 每个worker处理的PDF数量。如果为None则自动计算
                             （总PDF数 // worker数，至少为1）
        
        示例:
            假设有20个PDF，optimal_workers=5：
            - chunk_size = 20 // 5 = 4
            - 生成5个chunks，每个包含4个PDF
            - 5个worker并行处理，理论上提速5倍
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # 如果未提供输入路径，从目录扫描PDF文件
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        
        cpu_count = multiprocessing.cpu_count()
        
        # 如果未指定worker数，使用CPU核心数和PDF总数的较小值
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)
        
        if chunk_size is None:
            # 计算chunk大小（确保至少为1）
            chunk_size = max(1, total_pdfs // optimal_workers)
        
        # 将文档分割为chunks
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0
        
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # 提交所有任务到进程池
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path,
                )
                for chunk in chunks
            ]
            
            # 等待任务完成并记录结果（as_completed按完成顺序返回）
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])  # 从"Processed X PDFs"中提取数字
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    """
    JSON报告处理器，负责将Docling的原始转换结果组装为标准化的报告结构。
    
    核心职责：
    1. 提取元信息（SHA1哈希、公司名称）
    2. 展开组引用（groups -> texts/tables/pictures）
    3. 按页面组织内容结构
    4. 转换表格为Markdown和HTML格式
    5. 提取图片及其子元素（标题、说明文字）
    
    输出结构：
    {
        "metainfo": {"sha1": "...", "company_name": "..."},
        "content": [{"page": 1, "content": [...], "page_dimensions": {...}}, ...],
        "tables": [{"table_id": 0, "page": 5, "markdown": "...", "html": "..."}, ...],
        "pictures": [{"picture_id": 0, "page": 3, "bbox": [...], "children": [...]}, ...]
    }
    """
    
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        """
        初始化JSON报告处理器。
        
        参数:
            metadata_lookup (dict, optional): 元数据查找字典，key为sha1，value包含company_name
            debug_data_path (Path, optional): 调试数据保存路径，用于存储Docling原始输出
        """
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """
        组装完整的报告结构，整合元信息、内容、表格和图片。
        
        数据加工流程：
        1. 从normalized_data或conv_result获取文档字典
        2. 调用assemble_metainfo提取元信息
        3. 调用assemble_content按页面组织文本/表格/图片引用
        4. 调用assemble_tables转换表格为多种格式
        5. 调用assemble_pictures提取图片及其子元素
        6. 可选：保存调试数据（Docling原始输出）
        
        参数:
            conv_result (ConversionResult): Docling转换结果对象
            normalized_data (dict, optional): 标准化后的文档字典。如果为None则从conv_result导出
        
        返回:
            dict: 组装完成的报告字典，包含metainfo、content、tables、pictures四个顶级字段
        """
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report
    
    def assemble_metainfo(self, data):
        """
        提取报告的元信息，包括SHA1哈希和公司名称。
        
        数据流向：
        1. 从data['origin']['sha1']获取PDF的唯一标识
        2. 使用sha1在metadata_lookup中查找公司名称
        3. 返回精简的元信息字典
        
        参数:
            data (dict): 文档字典，期望包含'origin'键，其中有'sha1'字段
        
        返回:
            dict: 元信息字典，结构为：
                  {
                      "sha1": "abc123...",
                      "company_name": "中芯国际"  # 如果metadata_lookup中存在
                  }
        """
        metainfo = {}
        if 'sha1' in data['origin']:
            metainfo['sha1'] = data['origin']['sha1']
        if self.metadata_lookup and metainfo.get('sha1') in self.metadata_lookup:
            csv_meta = self.metadata_lookup[metainfo['sha1']]
            metainfo['company_name'] = csv_meta['company_name']
        return metainfo

    def process_table(self, table_data):
        """
        处理表格数据（预留方法，当前未使用）。
        
        参数:
            table_data: 表格数据
        
        返回:
            str: 处理后的表格内容占位符
        """
        # Implement your table processing logic here
        return 'processed_table_content'

    def debug_data(self, data):
        """
        保存调试数据，存储Docling的原始输出以便排查问题。
        
        文件命名规则：{doc_name}.json
        其中doc_name来自data['name']（通常是PDF文件名不含扩展名）
        
        参数:
            data (dict): 要保存的文档字典
        """
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)    
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        """
        展开body_children中的组引用，将groups的子元素扁平化到body_children中。
        
        问题场景：
        Docling的输出中，body.children可能包含对groups的引用：
        [{"$ref": "#/groups/0"}, {"$ref": "#/texts/0"}]
        
        groups[0]可能包含多个子元素：
        {"name": "财务报表", "children": [{"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}]}
        
        该方法将groups展开，使每个子元素都带上group信息：
        [{"$ref": "#/texts/1", "group_id": 0, "group_name": "财务报表", ...}, ...]
        
        参数:
            body_children (list): body的子元素列表，可能包含对groups的引用
            groups (list): 文档中的所有逻辑分组（章节、段落组等）
        
        返回:
            list: 展开后的子元素列表，所有原本在groups中的元素都被提升到顶层，
                 并附加group_id、group_name、group_label字段
        """
        expanded_children = []

        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children
    
    def _process_text_reference(self, ref_num, data):
        """
        处理文本引用并创建内容项字典。
        
        数据加工逻辑：
        1. 从data['texts'][ref_num]获取文本项
        2. 提取核心字段：text（清理后文本）、type（元素类型）、text_id（索引）
        3. 条件添加orig字段（仅当与text不同时）
        4. 添加可选字段：enumerated（是否为枚举项）、marker（列表标记）
        
        参数:
            ref_num (int): 文本项在texts数组中的索引，如0表示texts[0]
            data (dict): 文档数据字典，期望包含'texts'键
        
        返回:
            dict: 处理后的内容项，结构为：
                  {
                      "text": "营业收入",
                      "type": "section_header",
                      "text_id": 5,
                      "orig": "营​业​收​入",  # 可选，含不可见字符
                      "enumerated": false,     # 可选
                      "marker": "•"            # 可选
                  }
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num,
        }
        
        # 仅当orig与text不同时才添加orig字段
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # 如果存在则添加额外字段
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']
            
        return content_item
    
    def assemble_content(self, data):
        """
        按页面组织文档内容，将分散的texts/tables/pictures引用归类到对应页面。
        
        数据加工流程：p
        1. 展开body.children中的groups引用
        2. 遍历展开后的元素列表
        3. 解析每个元素的$ref引用（如#/texts/5、#/tables/2）
        4. 根据引用类型分别处理：
           - texts: 提取文本内容，附加group信息
           - tables: 创建表格占位符（仅记录table_id）
           - pictures: 创建图片占位符（仅记录picture_id）
        5. 从prov字段获取页码，将元素归类到对应页面
        6. 按页码排序返回页面列表
        
        参数:
            data (dict): 文档字典，期望包含：
                        - body.children: 主体内容的引用列表
                        - groups: 逻辑分组列表
                        - texts: 所有文本块数组
                        - tables: 所有表格数组
                        - pictures: 所有图片数组
        
        返回:
            list[dict]: 按页码排序的页面列表，每个页面结构为：
                       {
                           "page": 1,
                           "content": [
                               {"text": "...", "type": "section_header", "text_id": 0, "group_id": 1, ...},
                               {"type": "table", "table_id": 2},
                               {"type": "picture", "picture_id": 0}
                           ],
                           "page_dimensions": {"l": 0, "t": 0, "r": 595, "b": 842}
                       }
        """
        pages = {}
        # 展开body children以包含groups引用
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # 处理body内容
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)

                    # 如果可用则添加group信息
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # 从prov获取页码
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # 如果页面不存在则初始化
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {}),
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num,
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {}),
                            }

                        pages[page_num]['content'].append(content_item)
                
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num,
                    }
                    
                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {}),
                            }
                        
                        pages[page_num]['content'].append(content_item)

        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        """
        组装表格数据，将每个表格转换为多种格式（JSON、Markdown、HTML）。
        
        数据加工步骤：
        1. 遍历所有表格对象
        2. 调用model_dump()获取表格的完整JSON表示
        3. 调用_table_to_md()将网格数据转换为Markdown表格
        4. 调用export_to_html()生成HTML格式
        5. 从prov字段提取页码和边界框坐标
        6. 从data字段获取行数和列数
        7. 从self_ref解析table_id
        8. 组装为统一的表格对象
        
        参数:
            tables (list[Table]): Docling的表格对象列表，每个对象支持model_dump()和export_to_html()
            data (dict): 文档字典，其中的'tables'键包含表格的元数据（页码、位置、行列数等）
        
        返回:
            list[dict]: 组装后的表格列表，每个表格结构为：
                       {
                           "table_id": 0,
                           "page": 10,
                           "bbox": [100, 200, 400, 500],  # [left, top, right, bottom]
                           "#-rows": 8,
                           "#-cols": 6,
                           "markdown": "| 年份 | 营收 |\\n|------|------|\\n| 2024 | 100 |",
                           "html": "<table>...</table>",
                           "json": {...}  # 完整的表格JSON结构
                       }
        """
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html()
            
            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'], 
                table_bbox['r'],
                table_bbox['b'],
            ]
            
            # 从表格数据结构获取行数和列数
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj,
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        """
        将表格JSON对象转换为Markdown格式的表格字符串。
        
        处理逻辑：
        1. 从table['data']['grid']提取二维单元格数组
        2. 遍历每个单元格，获取cell['text']字段
        3. 判断第一行是否为表头（数据行数>1且第一行非空）
        4. 使用tabulate库生成GitHub风格的Markdown表格
        5. 处理异常情况（禁用数字解析）
        
        Markdown表格示例：
        | 年份 | 营业收入 | 净利润 |
        |------|----------|--------|
        | 2024 | 100亿    | 20亿   |
        | 2023 | 90亿     | 18亿   |
        
        参数:
            table (dict): 表格JSON对象，期望包含'data.grid'字段，
                         结构为[[cell, cell, ...], ...]，每个cell包含'text'字段
        
        返回:
            str: Markdown格式的表格字符串
        """
        # 从网格单元格提取文本
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)
        
        # 检查表格是否有表头
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github",
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")
        
        return md_table

    def assemble_pictures(self, data):
        """
        组装图片数据，提取图片位置信息和子元素（标题、说明文字）。
        
        处理步骤：
        1. 遍历所有图片项
        2. 调用_process_picture_block提取图片的子元素（引用的文本）
        3. 从self_ref解析picture_id
        4. 从prov字段提取页码和边界框坐标
        5. 组装为图片对象（包含children列表）
        
        参数:
            data (dict): 文档字典，期望包含'pictures'键，值为图片数组
        
        返回:
            list[dict]: 组装后的图片列表，每个图片结构为：
                       {
                           "picture_id": 0,
                           "page": 3,
                           "bbox": [100, 200, 400, 500],
                           "children": [
                               {"text": "图1：营收增长趋势", "type": "caption", "text_id": 15}
                           ]
                       }
        """
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)
            
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'], 
                picture_bbox['r'],
                picture_bbox['b'],
            ]
            
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures
    
    def _process_picture_block(self, picture, data):
        """
        处理图片块，提取图片关联的文本子元素（如标题、说明文字）。
        
        处理逻辑：
        1. 遍历picture['children']中的引用
        2. 解析引用类型（期望为#/texts/X）
        3. 调用_process_text_reference获取文本内容
        4. 收集所有文本子元素
        
        参数:
            picture (dict): 图片数据字典，期望包含'children'字段，值为引用列表
            data (dict): 文档数据字典，用于查找引用的文本项
        
        返回:
            list[dict]: 图片的子元素列表，每个元素为处理后的文本项
        """
        children_list = []
        
        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)
                        
                    children_list.append(content_item)

        return children_list

    def export_to_markdown(self, reports_dir: Path, output_dir: Path):
        """
        将处理后的JSON报告导出为Markdown文件（此方法在当前版本中未被使用）。
        
        注意：此方法调用了未定义的process_report方法，可能需要修复。
        
        参数:
            reports_dir (Path): 包含JSON报告文件的目录
            output_dir (Path): Markdown文件的输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for report_path in reports_dir.glob("*.json"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            processed_report = self.process_report(report_data)
            document_text = ""
            for page in processed_report['pages']:
                document_text += f"\n\n---\n\n# Page {page['page']}\n\n"
                document_text += page['text']
            # 用 sha1 作为 markdown 文件名
            report_name = report_data['metainfo'].get('sha1', 'unknown')
            with open(output_dir / f"{report_name}.md", "w", encoding="utf-8") as f:
                f.write(document_text)
