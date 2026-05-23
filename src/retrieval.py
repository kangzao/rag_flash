"""检索模块：基于 LangChain 的 BM25、向量、混合检索."""
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pickle
import numpy as np
import time
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.reranking import LLMReranker
from src.exceptions import RetrievalError, DocumentNotFoundError

logger = logging.getLogger(__name__)


class BaseRetriever:
    """检索器基类"""
    
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        top_n: int = 3,
        **kwargs
    ) -> List[Dict]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """BM25 关键词检索器"""
    
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        self._doc_cache: Dict[str, dict] = {}
        self._pages_cache: Dict[str, Dict[int, dict]] = {}
        self._load_documents()

    def _load_documents(self):
        """预加载所有文档到内存"""
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                company_name = doc["metainfo"].get("company_name")
                if company_name:
                    self._doc_cache[company_name] = doc
                    pages = doc["content"].get("pages", [])
                    self._pages_cache[company_name] = {p["page"]: p for p in pages}
            except Exception as e:
                logger.error(f"加载文档失败 {path.name}: {e}")

    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        top_n: int = 3,
        **kwargs
    ) -> List[Dict]:
        document = self._doc_cache.get(company_name)
        if not document:
            raise DocumentNotFoundError(company_name)

        # 使用 base_name（file_name 去掉扩展名）与 BM25 文件名保持一致
        file_name = document["metainfo"].get("file_name", "")
        base_name = Path(file_name).stem if file_name else ""
        bm25_path = self.bm25_db_dir / f"{base_name}.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25索引不存在: {bm25_path}")
        
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        chunks = document["content"]["chunks"]
        pages_map = self._pages_cache.get(company_name, {})
        
        # BM25 评分
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        return self._build_results(
            top_indices, 
            chunks, 
            pages_map, 
            scores, 
            kwargs.get("return_parent_pages", False)
        )

    def _build_results(
        self, 
        indices: List[int], 
        chunks: List[dict], 
        pages_map: Dict[int, dict], 
        scores: List[float], 
        return_parent: bool
    ) -> List[Dict]:
        results, seen = [], set()
        for idx in indices:
            chunk = chunks[idx]
            parent = pages_map.get(chunk["page"])
            
            if return_parent and parent and parent["page"] not in seen:
                seen.add(parent["page"])
                results.append({
                    "distance": round(float(scores[idx]), 4),
                    "page": parent["page"],
                    "text": parent["text"]
                })
            else:
                results.append({
                    "distance": round(float(scores[idx]), 4),
                    "page": chunk["page"],
                    "text": chunk["text"]
                })
        return results


class VectorRetriever(BaseRetriever):
    """
    向量检索器（基于 LangChain FAISS）
    
    优势：
    - 自动处理 FAISS 索引加载
    - 内置缓存机制
    - 支持异步检索
    - 统一的 Document 抽象
    """
    
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
    ):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir

        # 初始化 DashScope Embeddings
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 加载所有向量存储
        self.vector_stores = self._load_vector_stores()
        logger.info(f"加载了 {len(self.vector_stores)} 个向量数据库")

    def _load_vector_stores(self) -> Dict[str, dict]:
        """
        加载所有 FAISS 向量存储

        返回：
            Dict[base_name, {store, company_name, file_name, pages}]
        """
        stores = {}

        for faiss_path in self.vector_db_dir.glob("*.faiss"):
            base_name = faiss_path.stem  # 文件名（不含扩展名），与 JSON 文件名一致
            doc_path = self.documents_dir / f"{base_name}.json"

            if not doc_path.exists():
                logger.warning(f"找不到对应的文档文件: {doc_path}")
                continue

            try:
                # 加载原始文档获取元数据
                with open(doc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                company_name = data["metainfo"].get("company_name", "")
                file_name = data["metainfo"].get("file_name", "")

                # ✅ LangChain 一行代码加载 FAISS 索引
                store = FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # 提取页面信息
                pages = data["content"].get("pages", [])

                stores[base_name] = {
                    "store": store,
                    "company_name": company_name,
                    "file_name": file_name,
                    "pages": {p["page"]: p for p in pages},
                }

            except Exception as e:
                logger.error(f"加载向量数据库失败 {faiss_path.name}: {e}")

        return stores

    def _find_store(self, company_name: str) -> tuple:
        """根据公司名查找对应的向量存储"""
        for base_name, data in self.vector_stores.items():
            if (data["company_name"] == company_name or
                company_name in data["file_name"]):
                return base_name, data

        raise DocumentNotFoundError(company_name)

    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        top_n: int = 3,
        **kwargs
    ) -> List[Dict]:
        """
        基于公司名和查询语句进行向量检索
        
        参数：
            company_name: 公司名称
            query: 查询问题
            top_n: 返回结果数量
            return_parent_pages: 是否返回父页面（更大上下文）
        
        返回：
            List[Dict] 包含 distance, page, text
        """
        try:
            base_name, store_data = self._find_store(company_name)
        except DocumentNotFoundError:
            raise
        
        store = store_data["store"]
        pages_map = store_data["pages"]
        return_parent = kwargs.get("return_parent_pages", False)
        
        # ✅ LangChain 内置相似度搜索（带分数）
        docs_with_scores = store.similarity_search_with_score(query, k=top_n)
        
        # 格式化结果
        results = []
        seen_pages = set()
        
        for doc, score in docs_with_scores:
            # 从 metadata 获取页码（如果有的话）
            page = doc.metadata.get("page", 0)
            
            # 如果需要返回父页面
            if return_parent and page in pages_map:
                parent = pages_map[page]
                if parent["page"] not in seen_pages:
                    seen_pages.add(parent["page"])
                    results.append({
                        "distance": round(float(score), 4),
                        "page": parent["page"],
                        "text": parent["text"]
                    })
            else:
                results.append({
                    "distance": round(float(score), 4),
                    "page": page,
                    "text": doc.page_content
                })
        
        return results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        """返回公司的所有页面（用于 full_context 模式）"""
        _, store_data = self._find_store(company_name)
        pages = store_data["pages"]
        
        return [
            {"distance": 0.5, "page": p["page"], "text": p["text"]}
            for p in sorted(pages.values(), key=lambda x: x["page"])
        ]


class HybridRetriever(BaseRetriever):
    """
    混合检索器（向量检索 + LLM 重排序）
    
    工作流程：
    1. 向量检索获取候选集（扩大召回）
    2. LLM 重排序提升相关性
    3. 返回 Top-N 结果
    """
    
    def __init__(
        self, 
        vector_db_dir: Path, 
        documents_dir: Path,
        reranker: Optional[LLMReranker] = None
    ):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = reranker or LLMReranker()

    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        top_n: int = 6,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 10,
        llm_weight: float = 0.7,
        **kwargs
    ) -> List[Dict]:
        t0 = time.time()
        
        # Step 1: 向量检索（扩大候选集）
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name, 
            query, 
            top_n=llm_reranking_sample_size,
            **kwargs
        )
        t1 = time.time()
        logger.info(f"向量检索耗时: {t1 - t0:.2f}s, 候选数: {len(vector_results)}")

        if not vector_results:
            return []
        
        # Step 2: LLM 重排序
        reranked = self.reranker.rerank_documents(
            query, 
            vector_results, 
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        t2 = time.time()
        logger.info(f"LLM重排耗时: {t2 - t1:.2f}s, 总计: {t2 - t0:.2f}s")
        
        # Step 3: 返回 Top-N
        return reranked[:top_n]