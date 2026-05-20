"""检索模块：BM25、向量、混合检索."""
import json
import logging
from typing import List, Dict
from pathlib import Path
import pickle
import faiss
import numpy as np
import time

from src.services.embedding import EmbeddingService
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)


class BaseRetriever:
    def retrieve_by_company_name(self, company_name: str, query: str, **kwargs) -> List[Dict]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        self._doc_cache: Dict[str, dict] = {}
        self._pages_cache: Dict[str, Dict[int, dict]] = {}
        self._load_documents()

    def _load_documents(self):
        for path in self.documents_dir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            company_name = doc["metainfo"].get("company_name")
            if company_name:
                self._doc_cache[company_name] = doc
                pages = doc["content"].get("pages", [])
                self._pages_cache[company_name] = {p["page"]: p for p in pages}

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, **kwargs) -> List[Dict]:
        document = self._doc_cache.get(company_name)
        if not document:
            raise ValueError(f"No report found for '{company_name}'")

        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1']}.pkl"
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        chunks = document["content"]["chunks"]
        pages_map = self._pages_cache.get(company_name, {})
        scores = bm25_index.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        return self._build_results(top_indices, chunks, pages_map, scores, kwargs.get("return_parent_pages", False))

    def _build_results(self, indices: List[int], chunks: List[dict], pages_map: Dict[int, dict], scores: List[float], return_parent: bool) -> List[Dict]:
        results, seen = [], set()
        for idx in indices:
            chunk = chunks[idx]
            parent = pages_map.get(chunk["page"])
            if return_parent and parent and parent["page"] not in seen:
                seen.add(parent["page"])
                results.append({"distance": round(scores[idx], 4), "page": parent["page"], "text": parent["text"]})
            else:
                results.append({"distance": round(scores[idx], 4), "page": chunk["page"], "text": chunk["text"]})
        return results


class VectorRetriever(BaseRetriever):
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.embedding = EmbeddingService(embedding_provider)
        self.all_dbs = self._load_dbs()
        self._pages_cache: Dict[str, Dict[int, dict]] = {r["name"]: self._build_pages_cache(r["document"]) for r in self.all_dbs}

    def _build_pages_cache(self, doc: dict) -> Dict[int, dict]:
        pages = doc["content"].get("pages", [])
        return {p["page"]: p for p in pages}

    def _load_dbs(self) -> List[dict]:
        dbs = []
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                sha1 = doc.get("metainfo", {}).get("sha1")
                if not sha1:
                    continue
                faiss_path = self.vector_db_dir / f"{sha1}.faiss"
                if not faiss_path.exists():
                    continue
                dbs.append({"name": sha1, "vector_db": faiss.read_index(str(faiss_path)), "document": doc})
            except Exception as e:
                _log.error(f"Error loading {path.name}: {e}")
        return dbs

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, **kwargs) -> List[Dict]:
        target = self._find_report(company_name)
        chunks = target["document"]["content"]["chunks"]
        pages_map = self._pages_cache.get(target["name"], {})
        embedding = self.embedding.embed(query)
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = target["vector_db"].search(vec, min(top_n, len(chunks)))
        return self._format_results(distances[0], indices[0], chunks, pages_map, kwargs.get("return_parent_pages", False))

    def _find_report(self, company_name: str) -> dict:
        for report in self.all_dbs:
            meta = report["document"].get("metainfo", {})
            if meta.get("company_name") == company_name or company_name in meta.get("file_name", ""):
                return report
        raise ValueError(f"No report found for '{company_name}'")

    def _format_results(self, distances: np.ndarray, indices: np.ndarray, chunks: List[dict], pages_map: Dict[int, dict], return_parent: bool) -> List[Dict]:
        results, seen = [], set()
        for dist, idx in zip(distances, indices):
            chunk = chunks[idx]
            parent = pages_map.get(chunk.get("page")) if pages_map else None
            if return_parent and parent and parent["page"] not in seen:
                seen.add(parent["page"])
                results.append({"distance": round(float(dist), 4), "page": parent["page"], "text": parent["text"]})
            else:
                results.append({"distance": round(float(dist), 4), "page": chunk.get("page", 0), "text": chunk["text"]})
        return results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target = self._find_report(company_name)
        return [{"distance": 0.5, "page": p["page"], "text": p["text"]} for p in sorted(target["document"]["content"]["pages"], key=lambda x: x["page"])]


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.reranker = LLMReranker()

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 6, llm_reranking_sample_size: int = 28, documents_batch_size: int = 10, llm_weight: float = 0.7, **kwargs) -> List[Dict]:
        t0 = time.time()
        vector_results = self.vector_retriever.retrieve_by_company_name(company_name, query, top_n=llm_reranking_sample_size, **kwargs)
        t1 = time.time()
        print(f"[计时] 向量检索: {t1 - t0:.2f}s")

        reranked = self.reranker.rerank_documents(query, vector_results, documents_batch_size=documents_batch_size, llm_weight=llm_weight)
        t2 = time.time()
        print(f"[计时] LLM重排: {t2 - t1:.2f}s, 总计: {t2 - t0:.2f}s")
        return reranked[:top_n]