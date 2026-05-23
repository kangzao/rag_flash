"""数据导入模块：BM25 和 FAISS 向量库构建."""
import json
import pickle
from typing import List
from pathlib import Path
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.services.embedding import EmbeddingService


class BM25Ingestor:
    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        return BM25Okapi([c.split() for c in chunks])

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for report_path in tqdm(list(all_reports_dir.glob("*.json")), desc="BM25"):
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [c["text"] for c in data["content"]["chunks"]]
            # 使用 file_name（去掉扩展名）作为文件名，与 JSON 文件名保持一致
            file_name = data["metainfo"].get("file_name", "")
            if file_name:
                base_name = Path(file_name).stem
                with open(output_dir / f"{base_name}.pkl", "wb") as f:
                    pickle.dump(self.create_bm25_index(chunks), f)
        print(f"Processed {len(list(all_reports_dir.glob('*.json')))} reports")


class VectorDBIngestor:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [t[:2048] for t in texts if t]
        if not texts:
            raise ValueError("No valid texts")
        return self.embedding_service.embed(texts)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """使用 LangChain FAISS 的目录结构保存"""
        from langchain_community.embeddings import DashScopeEmbeddings
        import os

        output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 DashScope Embeddings
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        for report_path in tqdm(list(all_reports_dir.glob("*.json")), desc="FAISS"):
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks_data = data["content"]["chunks"]
            texts = [c["text"][:2000] for c in chunks_data]

            # 创建 LangChain Document 列表，带 metadata
            docs = []
            for i, (chunk, text) in enumerate(zip(chunks_data, texts)):
                page = chunk.get("page", 0)
                doc = Document(
                    page_content=text,
                    metadata={"page": page, "chunk_id": i}
                )
                docs.append(doc)

            # 使用 file_name（去掉扩展名）作为目录名
            file_name = data["metainfo"].get("file_name", "")
            if file_name:
                base_name = Path(file_name).stem
                faiss_dir = output_dir / f"{base_name}.faiss"

                # 创建 FAISS 向量库并保存为目录结构
                vector_store = FAISS.from_documents(docs, embeddings)
                vector_store.save_local(str(faiss_dir))

        print(f"Processed {len(list(all_reports_dir.glob('*.json')))} reports")