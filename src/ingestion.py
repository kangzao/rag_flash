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
            sha1 = data["metainfo"]["sha1"]
            with open(output_dir / f"{sha1}.pkl", "wb") as f:
                pickle.dump(self.create_bm25_index(chunks), f)
        print(f"Processed {len(list(all_reports_dir.glob('*.json')))} reports")


class VectorDBIngestor:
    def __init__(self):
        self.embedding = EmbeddingService("dashscope")

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [t[:2048] for t in texts if t]
        if not texts:
            raise ValueError("No valid texts")
        return self.embedding.embed(texts)

    def _create_index(self, embeddings: List[List[float]]) -> faiss.Index:
        arr = np.array(embeddings, dtype=np.float32)
        return faiss.IndexFlatIP(len(embeddings[0])), arr

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for report_path in tqdm(list(all_reports_dir.glob("*.json")), desc="FAISS"):
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [c["text"] for c in data["content"]["chunks"]]
            embeddings = self._get_embeddings(chunks)
            index, arr = self._create_index(embeddings)
            index.add(arr)
            sha1 = data["metainfo"].get("sha1")
            if sha1:
                faiss.write_index(index, str(output_dir / f"{sha1}.faiss"))
        print(f"Processed {len(list(all_reports_dir.glob('*.json')))} reports")