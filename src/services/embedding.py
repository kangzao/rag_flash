"""统一 Embedding 服务，封装 DashScope."""
import os
import numpy as np
from typing import Union, List
from dotenv import load_dotenv


class EmbeddingService:
    """通义千问 Embedding 服务"""

    def __init__(self, provider: str = "dashscope"):
        load_dotenv()
        self.provider = provider.lower()
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self._client = None

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """获取文本嵌入向量，支持单条或批量。"""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("No valid text provided")

        return self._embed_dashscope(texts)

    def _embed_dashscope(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        from dashscope import TextEmbedding
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = TextEmbedding.call(model="text-embedding-v1", input=batch)
            output = resp.get("output")
            if output is None:
                raise RuntimeError(f"DashScope API error: {resp}")

            if "embeddings" in output:
                for emb in output["embeddings"]:
                    if not emb.get("embedding"):
                        raise RuntimeError(f"Empty embedding at index {emb.get('text_index')}")
                    embeddings.append(emb["embedding"])
            elif "embedding" in output:
                emb = output["embedding"]
                if not emb:
                    raise RuntimeError("Empty embedding")
                embeddings.append(emb)
            else:
                raise RuntimeError(f"Unexpected DashScope response: {resp}")
        return embeddings

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        v1, v2 = np.array(vec1), np.array(vec2)
        return round(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), 4)
