"""统一 Embedding 服务，封装 DashScope 和 OpenAI."""
import os
import numpy as np
from typing import Union, List
from dotenv import load_dotenv


class EmbeddingService:
    PROVIDERS = ("openai", "dashscope")

    def __init__(self, provider: str = "dashscope"):
        load_dotenv()
        self.provider = provider.lower()
        if self.provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self._setup_client()

    def _setup_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=None, max_retries=2)
        elif self.provider == "dashscope":
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

        if self.provider == "openai":
            return self._embed_openai(texts)
        return self._embed_dashscope(texts)

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model="text-embedding-3-large")
        return [item.embedding for item in response.data]

    def _embed_dashscope(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        from dashscope import TextEmbedding
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v1, input=batch)
            if "embeddings" in resp.get("output", {}):
                for emb in resp["output"]["embeddings"]:
                    if not emb.get("embedding"):
                        raise RuntimeError(f"Empty embedding at index {emb.get('text_index')}")
                    embeddings.append(emb["embedding"])
            elif "embedding" in resp.get("output", {}):
                emb = resp["output"]["embedding"]
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