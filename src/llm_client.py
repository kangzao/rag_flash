from abc import ABC, abstractmethod
from typing import Dict, Any
import os
from dotenv import load_dotenv


class LLMClient(ABC):
    """LLM客户端统一接口"""

    @abstractmethod
    def chat_completion(self, messages: list, temperature: float = 0.5,
                        response_format=None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> list:
        pass


class DashscopeClient(LLMClient):
    """阿里云DashScope客户端（通义千问）"""

    def __init__(self):
        load_dotenv()
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.client = dashscope

    def chat_completion(self, messages, temperature=0.5, response_format=None):
        response = self.client.Generation.call(
            model="qwen-turbo",
            messages=messages,
            temperature=temperature,
            result_format='message',
        )
        return {
            "content": response.output.choices[0].message.content,
            "model": "qwen-turbo",
        }

    def get_embedding(self, text: str):
        rsp = self.client.TextEmbedding.call(
            model="text-embedding-v2",
            input=[text],
        )
        return rsp['output']['embeddings'][0]['embedding']
