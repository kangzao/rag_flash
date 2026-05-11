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
    """阿里云DashScope客户端"""

    def __init__(self):
        load_dotenv()
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.client = dashscope

    def chat_completion(self, messages, temperature=0.5, response_format=None):
        response = self.client.Generation.call(
            model="qwen-turbo-latest",
            messages=messages,
            temperature=temperature,
            result_format='message',
        )
        return {
            "content": response.output.choices[0].message.content,
            "model": "qwen-turbo-latest",
        }

    def get_embedding(self, text: str):
        rsp = self.client.TextEmbedding.call(
            model="text-embedding-v1",
            input=[text],
        )
        return rsp['output']['embeddings'][0]['embedding']


class OpenAIClient(LLMClient):
    """OpenAI客户端"""

    def __init__(self):
        from openai import OpenAI
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat_completion(self, messages, temperature=0.5, response_format=None):
        params = {"model": "gpt-4o-mini", "messages": messages, "temperature": temperature}
        if response_format:
            params["response_format"] = response_format

        completion = self.client.chat.completions.create(**params)
        return {
            "content": completion.choices[0].message.content,
            "model": completion.model,
            "usage": completion.usage.dict(),
        }

    def get_embedding(self, text: str):
        embedding = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large",
        )
        return embedding.data[0].embedding


def create_llm_client(provider: str = "dashscope") -> LLMClient:
    """工厂函数创建LLM客户端"""
    clients = {
        "dashscope": DashscopeClient,
        "openai": OpenAIClient,
    }
    client_class = clients.get(provider.lower())
    if not client_class:
        raise ValueError(f"不支持的LLM提供商: {provider}")
    return client_class()
