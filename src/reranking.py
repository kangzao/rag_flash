import os
from dotenv import load_dotenv
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
import dashscope


class JinaReranker:
    """基于Jina API的重排器，适用于多语言场景"""

    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()

    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers

    def rerank(self, query, documents, top_n=10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents,
        }
        response = requests.post(url=self.url, headers=self.headers, json=data)
        return response.json()


class LLMReranker:
    """基于通义千问(LangChain)的大模型重排器"""

    def __init__(self):
        self.provider = "dashscope"
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        load_dotenv()
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        return dashscope

    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        messages = [
            {"role": "system", "content": self.system_prompt_rerank_single_block},
            {"role": "user", "content": user_prompt},
        ]
        rsp = self.llm.Generation.call(
            model="qwen-turbo",
            messages=messages,
            temperature=0,
            result_format='message',
        )
        if not rsp or not isinstance(rsp, dict):
            raise RuntimeError(f"DashScope返回None或非dict: {rsp}")
        if 'output' in rsp and 'choices' in rsp['output']:
            content = rsp['output']['choices'][0]['message']['content']
            return {"relevance_score": 0.0, "reasoning": content}
        else:
            raise RuntimeError(f"DashScope返回格式异常: {rsp}")

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        messages = [
            {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
            {"role": "user", "content": user_prompt},
        ]
        rsp = self.llm.Generation.call(
            model="qwen-turbo",
            messages=messages,
            temperature=0,
            result_format='message',
        )
        if not rsp or not isinstance(rsp, dict):
            raise RuntimeError(f"DashScope返回None或非dict: {rsp}")
        if 'output' in rsp and 'choices' in rsp['output']:
            content = rsp['output']['choices'][0]['message']['content']
            return {"block_rankings": [{"relevance_score": 0.0, "reasoning": content} for _ in retrieved_documents]}
        else:
            raise RuntimeError(f"DashScope返回格式异常: {rsp}")

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        使用多线程并行方式对多个文档进行重排。
        结合向量相似度和LLM相关性分数，采用加权平均融合。
        """
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight

        if documents_batch_size == 1:
            def process_single_doc(doc):
                ranking = self.get_rank_for_single_block(query, doc['text'])
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] +
                    vector_weight * doc['distance'],
                    4,
                )
                return doc_with_score

            with ThreadPoolExecutor(max_workers=min(4, len(documents))) as executor:
                all_results = list(executor.map(process_single_doc, documents))

        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])

                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")

                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0,
                            "reasoning": "Default ranking due to missing LLM response",
                        })

                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] +
                        vector_weight * doc['distance'],
                        4,
                    )
                    results.append(doc_with_score)
                return results

            with ThreadPoolExecutor(max_workers=min(4, len(doc_batches))) as executor:
                batch_results = list(executor.map(process_batch, doc_batches))

            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
