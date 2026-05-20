import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional
import dashscope
import requests
from json_repair import repair_json
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed


class BaseDashscopeProcessor:
    """DashScope基础处理器，支持Qwen大模型对话"""

    def __init__(self):
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.default_model = 'qwen-turbo'

    def send_message(
            self,
            model="qwen-turbo",
            temperature=0.1,
            seed=None,  # 兼容参数，暂不使用
            system_content='You are a helpful assistant.',
            human_content='Hello!',
            is_structured=False,
            response_format=None,
            **kwargs,
    ):
        """
        发送消息到DashScope Qwen大模型，支持 system_content + human_content 拼接为 messages。
        暂不支持结构化输出。
        """
        if model is None:
            model = self.default_model
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if human_content:
            messages.append({"role": "user", "content": human_content})

        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format='message',
        )
        print('dashscope.api_key=', dashscope.api_key)
        print('model=', model)
        print('response=', response)

        if hasattr(response, 'output') and hasattr(response.output, 'choices'):
            content = response.output.choices[0].message.content
        else:
            content = str(response)

        self.response_data = {"model": model, "input_tokens": None, "output_tokens": None}
        print('content=', content)
        return {"final_answer": content}


class APIProcessor:
    """API处理器，仅支持DashScope/通义千问"""

    def __init__(self):
        self.provider = "dashscope"
        self.processor = BaseDashscopeProcessor()

    def send_message(
            self,
            model=None,
            temperature=0.5,
            seed=None,
            system_content="You are a helpful assistant.",
            human_content="Hello!",
            is_structured=False,
            response_format=None,
            **kwargs,
    ):
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs,
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        import src.prompts as prompts
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)

        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format,
        )
        self.response_data = self.processor.response_data
        if 'step_by_step_analysis' not in answer_dict:
            answer_dict = {
                "step_by_step_analysis": "",
                "reasoning_summary": "",
                "relevant_pages": [],
                "final_answer": answer_dict.get("final_answer", "N/A"),
            }
        return answer_dict

    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        import src.prompts as prompts

        entry = prompts.SCHEMA_MAP.get(schema)
        if not entry:
            raise ValueError(f"Unsupported schema: {schema}")
        prompt_cls, schema_cls = entry
        return prompt_cls.system_prompt, schema_cls, prompt_cls.user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        import src.prompts as prompts
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies]),
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions,
        )

        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        return questions_dict
