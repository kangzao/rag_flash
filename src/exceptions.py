"""统一异常处理模块"""


class RAGError(Exception):
    """RAG 系统基础异常类"""

    def __init__(self, message: str, error_type: str = "RAG_ERROR", details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class RetrievalError(RAGError):
    """检索失败异常"""

    def __init__(self, company: str, query: str, cause: Exception = None):
        message = f"检索失败: 公司={company}, 查询={query}"
        details = {"company": company, "query": query}
        if cause:
            details["cause"] = str(cause)
            message += f", 原因: {str(cause)}"
        super().__init__(message, "RETRIEVAL_ERROR", details)
        self.cause = cause


class EmbeddingError(RAGError):
    """Embedding 生成失败"""

    def __init__(self, text_length: int, cause: Exception = None):
        message = f"Embedding生成失败: 文本长度={text_length}"
        details = {"text_length": text_length}
        if cause:
            details["cause"] = str(cause)
            message += f", 原因: {str(cause)}"
        super().__init__(message, "EMBEDDING_ERROR", details)
        self.cause = cause


class LLMCallError(RAGError):
    """LLM 调用失败"""

    def __init__(self, model: str, prompt_length: int = None, cause: Exception = None):
        message = f"LLM调用失败: 模型={model}"
        details = {"model": model}
        if prompt_length:
            details["prompt_length"] = prompt_length
        if cause:
            details["cause"] = str(cause)
            message += f", 原因: {str(cause)}"
        super().__init__(message, "LLM_CALL_ERROR", details)
        self.cause = cause


class DocumentNotFoundError(RAGError):
    """文档未找到"""

    def __init__(self, company_name: str):
        message = f"未找到公司文档: {company_name}"
        super().__init__(message, "DOCUMENT_NOT_FOUND", {"company_name": company_name})


class ConfigurationError(RAGError):
    """配置错误"""

    def __init__(self, key: str, value: str, expected: str = None):
        message = f"配置错误: {key}={value}"
        details = {"key": key, "value": value}
        if expected:
            details["expected"] = expected
            message += f" (期望: {expected})"
        super().__init__(message, "CONFIGURATION_ERROR", details)
