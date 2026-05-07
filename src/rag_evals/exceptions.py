"""Custom exceptions for the RAG evaluation framework."""


class RagEvalsError(Exception):
    """Base exception for all rag-evals errors."""


class RagAPIError(RagEvalsError):
    """Raised when the RAG API returns an unexpected response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RagAPITimeoutError(RagEvalsError):
    """Raised when a call to the RAG API times out."""


class ConfigurationError(RagEvalsError):
    """Raised when required configuration is missing or invalid."""
