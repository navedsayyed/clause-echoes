"""
Custom exceptions for the Clause Echoes system.
Provides structured error handling across all components.
"""
from typing import Any, Dict, Optional, Union


class ClauseEchoesException(Exception):
    """Base exception for all Clause Echoes errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


# API Exceptions
class ValidationError(ClauseEchoesException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, "VALIDATION_ERROR", details, 400)


class AuthenticationError(ClauseEchoesException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR", {}, 401)


class AuthorizationError(ClauseEchoesException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "AUTHORIZATION_ERROR", {}, 403)


class NotFoundError(ClauseEchoesException):
    """Raised when a resource is not found."""
    
    def __init__(self, resource: str, identifier: Optional[str] = None):
        message = f"{resource} not found"
        details = {"resource": resource}
        if identifier:
            message += f" with identifier: {identifier}"
            details["identifier"] = identifier
        super().__init__(message, "NOT_FOUND_ERROR", details, 404)


class RateLimitError(ClauseEchoesException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window: str):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        details = {"limit": limit, "window": window}
        super().__init__(message, "RATE_LIMIT_ERROR", details, 429)


# Agent System Exceptions
class AgentError(ClauseEchoesException):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, agent_type: str, agent_id: Optional[str] = None):
        details = {"agent_type": agent_type}
        if agent_id:
            details["agent_id"] = agent_id
        super().__init__(message, "AGENT_ERROR", details, 500)


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    
    def __init__(self, agent_type: str, timeout_seconds: int, agent_id: Optional[str] = None):
        message = f"Agent {agent_type} timed out after {timeout_seconds} seconds"
        details = {"agent_type": agent_type, "timeout_seconds": timeout_seconds}
        if agent_id:
            details["agent_id"] = agent_id
        super(AgentError, self).__init__(message, "AGENT_TIMEOUT_ERROR", details, 500)


class AgentCommunicationError(AgentError):
    """Raised when agents fail to communicate."""
    
    def __init__(self, from_agent: str, to_agent: str, message: str = "Communication failed"):
        details = {"from_agent": from_agent, "to_agent": to_agent}
        super(AgentError, self).__init__(message, "AGENT_COMMUNICATION_ERROR", details, 500)


# LLM Provider Exceptions
class LLMProviderError(ClauseEchoesException):
    """Base exception for LLM provider errors."""
    
    def __init__(self, message: str, provider: str, model: Optional[str] = None):
        details = {"provider": provider}
        if model:
            details["model"] = model
        super().__init__(message, "LLM_PROVIDER_ERROR", details, 500)


class LLMQuotaExceededError(LLMProviderError):
    """Raised when LLM API quota is exceeded."""
    
    def __init__(self, provider: str, model: Optional[str] = None):
        message = f"API quota exceeded for provider: {provider}"
        super().__init__(message, provider, model)
        self.status_code = 429


class LLMModelNotFoundError(LLMProviderError):
    """Raised when requested LLM model is not found."""
    
    def __init__(self, provider: str, model: str):
        message = f"Model '{model}' not found for provider: {provider}"
        super().__init__(message, provider, model)
        self.status_code = 404


# Storage Exceptions
class StorageError(ClauseEchoesException):
    """Base exception for storage-related errors."""
    
    def __init__(self, message: str, storage_type: str, operation: Optional[str] = None):
        details = {"storage_type": storage_type}
        if operation:
            details["operation"] = operation
        super().__init__(message, "STORAGE_ERROR", details, 500)


class VectorStoreError(StorageError):
    """Raised for vector store operations."""
    
    def __init__(self, message: str, operation: str, collection: Optional[str] = None):
        details = {"operation": operation}
        if collection:
            details["collection"] = collection
        super(StorageError, self).__init__(message, "vector_store", operation)


class DatabaseError(StorageError):
    """Raised for database operations."""
    
    def __init__(self, message: str, operation: str, table: Optional[str] = None):
        details = {"operation": operation}
        if table:
            details["table"] = table
        super(StorageError, self).__init__(message, "database", operation)


# Retrieval Exceptions
class RetrievalError(ClauseEchoesException):
    """Base exception for retrieval-related errors."""
    
    def __init__(self, message: str, query: Optional[str] = None):
        details = {}
        if query:
            details["query"] = query[:100]  # Truncate long queries
        super().__init__(message, "RETRIEVAL_ERROR", details, 500)


class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, text: Optional[str] = None, model: Optional[str] = None):
        details = {}
        if text:
            details["text_length"] = len(text)
        if model:
            details["model"] = model
        super(RetrievalError, self).__init__(message, None)
        self.error_code = "EMBEDDING_ERROR"
        self.details.update(details)


# Reasoning Exceptions
class ReasoningError(ClauseEchoesException):
    """Base exception for reasoning-related errors."""
    
    def __init__(self, message: str, reasoning_type: str, context: Optional[Dict[str, Any]] = None):
        details = {"reasoning_type": reasoning_type}
        if context:
            details.update(context)
        super().__init__(message, "REASONING_ERROR", details, 500)


class LogicError(ReasoningError):
    """Raised when logical inconsistencies are detected."""
    
    def __init__(self, message: str, contradictions: Optional[list] = None):
        details = {}
        if contradictions:
            details["contradictions"] = contradictions
        super(ReasoningError, self).__init__(message, "logic", details)
        self.error_code = "LOGIC_ERROR"


class UncertaintyError(ReasoningError):
    """Raised when uncertainty calculations fail."""
    
    def __init__(self, message: str, confidence_score: Optional[float] = None):
        details = {}
        if confidence_score is not None:
            details["confidence_score"] = confidence_score
        super(ReasoningError, self).__init__(message, "uncertainty", details)
        self.error_code = "UNCERTAINTY_ERROR"


# Configuration Exceptions
class ConfigurationError(ClauseEchoesException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, config_value: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value:
            details["config_value"] = str(config_value)
        super().__init__(message, "CONFIGURATION_ERROR", details, 500)
