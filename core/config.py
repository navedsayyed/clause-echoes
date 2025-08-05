"""
Core configuration management for the Clause Echoes system.
Handles environment variables, validation, and configuration loading.
"""
import os
from functools import lru_cache
from typing import List, Optional, Union
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field, validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW")
    echo: bool = Field(False, env="DATABASE_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")
    max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_prefix = "REDIS_"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings."""
    
    type: str = Field("chroma", env="VECTOR_STORE_TYPE")
    chroma_host: str = Field("localhost", env="CHROMA_HOST")
    chroma_port: int = Field(8001, env="CHROMA_PORT")
    collection_name: str = Field("clause_embeddings", env="CHROMA_COLLECTION_NAME")
    
    class Config:
        env_prefix = "VECTOR_STORE_"


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    default_provider: str = Field("openai", env="DEFAULT_LLM_PROVIDER")
    default_model: str = Field("gpt-4-turbo-preview", env="DEFAULT_LLM_MODEL")
    max_tokens: int = Field(4000, env="MAX_TOKENS")
    temperature: float = Field(0.1, env="TEMPERATURE")
    
    @validator("default_provider")
    def validate_provider(cls, v):
        allowed_providers = ["openai", "anthropic", "local", "ensemble"]
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of: {allowed_providers}")
        return v
    
    class Config:
        env_prefix = "LLM_"


class EmbeddingSettings(BaseSettings):
    """Embedding configuration settings."""
    
    model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    dimension: int = Field(384, env="EMBEDDING_DIMENSION")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    
    class Config:
        env_prefix = "EMBEDDING_"


class AgentSettings(BaseSettings):
    """Agent system configuration settings."""
    
    max_iterations: int = Field(5, env="MAX_AGENT_ITERATIONS")
    timeout_seconds: int = Field(30, env="AGENT_TIMEOUT_SECONDS")
    enable_logging: bool = Field(True, env="ENABLE_AGENT_LOGGING")
    
    class Config:
        env_prefix = "AGENT_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    metrics_endpoint: str = Field("/metrics", env="METRICS_ENDPOINT")
    enable_tracing: bool = Field(True, env="ENABLE_TRACING")
    jaeger_endpoint: str = Field("http://localhost:14268/api/traces", env="JAEGER_ENDPOINT")
    
    class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(30, env="JWT_EXPIRE_MINUTES")
    rate_limit_per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(PydanticSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = Field("clause-echoes", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_prefix: str = Field("/api/v1", env="API_PREFIX")
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8080"], 
        env="CORS_ORIGINS"
    )
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    agent: AgentSettings = AgentSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    
    @validator("log_level")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Handle string format like '["http://localhost:3000", "http://localhost:8080"]'
            try:
                import json
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle comma-separated string
                return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
