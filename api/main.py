"""
Main FastAPI application for the Clause Echoes system.
Provides the public API interface for the multi-agent reasoning ecosystem.
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import structlog

from core.config import settings
from core.exceptions import ClauseEchoesException
from api.middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    MetricsMiddleware
)
from api.dependencies import get_database_session, get_redis_client, get_current_user
from api.v1.routes import query, admin, health, feedback
from storage.database import init_database
from storage.redis_client import init_redis
from llm.providers import init_llm_providers
from agents.orchestrator import AgentOrchestrator

# Configure structured logging
logger = structlog.get_logger(__name__)

# Metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    logger.info("Starting Clause Echoes API server", version=settings.app_version)
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Initialize Redis
        await init_redis()
        logger.info("Redis initialized successfully")
        
        # Initialize LLM providers
        await init_llm_providers()
        logger.info("LLM providers initialized successfully")
        
        # Initialize agent orchestrator
        app.state.orchestrator = AgentOrchestrator()
        await app.state.orchestrator.initialize()
        logger.info("Agent orchestrator initialized successfully")
        
        logger.info("All systems initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e), exc_info=True)
        raise
    
    yield
    
    # Cleanup
    try:
        if hasattr(app.state, 'orchestrator'):
            await app.state.orchestrator.cleanup()
        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        # Clause Echoes: Self-Critiquing Meta-Agents for Policy Analysis
        
        An advanced multi-agent LLM reasoning ecosystem that provides:
        
        ## 🎯 Core Features
        - **Multi-agent processing** with specialized micro-agents
        - **Self-critiquing meta-agents** with echo feedback loops
        - **Dynamic clause synthesis** and zero-shot hypothesizing
        - **Uncertainty-aware retrieval** with confidence scoring
        - **Curriculum-driven query refinement**
        - **Federated multi-source consensus**
        
        ## 🏗️ Architecture
        - **Query Parser Agent**: Natural language understanding
        - **Clause Retriever Agent**: Semantic search and retrieval
        - **Consistency Checker Agent**: Contradiction detection
        - **Logic Evaluator Agent**: Reasoning and inference
        - **Synthesis Agent**: Answer generation and explanation
        - **Meta-Critic Agent**: Self-improvement and refinement
        
        ## 🔍 Advanced Capabilities
        - Uncertainty quantification and confidence scoring
        - Implicit assumption detection and analysis
        - Multi-document consensus with conflict resolution
        - Progressive query refinement with branching paths
        - Real-time performance monitoring and metrics
        """,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Configure middleware
    configure_middleware(app)
    
    # Include routers
    include_routers(app)
    
    # Add custom exception handlers
    add_exception_handlers(app)
    
    # Add custom routes
    add_custom_routes(app)
    
    return app


def configure_middleware(app: FastAPI):
    """Configure all middleware for the application."""
    
    # Trusted Host middleware (security)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom middleware (in reverse order of execution)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)


def include_routers(app: FastAPI):
    """Include all API routers."""
    
    app.include_router(
        health.router,
        prefix=f"{settings.api_prefix}/health",
        tags=["Health"]
    )
    
    app.include_router(
        query.router,
        prefix=f"{settings.api_prefix}/query",
        tags=["Query Processing"]
    )
    
    app.include_router(
        feedback.router,
        prefix=f"{settings.api_prefix}/feedback",
        tags=["Feedback"]
    )
    
    app.include_router(
        admin.router,
        prefix=f"{settings.api_prefix}/admin",
        tags=["Administration"],
        dependencies=[Depends(get_current_user)]  # Require authentication
    )


def add_exception_handlers(app: FastAPI):
    """Add custom exception handlers."""
    
    @app.exception_handler(ClauseEchoesException)
    async def clause_echoes_exception_handler(request: Request, exc: ClauseEchoesException):
        """Handle custom application exceptions."""
        logger.error(
            "Application exception occurred",
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured logging."""
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        error_id = str(uuid.uuid4())
        logger.error(
            "Unexpected exception occurred",
            error_id=error_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method,
            exc_info=True
        )
        
        # Don't expose internal errors in production
        if settings.debug:
            message = str(exc)
        else:
            message = "An internal server error occurred"
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": message,
                "error_id": error_id,
                "status_code": 500
            }
        )


def add_custom_routes(app: FastAPI):
    """Add custom routes and endpoints."""
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with basic API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "Clause Echoes: Self-Critiquing Meta-Agents for Policy Analysis",
            "api_prefix": settings.api_prefix,
            "docs_url": f"{settings.api_prefix}/docs",
            "health_url": f"{settings.api_prefix}/health",
            "timestamp": time.time()
        }
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    
    @app.get(f"{settings.api_prefix}/info")
    async def api_info():
        """Detailed API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": "development" if settings.debug else "production",
            "features": [
                "Multi-agent LLM ecosystem",
                "Self-critiquing meta-agents",
                "Uncertainty-aware retrieval",
                "Dynamic clause synthesis",
                "Federated consensus",
                "Real-time monitoring"
            ],
            "endpoints": {
                "query": f"{settings.api_prefix}/query",
                "health": f"{settings.api_prefix}/health",
                "feedback": f"{settings.api_prefix}/feedback",
                "admin": f"{settings.api_prefix}/admin",
                "docs": f"{settings.api_prefix}/docs",
                "metrics": "/metrics"
            }
        }


def get_openapi_schema():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        Advanced multi-agent LLM reasoning ecosystem for policy analysis.
        
        This API provides access to a sophisticated system of specialized agents
        that work together to analyze policy documents, detect contradictions,
        quantify uncertainty, and provide comprehensive answers with confidence scores.
        """,
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create the FastAPI application instance
app = create_application()
app.openapi = get_openapi_schema


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )
