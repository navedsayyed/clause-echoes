"""
Base class for LLM-powered agents.
Provides common functionality for agents that use language models.
"""
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import structlog
from pydantic import BaseModel

from .agent import BaseAgent, AgentCapability
from .message import AgentMessage, MessageType, ResponseMessage
from llm.providers import get_llm_provider, LLMProvider
from llm.prompts import PromptTemplate
from core.exceptions import LLMProviderError
from core.config import settings

logger = structlog.get_logger(__name__)


class LLMResponse(BaseModel):
    """Response from LLM provider."""
    
    content: str
    tokens_used: int
    model_used: str
    processing_time: float
    metadata: Dict[str, Any] = {}


class LLMAgent(BaseAgent):
    """Base class for agents that use Large Language Models."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id, agent_type, config)
        
        # LLM configuration
        self.llm_provider: Optional[LLMProvider] = None
        self.model_name = self.config.get('model_name', settings.llm.default_model)
        self.temperature = self.config.get('temperature', settings.llm.temperature)
        self.max_tokens = self.config.get('max_tokens', settings.llm.max_tokens)
        
        # Prompt templates
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        
        # LLM metrics
        self.total_tokens_used = 0
        self.total_llm_calls = 0
        self.llm_call_failures = 0
        
    async def _initialize(self):
        """Initialize LLM-specific components."""
        try:
            # Get LLM provider
            provider_type = self.config.get('llm_provider', settings.llm.default_provider)
            self.llm_provider = await get_llm_provider(provider_type)
            
            # Initialize prompt templates
            self.prompt_templates = await self._load_prompt_templates()
            
            logger.info(
                "LLM agent initialized",
                agent_id=self.agent_id,
                provider=provider_type,
                model=self.model_name,
                templates=len(self.prompt_templates)
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize LLM agent",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup LLM resources."""
        try:
            if self.llm_provider:
                await self.llm_provider.cleanup()
            
            logger.info(
                "LLM agent cleanup complete",
                agent_id=self.agent_id,
                total_tokens=self.total_tokens_used,
                total_calls=self.total_llm_calls,
                failures=self.llm_call_failures
            )
            
        except Exception as e:
            logger.error(
                "Error during LLM agent cleanup",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
    
    async def generate_response(
        self,
        prompt: str,
        template_name: Optional[str] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response using the LLM."""
        start_time = datetime.utcnow()
        
        try:
            # Build final prompt
            final_prompt = await self._build_prompt(
                prompt,
                template_name,
                template_variables or {}
            )
            
            # Prepare LLM parameters
            llm_params = {
                'temperature': temperature or self.temperature,
                'max_tokens': max_tokens or self.max_tokens,
                'model': self.model_name
            }
            
            if system_message:
                llm_params['system_message'] = system_message
            
            # Make LLM call
            logger.debug(
                "Making LLM call",
                agent_id=self.agent_id,
                model=self.model_name,
                prompt_length=len(final_prompt),
                template=template_name
            )
            
            response = await self.llm_provider.generate(
                prompt=final_prompt,
                **llm_params
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self.total_llm_calls += 1
            self.total_tokens_used += response.get('tokens_used', 0)
            
            # Create response object
            llm_response = LLMResponse(
                content=response['content'],
                tokens_used=response.get('tokens_used', 0),
                model_used=response.get('model', self.model_name),
                processing_time=processing_time,
                metadata=response.get('metadata', {})
            )
            
            logger.debug(
                "LLM response generated",
                agent_id=self.agent_id,
                tokens_used=llm_response.tokens_used,
                processing_time=processing_time,
                response_length=len(llm_response.content)
            )
            
            return llm_response
            
        except Exception as e:
            self.llm_call_failures += 1
            
            logger.error(
                "LLM call failed",
                agent_id=self.agent_id,
                model=self.model_name,
                error=str(e),
                template=template_name,
                exc_info=True
            )
            
            raise LLMProviderError(
                f"LLM generation failed: {str(e)}",
                self.llm_provider.provider_name,
                self.model_name
            )
    
    async def analyze_with_llm(
        self,
        text: str,
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform analysis using LLM with structured output."""
        template_name = f"{analysis_type}_analysis"
        template_variables = {
            'text': text,
            'context': context or {}
        }
        
        try:
            response = await self.generate_response(
                prompt="",  # Prompt will be built from template
                template_name=template_name,
                template_variables=template_variables
            )
            
            # Parse structured response
            analysis_result = await self._parse_analysis_response(
                response.content,
                analysis_type
            )
            
            # Add metadata
            analysis_result['_metadata'] = {
                'tokens_used': response.tokens_used,
                'model_used': response.model_used,
                'processing_time': response.processing_time,
                'analysis_type': analysis_type
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(
                "LLM analysis failed",
                agent_id=self.agent_id,
                analysis_type=analysis_type,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _build_prompt(
        self,
        base_prompt: str,
        template_name: Optional[str],
        variables: Dict[str, Any]
    ) -> str:
        """Build final prompt from base prompt and template."""
        if template_name and template_name in self.prompt_templates:
            template = self.prompt_templates[template_name]
            
            # Add base prompt to variables if not empty
            if base_prompt.strip():
                variables['base_prompt'] = base_prompt
            
            # Add agent context
            variables.update({
                'agent_type': self.agent_type,
                'agent_id': self.agent_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return await template.render(variables)
        else:
            return base_prompt
    
    async def _parse_analysis_response(
        self,
        response_content: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Parse structured analysis response from LLM."""
        try:
            # Try to parse as JSON first
            import json
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Fall back to text analysis
            logger.warning(
                "Could not parse LLM response as JSON, using text",
                agent_id=self.agent_id,
                analysis_type=analysis_type
            )
            return {
                'analysis': response_content,
                'structured': False
            }
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for this agent."""
        # This will be implemented based on the specific agent type
        # Base implementation returns empty dict
        return {}
    
    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM-specific metrics."""
        return {
            'total_tokens_used': self.total_tokens_used,
            'total_llm_calls': self.total_llm_calls,
            'llm_call_failures': self.llm_call_failures,
            'average_tokens_per_call': (
                self.total_tokens_used / max(self.total_llm_calls, 1)
            ),
            'success_rate': (
                (self.total_llm_calls - self.llm_call_failures) / max(self.total_llm_calls, 1)
            )
        }
