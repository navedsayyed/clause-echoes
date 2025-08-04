"""
Self-Critic Meta-Agent - Provides self-critique and answer refinement capabilities.
The core meta-agent that analyzes responses and suggests improvements through reflection.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, CritiqueMessage, ResponseMessage
from core.exceptions import ReasoningError

logger = structlog.get_logger(__name__)


class CritiqueType(str, Enum):
    """Types of critiques that can be performed."""
    
    ACCURACY = "accuracy"           # Factual accuracy assessment
    COMPLETENESS = "completeness"   # Coverage completeness
    CLARITY = "clarity"            # Communication clarity
    LOGICAL = "logical"            # Logical consistency
    BIAS = "bias"                  # Bias detection
    RELEVANCE = "relevance"        # Query relevance
    EVIDENCE = "evidence"          # Evidence quality
    ASSUMPTIONS = "assumptions"    # Implicit assumptions


class CritiqueSeverity(str, Enum):
    """Severity levels for critique findings."""
    
    MINOR = "minor"         # Minor issues, suggestions
    MODERATE = "moderate"   # Notable problems requiring attention
    MAJOR = "major"        # Significant issues affecting quality
    CRITICAL = "critical"  # Critical flaws requiring major revision


class CritiquePoint(BaseModel):
    """Individual critique point with specific feedback."""
    
    critique_type: CritiqueType
    severity: CritiqueSeverity
    description: str
    evidence: List[str] = Field(default_factory=list)
    
    # Location information
    text_section: Optional[str] = None
    line_numbers: Optional[Tuple[int, int]] = None
    
    # Improvement suggestions
    suggestions: List[str] = Field(default_factory=list)
    example_improvement: Optional[str] = None
    
    # Confidence in critique
    confidence: float = Field(..., ge=0.0, le=1.0)


class SelfCritiqueResult(BaseModel):
    """Complete self-critique analysis result."""
    
    # Target information
    original_response: str
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Critique findings
    critique_points: List[CritiquePoint]
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    
    # Categorized feedback
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvement_priorities: List[str] = Field(default_factory=list)
    
    # Refined response
    suggested_revision: str
    revision_rationale: str
    key_changes: List[str] = Field(default_factory=list)
    
    # Meta-analysis
    confidence_in_original: float = Field(..., ge=0.0, le=1.0)
    confidence_in_revision: float = Field(..., ge=0.0, le=1.0)
    improvement_score: float = Field(..., ge=0.0, le=1.0)
    
    # Alternative perspectives
    alternative_approaches: List[str] = Field(default_factory=list)
    overlooked_aspects: List[str] = Field(default_factory=list)
    
    # Processing metadata
    critique_time: float = 0.0
    critiqued_at: datetime = Field(default_factory=datetime.utcnow)


class SelfCriticAgent(LLMAgent):
    """Meta-agent that provides self-critique and continuous improvement through reflection[7]."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="SelfCriticAgent",
            config=config or {}
        )
        
        # Critique configuration
        self.critique_types = set(CritiqueType)
        self.severity_threshold = CritiqueSeverity(self.config.get('severity_threshold', 'moderate'))
        self.enable_revision = self.config.get('enable_revision', True)
        self.max_revision_attempts = self.config.get('max_revision_attempts', 3)
        
        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.7)
        self.improvement_threshold = self.config.get('improvement_threshold', 0.1)
        
        logger.info("Self-Critic Meta-Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize self-critique capabilities."""
        return [
            AgentCapability(
                name="response_critique",
                description="Analyze responses for quality, accuracy, and completeness",
                input_types=["responses", "answers", "text"],
                output_types=["critique_analysis", "quality_assessment"],
                confidence_level=0.9,
                estimated_processing_time=12.0
            ),
            AgentCapability(
                name="assumption_detection",
                description="Identify implicit assumptions in reasoning",
                input_types=["reasoning_chains", "arguments"],
                output_types=["assumption_analysis", "implicit_assumptions"],
                confidence_level=0.85,
                estimated_processing_time=8.0
            ),
            AgentCapability(
                name="bias_detection",
                description="Detect potential biases in responses and reasoning",
                input_types=["responses", "reasoning"],
                output_types=["bias_analysis", "fairness_assessment"],
                confidence_level=0.8,
                estimated_processing_time=10.0
            ),
            AgentCapability(
                name="response_refinement",
                description="Generate improved versions of responses based on critique",
                input_types=["original_responses", "critique_points"],
                output_types=["refined_responses", "improvements"],
                confidence_level=0.85,
                estimated_processing_time=15.0
            ),
            AgentCapability(
                name="alternative_generation",
                description="Generate alternative approaches and perspectives",
                input_types=["original_approaches", "context"],
                output_types=["alternative_approaches", "different_perspectives"],
                confidence_level=0.75,
                estimated_processing_time=10.0
            )
        ]
    
    async def _initialize(self):
        """Initialize self-critic components."""
        await super()._initialize()
        
        # Initialize critique-specific configurations
        self.critique_weights = self.config.get('critique_weights', {
            CritiqueType.ACCURACY.value: 1.0,
            CritiqueType.COMPLETENESS.value: 0.9,
            CritiqueType.CLARITY.value: 0.8,
            CritiqueType.LOGICAL.value: 1.0,
            CritiqueType.BIAS.value: 0.7,
            CritiqueType.RELEVANCE.value: 0.9,
            CritiqueType.EVIDENCE.value: 0.8,
            CritiqueType.ASSUMPTIONS.value: 0.6
        })
        
        logger.info("Self-critic meta-agent initialized successfully")
    
    async def _cleanup(self):
        """Cleanup self-critic resources."""
        await super()._cleanup()
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for self-critique."""
        if message.type == MessageType.REQUEST:
            request_type = message.content.get('request_type', '')
            
            if request_type == 'critique_response':
                return await self._handle_critique_request(message)
            elif request_type == 'detect_assumptions':
                return await self._handle_assumption_detection(message)
            elif request_type == 'detect_bias':
                return await self._handle_bias_detection(message)
            elif request_type == 'refine_response':
                return await self._handle_refinement_request(message)
            elif request_type == 'generate_alternatives':
                return await self._handle_alternative_generation(message)
            else:
                logger.debug(
                    "Unsupported request type for self-critic",
                    agent_id=self.agent_id,
                    request_type=request_type
                )
                return None
        elif message.type == MessageType.CRITIQUE:
            # Handle direct critique messages
            return await self._handle_direct_critique(message)
        else:
            logger.debug(
                "Unsupported message type for self-critic",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_critique_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle self-critique requests[42]."""
        start_time = datetime.utcnow()
        
        try:
            # Extract critique parameters
            response_text = message.content.get('response', '')
            original_query = message.content.get('original_query', '')
            response_metadata = message.content.get('metadata', {})
            critique_types = message.content.get('critique_types', list(self.critique_types))
            
            if not response_text:
                raise ValueError("Response text is required for critique")
            
            logger.info(
                "Processing critique request",
                agent_id=self.agent_id,
                response_length=len(response_text),
                critique_types=critique_types
            )
            
            # Perform self-critique
            critique_result = await self._perform_critique(
                response_text, original_query, response_metadata, critique_types
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            critique_result.critique_time = processing_time
            
            # Create response
            response_content = {
                'critique_result': critique_result.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=critique_result.confidence_in_revision,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Critique completed with quality score: {critique_result.overall_quality_score:.2f}"
            )
            
        except Exception as e:
            logger.error(
                "Critique request failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content={
                    'success': False,
                    'error': str(e)
                },
                confidence_score=0.0,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Critique failed: {str(e)}"
            )
    
    async def _perform_critique(
        self,
        response_text: str,
        original_query: str,
        metadata: Dict[str, Any],
        critique_types: List[str]
    ) -> SelfCritiqueResult:
        """Perform comprehensive self-critique analysis."""
        
        # Step 1: Analyze each critique type
        critique_points = []
        for critique_type in critique_types:
            if critique_type in [ct.value for ct in CritiqueType]:
                points = await self._analyze_critique_type(
                    response_text, original_query, CritiqueType(critique_type), metadata
                )
                critique_points.extend(points)
        
        # Step 2: Calculate overall quality score
        quality_score = await self._calculate_quality_score(critique_points, response_text)
        
        # Step 3: Identify strengths and weaknesses
        strengths, weaknesses, priorities = await self._categorize_feedback(critique_points)
        
        # Step 4: Generate refined response if enabled
        suggested_revision = response_text
        revision_rationale = "No revision needed"
        key_changes = []
        confidence_in_revision = quality_score
        improvement_score = 0.0
        
        if self.enable_revision and quality_score < self.min_quality_score:
            revision_result = await self._generate_revision(
                response_text, original_query, critique_points, metadata
            )
            
            suggested_revision = revision_result.get('revised_response', response_text)
            revision_rationale = revision_result.get('rationale', revision_rationale)
            key_changes = revision_result.get('changes', [])
            confidence_in_revision = revision_result.get('confidence', quality_score)
            improvement_score = revision_result.get('improvement_score', 0.0)
        
        # Step 5: Generate alternative approaches
        alternatives, overlooked = await self._generate_alternatives(
            response_text, original_query, critique_points
        )
        
        return SelfCritiqueResult(
            original_response=response_text,
            response_metadata=metadata,
            critique_points=critique_points,
            overall_quality_score=quality_score,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_priorities=priorities,
            suggested_revision=suggested_revision,
            revision_rationale=revision_rationale,
            key_changes=key_changes,
            confidence_in_original=quality_score,
            confidence_in_revision=confidence_in_revision,
            improvement_score=improvement_score,
            alternative_approaches=alternatives,
            overlooked_aspects=overlooked
        )
    
    async def _analyze_critique_type(
        self,
        response: str,
        query: str,
        critique_type: CritiqueType,
        metadata: Dict[str, Any]
    ) -> List[CritiquePoint]:
        """Analyze a specific type of critique."""
        try:
            # Use LLM for critique analysis
            llm_response = await self.generate_response(
                response,
                template_name=f'critique_{critique_type.value}',
                template_variables={
                    'response': response,
                    'query': query,
                    'metadata': metadata,
                    'critique_type': critique_type.value
                }
            )
            
            # Parse critique results
            import json
            critique_data = json.loads(llm_response.content)
            
            critique_points = []
            for point_data in critique_data.get('critique_points', []):
                point = CritiquePoint(
                    critique_type=critique_type,
                    severity=CritiqueSeverity(point_data.get('severity', 'moderate')),
                    description=point_data['description'],
                    evidence=point_data.get('evidence', []),
                    text_section=point_data.get('text_section'),
                    suggestions=point_data.get('suggestions', []),
                    example_improvement=point_data.get('example_improvement'),
                    confidence=point_data.get('confidence', 0.7)
                )
                critique_points.append(point)
            
            return critique_points
            
        except Exception as e:
            logger.warning(
                "Failed to analyze critique type",
                agent_id=self.agent_id,
                critique_type=critique_type,
                error=str(e)
            )
            return []
    
    async def _calculate_quality_score(
        self, critique_points: List[CritiquePoint], response: str
    ) -> float:
        """Calculate overall quality score based on critique points."""
        if not critique_points:
            return 0.9  # High score if no issues found
        
        # Weight critiques by severity and type
        penalty = 0.0
        for point in critique_points:
            severity_weight = {
                CritiqueSeverity.MINOR: 0.1,
                CritiqueSeverity.MODERATE: 0.2,
                CritiqueSeverity.MAJOR: 0.4,
                CritiqueSeverity.CRITICAL: 0.8
            }.get(point.severity, 0.2)
            
            type_weight = self.critique_weights.get(point.critique_type.value, 0.5)
            confidence_weight = point.confidence
            
            penalty += severity_weight * type_weight * confidence_weight
        
        # Normalize penalty and convert to quality score
        max_penalty = len(critique_points) * 0.8  # Maximum possible penalty
        normalized_penalty = min(penalty / max_penalty, 1.0) if max_penalty > 0 else 0.0
        
        quality_score = 1.0 - normalized_penalty
        return max(0.0, min(1.0, quality_score))
    
    async def _categorize_feedback(
        self, critique_points: List[CritiquePoint]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Categorize critique points into strengths, weaknesses, and priorities."""
        weaknesses = []
        priorities = []
        
        # Extract weaknesses from critique points
        for point in critique_points:
            weaknesses.append(point.description)
            
            if point.severity in [CritiqueSeverity.MAJOR, CritiqueSeverity.CRITICAL]:
                priorities.append(f"Address {point.critique_type.value}: {point.description}")
        
        # Generate strengths (aspects not criticized)
        try:
            llm_response = await self.generate_response(
                "",
                template_name='identify_strengths',
                template_variables={
                    'critique_points': [p.dict() for p in critique_points],
                    'weaknesses': weaknesses
                }
            )
            
            import json
            strength_data = json.loads(llm_response.content)
            strengths = strength_data.get('strengths', [])
            
        except Exception as e:
            logger.warning("Failed to identify strengths", error=str(e))
            strengths = ["Well-structured response", "Addresses the main query"]
        
        return strengths, weaknesses, priorities
    
    async def _generate_revision(
        self,
        original_response: str,
        query: str,
        critique_points: List[CritiquePoint],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate revised response based on critique."""
        try:
            # Use LLM to generate revision
            llm_response = await self.generate_response(
                original_response,
                template_name='response_revision',
                template_variables={
                    'original_response': original_response,
                    'query': query,
                    'critique_points': [p.dict() for p in critique_points],
                    'metadata': metadata
                }
            )
            
            # Parse revision result
            import json
            revision_data = json.loads(llm_response.content)
            
            return {
                'revised_response': revision_data.get('revised_response', original_response),
                'rationale': revision_data.get('rationale', ''),
                'changes': revision_data.get('key_changes', []),
                'confidence': revision_data.get('confidence', 0.7),
                'improvement_score': revision_data.get('improvement_score', 0.1)
            }
            
        except Exception as e:
            logger.error("Failed to generate revision", agent_id=self.agent_id, error=str(e))
            return {
                'revised_response': original_response,
                'rationale': 'Revision failed',
                'changes': [],
                'confidence': 0.5,
                'improvement_score': 0.0
            }
    
    async def _generate_alternatives(
        self,
        response: str,
        query: str,
        critique_points: List[CritiquePoint]
    ) -> Tuple[List[str], List[str]]:
        """Generate alternative approaches and identify overlooked aspects."""
        try:
            # Use LLM to generate alternatives
            llm_response = await self.generate_response(
                response,
                template_name='alternative_generation',
                template_variables={
                    'response': response,
                    'query': query,
                    'critique_points': [p.dict() for p in critique_points]
                }
            )
            
            # Parse alternatives
            import json
            alt_data = json.loads(llm_response.content)
            
            alternatives = alt_data.get('alternative_approaches', [])
            overlooked = alt_data.get('overlooked_aspects', [])
            
            return alternatives, overlooked
            
        except Exception as e:
            logger.warning("Failed to generate alternatives", agent_id=self.agent_id, error=str(e))
            return [], []
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for self-critique."""
        templates = {}
        
        # Accuracy critique template
        templates['critique_accuracy'] = PromptTemplate(
            name='critique_accuracy',
            template="""
            Analyze this response for factual accuracy and correctness.
            
            Original Query: {query}
            Response: {response}
            Metadata: {metadata}
            
            Evaluate:
            1. Are the facts stated in the response accurate?
            2. Are there any factual errors or inaccuracies?
            3. Is the information up-to-date and reliable?
            4. Are claims properly supported by evidence?
            
            Respond with JSON:
            {{
                "critique_points": [
                    {{
                        "severity": "minor|moderate|major|critical",
                        "description": "detailed description of the issue",
                        "evidence": ["evidence1", "evidence2"],
                        "text_section": "problematic text section",
                        "suggestions": ["suggestion1", "suggestion2"],
                        "example_improvement": "example of how to fix this",
                        "confidence": 0.9
                    }}
                ]
            }}
            """.strip()
        )
        
        # Response revision template
        templates['response_revision'] = PromptTemplate(
            name='response_revision',
            template="""
            Revise this response based on the critique points provided.
            
            Original Response: {original_response}
            Query: {query}
            Critique Points: {critique_points}
            Metadata: {metadata}
            
            Create an improved version that:
            1. Addresses all major critique points
            2. Maintains the core message and intent
            3. Improves clarity, accuracy, and completeness
            4. Preserves what was good about the original
            
            Respond with JSON:
            {{
                "revised_response": "improved version of the response",
                "rationale": "explanation of changes made",
                "key_changes": ["change1", "change2", "change3"],
                "confidence": 0.85,
                "improvement_score": 0.3,
                "addressed_critiques": ["critique1", "critique2"]
            }}
            """.strip()
        )
        
        return templates
    
    # Additional handler methods
    async def _handle_assumption_detection(self, message: AgentMessage) -> ResponseMessage:
        """Handle assumption detection requests."""
        pass
    
    async def _handle_bias_detection(self, message: AgentMessage) -> ResponseMessage:
        """Handle bias detection requests."""
        pass
    
    async def _handle_refinement_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle response refinement requests."""
        pass
    
    async def _handle_alternative_generation(self, message: AgentMessage) -> ResponseMessage:
        """Handle alternative generation requests."""
        pass
    
    async def _handle_direct_critique(self, message: AgentMessage) -> ResponseMessage:
        """Handle direct critique messages."""
        pass
