"""
Assumption Detector Meta-Agent - Identifies implicit assumptions in reasoning and responses.
Detects hidden premises, unstated conditions, and background assumptions that influence conclusions.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, ResponseMessage
from core.exceptions import ReasoningError

logger = structlog.get_logger(__name__)


class AssumptionType(str, Enum):
    """Types of assumptions that can be detected."""
    
    IMPLICIT_PREMISE = "implicit_premise"           # Unstated logical premises
    BACKGROUND_KNOWLEDGE = "background_knowledge"   # Assumed background knowledge
    VALUE_JUDGMENT = "value_judgment"               # Implicit value judgments
    CAUSAL_RELATIONSHIP = "causal_relationship"     # Assumed causal connections
    SCOPE_ASSUMPTION = "scope_assumption"           # Assumptions about scope/context
    TEMPORAL_ASSUMPTION = "temporal_assumption"     # Time-related assumptions
    CULTURAL_ASSUMPTION = "cultural_assumption"     # Cultural or social assumptions
    DEFINITIONAL_ASSUMPTION = "definitional_assumption"  # Assumed definitions
    COMPLETENESS_ASSUMPTION = "completeness_assumption"  # Assumptions about information completeness


class AssumptionCriticality(str, Enum):
    """Criticality levels for assumptions."""
    
    LOW = "low"           # Minor assumptions, unlikely to affect conclusions
    MEDIUM = "medium"     # Moderate assumptions, may influence conclusions
    HIGH = "high"        # Important assumptions, likely to affect conclusions
    CRITICAL = "critical" # Critical assumptions, conclusions depend heavily on these


class DetectedAssumption(BaseModel):
    """A detected assumption with metadata."""
    
    assumption_id: str
    assumption_type: AssumptionType
    criticality: AssumptionCriticality
    
    # Content
    description: str
    implicit_statement: str  # What is implicitly assumed
    explicit_alternative: str  # How it could be made explicit
    
    # Context
    context_location: Optional[str] = None  # Where in the text this assumption appears
    related_reasoning: List[str] = Field(default_factory=list)
    
    # Impact analysis
    impact_on_conclusion: float = Field(..., ge=0.0, le=1.0)
    alternative_outcomes: List[str] = Field(default_factory=list)
    
    # Evidence and support
    supporting_indicators: List[str] = Field(default_factory=list)
    confidence_in_detection: float = Field(..., ge=0.0, le=1.0)
    
    # Validation questions
    validation_questions: List[str] = Field(default_factory=list)
    challenge_scenarios: List[str] = Field(default_factory=list)


class AssumptionAnalysis(BaseModel):
    """Complete assumption analysis result."""
    
    # Target analysis
    analyzed_content: str
    content_type: str  # response, reasoning_chain, argument, etc.
    analysis_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Detected assumptions
    detected_assumptions: List[DetectedAssumption]
    assumption_categories: Dict[str, int] = Field(default_factory=dict)  # Count by type
    
    # Risk assessment
    high_risk_assumptions: List[str] = Field(default_factory=list)
    assumption_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Overall assessment
    assumption_burden: float = Field(..., ge=0.0, le=1.0)  # How assumption-heavy the content is
    robustness_score: float = Field(..., ge=0.0, le=1.0)   # How robust conclusions are to assumptions
    
    # Recommendations
    assumptions_to_validate: List[str] = Field(default_factory=list)
    assumptions_to_make_explicit: List[str] = Field(default_factory=list)
    information_needed: List[str] = Field(default_factory=list)
    
    # Alternative scenarios
    assumption_free_conclusions: List[str] = Field(default_factory=list)
    worst_case_scenarios: List[str] = Field(default_factory=list)
    
    # Processing metadata
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    analysis_time: float = 0.0
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class AssumptionDetectorAgent(LLMAgent):
    """Meta-agent specialized in detecting implicit assumptions in reasoning and responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="AssumptionDetectorAgent",
            config=config or {}
        )
        
        # Detection configuration
        self.detection_sensitivity = self.config.get('detection_sensitivity', 0.7)
        self.criticality_threshold = AssumptionCriticality(self.config.get('criticality_threshold', 'medium'))
        self.max_assumptions_per_analysis = self.config.get('max_assumptions', 20)
        
        # Analysis modes
        self.deep_analysis_mode = self.config.get('deep_analysis', True)
        self.cross_cultural_analysis = self.config.get('cross_cultural', False)
        self.domain_specific_assumptions = self.config.get('domain_specific', True)
        
        # Assumption patterns (would be loaded from knowledge base)
        self.assumption_patterns = {}
        self.cultural_assumption_markers = set()
        self.logical_fallacy_patterns = {}
        
        logger.info("Assumption Detector Meta-Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize assumption detection capabilities."""
        return [
            AgentCapability(
                name="implicit_assumption_detection",
                description="Detect implicit assumptions in reasoning and responses",
                input_types=["responses", "arguments", "reasoning_chains"],
                output_types=["detected_assumptions", "assumption_analysis"],
                confidence_level=0.85,
                estimated_processing_time=10.0
            ),
            AgentCapability(
                name="assumption_impact_analysis",
                description="Analyze how assumptions impact conclusions and reasoning",
                input_types=["assumptions", "conclusions"],
                output_types=["impact_analysis", "robustness_assessment"],
                confidence_level=0.8,
                estimated_processing_time=8.0
            ),
            AgentCapability(
                name="assumption_validation",
                description="Generate questions and scenarios to validate assumptions",
                input_types=["detected_assumptions"],
                output_types=["validation_questions", "test_scenarios"],
                confidence_level=0.9,
                estimated_processing_time=6.0
            ),
            AgentCapability(
                name="bias_assumption_detection",
                description="Detect assumptions that may introduce bias",
                input_types=["content", "context"],
                output_types=["bias_assumptions", "fairness_analysis"],
                confidence_level=0.75,
                estimated_processing_time=12.0
            ),
            AgentCapability(
                name="cultural_assumption_analysis",
                description="Identify culturally-specific assumptions",
                input_types=["content", "cultural_context"],
                output_types=["cultural_assumptions", "universality_analysis"],
                confidence_level=0.7,
                estimated_processing_time=9.0
            )
        ]
    
    async def _initialize(self):
        """Initialize assumption detector components."""
        await super()._initialize()
        
        try:
            # Load assumption detection patterns
            await self._load_assumption_patterns()
            
            # Initialize cultural markers if cross-cultural analysis is enabled
            if self.cross_cultural_analysis:
                await self._load_cultural_markers()
            
            logger.info("Assumption detector components initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize assumption detector components",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup assumption detector resources."""
        await super()._cleanup()
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for assumption detection."""
        if message.type == MessageType.REQUEST:
            request_type = message.content.get('request_type', '')
            
            if request_type == 'detect_assumptions':
                return await self._handle_assumption_detection(message)
            elif request_type == 'analyze_assumption_impact':
                return await self._handle_impact_analysis(message)
            elif request_type == 'validate_assumptions':
                return await self._handle_assumption_validation(message)
            elif request_type == 'detect_bias_assumptions':
                return await self._handle_bias_assumption_detection(message)
            elif request_type == 'cultural_assumption_analysis':
                return await self._handle_cultural_analysis(message)
            else:
                logger.debug(
                    "Unsupported request type for assumption detector",
                    agent_id=self.agent_id,
                    request_type=request_type
                )
                return None
        else:
            logger.debug(
                "Unsupported message type for assumption detector",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_assumption_detection(self, message: AgentMessage) -> ResponseMessage:
        """Handle assumption detection requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract analysis parameters
            content = message.content.get('content', '')
            content_type = message.content.get('content_type', 'response')
            context = message.content.get('context', {})
            sensitivity = message.content.get('sensitivity', self.detection_sensitivity)
            
            if not content:
                raise ValueError("Content is required for assumption detection")
            
            logger.info(
                "Processing assumption detection",
                agent_id=self.agent_id,
                content_length=len(content),
                content_type=content_type,
                sensitivity=sensitivity
            )
            
            # Perform assumption detection
            analysis_result = await self._detect_assumptions(content, content_type, context, sensitivity)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            analysis_result.analysis_time = processing_time
            
            # Create response
            response_content = {
                'assumption_analysis': analysis_result.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=analysis_result.detection_confidence,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Detected {len(analysis_result.detected_assumptions)} assumptions with {analysis_result.assumption_burden:.2f} burden score"
            )
            
        except Exception as e:
            logger.error(
                "Assumption detection failed",
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
                answer=f"Assumption detection failed: {str(e)}"
            )
    
    async def _detect_assumptions(
        self,
        content: str,
        content_type: str,
        context: Dict[str, Any],
        sensitivity: float
    ) -> AssumptionAnalysis:
        """Perform comprehensive assumption detection."""
        
        # Step 1: Detect assumptions by type
        detected_assumptions = []
        
        for assumption_type in AssumptionType:
            type_assumptions = await self._detect_assumption_type(
                content, assumption_type, context, sensitivity
            )
            detected_assumptions.extend(type_assumptions)
        
        # Step 2: Categorize assumptions
        assumption_categories = {}
        for assumption in detected_assumptions:
            cat = assumption.assumption_type.value
            assumption_categories[cat] = assumption_categories.get(cat, 0) + 1
        
        # Step 3: Assess risk and dependencies
        high_risk_assumptions, dependencies = await self._assess_assumption_risks(detected_assumptions)
        
        # Step 4: Calculate overall scores
        assumption_burden = await self._calculate_assumption_burden(detected_assumptions, content)
        robustness_score = await self._calculate_robustness_score(detected_assumptions, content)
        
        # Step 5: Generate recommendations
        to_validate, to_make_explicit, info_needed = await self._generate_assumption_recommendations(
            detected_assumptions, context
        )
        
        # Step 6: Generate alternative scenarios
        assumption_free, worst_case = await self._generate_alternative_scenarios(
            detected_assumptions, content, context
        )
        
        # Step 7: Calculate overall detection confidence
        if detected_assumptions:
            detection_confidence = sum(a.confidence_in_detection for a in detected_assumptions) / len(detected_assumptions)
        else:
            detection_confidence = 0.9  # High confidence when no assumptions detected
        
        return AssumptionAnalysis(
            analyzed_content=content,
            content_type=content_type,
            analysis_context=context,
            detected_assumptions=detected_assumptions[:self.max_assumptions_per_analysis],
            assumption_categories=assumption_categories,
            high_risk_assumptions=high_risk_assumptions,
            assumption_dependencies=dependencies,
            assumption_burden=assumption_burden,
            robustness_score=robustness_score,
            assumptions_to_validate=to_validate,
            assumptions_to_make_explicit=to_make_explicit,
            information_needed=info_needed,
            assumption_free_conclusions=assumption_free,
            worst_case_scenarios=worst_case,
            detection_confidence=detection_confidence
        )
    
    async def _detect_assumption_type(
        self,
        content: str,
        assumption_type: AssumptionType,
        context: Dict[str, Any],
        sensitivity: float
    ) -> List[DetectedAssumption]:
        """Detect assumptions of a specific type."""
        try:
            # Use LLM to detect assumptions of this type
            llm_response = await self.generate_response(
                content,
                template_name=f'detect_{assumption_type.value}',
                template_variables={
                    'content': content,
                    'assumption_type': assumption_type.value,
                    'context': context,
                    'sensitivity': sensitivity
                }
            )
            
            # Parse detection results
            import json
            detection_data = json.loads(llm_response.content)
            
            assumptions = []
            for i, assumption_data in enumerate(detection_data.get('assumptions', [])):
                if assumption_data.get('confidence', 0) >= sensitivity:
                    assumption = DetectedAssumption(
                        assumption_id=f"{assumption_type.value}_{i}",
                        assumption_type=assumption_type,
                        criticality=AssumptionCriticality(assumption_data.get('criticality', 'medium')),
                        description=assumption_data['description'],
                        implicit_statement=assumption_data['implicit_statement'],
                        explicit_alternative=assumption_data.get('explicit_alternative', ''),
                        context_location=assumption_data.get('location'),
                        related_reasoning=assumption_data.get('related_reasoning', []),
                        impact_on_conclusion=assumption_data.get('impact', 0.5),
                        alternative_outcomes=assumption_data.get('alternatives', []),
                        supporting_indicators=assumption_data.get('indicators', []),
                        confidence_in_detection=assumption_data.get('confidence', 0.5),
                        validation_questions=assumption_data.get('validation_questions', []),
                        challenge_scenarios=assumption_data.get('challenge_scenarios', [])
                    )
                    assumptions.append(assumption)
            
            return assumptions
            
        except Exception as e:
            logger.warning(
                "Failed to detect assumption type",
                agent_id=self.agent_id,
                assumption_type=assumption_type,
                error=str(e)
            )
            return []
    
    async def _assess_assumption_risks(
        self, assumptions: List[DetectedAssumption]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Assess risks and dependencies among assumptions."""
        high_risk = []
        dependencies = {}
        
        for assumption in assumptions:
            # High risk if critical or high impact
            if (assumption.criticality in [AssumptionCriticality.HIGH, AssumptionCriticality.CRITICAL] or
                assumption.impact_on_conclusion > 0.7):
                high_risk.append(assumption.description)
            
            # Simple dependency detection based on related reasoning
            if assumption.related_reasoning:
                dependencies[assumption.assumption_id] = assumption.related_reasoning
        
        return high_risk, dependencies
    
    async def _calculate_assumption_burden(self, assumptions: List[DetectedAssumption], content: str) -> float:
        """Calculate how assumption-heavy the content is."""
        if not assumptions:
            return 0.0
        
        # Weight by criticality and impact
        weighted_sum = 0.0
        total_weight = 0.0
        
        criticality_weights = {
            AssumptionCriticality.LOW: 0.25,
            AssumptionCriticality.MEDIUM: 0.5,
            AssumptionCriticality.HIGH: 0.75,
            AssumptionCriticality.CRITICAL: 1.0
        }
        
        for assumption in assumptions:
            weight = criticality_weights.get(assumption.criticality, 0.5)
            impact = assumption.impact_on_conclusion
            confidence = assumption.confidence_in_detection
            
            assumption_score = weight * impact * confidence
            weighted_sum += assumption_score
            total_weight += weight
        
        # Normalize by content length and total possible weight
        content_factor = min(1.0, len(assumptions) / (len(content.split()) / 10))  # Rough heuristic
        burden_score = (weighted_sum / max(total_weight, 1.0)) * content_factor
        
        return min(1.0, burden_score)
    
    async def _calculate_robustness_score(self, assumptions: List[DetectedAssumption], content: str) -> float:
        """Calculate how robust conclusions are to the detected assumptions."""
        if not assumptions:
            return 0.95  # High robustness if no significant assumptions
        
        # Calculate average impact of assumptions
        avg_impact = sum(a.impact_on_conclusion for a in assumptions) / len(assumptions)
        
        # Count critical assumptions
        critical_count = sum(1 for a in assumptions if a.criticality == AssumptionCriticality.CRITICAL)
        critical_penalty = critical_count * 0.2
        
        # Robustness is inverse of assumption impact
        robustness = 1.0 - (avg_impact * 0.7) - critical_penalty
        
        return max(0.0, min(1.0, robustness))
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for assumption detection."""
        templates = {}
        
        # Generic assumption detection template
        templates['detect_implicit_premise'] = PromptTemplate(
            name='detect_implicit_premise',
            template="""
            Analyze the content for implicit premises - unstated logical assumptions that are necessary for the reasoning to work.
            
            Content: {content}
            Context: {context}
            Sensitivity: {sensitivity}
            
            Look for:
            1. Unstated premises that are necessary for arguments to be valid
            2. Hidden logical steps in reasoning chains
            3. Assumptions about relationships between concepts
            4. Unstated conditions or qualifications
            
            For each implicit premise found:
            - Describe what is being assumed
            - Explain why it's necessary for the reasoning
            - Assess how critical this assumption is
            - Suggest how to make it explicit
            - Consider alternative assumptions
            
            Respond with JSON:
            {{
                "assumptions": [
                    {{
                        "description": "detailed description of the assumption",
                        "implicit_statement": "what is implicitly assumed",
                        "explicit_alternative": "how to make it explicit",
                        "criticality": "low|medium|high|critical",
                        "location": "where in content this appears",
                        "related_reasoning": ["related reasoning step 1"],
                        "impact": 0.8,
                        "alternatives": ["alternative assumption 1"],
                        "indicators": ["textual indicator 1"],
                        "confidence": 0.9,
                        "validation_questions": ["question to test assumption"],
                        "challenge_scenarios": ["scenario that challenges assumption"]
                    }}
                ]
            }}
            """.strip()
        )
        
        # Background knowledge assumption template  
        templates['detect_background_knowledge'] = PromptTemplate(
            name='detect_background_knowledge',
            template="""
            Identify assumptions about background knowledge that the reader is expected to have.
            
            Content: {content}
            Context: {context}
            
            Look for:
            1. Unexplained technical terms or concepts
            2. References to domain-specific knowledge
            3. Cultural or contextual references
            4. Historical or factual assumptions
            5. Assumptions about reader expertise level
            
            Assess whether these assumptions are reasonable given the context.
            Consider what happens if readers don't have this background knowledge.
            
            Follow the same JSON format as the implicit_premise template.
            """.strip()
        )
        
        return templates
    
    # Helper method stubs
    async def _load_assumption_patterns(self):
        """Load patterns for assumption detection."""
        # Would load from knowledge base or configuration
        self.assumption_patterns = {
            'causal_words': ['because', 'since', 'therefore', 'thus', 'consequently'],
            'certainty_markers': ['obviously', 'clearly', 'certainly', 'definitely'],
            'universality_markers': ['all', 'every', 'always', 'never', 'none'],
            'value_judgments': ['should', 'ought to', 'better', 'worse', 'good', 'bad']
        }
    
    async def _load_cultural_markers(self):
        """Load cultural assumption markers."""
        self.cultural_assumption_markers = {
            'western_values', 'individual_focus', 'time_orientation',
            'authority_relations', 'gender_roles', 'family_structures'
        }
    
    async def _generate_assumption_recommendations(
        self, assumptions: List[DetectedAssumption], context: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate recommendations for handling assumptions."""
        pass
    
    async def _generate_alternative_scenarios(
        self, assumptions: List[DetectedAssumption], content: str, context: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Generate alternative scenarios with different assumptions."""
        pass
    
    # Additional handler method stubs
    async def _handle_impact_analysis(self, message: AgentMessage) -> ResponseMessage:
        """Handle assumption impact analysis."""
        pass
    
    async def _handle_assumption_validation(self, message: AgentMessage) -> ResponseMessage:
        """Handle assumption validation requests."""
        pass
    
    async def _handle_bias_assumption_detection(self, message: AgentMessage) -> ResponseMessage:
        """Handle bias assumption detection."""
        pass
    
    async def _handle_cultural_analysis(self, message: AgentMessage) -> ResponseMessage:
        """Handle cultural assumption analysis."""
        pass
