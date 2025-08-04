"""
Consistency Checker Agent - Specialized agent for detecting contradictions and inconsistencies.
Handles logical contradiction detection, consistency verification, and conflict resolution.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, ResponseMessage
from reasoning.logic.contradiction_detector import ContradictionDetector
from reasoning.logic.consistency_verifier import ConsistencyVerifier
from core.exceptions import ReasoningError, LogicError

logger = structlog.get_logger(__name__)


class ContradictionType(str, Enum):
    """Types of contradictions that can be detected."""
    
    DIRECT = "direct"  # A directly contradicts B
    IMPLICIT = "implicit"  # A implies something that contradicts B
    CONDITIONAL = "conditional"  # A contradicts B under certain conditions
    TEMPORAL = "temporal"  # A and B contradict across different time periods
    SCOPE = "scope"  # A and B contradict within overlapping scopes


class ConsistencyLevel(str, Enum):
    """Levels of consistency analysis."""
    
    STRICT = "strict"  # Formal logical consistency
    PRACTICAL = "practical"  # Real-world interpretation consistency
    CONTEXTUAL = "contextual"  # Context-dependent consistency


class ContradictionResult(BaseModel):
    """Result of contradiction detection between clauses."""
    
    # Clause identification
    clause_1_id: str
    clause_2_id: str
    
    # Contradiction details
    contradiction_type: ContradictionType
    severity: float = Field(..., ge=0.0, le=1.0)  # How severe the contradiction is
    confidence: float = Field(..., ge=0.0, le=1.0)  # Confidence in detection
    
    # Evidence
    contradicting_statements: List[str]
    evidence_text: List[str]
    logical_reasoning: str
    
    # Context
    context_conditions: List[str] = Field(default_factory=list)
    affected_scenarios: List[str] = Field(default_factory=list)
    
    # Resolution suggestions
    resolution_suggestions: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    
    # Metadata
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    detection_method: str = "llm_analysis"


class ConsistencyAnalysis(BaseModel):
    """Complete consistency analysis of a set of clauses."""
    
    # Input information
    analyzed_clauses: List[str]  # Clause IDs
    analysis_scope: str
    consistency_level: ConsistencyLevel
    
    # Results
    contradictions: List[ContradictionResult]
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    
    # Summary
    total_contradictions: int = 0
    severe_contradictions: int = 0
    resolvable_contradictions: int = 0
    
    # Recommendations
    priority_issues: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    clarification_needed: List[str] = Field(default_factory=list)
    
    # Analysis metadata
    analysis_time: float = 0.0
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class ConsistencyCheckerAgent(LLMAgent):
    """Agent specialized in detecting contradictions and verifying consistency."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="ConsistencyCheckerAgent",
            config=config or {}
        )
        
        # Analysis components
        self.contradiction_detector: Optional[ContradictionDetector] = None
        self.consistency_verifier: Optional[ConsistencyVerifier] = None
        
        # Configuration
        self.default_confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_clause_pairs = self.config.get('max_clause_pairs', 100)
        self.enable_deep_analysis = self.config.get('enable_deep_analysis', True)
        self.parallel_analysis = self.config.get('parallel_analysis', True)
        
        # Caching for performance
        self.analysis_cache: Dict[str, ContradictionResult] = {}
        self.cache_ttl = self.config.get('cache_ttl_hours', 24)
        
        logger.info("Consistency Checker Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize consistency checker capabilities."""
        return [
            AgentCapability(
                name="contradiction_detection",
                description="Detect direct and implicit contradictions between clauses",
                input_types=["clause_pairs", "clause_set"],
                output_types=["contradictions", "analysis_results"],
                confidence_level=0.85,
                estimated_processing_time=5.0
            ),
            AgentCapability(
                name="consistency_verification",
                description="Verify logical consistency across multiple clauses",
                input_types=["clause_set", "policy_document"],
                output_types=["consistency_analysis", "verification_results"],
                confidence_level=0.8,
                estimated_processing_time=8.0
            ),
            AgentCapability(
                name="conflict_resolution",
                description="Suggest resolutions for detected contradictions",
                input_types=["contradictions", "context"],
                output_types=["resolution_suggestions", "recommendations"],
                confidence_level=0.75,
                estimated_processing_time=6.0
            ),
            AgentCapability(
                name="temporal_consistency",
                description="Check consistency across different time periods or versions",
                input_types=["versioned_clauses", "temporal_context"],
                output_types=["temporal_analysis", "version_conflicts"],
                confidence_level=0.8,
                estimated_processing_time=7.0
            ),
            AgentCapability(
                name="contextual_analysis",
                description="Analyze consistency within specific contexts or scenarios",
                input_types=["clause_set", "context_scenarios"],
                output_types=["contextual_consistency", "scenario_analysis"],
                confidence_level=0.75,
                estimated_processing_time=6.0
            )
        ]
    
    async def _initialize(self):
        """Initialize consistency checker components."""
        await super()._initialize()
        
        try:
            # Initialize analysis components
            self.contradiction_detector = ContradictionDetector(
                llm_provider=self.llm_provider,
                config=self.config.get('contradiction_detection', {})
            )
            await self.contradiction_detector.initialize()
            
            self.consistency_verifier = ConsistencyVerifier(
                llm_provider=self.llm_provider,
                config=self.config.get('consistency_verification', {})
            )
            await self.consistency_verifier.initialize()
            
            logger.info("Consistency checker components initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize consistency checker components",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup consistency checker resources."""
        try:
            if self.contradiction_detector:
                await self.contradiction_detector.cleanup()
            
            if self.consistency_verifier:
                await self.consistency_verifier.cleanup()
            
            await super()._cleanup()
            
        except Exception as e:
            logger.error(
                "Error during consistency checker cleanup",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for consistency checking."""
        if message.type == MessageType.REQUEST:
            request_type = message.content.get('request_type', '')
            
            if request_type == 'check_contradictions':
                return await self._handle_contradiction_check(message)
            elif request_type == 'verify_consistency':
                return await self._handle_consistency_verification(message)
            elif request_type == 'analyze_conflicts':
                return await self._handle_conflict_analysis(message)
            elif request_type == 'temporal_consistency':
                return await self._handle_temporal_analysis(message)
            else:
                logger.debug(
                    "Unsupported request type for consistency checker",
                    agent_id=self.agent_id,
                    request_type=request_type
                )
                return None
        else:
            logger.debug(
                "Unsupported message type for consistency checker",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_contradiction_check(self, message: AgentMessage) -> ResponseMessage:
        """Handle contradiction detection requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract clause information
            clauses = message.content.get('clauses', [])
            clause_pairs = message.content.get('clause_pairs', [])
            analysis_level = message.content.get('analysis_level', 'practical')
            
            if not clauses and not clause_pairs:
                raise ValueError("Either 'clauses' or 'clause_pairs' must be provided")
            
            logger.info(
                "Processing contradiction check",
                agent_id=self.agent_id,
                clauses_count=len(clauses),
                pairs_count=len(clause_pairs)
            )
            
            # Perform contradiction detection
            if clause_pairs:
                contradictions = await self._check_clause_pairs(clause_pairs, analysis_level)
            else:
                contradictions = await self._check_clause_set(clauses, analysis_level)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate overall confidence
            avg_confidence = sum(c.confidence for c in contradictions) / max(len(contradictions), 1)
            
            # Create response
            response_content = {
                'contradictions': [c.dict() for c in contradictions],
                'total_contradictions': len(contradictions),
                'severe_contradictions': len([c for c in contradictions if c.severity > 0.7]),
                'analysis_level': analysis_level,
                'processing_time': processing_time,
                'success': True
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=avg_confidence,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Found {len(contradictions)} contradictions with {avg_confidence:.2f} average confidence"
            )
            
        except Exception as e:
            logger.error(
                "Contradiction check failed",
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
                answer=f"Contradiction check failed: {str(e)}"
            )
    
    async def _check_clause_pairs(
        self, 
        clause_pairs: List[Tuple[str, str]], 
        analysis_level: str
    ) -> List[ContradictionResult]:
        """Check specific clause pairs for contradictions."""
        contradictions = []
        
        if self.parallel_analysis and len(clause_pairs) > 5:
            # Process in parallel for better performance
            tasks = []
            for pair in clause_pairs[:self.max_clause_pairs]:
                task = self._analyze_clause_pair(pair[0], pair[1], analysis_level)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ContradictionResult):
                    contradictions.append(result)
                elif isinstance(result, Exception):
                    logger.warning(
                        "Clause pair analysis failed",
                        agent_id=self.agent_id,
                        error=str(result)
                    )
        else:
            # Sequential processing
            for pair in clause_pairs[:self.max_clause_pairs]:
                try:
                    result = await self._analyze_clause_pair(pair[0], pair[1], analysis_level)
                    if result:
                        contradictions.append(result)
                except Exception as e:
                    logger.warning(
                        "Clause pair analysis failed",
                        agent_id=self.agent_id,
                        pair=pair,
                        error=str(e)
                    )
        
        return contradictions
    
    async def _check_clause_set(
        self, 
        clauses: List[Dict[str, Any]], 
        analysis_level: str
    ) -> List[ContradictionResult]:
        """Check all pairs within a set of clauses for contradictions."""
        contradictions = []
        
        # Generate all pairs
        clause_pairs = []
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                clause_pairs.append((clauses[i]['id'], clauses[j]['id']))
        
        # Limit pairs to avoid excessive processing
        if len(clause_pairs) > self.max_clause_pairs:
            logger.warning(
                "Too many clause pairs, limiting analysis",
                agent_id=self.agent_id,
                total_pairs=len(clause_pairs),
                max_pairs=self.max_clause_pairs
            )
            clause_pairs = clause_pairs[:self.max_clause_pairs]
        
        # Check pairs
        return await self._check_clause_pairs(clause_pairs, analysis_level)
    
    async def _analyze_clause_pair(
        self, 
        clause_id_1: str, 
        clause_id_2: str, 
        analysis_level: str
    ) -> Optional[ContradictionResult]:
        """Analyze a specific pair of clauses for contradictions."""
        # Check cache first
        cache_key = f"{clause_id_1}:{clause_id_2}:{analysis_level}"
        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            # Check if cache is still valid
            if (datetime.utcnow() - cached_result.detected_at).total_seconds() < self.cache_ttl * 3600:
                return cached_result
        
        try:
            # Use LLM for contradiction analysis
            llm_response = await self.generate_response(
                "",  # Prompt built from template
                template_name='contradiction_analysis',
                template_variables={
                    'clause_1_id': clause_id_1,
                    'clause_2_id': clause_id_2,
                    'analysis_level': analysis_level
                }
            )
            
            # Parse LLM response
            import json
            analysis_data = json.loads(llm_response.content)
            
            # Only create result if contradiction detected
            if analysis_data.get('has_contradiction', False):
                contradiction = ContradictionResult(
                    clause_1_id=clause_id_1,
                    clause_2_id=clause_id_2,
                    contradiction_type=ContradictionType(analysis_data.get('type', 'direct')),
                    severity=analysis_data.get('severity', 0.5),
                    confidence=analysis_data.get('confidence', 0.5),
                    contradicting_statements=analysis_data.get('contradicting_statements', []),
                    evidence_text=analysis_data.get('evidence', []),
                    logical_reasoning=analysis_data.get('reasoning', ''),
                    context_conditions=analysis_data.get('conditions', []),
                    affected_scenarios=analysis_data.get('scenarios', []),
                    resolution_suggestions=analysis_data.get('resolutions', []),
                    requires_clarification=analysis_data.get('needs_clarification', False),
                    detection_method='llm_analysis'
                )
                
                # Cache the result
                self.analysis_cache[cache_key] = contradiction
                
                return contradiction
            
            return None
            
        except Exception as e:
            logger.error(
                "Clause pair analysis failed",
                agent_id=self.agent_id,
                clause_1=clause_id_1,
                clause_2=clause_id_2,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def _handle_consistency_verification(self, message: AgentMessage) -> ResponseMessage:
        """Handle consistency verification requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract parameters
            clauses = message.content.get('clauses', [])
            consistency_level = ConsistencyLevel(message.content.get('level', 'practical'))
            scope = message.content.get('scope', 'document')
            
            if not clauses:
                raise ValueError("Clauses list is required for consistency verification")
            
            logger.info(
                "Processing consistency verification",
                agent_id=self.agent_id,
                clauses_count=len(clauses),
                level=consistency_level,
                scope=scope
            )
            
            # Perform consistency analysis
            analysis = await self._verify_consistency(clauses, consistency_level, scope)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            analysis.analysis_time = processing_time
            
            # Create response
            response_content = {
                'consistency_analysis': analysis.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=analysis.confidence_level,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Consistency score: {analysis.consistency_score:.2f} with {analysis.total_contradictions} contradictions found"
            )
            
        except Exception as e:
            logger.error(
                "Consistency verification failed",
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
                answer=f"Consistency verification failed: {str(e)}"
            )
    
    async def _verify_consistency(
        self, 
        clauses: List[Dict[str, Any]], 
        level: ConsistencyLevel, 
        scope: str
    ) -> ConsistencyAnalysis:
        """Perform comprehensive consistency verification."""
        # Get all contradictions
        contradictions = await self._check_clause_set(clauses, level.value)
        
        # Calculate consistency score
        total_pairs = len(clauses) * (len(clauses) - 1) // 2
        contradiction_rate = len(contradictions) / max(total_pairs, 1)
        consistency_score = max(0.0, 1.0 - contradiction_rate)
        
        # Categorize contradictions
        severe_contradictions = [c for c in contradictions if c.severity > 0.7]
        resolvable_contradictions = [c for c in contradictions if c.resolution_suggestions]
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(contradictions, clauses)
        
        # Calculate overall confidence
        if contradictions:
            confidence = sum(c.confidence for c in contradictions) / len(contradictions)
        else:
            confidence = 0.9  # High confidence when no contradictions found
        
        return ConsistencyAnalysis(
            analyzed_clauses=[c['id'] for c in clauses],
            analysis_scope=scope,
            consistency_level=level,
            contradictions=contradictions,
            consistency_score=consistency_score,
            total_contradictions=len(contradictions),
            severe_contradictions=len(severe_contradictions),
            resolvable_contradictions=len(resolvable_contradictions),
            priority_issues=recommendations.get('priority_issues', []),
            recommended_actions=recommendations.get('actions', []),
            clarification_needed=recommendations.get('clarifications', []),
            confidence_level=confidence
        )
    
    async def _generate_recommendations(
        self, 
        contradictions: List[ContradictionResult], 
        clauses: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate recommendations based on consistency analysis."""
        try:
            # Use LLM to generate comprehensive recommendations
            llm_response = await self.generate_response(
                "",
                template_name='consistency_recommendations',
                template_variables={
                    'contradictions': [c.dict() for c in contradictions],
                    'clause_count': len(clauses),
                    'severe_count': len([c for c in contradictions if c.severity > 0.7])
                }
            )
            
            # Parse recommendations
            import json
            recommendations = json.loads(llm_response.content)
            return recommendations
            
        except Exception as e:
            logger.warning(
                "Failed to generate LLM recommendations",
                agent_id=self.agent_id,
                error=str(e)
            )
            
            # Fallback to rule-based recommendations
            return self._generate_fallback_recommendations(contradictions)
    
    def _generate_fallback_recommendations(
        self, 
        contradictions: List[ContradictionResult]
    ) -> Dict[str, List[str]]:
        """Generate basic recommendations when LLM fails."""
        recommendations = {
            'priority_issues': [],
            'actions': [],
            'clarifications': []
        }
        
        # Identify priority issues
        severe_contradictions = [c for c in contradictions if c.severity > 0.7]
        if severe_contradictions:
            recommendations['priority_issues'] = [
                f"Severe contradiction between clauses {c.clause_1_id} and {c.clause_2_id}"
                for c in severe_contradictions[:3]
            ]
        
        # Basic actions
        if contradictions:
            recommendations['actions'] = [
                "Review contradicting clauses for potential conflicts",
                "Consider clause prioritization or exception handling",
                "Evaluate need for clause amendments or clarifications"
            ]
        
        # Clarifications needed
        clarification_needed = [c for c in contradictions if c.requires_clarification]
        if clarification_needed:
            recommendations['clarifications'] = [
                f"Clarification needed for contradiction between {c.clause_1_id} and {c.clause_2_id}"
                for c in clarification_needed[:3]
            ]
        
        return recommendations
    
    async def _handle_conflict_analysis(self, message: AgentMessage) -> ResponseMessage:
        """Handle conflict analysis and resolution requests."""
        # Implementation for conflict analysis
        # This would involve deeper analysis of contradictions and resolution strategies
        pass
    
    async def _handle_temporal_analysis(self, message: AgentMessage) -> ResponseMessage:
        """Handle temporal consistency analysis."""
        # Implementation for temporal consistency checking
        # This would analyze consistency across different versions or time periods
        pass
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for consistency checking."""
        templates = {}
        
        # Contradiction analysis template
        templates['contradiction_analysis'] = PromptTemplate(
            name='contradiction_analysis',
            template="""
            Analyze these two clauses for potential contradictions or conflicts.
            
            Clause 1 ID: {clause_1_id}
            Clause 2 ID: {clause_2_id}
            Analysis Level: {analysis_level}
            
            Consider:
            1. Direct contradictions (A says yes, B says no)
            2. Implicit contradictions (A implies X, B implies not-X)
            3. Conditional contradictions (A and B conflict under certain conditions)
            4. Scope conflicts (A and B apply to overlapping but different scopes)
            
            Respond with JSON:
            {{
                "has_contradiction": true/false,
                "type": "direct|implicit|conditional|temporal|scope",
                "severity": 0.8,
                "confidence": 0.9,
                "contradicting_statements": ["statement from A", "statement from B"],
                "evidence": ["evidence text 1", "evidence text 2"],
                "reasoning": "detailed logical reasoning",
                "conditions": ["condition under which contradiction occurs"],
                "scenarios": ["affected scenarios"],
                "resolutions": ["suggested resolution 1", "suggested resolution 2"],
                "needs_clarification": false
            }}
            """.strip()
        )
        
        # Consistency recommendations template
        templates['consistency_recommendations'] = PromptTemplate(
            name='consistency_recommendations',
            template="""
            Generate recommendations based on consistency analysis results.
            
            Contradictions Found: {contradictions}
            Total Clauses: {clause_count}
            Severe Contradictions: {severe_count}
            
            Provide:
            1. Priority issues that need immediate attention
            2. Recommended actions to resolve conflicts
            3. Areas that need clarification
            
            Respond with JSON:
            {{
                "priority_issues": ["priority issue 1", "priority issue 2"],
                "actions": ["action 1", "action 2", "action 3"],
                "clarifications": ["clarification needed 1", "clarification needed 2"]
            }}
            """.strip()
        )
        
        return templates
