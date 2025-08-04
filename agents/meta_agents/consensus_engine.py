"""
Consensus Engine Meta-Agent - Manages multi-source consensus and conflict resolution.
Handles federated decision making, source weighting, and consensus building across multiple agents and sources.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from enum import Enum
import statistics

import structlog
from pydantic import BaseModel, Field

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, ResponseMessage
from reasoning.consensus.voting_mechanism import VotingMechanism
from reasoning.consensus.conflict_resolver import ConflictResolver
from reasoning.consensus.trust_scorer import TrustScorer
from core.exceptions import ReasoningError

logger = structlog.get_logger(__name__)


class ConsensusMethod(str, Enum):
    """Methods for reaching consensus."""
    
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_VOTING = "weighted_voting"
    EXPERT_CONSENSUS = "expert_consensus"
    BAYESIAN_CONSENSUS = "bayesian_consensus"
    DELPHI_METHOD = "delphi_method"
    RANKED_CHOICE = "ranked_choice"
    THRESHOLD_CONSENSUS = "threshold_consensus"


class ConflictType(str, Enum):
    """Types of conflicts between sources."""
    
    FACTUAL_DISAGREEMENT = "factual_disagreement"
    VALUE_CONFLICT = "value_conflict"
    INTERPRETATION_DIFFERENCE = "interpretation_difference"
    SCOPE_DISAGREEMENT = "scope_disagreement"
    TEMPORAL_CONFLICT = "temporal_conflict"
    AUTHORITY_CONFLICT = "authority_conflict"
    METHODOLOGY_CONFLICT = "methodology_conflict"


class ConsensusParticipant(BaseModel):
    """A participant in the consensus process."""
    
    participant_id: str
    participant_type: str  # agent, source, expert, document, etc.
    
    # Position and confidence
    position: str  # Their stance or opinion
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    
    # Trust and weighting
    trust_score: float = Field(..., ge=0.0, le=1.0)
    expertise_level: float = Field(..., ge=0.0, le=1.0)
    voting_weight: float = Field(..., ge=0.0, le=1.0)
    
    # Context
    supporting_evidence: List[str] = Field(default_factory=list)
    dissenting_views: List[str] = Field(default_factory=list)
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    participation_history: Dict[str, Any] = Field(default_factory=dict)


class ConflictResolution(BaseModel):
    """A resolved conflict between participants."""
    
    conflict_id: str
    conflict_type: ConflictType
    
    # Participants in conflict
    conflicting_participants: List[str]
    conflict_description: str
    
    # Resolution
    resolution_method: str
    resolution_explanation: str
    resolved_position: str
    resolution_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Compromise and trade-offs
    compromises_made: List[str] = Field(default_factory=list)
    remaining_disagreements: List[str] = Field(default_factory=list)
    
    # Validation
    participant_acceptance: Dict[str, bool] = Field(default_factory=dict)
    resolution_stability: float = Field(..., ge=0.0, le=1.0)


class ConsensusResult(BaseModel):
    """Complete consensus building result."""
    
    # Consensus query/topic
    consensus_topic: str
    participants: List[ConsensusParticipant]
    consensus_method: ConsensusMethod
    
    # Consensus outcome
    consensus_reached: bool
    consensus_position: str
    consensus_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Voting details
    voting_results: Dict[str, Any] = Field(default_factory=dict)
    participation_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Conflict resolution
    identified_conflicts: List[ConflictResolution]
    unresolved_conflicts: List[str] = Field(default_factory=list)
    
    # Quality metrics
    consensus_strength: float = Field(..., ge=0.0, le=1.0)  # How strong the consensus is
    diversity_score: float = Field(..., ge=0.0, le=1.0)    # How diverse the input was
    convergence_rate: float = Field(..., ge=0.0, le=1.0)   # How quickly consensus emerged
    
    # Alternative positions
    minority_positions: List[Dict[str, Any]] = Field(default_factory=list)
    dissenting_opinions: List[str] = Field(default_factory=list)
    
    # Recommendations
    consensus_recommendations: List[str] = Field(default_factory=list)
    areas_needing_clarification: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    
    # Processing metadata
    consensus_iterations: int = 0
    consensus_time: float = 0.0
    final_consensus_at: datetime = Field(default_factory=datetime.utcnow)


class ConsensusEngineAgent(LLMAgent):
    """Meta-agent specialized in building consensus from multiple sources and resolving conflicts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="ConsensusEngineAgent",
            config=config or {}
        )
        
        # Consensus components
        self.voting_mechanism: Optional[VotingMechanism] = None
        self.conflict_resolver: Optional[ConflictResolver] = None
        self.trust_scorer: Optional[TrustScorer] = None
        
        # Configuration
        self.default_consensus_method = ConsensusMethod(self.config.get('default_method', 'weighted_voting'))
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        self.max_iterations = self.config.get('max_iterations', 5)
        self.min_participants = self.config.get('min_participants', 2)
        
        # Conflict resolution settings
        self.auto_resolve_conflicts = self.config.get('auto_resolve', True)
        self.conflict_resolution_strategy = self.config.get('conflict_strategy', 'weighted_compromise')
        
        logger.info("Consensus Engine Meta-Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize consensus building capabilities."""
        return [
            AgentCapability(
                name="multi_source_consensus",
                description="Build consensus from multiple sources and participants",
                input_types=["multiple_sources", "agent_responses", "expert_opinions"],
                output_types=["consensus_result", "unified_position"],
                confidence_level=0.85,
                estimated_processing_time=15.0
            ),
            AgentCapability(
                name="conflict_identification",
                description="Identify and categorize conflicts between sources",
                input_types=["conflicting_sources", "disagreements"],
                output_types=["conflict_analysis", "conflict_categories"],
                confidence_level=0.9,
                estimated_processing_time=10.0
            ),
            AgentCapability(
                name="conflict_resolution",
                description="Resolve conflicts through various consensus mechanisms",
                input_types=["identified_conflicts", "participant_preferences"],
                output_types=["resolved_conflicts", "compromise_solutions"],
                confidence_level=0.8,
                estimated_processing_time=20.0
            ),
            AgentCapability(
                name="trust_assessment",
                description="Assess trust and reliability of consensus participants",
                input_types=["participant_history", "source_reliability"],
                output_types=["trust_scores", "weighting_recommendations"],
                confidence_level=0.85,
                estimated_processing_time=8.0
            ),
            AgentCapability(
                name="consensus_validation",
                description="Validate and test the stability of reached consensus",
                input_types=["consensus_results", "test_scenarios"],
                output_types=["validation_results", "stability_analysis"],
                confidence_level=0.8,
                estimated_processing_time=12.0
            )
        ]
    
    async def _initialize(self):
        """Initialize consensus engine components."""
        await super()._initialize()
        
        try:
            # Initialize consensus components
            self.voting_mechanism = VotingMechanism(
                config=self.config.get('voting_mechanism', {})
            )
            await self.voting_mechanism.initialize()
            
            self.conflict_resolver = ConflictResolver(
                llm_provider=self.llm_provider,
                config=self.config.get('conflict_resolution', {})
            )
            await self.conflict_resolver.initialize()
            
            self.trust_scorer = TrustScorer(
                config=self.config.get('trust_scoring', {})
            )
            await self.trust_scorer.initialize()
            
            logger.info("Consensus engine components initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize consensus engine components",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup consensus engine resources."""
        try:
            if self.voting_mechanism:
                await self.voting_mechanism.cleanup()
            
            if self.conflict_resolver:
                await self.conflict_resolver.cleanup()
            
            if self.trust_scorer:
                await self.trust_scorer.cleanup()
            
            await super()._cleanup()
            
        except Exception as e:
            logger.error(
                "Error during consensus engine cleanup",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for consensus building."""
        if message.type == MessageType.REQUEST:
            request_type = message.content.get('request_type', '')
            
            if request_type == 'build_consensus':
                return await self._handle_consensus_request(message)
            elif request_type == 'resolve_conflicts':
                return await self._handle_conflict_resolution(message)
            elif request_type == 'assess_trust':
                return await self._handle_trust_assessment(message)
            elif request_type == 'validate_consensus':
                return await self._handle_consensus_validation(message)
            elif request_type == 'weighted_voting':
                return await self._handle_weighted_voting(message)
            else:
                logger.debug(
                    "Unsupported request type for consensus engine",
                    agent_id=self.agent_id,
                    request_type=request_type
                )
                return None
        else:
            logger.debug(
                "Unsupported message type for consensus engine",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_consensus_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle consensus building requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract consensus parameters
            topic = message.content.get('topic', '')
            participants_data = message.content.get('participants', [])
            consensus_method = ConsensusMethod(message.content.get('method', self.default_consensus_method.value))
            context = message.content.get('context', {})
            
            if not participants_data:
                raise ValueError("Participants are required for consensus building")
            
            logger.info(
                "Processing consensus request",
                agent_id=self.agent_id,
                topic=topic,
                participants_count=len(participants_data),
                method=consensus_method
            )
            
            # Build consensus
            consensus_result = await self._build_consensus(
                topic, participants_data, consensus_method, context
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            consensus_result.consensus_time = processing_time
            
            # Create response
            response_content = {
                'consensus_result': consensus_result.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=consensus_result.consensus_confidence,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Consensus {'reached' if consensus_result.consensus_reached else 'not reached'}: {consensus_result.consensus_position}"
            )
            
        except Exception as e:
            logger.error(
                "Consensus building failed",
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
                answer=f"Consensus building failed: {str(e)}"
            )
    
    async def _build_consensus(
        self,
        topic: str,
        participants_data: List[Dict[str, Any]],
        method: ConsensusMethod,
        context: Dict[str, Any]
    ) -> ConsensusResult:
        """Build consensus using specified method."""
        
        # Step 1: Create participant objects
        participants = await self._create_participants(participants_data, context)
        
        if len(participants) < self.min_participants:
            raise ValueError(f"Minimum {self.min_participants} participants required")
        
        # Step 2: Identify conflicts
        conflicts = await self._identify_conflicts(participants, topic)
        
        # Step 3: Resolve conflicts if auto-resolution is enabled
        resolved_conflicts = []
        if self.auto_resolve_conflicts and conflicts:
            resolved_conflicts = await self._resolve_conflicts(conflicts, participants)
        
        # Step 4: Perform consensus building iterations
        consensus_result = await self._perform_consensus_iterations(
            topic, participants, method, context
        )
        
        # Step 5: Analyze consensus quality
        consensus_result = await self._analyze_consensus_quality(consensus_result, participants)
        
        # Step 6: Generate recommendations
        recommendations, clarifications, follow_ups = await self._generate_consensus_recommendations(
            consensus_result, resolved_conflicts
        )
        
        # Update final result
        consensus_result.identified_conflicts = resolved_conflicts
        consensus_result.unresolved_conflicts = [c.conflict_description for c in conflicts if c not in resolved_conflicts]
        consensus_result.consensus_recommendations = recommendations
        consensus_result.areas_needing_clarification = clarifications
        consensus_result.follow_up_questions = follow_ups
        
        return consensus_result
    
    async def _create_participants(
        self, participants_data: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[ConsensusParticipant]:
        """Create participant objects from input data."""
        participants = []
        
        for i, participant_data in enumerate(participants_data):
            # Calculate trust score
            trust_score = await self._calculate_trust_score(participant_data, context)
            
            # Determine voting weight
            voting_weight = await self._calculate_voting_weight(participant_data, trust_score)
            
            participant = ConsensusParticipant(
                participant_id=participant_data.get('id', f"participant_{i}"),
                participant_type=participant_data.get('type', 'unknown'),
                position=participant_data.get('position', ''),
                confidence=participant_data.get('confidence', 0.5),
                reasoning=participant_data.get('reasoning', ''),
                trust_score=trust_score,
                expertise_level=participant_data.get('expertise', 0.5),
                voting_weight=voting_weight,
                supporting_evidence=participant_data.get('evidence', []),
                dissenting_views=participant_data.get('dissenting_views', [])
            )
            
            participants.append(participant)
        
        return participants
    
    async def _identify_conflicts(
        self, participants: List[ConsensusParticipant], topic: str
    ) -> List[ConflictResolution]:
        """Identify conflicts between participants."""
        conflicts = []
        
        # Compare each pair of participants
        for i, p1 in enumerate(participants):
            for j, p2 in enumerate(participants[i+1:], i+1):
                conflict = await self._detect_pairwise_conflict(p1, p2, topic)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_pairwise_conflict(
        self, p1: ConsensusParticipant, p2: ConsensusParticipant, topic: str
    ) -> Optional[ConflictResolution]:
        """Detect conflict between two participants."""
        try:
            # Use LLM to detect conflicts
            llm_response = await self.generate_response(
                "",
                template_name='conflict_detection',
                template_variables={
                    'topic': topic,
                    'participant_1': p1.dict(),
                    'participant_2': p2.dict()
                }
            )
            
            # Parse conflict detection
            import json
            conflict_data = json.loads(llm_response.content)
            
            if conflict_data.get('has_conflict', False):
                return ConflictResolution(
                    conflict_id=f"conflict_{p1.participant_id}_{p2.participant_id}",
                    conflict_type=ConflictType(conflict_data.get('conflict_type', 'factual_disagreement')),
                    conflicting_participants=[p1.participant_id, p2.participant_id],
                    conflict_description=conflict_data.get('description', ''),
                    resolution_method='pending',
                    resolution_explanation='',
                    resolved_position='',
                    resolution_confidence=0.0,
                    resolution_stability=0.0
                )
            
            return None
            
        except Exception as e:
            logger.warning(
                "Failed to detect conflict between participants",
                agent_id=self.agent_id,
                p1_id=p1.participant_id,
                p2_id=p2.participant_id,
                error=str(e)
            )
            return None
    
    async def _perform_consensus_iterations(
        self,
        topic: str,
        participants: List[ConsensusParticipant],
        method: ConsensusMethod,
        context: Dict[str, Any]
    ) -> ConsensusResult:
        """Perform iterative consensus building."""
        
        iteration = 0
        consensus_reached = False
        
        # Initial voting results
        voting_results = await self._conduct_voting(participants, method)
        
        # Check if consensus is reached
        consensus_reached, consensus_position, confidence = await self._check_consensus(
            voting_results, participants, method
        )
        
        # Calculate participation rate
        participation_rate = len([p for p in participants if p.position.strip()]) / len(participants)
        
        # Identify minority positions
        minority_positions, dissenting_opinions = await self._identify_minority_positions(
            participants, consensus_position
        )
        
        return ConsensusResult(
            consensus_topic=topic,
            participants=participants,
            consensus_method=method,
            consensus_reached=consensus_reached,
            consensus_position=consensus_position,
            consensus_confidence=confidence,
            voting_results=voting_results,
            participation_rate=participation_rate,
            identified_conflicts=[],  # Will be filled by caller
            minority_positions=minority_positions,
            dissenting_opinions=dissenting_opinions,
            consensus_iterations=iteration + 1
        )
    
    async def _conduct_voting(
        self, participants: List[ConsensusParticipant], method: ConsensusMethod
    ) -> Dict[str, Any]:
        """Conduct voting according to specified method."""
        
        if method == ConsensusMethod.MAJORITY_VOTING:
            return await self._majority_voting(participants)
        elif method == ConsensusMethod.WEIGHTED_VOTING:
            return await self._weighted_voting(participants)
        elif method == ConsensusMethod.EXPERT_CONSENSUS:
            return await self._expert_consensus(participants)
        elif method == ConsensusMethod.BAYESIAN_CONSENSUS:
            return await self._bayesian_consensus(participants)
        else:
            # Default to weighted voting
            return await self._weighted_voting(participants)
    
    async def _weighted_voting(self, participants: List[ConsensusParticipant]) -> Dict[str, Any]:
        """Perform weighted voting based on trust and expertise."""
        position_weights = {}
        total_weight = 0.0
        
        for participant in participants:
            if participant.position.strip():
                weight = participant.voting_weight * participant.confidence
                position_weights[participant.position] = position_weights.get(participant.position, 0.0) + weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for position in position_weights:
                position_weights[position] /= total_weight
        
        return {
            'method': 'weighted_voting',
            'position_weights': position_weights,
            'total_participants': len(participants),
            'voting_participants': len([p for p in participants if p.position.strip()])
        }
    
    async def _check_consensus(
        self, voting_results: Dict[str, Any], participants: List[ConsensusParticipant], method: ConsensusMethod
    ) -> Tuple[bool, str, float]:
        """Check if consensus has been reached."""
        
        position_weights = voting_results.get('position_weights', {})
        
        if not position_weights:
            return False, "No consensus possible - no positions", 0.0
        
        # Find the position with highest weight
        top_position = max(position_weights, key=position_weights.get)
        top_weight = position_weights[top_position]
        
        # Check if it meets consensus threshold
        consensus_reached = top_weight >= self.consensus_threshold
        
        # Calculate consensus confidence
        if consensus_reached:
            # Weight by participant confidence
            supporting_participants = [p for p in participants if p.position == top_position]
            if supporting_participants:
                avg_confidence = sum(p.confidence for p in supporting_participants) / len(supporting_participants)
                consensus_confidence = (top_weight + avg_confidence) / 2
            else:
                consensus_confidence = top_weight
        else:
            consensus_confidence = top_weight
        
        return consensus_reached, top_position, consensus_confidence
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for consensus building."""
        templates = {}
        
        # Conflict detection template
        templates['conflict_detection'] = PromptTemplate(
            name='conflict_detection',
            template="""
            Analyze whether these two participants have conflicting positions on the topic.
            
            Topic: {topic}
            Participant 1: {participant_1}
            Participant 2: {participant_2}
            
            Look for:
            1. Direct contradictions in their positions
            2. Different interpretations of the same facts
            3. Value conflicts or different priorities
            4. Scope disagreements (what the topic includes/excludes)
            5. Temporal conflicts (different time perspectives)
            6. Authority conflicts (different sources of truth)
            7. Methodological conflicts (different approaches)
            
            If there is a conflict:
            - Categorize the type of conflict
            - Describe the nature of the disagreement
            - Assess the severity of the conflict
            - Suggest potential resolution approaches
            
            Respond with JSON:
            {{
                "has_conflict": true/false,
                "conflict_type": "factual_disagreement|value_conflict|interpretation_difference|scope_disagreement|temporal_conflict|authority_conflict|methodology_conflict",
                "description": "detailed description of the conflict",
                "severity": 0.8,
                "resolution_approaches": ["approach1", "approach2"],
                "common_ground": ["area of agreement 1"],
                "key_differences": ["difference 1", "difference 2"]
            }}
            """.strip()
        )
        
        return templates
    
    # Helper method stubs
    async def _calculate_trust_score(self, participant_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate trust score for a participant."""
        pass
    
    async def _calculate_voting_weight(self, participant_data: Dict[str, Any], trust_score: float) -> float:
        """Calculate voting weight for a participant."""
        pass
    
    async def _resolve_conflicts(
        self, conflicts: List[ConflictResolution], participants: List[ConsensusParticipant]
    ) -> List[ConflictResolution]:
        """Resolve identified conflicts."""
        pass
    
    async def _analyze_consensus_quality(
        self, result: ConsensusResult, participants: List[ConsensusParticipant]
    ) -> ConsensusResult:
        """Analyze the quality of the reached consensus."""
        pass
    
    async def _generate_consensus_recommendations(
        self, result: ConsensusResult, conflicts: List[ConflictResolution]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate recommendations based on consensus result."""
        pass
    
    async def _identify_minority_positions(
        self, participants: List[ConsensusParticipant], consensus_position: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Identify and document minority positions."""
        pass
    
    async def _majority_voting(self, participants: List[ConsensusParticipant]) -> Dict[str, Any]:
        """Perform simple majority voting."""
        pass
    
    async def _expert_consensus(self, participants: List[ConsensusParticipant]) -> Dict[str, Any]:
        """Perform expert-weighted consensus."""
        pass
    
    async def _bayesian_consensus(self, participants: List[ConsensusParticipant]) -> Dict[str, Any]:
        """Perform Bayesian consensus building."""
        pass
    
    # Additional handler method stubs
    async def _handle_conflict_resolution(self, message: AgentMessage) -> ResponseMessage:
        """Handle conflict resolution requests."""
        pass
    
    async def _handle_trust_assessment(self, message: AgentMessage) -> ResponseMessage:
        """Handle trust assessment requests."""
        pass
    
    async def _handle_consensus_validation(self, message: AgentMessage) -> ResponseMessage:
        """Handle consensus validation requests."""
        pass
    
    async def _handle_weighted_voting(self, message: AgentMessage) -> ResponseMessage:
        """Handle weighted voting requests."""
        pass
