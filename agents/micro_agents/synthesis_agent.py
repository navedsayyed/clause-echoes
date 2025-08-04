"""
Synthesis Agent - Specialized agent for answer generation and explanation synthesis.
Handles response generation, explanation building, provenance tracking, and multi-source integration.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, ResponseMessage
from reasoning.synthesis.answer_generator import AnswerGenerator
from reasoning.synthesis.explanation_builder import ExplanationBuilder
from reasoning.synthesis.provenance_tracker import ProvenanceTracker
from core.exceptions import ReasoningError

logger = structlog.get_logger(__name__)


class SynthesisStrategy(str, Enum):
    """Strategies for synthesizing information."""
    
    CONCATENATIVE = "concatenative"  # Simple concatenation
    HIERARCHICAL = "hierarchical"   # Structured hierarchy
    NARRATIVE = "narrative"         # Story-like flow
    ARGUMENTATIVE = "argumentative" # Argument-based structure
    COMPARATIVE = "comparative"     # Comparison-based
    CAUSAL = "causal"              # Cause-effect relationships


class ConfidenceLevel(str, Enum):
    """Confidence levels for synthesized responses."""
    
    VERY_LOW = "very_low"    # 0.0 - 0.2
    LOW = "low"              # 0.2 - 0.4
    MEDIUM = "medium"        # 0.4 - 0.6
    HIGH = "high"           # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


class SourceContribution(BaseModel):
    """Information about how a source contributed to the synthesis."""
    
    source_id: str
    source_type: str
    content_used: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    contradictions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    weight_in_synthesis: float = Field(..., ge=0.0, le=1.0)


class SynthesisResult(BaseModel):
    """Complete synthesis result with metadata."""
    
    # Query information
    original_query: str
    query_intent: str
    
    # Synthesized response
    answer: str
    explanation: str
    summary: str
    
    # Source information
    sources_used: List[SourceContribution]
    primary_sources: List[str] = Field(default_factory=list)
    supporting_sources: List[str] = Field(default_factory=list)
    
    # Quality metrics
    confidence_level: ConfidenceLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Synthesis metadata
    synthesis_strategy: SynthesisStrategy
    processing_steps: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Uncertainty and limitations
    uncertainties: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    
    # Alternative perspectives
    alternative_answers: List[str] = Field(default_factory=list)
    conflicting_information: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Follow-up suggestions
    clarification_questions: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    
    # Processing metadata
    synthesis_time: float = 0.0
    synthesized_at: datetime = Field(default_factory=datetime.utcnow)


class SynthesisAgent(LLMAgent):
    """Agent specialized in synthesizing information from multiple sources into coherent responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="SynthesisAgent",
            config=config or {}
        )
        
        # Synthesis components
        self.answer_generator: Optional[AnswerGenerator] = None
        self.explanation_builder: Optional[ExplanationBuilder] = None
        self.provenance_tracker: Optional[ProvenanceTracker] = None
        
        # Configuration
        self.default_strategy = SynthesisStrategy(self.config.get('default_strategy', 'hierarchical'))
        self.max_sources = self.config.get('max_sources', 10)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.enable_alternative_answers = self.config.get('enable_alternatives', True)
        
        # Quality thresholds
        self.min_completeness = self.config.get('min_completeness', 0.7)
        self.min_coherence = self.config.get('min_coherence', 0.8)
        
        logger.info("Synthesis Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize synthesis capabilities."""
        return [
            AgentCapability(
                name="multi_source_synthesis",
                description="Synthesize information from multiple sources into coherent responses",
                input_types=["source_materials", "query_results"],
                output_types=["synthesized_response", "comprehensive_answer"],
                confidence_level=0.9,
                estimated_processing_time=8.0
            ),
            AgentCapability(
                name="explanation_generation",
                description="Generate detailed explanations for complex topics",
                input_types=["facts", "concepts", "relationships"],
                output_types=["explanations", "educational_content"],
                confidence_level=0.85,
                estimated_processing_time=6.0
            ),
            AgentCapability(
                name="provenance_tracking",
                description="Track and attribute information sources in responses",
                input_types=["source_documents", "citations"],
                output_types=["attributed_responses", "source_tracking"],
                confidence_level=0.95,
                estimated_processing_time=3.0
            ),
            AgentCapability(
                name="conflict_resolution",
                description="Resolve conflicts between contradictory sources",
                input_types=["conflicting_sources", "contradictions"],
                output_types=["resolved_conflicts", "consensus_views"],
                confidence_level=0.75,
                estimated_processing_time=10.0
            ),
            AgentCapability(
                name="uncertainty_communication",
                description="Clearly communicate uncertainties and limitations",
                input_types=["uncertain_information", "confidence_scores"],
                output_types=["uncertainty_aware_responses", "qualified_answers"],
                confidence_level=0.8,
                estimated_processing_time=4.0
            )
        ]
    
    async def _initialize(self):
        """Initialize synthesis components."""
        await super()._initialize()
        
        try:
            # Initialize synthesis components
            self.answer_generator = AnswerGenerator(
                llm_provider=self.llm_provider,
                config=self.config.get('answer_generation', {})
            )
            await self.answer_generator.initialize()
            
            self.explanation_builder = ExplanationBuilder(
                llm_provider=self.llm_provider,
                config=self.config.get('explanation_building', {})
            )
            await self.explanation_builder.initialize()
            
            self.provenance_tracker = ProvenanceTracker(
                config=self.config.get('provenance_tracking', {})
            )
            await self.provenance_tracker.initialize()
            
            logger.info("Synthesis agent components initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize synthesis components",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup synthesis resources."""
        try:
            if self.answer_generator:
                await self.answer_generator.cleanup()
            
            if self.explanation_builder:
                await self.explanation_builder.cleanup()
            
            if self.provenance_tracker:
                await self.provenance_tracker.cleanup()
            
            await super()._cleanup()
            
        except Exception as e:
            logger.error(
                "Error during synthesis cleanup",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for synthesis."""
        if message.type == MessageType.REQUEST:
            request_type = message.content.get('request_type', '')
            
            if request_type == 'synthesize_response':
                return await self._handle_synthesis_request(message)
            elif request_type == 'generate_explanation':
                return await self._handle_explanation_request(message)
            elif request_type == 'resolve_conflicts':
                return await self._handle_conflict_resolution(message)
            elif request_type == 'track_provenance':
                return await self._handle_provenance_request(message)
            else:
                logger.debug(
                    "Unsupported request type for synthesis agent",
                    agent_id=self.agent_id,
                    request_type=request_type
                )
                return None
        else:
            logger.debug(
                "Unsupported message type for synthesis agent",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_synthesis_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle synthesis requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract synthesis parameters
            query = message.content.get('query', '')
            sources = message.content.get('sources', [])
            strategy = SynthesisStrategy(message.content.get('strategy', self.default_strategy.value))
            context = message.content.get('context', {})
            
            if not sources:
                raise ValueError("Sources are required for synthesis")
            
            logger.info(
                "Processing synthesis request",
                agent_id=self.agent_id,
                query_length=len(query),
                sources_count=len(sources),
                strategy=strategy
            )
            
            # Perform synthesis
            synthesis_result = await self._synthesize_response(query, sources, strategy, context)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            synthesis_result.synthesis_time = processing_time
            
            # Create response
            response_content = {
                'synthesis_result': synthesis_result.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=synthesis_result.confidence_score,
                sources=[{
                    'type': 'synthesized_response',
                    'sources_count': len(synthesis_result.sources_used),
                    'confidence': synthesis_result.confidence_score
                }],
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=synthesis_result.answer
            )
            
        except Exception as e:
            logger.error(
                "Synthesis request failed",
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
                answer=f"Synthesis failed: {str(e)}"
            )
    
    async def _synthesize_response(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        strategy: SynthesisStrategy,
        context: Dict[str, Any]
    ) -> SynthesisResult:
        """Main synthesis logic."""
        
        # Step 1: Analyze and prepare sources
        source_contributions = await self._analyze_sources(query, sources, context)
        
        # Step 2: Detect conflicts and resolve them
        conflicts, resolutions = await self._detect_and_resolve_conflicts(source_contributions)
        
        # Step 3: Generate synthesized answer
        answer = await self._generate_synthesized_answer(query, source_contributions, strategy, context)
        
        # Step 4: Build explanation
        explanation = await self._build_explanation(query, source_contributions, answer, strategy)
        
        # Step 5: Create summary
        summary = await self._create_summary(answer, explanation)
        
        # Step 6: Calculate quality scores
        confidence_score, completeness_score, coherence_score = await self._calculate_quality_scores(
            query, answer, source_contributions
        )
        
        # Step 7: Identify uncertainties and limitations
        uncertainties, assumptions, limitations = await self._identify_uncertainties(
            query, source_contributions, conflicts
        )
        
        # Step 8: Generate alternatives if enabled
        alternative_answers = []
        if self.enable_alternative_answers:
            alternative_answers = await self._generate_alternative_answers(
                query, source_contributions, answer
            )
        
        # Step 9: Generate follow-up suggestions
        clarification_questions, related_topics = await self._generate_follow_ups(
            query, answer, source_contributions
        )
        
        return SynthesisResult(
            original_query=query,
            query_intent=context.get('intent', 'information_request'),
            answer=answer,
            explanation=explanation,
            summary=summary,
            sources_used=source_contributions,
            primary_sources=[s.source_id for s in source_contributions if s.weight_in_synthesis > 0.5],
            supporting_sources=[s.source_id for s in source_contributions if 0.2 <= s.weight_in_synthesis <= 0.5],
            confidence_level=self._score_to_confidence_level(confidence_score),
            confidence_score=confidence_score,
            completeness_score=completeness_score,
            coherence_score=coherence_score,
            synthesis_strategy=strategy,
            uncertainties=uncertainties,
            assumptions=assumptions,
            limitations=limitations,
            alternative_answers=alternative_answers,
            conflicting_information=conflicts,
            clarification_questions=clarification_questions,
            related_topics=related_topics
        )
    
    async def _analyze_sources(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[SourceContribution]:
        """Analyze sources for relevance and contribution."""
        source_contributions = []
        
        for source in sources[:self.max_sources]:
            try:
                # Use LLM to analyze source relevance
                llm_response = await self.generate_response(
                    "",
                    template_name='source_analysis',
                    template_variables={
                        'query': query,
                        'source': source,
                        'context': context
                    }
                )
                
                # Parse analysis
                import json
                analysis = json.loads(llm_response.content)
                
                contribution = SourceContribution(
                    source_id=source.get('id', f"source_{len(source_contributions)}"),
                    source_type=source.get('type', 'unknown'),
                    content_used=analysis.get('relevant_content', source.get('content', '')),
                    relevance_score=analysis.get('relevance_score', 0.5),
                    confidence_score=analysis.get('confidence_score', 0.5),
                    contradictions=analysis.get('contradictions', []),
                    supporting_evidence=analysis.get('supporting_evidence', []),
                    weight_in_synthesis=analysis.get('synthesis_weight', 0.5)
                )
                
                source_contributions.append(contribution)
                
            except Exception as e:
                logger.warning(
                    "Failed to analyze source",
                    agent_id=self.agent_id,
                    source_id=source.get('id'),
                    error=str(e)
                )
        
        return source_contributions
    
    async def _generate_synthesized_answer(
        self,
        query: str,
        sources: List[SourceContribution],
        strategy: SynthesisStrategy,
        context: Dict[str, Any]
    ) -> str:
        """Generate synthesized answer using specified strategy."""
        try:
            # Use LLM with strategy-specific template
            llm_response = await self.generate_response(
                query,
                template_name=f'synthesis_{strategy.value}',
                template_variables={
                    'query': query,
                    'sources': [s.dict() for s in sources],
                    'context': context,
                    'strategy': strategy.value
                }
            )
            
            return llm_response.content.strip()
            
        except Exception as e:
            logger.error(
                "Failed to generate synthesized answer",
                agent_id=self.agent_id,
                strategy=strategy,
                error=str(e)
            )
            # Fallback to simple concatenation
            return await self._fallback_synthesis(query, sources)
    
    async def _detect_and_resolve_conflicts(
        self,
        sources: List[SourceContribution]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect conflicts between sources and attempt resolution."""
        conflicts = []
        resolutions = []
        
        # Simple conflict detection based on contradictions
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                if source1.contradictions or source2.contradictions:
                    # Check for contradictions between sources
                    conflict_found = await self._check_source_conflict(source1, source2)
                    
                    if conflict_found:
                        conflict = {
                            'source_1': source1.source_id,
                            'source_2': source2.source_id,
                            'conflict_type': 'factual_contradiction',
                            'description': conflict_found.get('description', ''),
                            'severity': conflict_found.get('severity', 0.5)
                        }
                        conflicts.append(conflict)
                        
                        # Attempt resolution
                        resolution = await self._resolve_conflict(source1, source2, conflict_found)
                        if resolution:
                            resolutions.append(resolution)
        
        return conflicts, resolutions
    
    async def _calculate_quality_scores(
        self,
        query: str,
        answer: str,
        sources: List[SourceContribution]
    ) -> Tuple[float, float, float]:
        """Calculate confidence, completeness, and coherence scores."""
        
        # Confidence: Based on source reliability and consensus
        confidence_scores = [s.confidence_score * s.weight_in_synthesis for s in sources]
        confidence_score = sum(confidence_scores) / max(len(confidence_scores), 1)
        
        # Completeness: How well the answer addresses the query
        try:
            completeness_response = await self.generate_response(
                "",
                template_name='completeness_evaluation',
                template_variables={
                    'query': query,
                    'answer': answer,
                    'sources': [s.dict() for s in sources]
                }
            )
            
            import json
            completeness_data = json.loads(completeness_response.content)
            completeness_score = completeness_data.get('completeness_score', 0.5)
            
        except Exception:
            # Fallback: Simple heuristic
            completeness_score = min(1.0, len(answer.split()) / 100)  # Rough estimate
        
        # Coherence: How well-structured and logical the answer is
        try:
            coherence_response = await self.generate_response(
                "",
                template_name='coherence_evaluation',
                template_variables={
                    'answer': answer,
                    'query': query
                }
            )
            
            import json
            coherence_data = json.loads(coherence_response.content)
            coherence_score = coherence_data.get('coherence_score', 0.5)
            
        except Exception:
            # Fallback: Simple heuristic based on answer structure
            sentences = answer.split('.')
            coherence_score = min(1.0, len(sentences) / 10)  # Rough estimate
        
        return confidence_score, completeness_score, coherence_score
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numerical score to confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for synthesis."""
        templates = {}
        
        # Source analysis template
        templates['source_analysis'] = PromptTemplate(
            name='source_analysis',
            template="""
            Analyze how relevant this source is to answering the query.
            
            Query: {query}
            Source: {source}
            Context: {context}
            
            Evaluate:
            1. How relevant is this source to the query?
            2. What content from this source is most useful?
            3. How confident are you in this source's information?
            4. Are there any contradictions with other sources?
            5. What weight should this source have in the final synthesis?
            
            Respond with JSON:
            {{
                "relevant_content": "key content from source",
                "relevance_score": 0.9,
                "confidence_score": 0.8,
                "contradictions": ["contradiction1", "contradiction2"],
                "supporting_evidence": ["evidence1", "evidence2"],
                "synthesis_weight": 0.7,
                "reasoning": "explanation of analysis"
            }}
            """.strip()
        )
        
        # Hierarchical synthesis template
        templates['synthesis_hierarchical'] = PromptTemplate(
            name='synthesis_hierarchical',
            template="""
            Synthesize information from multiple sources using a hierarchical structure.
            
            Query: {query}
            Sources: {sources}
            Context: {context}
            
            Create a comprehensive answer that:
            1. Directly answers the main query
            2. Organizes information hierarchically (main points → sub-points → details)
            3. Integrates information from all relevant sources
            4. Handles conflicts by presenting different perspectives
            5. Maintains logical flow and coherence
            
            Structure your response with clear sections and subsections.
            Cite sources naturally within the text.
            """.strip()
        )
        
        # Completeness evaluation template
        templates['completeness_evaluation'] = PromptTemplate(
            name='completeness_evaluation',
            template="""
            Evaluate how completely this answer addresses the query.
            
            Query: {query}
            Answer: {answer}
            Available Sources: {sources}
            
            Consider:
            1. Does the answer address all aspects of the query?
            2. Are there important missing elements?
            3. Is the depth of coverage appropriate?
            4. Were all relevant sources utilized?
            
            Respond with JSON:
            {{
                "completeness_score": 0.85,
                "addressed_aspects": ["aspect1", "aspect2"],
                "missing_aspects": ["missing1", "missing2"],
                "depth_assessment": "adequate|shallow|deep",
                "unused_sources": ["source1", "source2"],
                "improvement_suggestions": ["suggestion1", "suggestion2"]
            }}
            """.strip()
        )
        
        return templates
    
    # Additional helper methods
    async def _build_explanation(
        self, query: str, sources: List[SourceContribution], answer: str, strategy: SynthesisStrategy
    ) -> str:
        """Build detailed explanation of the answer."""
        pass
    
    async def _create_summary(self, answer: str, explanation: str) -> str:
        """Create concise summary of the answer."""
        pass
    
    async def _identify_uncertainties(
        self, query: str, sources: List[SourceContribution], conflicts: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Identify uncertainties, assumptions, and limitations."""
        pass
    
    async def _generate_alternative_answers(
        self, query: str, sources: List[SourceContribution], primary_answer: str
    ) -> List[str]:
        """Generate alternative interpretations or answers."""
        pass
    
    async def _generate_follow_ups(
        self, query: str, answer: str, sources: List[SourceContribution]
    ) -> Tuple[List[str], List[str]]:
        """Generate clarification questions and related topics."""
        pass
    
    async def _fallback_synthesis(self, query: str, sources: List[SourceContribution]) -> str:
        """Simple fallback synthesis when advanced methods fail."""
        pass
    
    async def _check_source_conflict(
        self, source1: SourceContribution, source2: SourceContribution
    ) -> Optional[Dict[str, Any]]:
        """Check if two sources conflict with each other."""
        pass
    
    async def _resolve_conflict(
        self, source1: SourceContribution, source2: SourceContribution, conflict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to resolve a conflict between sources."""
        pass
    
    # Handler method stubs
    async def _handle_explanation_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle explanation generation requests."""
        pass
    
    async def _handle_conflict_resolution(self, message: AgentMessage) -> ResponseMessage:
        """Handle conflict resolution requests."""
        pass
    
    async def _handle_provenance_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle provenance tracking requests."""
        pass
