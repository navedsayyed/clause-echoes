"""
Curriculum Agent - Manages progressive query refinement and educational dialogue.
Generates curriculum-driven follow-up questions and guides users through complex topics.
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


class LearningObjective(str, Enum):
    """Learning objectives for curriculum development."""
    
    CLARIFICATION = "clarification"           # Clarify ambiguous queries
    EXPLORATION = "exploration"               # Explore related topics
    DEEPENING = "deepening"                  # Deepen understanding
    COMPARISON = "comparison"                # Compare alternatives
    APPLICATION = "application"              # Apply knowledge
    SYNTHESIS = "synthesis"                  # Synthesize information
    EVALUATION = "evaluation"                # Evaluate options


class QuestionType(str, Enum):
    """Types of questions in the curriculum."""
    
    CLARIFYING = "clarifying"                # Clarify user intent
    PROBING = "probing"                     # Probe deeper understanding
    HYPOTHETICAL = "hypothetical"            # Explore hypothetical scenarios
    COMPARATIVE = "comparative"              # Compare options
    CONSEQUENTIAL = "consequential"          # Explore consequences
    DEFINITIONAL = "definitional"            # Define terms and concepts
    PROCEDURAL = "procedural"               # Understand procedures
    EVALUATIVE = "evaluative"               # Make judgments


class DifficultyLevel(str, Enum):
    """Difficulty levels for curriculum progression."""
    
    BASIC = "basic"                         # Basic understanding
    INTERMEDIATE = "intermediate"            # Intermediate complexity
    ADVANCED = "advanced"                   # Advanced concepts
    EXPERT = "expert"                       # Expert-level analysis


class CurriculumQuestion(BaseModel):
    """A question within a learning curriculum."""
    
    question_id: str
    question_type: QuestionType
    difficulty_level: DifficultyLevel
    
    # Content
    question_text: str
    context: str = ""
    explanation: str = ""
    
    # Learning design
    learning_objective: LearningObjective
    prerequisite_questions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    
    # Answer handling
    expected_answer_types: List[str] = Field(default_factory=list)
    evaluation_criteria: List[str] = Field(default_factory=list)
    
    # Adaptivity
    adaptation_triggers: Dict[str, str] = Field(default_factory=dict)
    alternative_phrasings: List[str] = Field(default_factory=list)
    
    # Metadata
    estimated_time_minutes: int = 2
    importance_score: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearningPath(BaseModel):
    """A structured learning path through a topic."""
    
    path_id: str
    topic: str
    description: str
    
    # Path structure
    questions: List[CurriculumQuestion]
    question_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Learning progression
    difficulty_progression: List[DifficultyLevel]
    objective_sequence: List[LearningObjective]
    
    # Adaptivity
    branching_points: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    personalization_factors: List[str] = Field(default_factory=list)
    
    # Quality metrics
    estimated_duration_minutes: int = 15
    completion_rate: float = 1.0
    effectiveness_score: float = 0.8
    
    # Metadata
    target_audience: str = "general"
    prerequisites: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearnerProfile(BaseModel):
    """Profile of a learner's progress and preferences."""
    
    learner_id: str
    
    # Learning characteristics
    expertise_level: DifficultyLevel = DifficultyLevel.BASIC
    learning_style: str = "balanced"  # visual, auditory, kinesthetic, reading, balanced
    pace_preference: str = "normal"   # slow, normal, fast
    
    # Progress tracking
    completed_questions: Set[str] = Field(default_factory=set)
    current_path: Optional[str] = None
    current_question: Optional[str] = None
    
    # Performance metrics
    correct_answers: int = 0
    total_questions: int = 0
    average_response_time: float = 0.0
    
    # Preferences and interests
    preferred_question_types: List[QuestionType] = Field(default_factory=list)
    interest_areas: List[str] = Field(default_factory=list)
    
    # Adaptation data
    confusion_indicators: List[str] = Field(default_factory=list)
    strength_areas: List[str] = Field(default_factory=list)
    
    # Temporal data
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    total_learning_time: float = 0.0


class CurriculumSession(BaseModel):
    """An active learning session."""
    
    session_id: str
    learner_id: str
    path_id: str
    
    # Session state
    current_question_index: int = 0
    questions_completed: int = 0
    session_start: datetime = Field(default_factory=datetime.utcnow)
    
    # Interaction history
    question_responses: List[Dict[str, Any]] = Field(default_factory=list)
    adaptation_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Progress tracking
    progress_percentage: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    
    # Quality metrics
    engagement_score: float = 0.5
    understanding_score: float = 0.5
    satisfaction_score: float = 0.5
    
    # Adaptations made
    difficulty_adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    path_modifications: List[Dict[str, Any]] = Field(default_factory=list)


class CurriculumGenerationResult(BaseModel):
    """Result of curriculum generation."""
    
    # Generated curriculum
    learning_path: LearningPath
    alternative_paths: List[LearningPath] = Field(default_factory=list)
    
    # Quality assessment
    path_quality_score: float = Field(..., ge=0.0, le=1.0)
    coverage_completeness: float = Field(..., ge=0.0, le=1.0)
    difficulty_progression_score: float = Field(..., ge=0.0, le=1.0)
    
    # Recommendations
    personalization_suggestions: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    
    # Metadata
    generation_time: float = 0.0
    generation_confidence: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CurriculumAgent(LLMAgent):
    """Agent specialized in curriculum-driven query refinement and progressive learning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="CurriculumAgent",
            config=config or {}
        )
        
        # Curriculum configuration
        self.default_path_length = self.config.get('default_path_length', 10)
        self.max_branching_factor = self.config.get('max_branching_factor', 3)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.3)
        
        # Learning design parameters
        self.enable_personalization = self.config.get('enable_personalization', True)
        self.enable_adaptive_difficulty = self.config.get('enable_adaptive_difficulty', True)
        self.enable_branching_paths = self.config.get('enable_branching_paths', True)
        
        # Data storage
        self.learning_paths: Dict[str, LearningPath] = {}
        self.learner_profiles: Dict[str, LearnerProfile] = {}
        self.active_sessions: Dict[str, CurriculumSession] = {}
        
        # Question templates and patterns
        self.question_templates: Dict[QuestionType, List[str]] = {}
        self.domain_specific_templates: Dict[str, Dict[QuestionType, List[str]]] = {}
        
        logger.info("Curriculum Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize curriculum development capabilities."""
        return [
            AgentCapability(
                name="curriculum_generation",
                description="Generate structured learning curricula for complex topics",
                input_types=["topics", "learning_objectives"],
                output_types=["learning_paths", "question_sequences"],
                confidence_level=0.85,
                estimated_processing_time=20.0
            ),
            AgentCapability(
                name="progressive_questioning",
                description="Generate progressive question sequences from basic to advanced",
                input_types=["topics", "difficulty_requirements"],
                output_types=["question_sequences", "learning_progression"],
                confidence_level=0.9,
                estimated_processing_time=15.0
            ),
            AgentCapability(
                name="adaptive_personalization",
                description="Adapt curricula based on learner profiles and performance",
                input_types=["learner_data", "performance_metrics"],
                output_types=["personalized_curricula", "adaptive_paths"],
                confidence_level=0.8,
                estimated_processing_time=12.0
            ),
            AgentCapability(
                name="clarification_sequencing",
                description="Generate sequences of clarifying questions for ambiguous queries",
                input_types=["ambiguous_queries", "context"],
                output_types=["clarification_sequences", "refinement_paths"],
                confidence_level=0.9,
                estimated_processing_time=10.0
            ),
            AgentCapability(
                name="learning_assessment",
                description="Assess learning progress and understanding",
                input_types=["responses", "interaction_history"],
                output_types=["progress_assessment", "understanding_metrics"],
                confidence_level=0.85,
                estimated_processing_time=8.0
            )
        ]
    
    async def _initialize(self):
        """Initialize curriculum agent components."""
        await super()._initialize()
        
        try:
            # Load question templates
            await self._load_question_templates()
            
            # Load domain-specific templates
            await self._load_domain_templates()
            
            # Initialize predefined learning paths
            await self._load_predefined_paths()
            
            logger.info("Curriculum agent components initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize curriculum agent components",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup curriculum agent resources."""
        # Save active sessions
        for session in self.active_sessions.values():
            try:
                await self._save_session_progress(session)
            except Exception as e:
                logger.warning("Failed to save session progress", session_id=session.session_id, error=str(e))
        
        await super()._cleanup()
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for curriculum management."""
        if message.type == MessageType.REQUEST:
            request_type = message.content.get('request_type', '')
            
            if request_type == 'generate_curriculum':
                return await self._handle_curriculum_generation(message)
            elif request_type == 'generate_clarification_sequence':
                return await self._handle_clarification_sequence(message)
            elif request_type == 'adaptive_questioning':
                return await self._handle_adaptive_questioning(message)
            elif request_type == 'assess_learning':
                return await self._handle_learning_assessment(message)
            elif request_type == 'personalize_path':
                return await self._handle_personalization(message)
            else:
                logger.debug(
                    "Unsupported request type for curriculum agent",
                    agent_id=self.agent_id,
                    request_type=request_type
                )
                return None
        else:
            logger.debug(
                "Unsupported message type for curriculum agent",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_curriculum_generation(self, message: AgentMessage) -> ResponseMessage:
        """Handle curriculum generation requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract generation parameters
            content = message.content
            topic = content.get('topic', '')
            learning_objectives = content.get('learning_objectives', [])
            target_audience = content.get('target_audience', 'general')
            difficulty_range = content.get('difficulty_range', ['basic', 'intermediate'])
            max_questions = content.get('max_questions', self.default_path_length)
            
            if not topic:
                raise ValueError("Topic is required for curriculum generation")
            
            logger.info(
                "Processing curriculum generation",
                agent_id=self.agent_id,
                topic=topic,
                objectives=len(learning_objectives),
                target_audience=target_audience
            )
            
            # Generate curriculum
            generation_result = await self._generate_curriculum(
                topic, learning_objectives, target_audience, difficulty_range, max_questions
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            generation_result.generation_time = processing_time
            
            # Store generated path
            path_id = generation_result.learning_path.path_id
            self.learning_paths[path_id] = generation_result.learning_path
            
            # Create response
            response_content = {
                'curriculum_result': generation_result.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=generation_result.generation_confidence,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Generated curriculum with {len(generation_result.learning_path.questions)} questions for topic: {topic}"
            )
            
        except Exception as e:
            logger.error(
                "Curriculum generation failed",
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
                answer=f"Curriculum generation failed: {str(e)}"
            )
    
    async def _generate_curriculum(
        self,
        topic: str,
        learning_objectives: List[str],
        target_audience: str,
        difficulty_range: List[str],
        max_questions: int
    ) -> CurriculumGenerationResult:
        """Generate a comprehensive curriculum for the topic."""
        
        # Step 1: Analyze topic and determine scope
        topic_analysis = await self._analyze_topic_scope(topic, target_audience)
        
        # Step 2: Generate question sequences for each objective
        all_questions = []
        for objective in learning_objectives:
            objective_questions = await self._generate_questions_for_objective(
                topic, objective, target_audience, difficulty_range
            )
            all_questions.extend(objective_questions)
        
        # Step 3: If no specific objectives, generate comprehensive coverage
        if not learning_objectives:
            comprehensive_questions = await self._generate_comprehensive_questions(
                topic, target_audience, difficulty_range, max_questions
            )
            all_questions.extend(comprehensive_questions)
        
        # Step 4: Limit and prioritize questions
        prioritized_questions = await self._prioritize_and_limit_questions(
            all_questions, max_questions, topic_analysis
        )
        
        # Step 5: Sequence questions logically
        sequenced_questions = await self._sequence_questions(prioritized_questions)
        
        # Step 6: Add dependencies and branching
        dependencies = await self._create_question_dependencies(sequenced_questions)
        branching_points = await self._create_branching_points(sequenced_questions)
        
        # Step 7: Create learning path
        path_id = f"curriculum_{hash(topic)}_{int(datetime.utcnow().timestamp())}"
        learning_path = LearningPath(
            path_id=path_id,
            topic=topic,
            description=f"Comprehensive curriculum for {topic}",
            questions=sequenced_questions,
            question_dependencies=dependencies,
            difficulty_progression=[DifficultyLevel(d) for d in difficulty_range],
            objective_sequence=[LearningObjective(obj) for obj in learning_objectives] if learning_objectives else [LearningObjective.EXPLORATION],
            branching_points=branching_points,
            estimated_duration_minutes=len(sequenced_questions) * 3,  # 3 minutes per question
            target_audience=target_audience
        )
        
        # Step 8: Generate alternatives
        alternative_paths = await self._generate_alternative_paths(learning_path, 2)
        
        # Step 9: Assess quality
        quality_scores = await self._assess_curriculum_quality(learning_path)
        
        return CurriculumGenerationResult(
            learning_path=learning_path,
            alternative_paths=alternative_paths,
            path_quality_score=quality_scores['overall'],
            coverage_completeness=quality_scores['coverage'],
            difficulty_progression_score=quality_scores['progression'],
            generation_confidence=0.8  # Would be calculated based on generation success
        )
    
    async def _generate_comprehensive_questions(
        self, topic: str, audience: str, difficulty_range: List[str], max_questions: int
    ) -> List[CurriculumQuestion]:
        """Generate comprehensive question coverage for a topic."""
        
        try:
            # Use LLM to generate comprehensive questions
            llm_response = await self.generate_response(
                topic,
                template_name='generate_comprehensive_curriculum',
                template_variables={
                    'topic': topic,
                    'target_audience': audience,
                    'difficulty_range': difficulty_range,
                    'max_questions': max_questions
                }
            )
            
            # Parse generated questions
            import json
            curriculum_data = json.loads(llm_response.content)
            
            questions = []
            for i, q_data in enumerate(curriculum_data.get('questions', [])):
                question = CurriculumQuestion(
                    question_id=f"q_{i}",
                    question_type=QuestionType(q_data.get('type', 'clarifying')),
                    difficulty_level=DifficultyLevel(q_data.get('difficulty', 'basic')),
                    question_text=q_data['text'],
                    context=q_data.get('context', ''),
                    explanation=q_data.get('explanation', ''),
                    learning_objective=LearningObjective(q_data.get('objective', 'clarification')),
                    expected_answer_types=q_data.get('expected_answers', []),
                    evaluation_criteria=q_data.get('evaluation_criteria', []),
                    alternative_phrasings=q_data.get('alternatives', []),
                    importance_score=q_data.get('importance', 0.5),
                    estimated_time_minutes=q_data.get('estimated_minutes', 2)
                )
                questions.append(question)
            
            return questions[:max_questions]
            
        except Exception as e:
            logger.error(
                "Failed to generate comprehensive questions",
                agent_id=self.agent_id,
                topic=topic,
                error=str(e)
            )
            return []
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for curriculum generation."""
        templates = {}
        
        # Comprehensive curriculum generation template
        templates['generate_comprehensive_curriculum'] = PromptTemplate(
            name='generate_comprehensive_curriculum',
            template="""
            Generate a comprehensive curriculum of questions for learning about this topic.
            
            Topic: {topic}
            Target Audience: {target_audience}
            Difficulty Range: {difficulty_range}
            Maximum Questions: {max_questions}
            
            Create a progressive sequence of questions that:
            1. Starts with basic clarification and understanding
            2. Progresses through intermediate concepts
            3. Explores advanced applications and implications
            4. Includes different question types (clarifying, probing, hypothetical, comparative, etc.)
            5. Builds understanding systematically
            
            For each question, consider:
            - What learning objective does it serve?
            - What difficulty level is appropriate?
            - What type of question is most effective?
            - How much time should learners spend on it?
            - How important is it to overall understanding?
            - What are alternative ways to ask the same thing?
            
            Create questions that are:
            - Clear and unambiguous
            - Appropriately challenging for the difficulty level
            - Building on previous questions
            - Engaging and relevant
            - Assessable (you can tell if they're answered well)
            
            Respond with JSON:
            {{
                "questions": [
                    {{
                        "text": "What specific aspect of [topic] are you most interested in learning about?",
                        "type": "clarifying",
                        "difficulty": "basic",
                        "context": "Opening question to understand learner interests",
                        "explanation": "This helps personalize the learning experience",
                        "objective": "clarification",
                        "expected_answers": ["specific_interest", "broad_overview"],
                        "evaluation_criteria": ["specificity", "relevance"],
                        "alternatives": ["Which part of [topic] would you like to explore first?"],
                        "importance": 0.8,
                        "estimated_minutes": 2
                    }}
                ],
                "progression_rationale": "Explanation of how questions build on each other",
                "adaptation_points": ["Where curriculum can branch based on responses"]
            }}
            """.strip()
        )
        
        return templates
    
    # Helper method stubs
    async def _analyze_topic_scope(self, topic: str, audience: str) -> Dict[str, Any]:
        """Analyze the scope and complexity of a topic."""
        pass
    
    async def _generate_questions_for_objective(
        self, topic: str, objective: str, audience: str, difficulty_range: List[str]
    ) -> List[CurriculumQuestion]:
        """Generate questions for a specific learning objective."""
        pass
    
    async def _prioritize_and_limit_questions(
        self, questions: List[CurriculumQuestion], max_questions: int, topic_analysis: Dict[str, Any]
    ) -> List[CurriculumQuestion]:
        """Prioritize questions and limit to maximum count."""
        pass
    
    async def _sequence_questions(self, questions: List[CurriculumQuestion]) -> List[CurriculumQuestion]:
        """Sequence questions in optimal learning order."""
        pass
    
    async def _create_question_dependencies(
        self, questions: List[CurriculumQuestion]
    ) -> Dict[str, List[str]]:
        """Create dependencies between questions."""
        pass
    
    async def _create_branching_points(
        self, questions: List[CurriculumQuestion]
    ) -> Dict[str, Dict[str, str]]:
        """Create branching points for adaptive paths."""
        pass
    
    async def _generate_alternative_paths(
        self, main_path: LearningPath, count: int
    ) -> List[LearningPath]:
        """Generate alternative learning paths."""
        pass
    
    async def _assess_curriculum_quality(self, path: LearningPath) -> Dict[str, float]:
        """Assess the quality of a generated curriculum."""
        pass
    
    async def _load_question_templates(self):
        """Load question templates by type."""
        self.question_templates = {
            QuestionType.CLARIFYING: [
                "What specifically do you mean by {term}?",
                "Could you clarify what you're looking for regarding {topic}?",
                "Are you asking about {interpretation1} or {interpretation2}?"
            ],
            QuestionType.PROBING: [
                "Why is {aspect} important in this context?",
                "What evidence supports {claim}?",
                "How does {concept} relate to {other_concept}?"
            ],
            QuestionType.HYPOTHETICAL: [
                "What would happen if {condition} were different?",
                "Suppose {scenario} occurred, how would that affect {outcome}?",
                "If you had to choose between {option1} and {option2}, which would you pick and why?"
            ]
        }
    
    async def _load_domain_templates(self):
        """Load domain-specific question templates."""
        # Would load domain-specific templates
        pass
    
    async def _load_predefined_paths(self):
        """Load predefined learning paths."""
        # Would load predefined curricula
        pass
    
    async def _save_session_progress(self, session: CurriculumSession):
        """Save session progress to persistent storage."""
        pass
    
    # Additional handler method stubs
    async def _handle_clarification_sequence(self, message: AgentMessage) -> ResponseMessage:
        """Handle clarification sequence generation."""
        pass
    
    async def _handle_adaptive_questioning(self, message: AgentMessage) -> ResponseMessage:
        """Handle adaptive questioning requests."""
        pass
    
    async def _handle_learning_assessment(self, message: AgentMessage) -> ResponseMessage:
        """Handle learning assessment requests."""
        pass
    
    async def _handle_personalization(self, message: AgentMessage) -> ResponseMessage:
        """Handle curriculum personalization requests."""
        pass
