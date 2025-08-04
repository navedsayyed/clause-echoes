"""
Query Parser Agent - Specialized agent for natural language query understanding.
Handles query parsing, intent detection, entity extraction, and query normalization.
"""
import json
import re
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, QueryMessage, ResponseMessage
from llm.prompts import PromptTemplate
from core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


class EntityExtraction(BaseModel):
    """Extracted entity information."""
    
    entity_type: str
    entity_value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    context: Optional[str] = None


class IntentClassification(BaseModel):
    """Intent classification result."""
    
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sub_intent: Optional[str] = None
    intent_context: Dict[str, Any] = Field(default_factory=dict)


class QueryStructure(BaseModel):
    """Structured representation of a parsed query."""
    
    # Original query
    original_text: str
    normalized_text: str
    
    # Intent classification
    primary_intent: IntentClassification
    secondary_intents: List[IntentClassification] = Field(default_factory=list)
    
    # Entity extraction
    entities: List[EntityExtraction] = Field(default_factory=list)
    
    # Query characteristics
    query_type: str  # factual, procedural, comparison, hypothetical, etc.
    complexity_level: str  # simple, medium, complex
    domain: str  # medical, legal, technical, etc.
    
    # Semantic analysis
    keywords: List[str] = Field(default_factory=list)
    key_phrases: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    
    # Ambiguity detection
    ambiguous_terms: List[str] = Field(default_factory=list)
    clarification_needed: bool = False
    suggested_clarifications: List[str] = Field(default_factory=list)
    
    # Context requirements
    requires_context: bool = False
    context_types: List[str] = Field(default_factory=list)
    
    # Processing metadata
    parsing_confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = 0.0
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


class QueryParserAgent(LLMAgent):
    """Agent specialized in parsing and understanding natural language queries."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="QueryParserAgent",
            config=config or {}
        )
        
        # Parser configuration
        self.supported_languages = self.config.get('languages', ['en'])
        self.intent_threshold = self.config.get('intent_threshold', 0.7)
        self.entity_threshold = self.config.get('entity_threshold', 0.6)
        
        # Pre-defined intents and entities
        self.known_intents: Set[str] = set()
        self.known_entities: Set[str] = set()
        
        # Pattern matching rules
        self.intent_patterns: Dict[str, List[str]] = {}
        self.entity_patterns: Dict[str, str] = {}
        
        logger.info("Query Parser Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize parser capabilities."""
        return [
            AgentCapability(
                name="query_parsing",
                description="Parse natural language queries into structured format",
                input_types=["text", "query"],
                output_types=["structured_query", "parsed_query"],
                confidence_level=0.9,
                estimated_processing_time=2.0
            ),
            AgentCapability(
                name="intent_detection",
                description="Detect user intent from natural language queries",
                input_types=["text", "query"],
                output_types=["intent", "classification"],
                confidence_level=0.85,
                estimated_processing_time=1.5
            ),
            AgentCapability(
                name="entity_extraction",
                description="Extract named entities and key information",
                input_types=["text", "query"],
                output_types=["entities", "extracted_info"],
                confidence_level=0.8,
                estimated_processing_time=1.0
            ),
            AgentCapability(
                name="query_normalization",
                description="Normalize and clean query text",
                input_types=["text", "query"],
                output_types=["normalized_text"],
                confidence_level=0.95,
                estimated_processing_time=0.5
            ),
            AgentCapability(
                name="ambiguity_detection",
                description="Detect ambiguous terms and suggest clarifications",
                input_types=["text", "query"],
                output_types=["ambiguity_analysis", "clarifications"],
                confidence_level=0.75,
                estimated_processing_time=1.5
            )
        ]
    
    async def _initialize(self):
        """Initialize parser-specific components."""
        await super()._initialize()
        
        # Load intent patterns
        self.intent_patterns = await self._load_intent_patterns()
        
        # Load entity patterns
        self.entity_patterns = await self._load_entity_patterns()
        
        # Load known intents and entities
        await self._load_knowledge_base()
    
    async def _cleanup(self):
        """Cleanup parser resources."""
        await super()._cleanup()
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for query parsing."""
        if message.type == MessageType.QUERY:
            return await self._handle_query_parsing(message)
        elif message.type == MessageType.REQUEST:
            return await self._handle_specific_request(message)
        else:
            logger.debug(
                "Unsupported message type for query parser",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_query_parsing(self, message: AgentMessage) -> ResponseMessage:
        """Handle query parsing requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract query text
            if isinstance(message, QueryMessage):
                query_text = message.query_text
                context = message.query_context
            else:
                query_text = message.content.get('query', message.content.get('text', ''))
                context = message.content.get('context', {})
            
            if not query_text:
                raise ValidationError("Query text is required for parsing")
            
            # Parse the query
            parsed_query = await self._parse_query(query_text, context)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            parsed_query.processing_time = processing_time
            
            # Create response
            response_content = {
                'parsed_query': parsed_query.dict(),
                'success': True,
                'processing_time': processing_time
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=parsed_query.parsing_confidence,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Query parsed successfully with {parsed_query.parsing_confidence:.2f} confidence"
            )
            
        except Exception as e:
            logger.error(
                "Query parsing failed",
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
                answer=f"Query parsing failed: {str(e)}"
            )
    
    async def _parse_query(self, query_text: str, context: Dict[str, Any]) -> QueryStructure:
        """Main query parsing logic."""
        logger.info(
            "Parsing query",
            agent_id=self.agent_id,
            query_length=len(query_text)
        )
        
        # Step 1: Normalize query text
        normalized_text = await self._normalize_query(query_text)
        
        # Step 2: Detect intent
        primary_intent, secondary_intents = await self._detect_intent(normalized_text, context)
        
        # Step 3: Extract entities
        entities = await self._extract_entities(normalized_text, context)
        
        # Step 4: Classify query characteristics
        query_type, complexity, domain = await self._classify_query(normalized_text, primary_intent)
        
        # Step 5: Extract keywords and phrases
        keywords, key_phrases, topics = await self._extract_semantic_features(normalized_text)
        
        # Step 6: Detect ambiguity
        ambiguous_terms, clarifications = await self._detect_ambiguity(normalized_text, entities)
        
        # Step 7: Analyze context requirements
        requires_context, context_types = await self._analyze_context_requirements(
            normalized_text, primary_intent, entities
        )
        
        # Step 8: Calculate overall confidence
        parsing_confidence = await self._calculate_parsing_confidence(
            primary_intent, entities, ambiguous_terms
        )
        
        # Create structured query
        parsed_query = QueryStructure(
            original_text=query_text,
            normalized_text=normalized_text,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            entities=entities,
            query_type=query_type,
            complexity_level=complexity,
            domain=domain,
            keywords=keywords,
            key_phrases=key_phrases,
            topics=topics,
            ambiguous_terms=ambiguous_terms,
            clarification_needed=len(ambiguous_terms) > 0,
            suggested_clarifications=clarifications,
            requires_context=requires_context,
            context_types=context_types,
            parsing_confidence=parsing_confidence
        )
        
        logger.info(
            "Query parsed successfully",
            agent_id=self.agent_id,
            intent=primary_intent.intent,
            entities_found=len(entities),
            confidence=parsing_confidence
        )
        
        return parsed_query
    
    async def _normalize_query(self, query_text: str) -> str:
        """Normalize query text."""
        # Basic normalization
        normalized = query_text.strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Fix common punctuation issues
        normalized = re.sub(r'\s+([.!?])', r'\1', normalized)
        normalized = re.sub(r'([.!?])\s*$', r'\1', normalized)
        
        # Use LLM for advanced normalization if needed
        if len(normalized) > 100 or any(char in normalized for char in ['...', '???', '!!!']):
            try:
                llm_response = await self.generate_response(
                    normalized,
                    template_name='query_normalization',
                    template_variables={'query': normalized}
                )
                
                # Validate LLM normalized version
                llm_normalized = llm_response.content.strip()
                if llm_normalized and len(llm_normalized) < len(normalized) * 1.5:
                    normalized = llm_normalized
                    
            except Exception as e:
                logger.warning(
                    "LLM normalization failed, using basic normalization",
                    agent_id=self.agent_id,
                    error=str(e)
                )
        
        return normalized
    
    async def _detect_intent(
        self,
        query_text: str,
        context: Dict[str, Any]
    ) -> tuple[IntentClassification, List[IntentClassification]]:
        """Detect user intent from query text."""
        # First try pattern matching for common intents
        pattern_intent = await self._pattern_match_intent(query_text)
        
        # Use LLM for intent detection
        try:
            llm_response = await self.generate_response(
                query_text,
                template_name='intent_detection',
                template_variables={
                    'query': query_text,
                    'context': context,
                    'known_intents': list(self.known_intents)
                }
            )
            
            # Parse LLM response
            intent_data = json.loads(llm_response.content)
            
            # Primary intent
            primary_intent = IntentClassification(
                intent=intent_data.get('primary_intent', 'information_request'),
                confidence=intent_data.get('confidence', 0.5),
                sub_intent=intent_data.get('sub_intent'),
                intent_context=intent_data.get('context', {})
            )
            
            # Secondary intents
            secondary_intents = []
            for secondary in intent_data.get('secondary_intents', []):
                secondary_intents.append(IntentClassification(
                    intent=secondary['intent'],
                    confidence=secondary['confidence'],
                    sub_intent=secondary.get('sub_intent'),
                    intent_context=secondary.get('context', {})
                ))
            
            # Use pattern intent if LLM confidence is low
            if primary_intent.confidence < self.intent_threshold and pattern_intent:
                primary_intent.intent = pattern_intent
                primary_intent.confidence = 0.8
            
            return primary_intent, secondary_intents
            
        except Exception as e:
            logger.warning(
                "LLM intent detection failed, using fallback",
                agent_id=self.agent_id,
                error=str(e)
            )
            
            # Fallback to pattern matching or default
            fallback_intent = IntentClassification(
                intent=pattern_intent or 'information_request',
                confidence=0.6 if pattern_intent else 0.3
            )
            
            return fallback_intent, []
    
    async def _extract_entities(
        self,
        query_text: str,
        context: Dict[str, Any]
    ) -> List[EntityExtraction]:
        """Extract entities from query text."""
        entities = []
        
        # Pattern-based entity extraction
        pattern_entities = await self._pattern_extract_entities(query_text)
        entities.extend(pattern_entities)
        
        # LLM-based entity extraction
        try:
            llm_response = await self.generate_response(
                query_text,
                template_name='entity_extraction',
                template_variables={
                    'query': query_text,
                    'context': context,
                    'known_entities': list(self.known_entities)
                }
            )
            
            # Parse LLM response
            entity_data = json.loads(llm_response.content)
            
            for entity_info in entity_data.get('entities', []):
                if entity_info.get('confidence', 0) >= self.entity_threshold:
                    entities.append(EntityExtraction(
                        entity_type=entity_info['type'],
                        entity_value=entity_info['value'],
                        confidence=entity_info['confidence'],
                        start_position=entity_info.get('start'),
                        end_position=entity_info.get('end'),
                        context=entity_info.get('context')
                    ))
            
        except Exception as e:
            logger.warning(
                "LLM entity extraction failed",
                agent_id=self.agent_id,
                error=str(e)
            )
        
        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity.entity_type, entity.entity_value.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def _classify_query(
        self,
        query_text: str,
        intent: IntentClassification
    ) -> tuple[str, str, str]:
        """Classify query characteristics."""
        # Determine query type based on patterns and intent
        query_type = await self._determine_query_type(query_text, intent)
        
        # Assess complexity
        complexity = await self._assess_complexity(query_text)
        
        # Identify domain
        domain = await self._identify_domain(query_text, intent)
        
        return query_type, complexity, domain
    
    async def _extract_semantic_features(self, query_text: str) -> tuple[List[str], List[str], List[str]]:
        """Extract semantic features from query."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query_text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Extract key phrases (simple n-gram approach)
        key_phrases = []
        words_list = query_text.split()
        for i in range(len(words_list) - 1):
            phrase = ' '.join(words_list[i:i+2])
            if len(phrase) > 5:
                key_phrases.append(phrase)
        
        # Use LLM for topic extraction
        topics = []
        try:
            llm_response = await self.generate_response(
                query_text,
                template_name='topic_extraction',
                template_variables={'query': query_text}
            )
            
            topic_data = json.loads(llm_response.content)
            topics = topic_data.get('topics', [])
            
        except Exception as e:
            logger.warning(
                "Topic extraction failed",
                agent_id=self.agent_id,
                error=str(e)
            )
        
        return keywords[:10], key_phrases[:5], topics[:5]
    
    async def _detect_ambiguity(
        self,
        query_text: str,
        entities: List[EntityExtraction]
    ) -> tuple[List[str], List[str]]:
        """Detect ambiguous terms and generate clarifications."""
        ambiguous_terms = []
        clarifications = []
        
        # Pattern-based ambiguity detection
        ambiguity_patterns = [
            r'\b(it|that|this|they|them)\b',  # Pronouns without clear referents
            r'\b(some|many|few|several)\b',   # Vague quantifiers
            r'\b(recently|soon|later)\b',     # Vague time references
            r'\b(here|there|nearby)\b',       # Vague location references
        ]
        
        for pattern in ambiguity_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            ambiguous_terms.extend(matches)
        
        # Use LLM for advanced ambiguity detection
        try:
            llm_response = await self.generate_response(
                query_text,
                template_name='ambiguity_detection',
                template_variables={
                    'query': query_text,
                    'entities': [e.dict() for e in entities]
                }
            )
            
            ambiguity_data = json.loads(llm_response.content)
            
            llm_ambiguous = ambiguity_data.get('ambiguous_terms', [])
            ambiguous_terms.extend(llm_ambiguous)
            
            clarifications = ambiguity_data.get('clarifications', [])
            
        except Exception as e:
            logger.warning(
                "LLM ambiguity detection failed",
                agent_id=self.agent_id,
                error=str(e)
            )
        
        return list(set(ambiguous_terms)), clarifications
    
    async def _analyze_context_requirements(
        self,
        query_text: str,
        intent: IntentClassification,
        entities: List[EntityExtraction]
    ) -> tuple[bool, List[str]]:
        """Analyze what context the query requires."""
        requires_context = False
        context_types = []
        
        # Check for context-requiring patterns
        context_patterns = {
            'temporal': r'\b(today|yesterday|last\s+\w+|this\s+\w+|recently)\b',
            'personal': r'\b(my|mine|our|ours|I|we)\b',
            'location': r'\b(here|there|nearby|local)\b',
            'document': r'\b(this\s+document|current\s+policy|above|below)\b'
        }
        
        for context_type, pattern in context_patterns.items():
            if re.search(pattern, query_text, re.IGNORECASE):
                requires_context = True
                context_types.append(context_type)
        
        # Check entities for context requirements
        for entity in entities:
            if entity.entity_type in ['person', 'organization']:
                requires_context = True
                if 'personal' not in context_types:
                    context_types.append('personal')
        
        return requires_context, context_types
    
    async def _calculate_parsing_confidence(
        self,
        intent: IntentClassification,
        entities: List[EntityExtraction],
        ambiguous_terms: List[str]
    ) -> float:
        """Calculate overall parsing confidence."""
        # Base confidence from intent
        confidence = intent.confidence
        
        # Adjust for entities
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            confidence = (confidence + avg_entity_confidence) / 2
        
        # Reduce for ambiguity
        ambiguity_penalty = min(0.1 * len(ambiguous_terms), 0.4)
        confidence = max(0.1, confidence - ambiguity_penalty)
        
        return round(confidence, 3)
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for query parsing."""
        templates = {}
        
        # Query normalization template
        templates['query_normalization'] = PromptTemplate(
            name='query_normalization',
            template="""
            Clean and normalize the following query text. Fix grammar, punctuation, and formatting issues.
            Preserve the original meaning and intent.
            
            Query: {query}
            
            Return only the normalized query text without any explanation.
            """.strip()
        )
        
        # Intent detection template
        templates['intent_detection'] = PromptTemplate(
            name='intent_detection',
            template="""
            Analyze the following query and determine the user's intent.
            
            Query: {query}
            Context: {context}
            Known Intents: {known_intents}
            
            Respond with JSON in this format:
            {{
                "primary_intent": "intent_name",
                "confidence": 0.85,
                "sub_intent": "specific_sub_intent",
                "context": {{}},
                "secondary_intents": [
                    {{"intent": "secondary_intent", "confidence": 0.6}}
                ]
            }}
            """.strip()
        )
        
        # Entity extraction template
        templates['entity_extraction'] = PromptTemplate(
            name='entity_extraction',
            template="""
            Extract named entities and key information from the following query.
            
            Query: {query}
            Context: {context}
            Known Entity Types: {known_entities}
            
            Respond with JSON in this format:
            {{
                "entities": [
                    {{
                        "type": "entity_type",
                        "value": "entity_value",
                        "confidence": 0.9,
                        "start": 0,
                        "end": 10,
                        "context": "surrounding_context"
                    }}
                ]
            }}
            """.strip()
        )
        
        # Topic extraction template
        templates['topic_extraction'] = PromptTemplate(
            name='topic_extraction',
            template="""
            Extract the main topics from the following query.
            
            Query: {query}
            
            Respond with JSON in this format:
            {{
                "topics": ["topic1", "topic2", "topic3"]
            }}
            """.strip()
        )
        
        # Ambiguity detection template
        templates['ambiguity_detection'] = PromptTemplate(
            name='ambiguity_detection',
            template="""
            Identify ambiguous terms and suggest clarifications for the following query.
            
            Query: {query}
            Entities: {entities}
            
            Respond with JSON in this format:
            {{
                "ambiguous_terms": ["term1", "term2"],
                "clarifications": [
                    "Did you mean X or Y when you said 'term1'?",
                    "Could you specify what you mean by 'term2'?"
                ]
            }}
            """.strip()
        )
        
        return templates
    
    # Helper methods for pattern matching
    async def _pattern_match_intent(self, query_text: str) -> Optional[str]:
        """Pattern-based intent detection."""
        query_lower = query_text.lower()
        
        intent_patterns = {
            'coverage_check': [
                r'is\s+\w+\s+covered',
                r'does\s+\w+\s+cover',
                r'coverage\s+for',
                r'covered\s+by'
            ],
            'cost_inquiry': [
                r'how\s+much\s+cost',
                r'price\s+of',
                r'cost\s+for',
                r'copay',
                r'deductible'
            ],
            'procedure_question': [
                r'how\s+to',
                r'steps\s+to',
                r'process\s+for',
                r'how\s+do\s+I'
            ],
            'eligibility_check': [
                r'eligible\s+for',
                r'qualify\s+for',
                r'am\s+I\s+covered',
                r'can\s+I\s+get'
            ]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return None
    
    async def _pattern_extract_entities(self, query_text: str) -> List[EntityExtraction]:
        """Pattern-based entity extraction."""
        entities = []
        
        # Medical procedures
        medical_pattern = r'\b(?:surgery|procedure|operation|treatment|therapy|exam|test|screening)\b'
        for match in re.finditer(medical_pattern, query_text, re.IGNORECASE):
            entities.append(EntityExtraction(
                entity_type='medical_procedure',
                entity_value=match.group(),
                confidence=0.8,
                start_position=match.start(),
                end_position=match.end()
            ))
        
        # Dollar amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        for match in re.finditer(money_pattern, query_text):
            entities.append(EntityExtraction(
                entity_type='money_amount',
                entity_value=match.group(),
                confidence=0.9,
                start_position=match.start(),
                end_position=match.end()
            ))
        
        # Dates
        date_pattern = r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
        for match in re.finditer(date_pattern, query_text, re.IGNORECASE):
            entities.append(EntityExtraction(
                entity_type='date',
                entity_value=match.group(),
                confidence=0.85,
                start_position=match.start(),
                end_position=match.end()
            ))
        
        return entities
    
    async def _determine_query_type(self, query_text: str, intent: IntentClassification) -> str:
        """Determine the type of query."""
        query_lower = query_text.lower()
        
        # Question words
        if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'why', 'how']):
            if 'how' in query_lower:
                return 'procedural'
            elif any(word in query_lower for word in ['what', 'who', 'where', 'when']):
                return 'factual'
            else:
                return 'explanatory'
        
        # Comparison indicators
        if any(phrase in query_lower for phrase in ['vs', 'versus', 'compare', 'difference', 'better']):
            return 'comparison'
        
        # Hypothetical scenarios
        if any(phrase in query_lower for phrase in ['what if', 'suppose', 'assume', 'hypothetical']):
            return 'hypothetical'
        
        # Default based on intent
        intent_to_type = {
            'coverage_check': 'factual',
            'cost_inquiry': 'factual',
            'procedure_question': 'procedural',
            'eligibility_check': 'factual'
        }
        
        return intent_to_type.get(intent.intent, 'factual')
    
    async def _assess_complexity(self, query_text: str) -> str:
        """Assess query complexity."""
        # Simple metrics
        word_count = len(query_text.split())
        sentence_count = len(re.split(r'[.!?]+', query_text))
        
        # Check for complex structures
        complex_indicators = [
            'and', 'or', 'but', 'however', 'although', 'because', 'since',
            'if', 'unless', 'while', 'whereas', 'moreover', 'furthermore'
        ]
        
        complex_count = sum(1 for word in query_text.lower().split() if word in complex_indicators)
        
        # Scoring
        if word_count < 10 and sentence_count == 1 and complex_count == 0:
            return 'simple'
        elif word_count > 30 or sentence_count > 2 or complex_count > 2:
            return 'complex'
        else:
            return 'medium'
    
    async def _identify_domain(self, query_text: str, intent: IntentClassification) -> str:
        """Identify the domain of the query."""
        query_lower = query_text.lower()
        
        # Domain keywords
        domain_keywords = {
            'medical': ['health', 'medical', 'doctor', 'hospital', 'treatment', 'surgery', 'medicine', 'disease', 'diagnosis'],
            'dental': ['dental', 'teeth', 'tooth', 'dentist', 'orthodontic', 'oral', 'cavity', 'crown'],
            'vision': ['vision', 'eye', 'glasses', 'contacts', 'optometry', 'sight', 'visual'],
            'legal': ['legal', 'law', 'contract', 'agreement', 'court', 'attorney', 'lawsuit'],
            'financial': ['money', 'cost', 'price', 'payment', 'insurance', 'claim', 'coverage', 'premium'],
            'technical': ['system', 'software', 'hardware', 'technical', 'computer', 'network', 'database']
        }
        
        # Score domains
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return highest scoring domain
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        # Default domain based on intent
        intent_to_domain = {
            'coverage_check': 'medical',
            'cost_inquiry': 'financial',
            'procedure_question': 'procedural'
        }
        
        return intent_to_domain.get(intent.intent, 'general')
    
    async def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns."""
        # This would typically load from a configuration file or database
        return {
            'coverage_check': [
                'is * covered',
                'does * cover',
                'coverage for *',
                'am i covered for *'
            ],
            'cost_inquiry': [
                'how much *',
                'what does * cost',
                'price of *',
                'copay for *'
            ],
            'procedure_question': [
                'how to *',
                'what are the steps *',
                'how do i *',
                'process for *'
            ]
        }
    
    async def _load_entity_patterns(self) -> Dict[str, str]:
        """Load entity recognition patterns."""
        return {
            'medical_procedure': r'\b(?:surgery|procedure|operation|treatment|therapy|exam|test|screening)\b',
            'money_amount': r'\$[\d,]+(?:\.\d{2})?',
            'date': r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b',
            'person': r'\b(?:Dr\.?|Doctor|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+\b',
            'organization': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Hospital|Clinic|Center|Institute)\b'
        }
    
    async def _load_knowledge_base(self):
        """Load known intents and entities."""
        self.known_intents.update([
            'coverage_check', 'cost_inquiry', 'procedure_question', 'eligibility_check',
            'claim_status', 'provider_search', 'appointment_scheduling', 'information_request',
            'complaint', 'compliment', 'comparison', 'explanation'
        ])
        
        self.known_entities.update([
            'medical_procedure', 'money_amount', 'date', 'person', 'organization',
            'body_part', 'medication', 'symptom', 'diagnosis', 'insurance_plan'
        ])
    
    async def _handle_specific_request(self, message: AgentMessage) -> Optional[ResponseMessage]:
        """Handle specific parsing requests."""
        request_type = message.content.get('request_type')
        
        if request_type == 'intent_only':
            # Only detect intent
            query_text = message.content.get('query', '')
            context = message.content.get('context', {})
            
            primary_intent, secondary_intents = await self._detect_intent(query_text, context)
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content={
                    'primary_intent': primary_intent.dict(),
                    'secondary_intents': [intent.dict() for intent in secondary_intents]
                },
                confidence_score=primary_intent.confidence,
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Intent detected: {primary_intent.intent}"
            )
        
        elif request_type == 'entities_only':
            # Only extract entities
            query_text = message.content.get('query', '')
            context = message.content.get('context', {})
            
            entities = await self._extract_entities(query_text, context)
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content={
                    'entities': [entity.dict() for entity in entities]
                },
                confidence_score=sum(e.confidence for e in entities) / max(len(entities), 1),
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Found {len(entities)} entities"
            )
        
        else:
            # Unsupported request type
            return None
