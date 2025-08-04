"""
Clause Retriever Agent - Specialized agent for semantic clause retrieval.
Handles vector search, hybrid search, ranking, and result filtering.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import structlog
from pydantic import BaseModel, Field
import numpy as np

from agents.base.llm_agent import LLMAgent, AgentCapability
from agents.base.message import AgentMessage, MessageType, ResponseMessage
from storage.interfaces.vector_store import VectorStoreInterface
from storage.interfaces.document_store import DocumentStoreInterface
from retrieval.engines.semantic_search import SemanticSearchEngine
from retrieval.engines.hybrid_search import HybridSearchEngine
from retrieval.ranking.relevance_scorer import RelevanceScorer
from retrieval.ranking.uncertainty_scorer import UncertaintyScorer
from core.exceptions import RetrievalError, VectorStoreError
from core.config import settings

logger = structlog.get_logger(__name__)


class SearchParameters(BaseModel):
    """Parameters for clause search."""
    
    # Search terms
    query: str
    keywords: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Search configuration
    search_type: str = "hybrid"  # semantic, keyword, hybrid
    max_results: int = 20
    similarity_threshold: float = 0.7
    
    # Filtering criteria
    document_types: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    date_range: Optional[Dict[str, str]] = None
    
    # Ranking preferences
    ranking_strategy: str = "relevance"  # relevance, uncertainty, hybrid
    boost_recent: bool = True
    boost_high_confidence: bool = True
    
    # Context
    user_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[str] = Field(default_factory=list)


class ClauseResult(BaseModel):
    """A single clause search result."""
    
    # Clause identification
    clause_id: str
    document_id: str
    clause_number: Optional[str] = None
    
    # Content
    title: Optional[str] = None
    content: str
    summary: Optional[str] = None
    
    # Metadata
    document_title: str
    document_type: str
    section: Optional[str] = None
    category: Optional[str] = None
    
    # Scoring
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Additional information
    keywords: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    
    # Search context
    match_type: str = "semantic"  # semantic, keyword, exact
    matched_terms: List[str] = Field(default_factory=list)
    highlight_positions: List[Tuple[int, int]] = Field(default_factory=list)


class SearchResults(BaseModel):
    """Complete search results with metadata."""
    
    # Query information
    original_query: str
    processed_query: str
    search_params: SearchParameters
    
    # Results
    results: List[ClauseResult]
    total_found: int
    
    # Search metadata
    search_time_ms: float
    retrieval_strategy: str
    ranking_strategy: str
    
    # Quality metrics
    average_relevance: float = 0.0
    average_confidence: float = 0.0
    coverage_score: float = 0.0  # How well the results cover the query
    
    # Suggestions
    suggested_refinements: List[str] = Field(default_factory=list)
    related_queries: List[str] = Field(default_factory=list)
    missing_coverage_areas: List[str] = Field(default_factory=list)


class ClauseRetrieverAgent(LLMAgent):
    """Agent specialized in retrieving relevant clauses from document stores."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_type="ClauseRetrieverAgent",
            config=config or {}
        )
        
        # Storage interfaces
        self.vector_store: Optional[VectorStoreInterface] = None
        self.document_store: Optional[DocumentStoreInterface] = None
        
        # Search engines
        self.semantic_engine: Optional[SemanticSearchEngine] = None
        self.hybrid_engine: Optional[HybridSearchEngine] = None
        
        # Scoring components
        self.relevance_scorer: Optional[RelevanceScorer] = None
        self.uncertainty_scorer: Optional[UncertaintyScorer] = None
        
        # Configuration
        self.default_max_results = self.config.get('max_results', 20)
        self.default_threshold = self.config.get('similarity_threshold', 0.7)
        self.enable_reranking = self.config.get('enable_reranking', True)
        self.enable_query_expansion = self.config.get('enable_query_expansion', True)
        
        logger.info("Clause Retriever Agent initialized", agent_id=self.agent_id)
    
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize retriever capabilities."""
        return [
            AgentCapability(
                name="semantic_search",
                description="Semantic similarity search using embeddings",
                input_types=["query", "text"],
                output_types=["clause_results", "ranked_results"],
                confidence_level=0.9,
                estimated_processing_time=3.0
            ),
            AgentCapability(
                name="hybrid_search",
                description="Combined semantic and keyword search",
                input_types=["query", "text", "keywords"],
                output_types=["clause_results", "ranked_results"],
                confidence_level=0.95,
                estimated_processing_time=4.0
            ),
            AgentCapability(
                name="contextual_retrieval",
                description="Context-aware clause retrieval with user preferences",
                input_types=["query", "context", "history"],
                output_types=["personalized_results"],
                confidence_level=0.85,
                estimated_processing_time=5.0
            ),
            AgentCapability(
                name="uncertainty_ranking",
                description="Rank results by uncertainty and confidence scores",
                input_types=["clause_results"],
                output_types=["ranked_results", "uncertainty_scores"],
                confidence_level=0.8,
                estimated_processing_time=2.0
            ),
            AgentCapability(
                name="result_filtering",
                description="Filter results by criteria and preferences",
                input_types=["clause_results", "filters"],
                output_types=["filtered_results"],
                confidence_level=0.9,
                estimated_processing_time=1.0
            )
        ]
    
    async def _initialize(self):
        """Initialize retriever-specific components."""
        await super()._initialize()
        
        try:
            # Initialize storage interfaces
            from storage.implementations.chroma_store import ChromaVectorStore
            from storage.implementations.postgres_store import PostgresDocumentStore
            
            self.vector_store = ChromaVectorStore()
            await self.vector_store.initialize()
            
            self.document_store = PostgresDocumentStore()
            await self.document_store.initialize()
            
            # Initialize search engines
            self.semantic_engine = SemanticSearchEngine(
                vector_store=self.vector_store,
                config=self.config.get('semantic_search', {})
            )
            await self.semantic_engine.initialize()
            
            self.hybrid_engine = HybridSearchEngine(
                vector_store=self.vector_store,
                document_store=self.document_store,
                config=self.config.get('hybrid_search', {})
            )
            await self.hybrid_engine.initialize()
            
            # Initialize scoring components
            self.relevance_scorer = RelevanceScorer(
                config=self.config.get('relevance_scoring', {})
            )
            
            self.uncertainty_scorer = UncertaintyScorer(
                config=self.config.get('uncertainty_scoring', {})
            )
            
            logger.info("Clause retriever components initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize clause retriever components",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _cleanup(self):
        """Cleanup retriever resources."""
        try:
            if self.semantic_engine:
                await self.semantic_engine.cleanup()
            
            if self.hybrid_engine:
                await self.hybrid_engine.cleanup()
            
            if self.vector_store:
                await self.vector_store.cleanup()
            
            if self.document_store:
                await self.document_store.cleanup()
            
            await super()._cleanup()
            
        except Exception as e:
            logger.error(
                "Error during clause retriever cleanup",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for clause retrieval."""
        if message.type == MessageType.QUERY:
            return await self._handle_search_request(message)
        elif message.type == MessageType.REQUEST:
            return await self._handle_specific_request(message)
        else:
            logger.debug(
                "Unsupported message type for clause retriever",
                agent_id=self.agent_id,
                message_type=message.type
            )
            return None
    
    async def _handle_search_request(self, message: AgentMessage) -> ResponseMessage:
        """Handle clause search requests."""
        start_time = datetime.utcnow()
        
        try:
            # Extract search parameters
            search_params = await self._extract_search_parameters(message)
            
            logger.info(
                "Processing clause search",
                agent_id=self.agent_id,
                query=search_params.query,
                search_type=search_params.search_type,
                max_results=search_params.max_results
            )
            
            # Perform search
            search_results = await self._perform_search(search_params)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            search_results.search_time_ms = processing_time * 1000
            
            # Create response
            response_content = {
                'search_results': search_results.dict(),
                'success': True,
                'processing_time_ms': processing_time * 1000
            }
            
            return ResponseMessage(
                from_agent_id=self.agent_id,
                from_agent_type=self.agent_type,
                to_agent_id=message.from_agent_id,
                to_agent_type=message.from_agent_type,
                content=response_content,
                confidence_score=search_results.average_confidence,
                sources=[{
                    'type': 'clause_search',
                    'results_count': len(search_results.results),
                    'average_relevance': search_results.average_relevance
                }],
                conversation_id=message.conversation_id,
                parent_message_id=message.id,
                answer=f"Found {len(search_results.results)} relevant clauses"
            )
            
        except Exception as e:
            logger.error(
                "Clause search failed",
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
                answer=f"Clause search failed: {str(e)}"
            )
    
    async def _extract_search_parameters(self, message: AgentMessage) -> SearchParameters:
        """Extract search parameters from message."""
        content = message.content
        
        # Extract basic query
        query = content.get('query', content.get('text', ''))
        if not query:
            raise ValidationError("Query text is required for search")
        
        # Extract parsed query information if available
        parsed_query = content.get('parsed_query', {})
        
        # Build search parameters
        search_params = SearchParameters(
            query=query,
            keywords=parsed_query.get('keywords', content.get('keywords', [])),
            entities=parsed_query.get('entities', content.get('entities', [])),
            search_type=content.get('search_type', 'hybrid'),
            max_results=content.get('max_results', self.default_max_results),
            similarity_threshold=content.get('similarity_threshold', self.default_threshold),
            document_types=content.get('document_types', []),
            domains=parsed_query.get('domain', content.get('domains', [])),
            date_range=content.get('date_range'),
            ranking_strategy=content.get('ranking_strategy', 'relevance'),
            boost_recent=content.get('boost_recent', True),
            boost_high_confidence=content.get('boost_high_confidence', True),
            user_context=content.get('user_context', {}),
            conversation_history=content.get('conversation_history', [])
        )
        
        return search_params
    
    async def _perform_search(self, params: SearchParameters) -> SearchResults:
        """Perform the actual clause search."""
        # Expand query if enabled
        expanded_query = params.query
        if self.enable_query_expansion:
            expanded_query = await self._expand_query(params)
        
        # Perform search based on type
        if params.search_type == "semantic":
            raw_results = await self._semantic_search(expanded_query, params)
        elif params.search_type == "keyword":
            raw_results = await self._keyword_search(expanded_query, params)
        else:  # hybrid
            raw_results = await self._hybrid_search(expanded_query, params)
        
        # Apply filters
        filtered_results = await self._apply_filters(raw_results, params)
        
        # Rank results
        if self.enable_reranking:
            ranked_results = await self._rank_results(filtered_results, params)
        else:
            ranked_results = filtered_results
        
        # Limit results
        final_results = ranked_results[:params.max_results]
        
        # Calculate metrics
        avg_relevance = sum(r.relevance_score for r in final_results) / max(len(final_results), 1)
        avg_confidence = sum(r.confidence_score for r in final_results) / max(len(final_results), 1)
        coverage_score = await self._calculate_coverage_score(params.query, final_results)
        
        # Generate suggestions
        suggestions = await self._generate_suggestions(params, final_results)
        
        return SearchResults(
            original_query=params.query,
            processed_query=expanded_query,
            search_params=params,
            results=final_results,
            total_found=len(raw_results),
            search_time_ms=0.0,  # Will be set by caller
            retrieval_strategy=params.search_type,
            ranking_strategy=params.ranking_strategy,
            average_relevance=avg_relevance,
            average_confidence=avg_confidence,
            coverage_score=coverage_score,
            suggested_refinements=suggestions.get('refinements', []),
            related_queries=suggestions.get('related_queries', []),
            missing_coverage_areas=suggestions.get('missing_areas', [])
        )
    
    async def _expand_query(self, params: SearchParameters) -> str:
        """Expand query with synonyms and related terms."""
        try:
            # Use LLM to expand query
            llm_response = await self.generate_response(
                params.query,
                template_name='query_expansion',
                template_variables={
                    'original_query': params.query,
                    'keywords': params.keywords,
                    'entities': params.entities,
                    'context': params.user_context
                }
            )
            
            # Parse expansion result
            expanded = llm_response.content.strip()
            
            # Validate expanded query
            if len(expanded) > len(params.query) * 3:
                logger.warning("Query expansion too large, using original")
                return params.query
            
            return expanded
            
        except Exception as e:
            logger.warning(
                "Query expansion failed, using original query",
                agent_id=self.agent_id,
                error=str(e)
            )
            return params.query
    
    async def _semantic_search(self, query: str, params: SearchParameters) -> List[ClauseResult]:
        """Perform semantic vector search."""
        try:
            # Get semantic search results
            semantic_results = await self.semantic_engine.search(
                query=query,
                limit=params.max_results * 2,  # Get more for filtering
                similarity_threshold=params.similarity_threshold,
                metadata_filters=self._build_metadata_filters(params)
            )
            
            # Convert to ClauseResult objects
            clause_results = []
            for result in semantic_results:
                clause_result = await self._convert_to_clause_result(
                    result, "semantic", params
                )
                if clause_result:
                    clause_results.append(clause_result)
            
            return clause_results
            
        except Exception as e:
            logger.error(
                "Semantic search failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise RetrievalError(f"Semantic search failed: {str(e)}", query)
    
    async def _keyword_search(self, query: str, params: SearchParameters) -> List[ClauseResult]:
        """Perform keyword-based search."""
        try:
            # Extract search terms
            search_terms = params.keywords + [query]
            
            # Perform keyword search using document store
            keyword_results = await self.document_store.search_text(
                terms=search_terms,
                limit=params.max_results * 2,
                filters=self._build_document_filters(params)
            )
            
            # Convert to ClauseResult objects
            clause_results = []
            for result in keyword_results:
                clause_result = await self._convert_to_clause_result(
                    result, "keyword", params
                )
                if clause_result:
                    clause_results.append(clause_result)
            
            return clause_results
            
        except Exception as e:
            logger.error(
                "Keyword search failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise RetrievalError(f"Keyword search failed: {str(e)}", query)
    
    async def _hybrid_search(self, query: str, params: SearchParameters) -> List[ClauseResult]:
        """Perform hybrid semantic + keyword search."""
        try:
            # Use hybrid search engine
            hybrid_results = await self.hybrid_engine.search(
                query=query,
                keywords=params.keywords,
                limit=params.max_results * 2,
                semantic_weight=0.7,
                keyword_weight=0.3,
                similarity_threshold=params.similarity_threshold,
                metadata_filters=self._build_metadata_filters(params)
            )
            
            # Convert to ClauseResult objects
            clause_results = []
            for result in hybrid_results:
                clause_result = await self._convert_to_clause_result(
                    result, "hybrid", params
                )
                if clause_result:
                    clause_results.append(clause_result)
            
            return clause_results
            
        except Exception as e:
            logger.error(
                "Hybrid search failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise RetrievalError(f"Hybrid search failed: {str(e)}", query)
    
    async def _convert_to_clause_result(
        self,
        search_result: Dict[str, Any],
        match_type: str,
        params: SearchParameters
    ) -> Optional[ClauseResult]:
        """Convert search engine result to ClauseResult."""
        try:
            # Extract basic information
            clause_id = search_result.get('id')
            if not clause_id:
                return None
            
            # Get additional clause metadata
            clause_metadata = await self.document_store.get_clause_metadata(clause_id)
            
            # Calculate scores
            similarity_score = search_result.get('similarity_score', 0.0)
            relevance_score = await self.relevance_scorer.score(
                query=params.query,
                clause_content=search_result.get('content', ''),
                metadata=clause_metadata
            )
            
            uncertainty_score = await self.uncertainty_scorer.score(
                clause_content=search_result.get('content', ''),
                metadata=clause_metadata,
                search_context={'query': params.query, 'match_type': match_type}
            )
            
            confidence_score = await self._calculate_confidence_score(
                similarity_score, relevance_score, uncertainty_score, clause_metadata
            )
            
            # Extract matched terms and positions
            matched_terms = search_result.get('matched_terms', [])
            highlight_positions = search_result.get('highlight_positions', [])
            
            return ClauseResult(
                clause_id=clause_id,
                document_id=clause_metadata.get('document_id', ''),
                clause_number=clause_metadata.get('clause_number'),
                title=clause_metadata.get('title'),
                content=search_result.get('content', ''),
                summary=clause_metadata.get('summary'),
                document_title=clause_metadata.get('document_title', ''),
                document_type=clause_metadata.get('document_type', ''),
                section=clause_metadata.get('section'),
                category=clause_metadata.get('category'),
                relevance_score=relevance_score,
                similarity_score=similarity_score,
                uncertainty_score=uncertainty_score,
                confidence_score=confidence_score,
                keywords=clause_metadata.get('keywords', []),
                entities=clause_metadata.get('entities', []),
                relationships=clause_metadata.get('relationships', []),
                match_type=match_type,
                matched_terms=matched_terms,
                highlight_positions=highlight_positions
            )
            
        except Exception as e:
            logger.warning(
                "Failed to convert search result",
                agent_id=self.agent_id,
                result_id=search_result.get('id'),
                error=str(e)
            )
            return None
    
    async def _apply_filters(
        self,
        results: List[ClauseResult],
        params: SearchParameters
    ) -> List[ClauseResult]:
        """Apply filtering criteria to results."""
        filtered = results
        
        # Filter by document types
        if params.document_types:
            filtered = [r for r in filtered if r.document_type in params.document_types]
        
        # Filter by domains (categories)
        if params.domains:
            filtered = [r for r in filtered if r.category in params.domains]
        
        # Filter by similarity threshold
        filtered = [r for r in filtered if r.similarity_score >= params.similarity_threshold]
        
        # Apply date range filtering if specified
        if params.date_range:
            # This would require additional metadata from document store
            pass
        
        return filtered
    
    async def _rank_results(
        self,
        results: List[ClauseResult],
        params: SearchParameters
    ) -> List[ClauseResult]:
        """Rank results according to strategy."""
        if params.ranking_strategy == "relevance":
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
        elif params.ranking_strategy == "uncertainty":
            return sorted(results, key=lambda r: (1 - r.uncertainty_score), reverse=True)
        elif params.ranking_strategy == "confidence":
            return sorted(results, key=lambda r: r.confidence_score, reverse=True)
        else:  # hybrid ranking
            return await self._hybrid_rank(results, params)
    
    async def _hybrid_rank(
        self,
        results: List[ClauseResult],
        params: SearchParameters
    ) -> List[ClauseResult]:
        """Perform hybrid ranking combining multiple factors."""
        for result in results:
            # Calculate composite score
            score = 0.0
            
            # Relevance component (40%)
            score += result.relevance_score * 0.4
            
            # Similarity component (30%)
            score += result.similarity_score * 0.3
            
            # Confidence component (20%)
            score += result.confidence_score * 0.2
            
            # Uncertainty component (10%, inverted)
            score += (1 - result.uncertainty_score) * 0.1
            
            # Apply boosts
            if params.boost_high_confidence and result.confidence_score > 0.8:
                score *= 1.1
            
            if params.boost_recent:
                # Would need document date information
                pass
            
            # Store composite score for sorting
            result.composite_score = score
        
        return sorted(results, key=lambda r: getattr(r, 'composite_score', 0), reverse=True)
    
    async def _calculate_confidence_score(
        self,
        similarity_score: float,
        relevance_score: float,
        uncertainty_score: float,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score for a result."""
        # Base confidence from similarity and relevance
        base_confidence = (similarity_score + relevance_score) / 2
        
        # Adjust for uncertainty (higher uncertainty = lower confidence)
        uncertainty_adjustment = 1 - uncertainty_score
        adjusted_confidence = base_confidence * uncertainty_adjustment
        
        # Adjust for metadata quality
        quality_score = metadata.get('quality_score', 1.0)
        final_confidence = adjusted_confidence * quality_score
        
        return min(1.0, max(0.0, final_confidence))
    
    async def _calculate_coverage_score(
        self,
        query: str,
        results: List[ClauseResult]
    ) -> float:
        """Calculate how well the results cover the query."""
        if not results:
            return 0.0
        
        try:
            # Use LLM to assess coverage
            llm_response = await self.generate_response(
                query,
                template_name='coverage_assessment',
                template_variables={
                    'query': query,
                    'result_summaries': [r.summary or r.content[:200] for r in results[:5]]
                }
            )
            
            # Parse coverage score from response
            import json
            coverage_data = json.loads(llm_response.content)
            return coverage_data.get('coverage_score', 0.5)
            
        except Exception as e:
            logger.warning(
                "Coverage score calculation failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            # Fallback: simple coverage based on result count and average relevance
            avg_relevance = sum(r.relevance_score for r in results) / len(results)
            result_count_factor = min(1.0, len(results) / 10)  # Assume 10 results = full coverage
            return avg_relevance * result_count_factor
    
    async def _generate_suggestions(
        self,
        params: SearchParameters,
        results: List[ClauseResult]
    ) -> Dict[str, List[str]]:
        """Generate search suggestions and improvements."""
        suggestions = {
            'refinements': [],
            'related_queries': [],
            'missing_areas': []
        }
        
        try:
            # Use LLM to generate suggestions
            llm_response = await self.generate_response(
                params.query,
                template_name='search_suggestions',
                template_variables={
                    'query': params.query,
                    'results_count': len(results),
                    'avg_relevance': sum(r.relevance_score for r in results) / max(len(results), 1),
                    'categories': list(set(r.category for r in results if r.category))
                }
            )
            
            # Parse suggestions
            import json
            suggestion_data = json.loads(llm_response.content)
            suggestions.update(suggestion_data)
            
        except Exception as e:
            logger.warning(
                "Suggestion generation failed",
                agent_id=self.agent_id,
                error=str(e)
            )
        
        return suggestions
    
    def _build_metadata_filters(self, params: SearchParameters) -> Dict[str, Any]:
        """Build metadata filters for vector store."""
        filters = {}
        
        if params.document_types:
            filters['document_type'] = {'$in': params.document_types}
        
        if params.domains:
            filters['category'] = {'$in': params.domains}
        
        return filters
    
    def _build_document_filters(self, params: SearchParameters) -> Dict[str, Any]:
        """Build filters for document store."""
        filters = {}
        
        if params.document_types:
            filters['document_type'] = params.document_types
        
        if params.domains:
            filters['domain'] = params.domains
        
        if params.date_range:
            filters['date_range'] = params.date_range
        
        return filters
    
    async def _handle_specific_request(self, message: AgentMessage) -> Optional[ResponseMessage]:
        """Handle specific retrieval requests."""
        request_type = message.content.get('request_type')
        
        if request_type == 'similar_clauses':
            # Find clauses similar to a given clause
            clause_id = message.content.get('clause_id')
            if not clause_id:
                return None
            
            try:
                # Get the source clause
                source_clause = await self.document_store.get_clause(clause_id)
                if not source_clause:
                    return None
                
                # Search for similar clauses
                similar_results = await self.semantic_engine.search(
                    query=source_clause['content'],
                    limit=10,
                    exclude_ids=[clause_id]
                )
                
                # Convert to ClauseResult objects
                clause_results = []
                for result in similar_results:
                    clause_result = await self._convert_to_clause_result(
                        result, "semantic", SearchParameters(query=source_clause['content'])
                    )
                    if clause_result:
                        clause_results.append(clause_result)
                
                return ResponseMessage(
                    from_agent_id=self.agent_id,
                    from_agent_type=self.agent_type,
                    to_agent_id=message.from_agent_id,
                    to_agent_type=message.from_agent_type,
                    content={
                        'similar_clauses': [r.dict() for r in clause_results],
                        'source_clause_id': clause_id
                    },
                    confidence_score=sum(r.similarity_score for r in clause_results) / max(len(clause_results), 1),
                    conversation_id=message.conversation_id,
                    parent_message_id=message.id,
                    answer=f"Found {len(clause_results)} similar clauses"
                )
                
            except Exception as e:
                logger.error(
                    "Similar clause search failed",
                    agent_id=self.agent_id,
                    clause_id=clause_id,
                    error=str(e)
                )
                return None
        
        elif request_type == 'clause_relationships':
            # Find related clauses based on explicit relationships
            clause_id = message.content.get('clause_id')
            relationship_types = message.content.get('relationship_types', [])
            
            try:
                related_clauses = await self.document_store.get_related_clauses(
                    clause_id=clause_id,
                    relationship_types=relationship_types
                )
                
                # return ResponseMessage(
                #     from_agent_id=self.agent_id,
                #     from_agent_type=self.agent_type,
                #     to_agent_id=message.from_agent_
                return ResponseMessage(
                    from_agent_id=self.agent_id,
                    from_agent_type=self.agent_type,
                    to_agent_id=message.from_agent_id,
                    to_agent_type=message.from_agent_type,
                    content={
                        'related_clauses': [clause.dict() for clause in related_clauses],
                        'source_clause_id': clause_id,
                        'relationship_types': relationship_types
                    },
                    confidence_score=0.9,  # High confidence for explicit relationships
                    conversation_id=message.conversation_id,
                    parent_message_id=message.id,
                    answer=f"Found {len(related_clauses)} related clauses"
                )
                
            except Exception as e:
                logger.error(
                    "Clause relationship search failed",
                    agent_id=self.agent_id,
                    clause_id=clause_id,
                    error=str(e)
                )
                return None
        
        else:
            # Unsupported request type
            return None
    
    async def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for clause retrieval."""
        templates = {}
        
        # Query expansion template
        templates['query_expansion'] = PromptTemplate(
            name='query_expansion',
            template="""
            Expand the following query with relevant synonyms, related terms, and context.
            Focus on policy and legal terminology that would help find relevant clauses.
            
            Original Query: {original_query}
            Keywords: {keywords}
            Entities: {entities}
            Context: {context}
            
            Provide an expanded query that includes:
            1. Original terms
            2. Synonyms and related terms
            3. Alternative phrasings
            4. Domain-specific terminology
            
            Keep the expansion focused and relevant. Return only the expanded query text.
            """.strip()
        )
        
        # Coverage assessment template
        templates['coverage_assessment'] = PromptTemplate(
            name='coverage_assessment',
            template="""
            Assess how well the search results cover the user's query.
            
            Query: {query}
            Result Summaries: {result_summaries}
            
            Analyze:
            1. What aspects of the query are well covered?
            2. What aspects are missing or poorly covered?
            3. How comprehensive are the results?
            
            Respond with JSON:
            {{
                "coverage_score": 0.85,
                "well_covered": ["aspect1", "aspect2"],
                "poorly_covered": ["aspect3"],
                "missing_areas": ["area1", "area2"]
            }}
            """.strip()
        )
        
        # Search suggestions template
        templates['search_suggestions'] = PromptTemplate(
            name='search_suggestions',
            template="""
            Generate search improvement suggestions based on the query and results.
            
            Query: {query}
            Results Count: {results_count}
            Average Relevance: {avg_relevance}
            Categories Found: {categories}
            
            Provide suggestions for:
            1. Query refinements (if results are poor)
            2. Related queries (alternative searches)
            3. Missing coverage areas
            
            Respond with JSON:
            {{
                "refinements": ["refined query 1", "refined query 2"],
                "related_queries": ["related query 1", "related query 2"],
                "missing_areas": ["missing area 1", "missing area 2"]
            }}
            """.strip()
        )
        
        return templates
