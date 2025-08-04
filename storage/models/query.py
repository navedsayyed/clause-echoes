"""
Database models for query processing and history.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from .base import BaseModel


class QuerySession(BaseModel):
    """Model for tracking query sessions and conversations."""
    
    __tablename__ = "query_sessions"
    
    # Session identification
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), index=True)  # External user identifier
    user_agent = Column(String(500))
    ip_address = Column(String(45))  # IPv6 compatible
    
    # Session metadata
    session_type = Column(String(50), default="interactive", index=True)  # interactive, api, batch
    language = Column(String(10), default="en")
    
    # Session state
    status = Column(String(50), default="active", index=True)  # active, completed, abandoned, error
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    
    # Session statistics
    total_queries = Column(Integer, default=0)
    total_responses = Column(Integer, default=0)
    avg_response_time_ms = Column(Float)
    
    # Context and preferences
    context = Column(JSON, default=dict)  # Session context and state
    preferences = Column(JSON, default=dict)  # User preferences
    
    # Relationships
    queries = relationship("Query", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_session_user_status", "user_id", "status"),
        Index("idx_session_activity", "last_activity_at", "status"),
    )


class Query(BaseModel):
    """Model for storing individual queries and their processing."""
    
    __tablename__ = "queries"
    
    # Query identification
    session_id = Column(UUID(as_uuid=True), ForeignKey("query_sessions.id"), nullable=False, index=True)
    query_sequence = Column(Integer, nullable=False)  # Order within session
    
    # Original query
    original_text = Column(Text, nullable=False)
    language = Column(String(10), default="en")
    
    # Processed query
    parsed_query = Column(JSON, default=dict)  # Structured query representation
    normalized_text = Column(Text)  # Cleaned and normalized query text
    intent = Column(String(100), index=True)  # Detected intent
    entities = Column(JSON, default=dict)  # Extracted entities
    
    # Query classification
    query_type = Column(String(50), index=True)  # factual, procedural, comparison, etc.
    complexity_level = Column(String(20), index=True)  # simple, medium, complex
    domain = Column(String(100), index=True)  # medical, legal, technical, etc.
    
    # Processing metadata
    processing_start = Column(DateTime, default=datetime.utcnow)
    processing_end = Column(DateTime)
    processing_time_ms = Column(Integer)
    
    # Query state
    status = Column(String(50), default="processing", index=True)  # processing, completed, failed, timeout
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Context
    context_used = Column(JSON, default=dict)  # Context available during processing
    follow_up_to = Column(UUID(as_uuid=True), ForeignKey("queries.id"))  # Previous query this follows up on
    
    # Relationships
    session = relationship("QuerySession", back_populates="queries")
    parent_query = relationship("Query", remote_side="Query.id", backref="follow_up_queries")
    agent_executions = relationship("AgentExecution", back_populates="query", cascade="all, delete-orphan")
    responses = relationship("QueryResponse", back_populates="query", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_query_session_sequence", "session_id", "query_sequence"),
        Index("idx_query_type_domain", "query_type", "domain"),
        Index("idx_query_status_created", "status", "created_at"),
    )


class AgentExecution(BaseModel):
    """Model for tracking agent executions within query processing."""
    
    __tablename__ = "agent_executions"
    
    # Execution identification
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False, index=True)
    agent_type = Column(String(100), nullable=False, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    execution_order = Column(Integer, nullable=False)
    
    # Parent execution (for nested agent calls)
    parent_execution_id = Column(UUID(as_uuid=True), ForeignKey("agent_executions.id"))
    
    # Execution details
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    execution_time_ms = Column(Integer)
    
    # Agent input/output
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    intermediate_steps = Column(JSON, default=list)  # Step-by-step execution log
    
    # Execution status
    status = Column(String(50), default="running", index=True)  # running, completed, failed, timeout, cancelled
    error_message = Column(Text)
    error_type = Column(String(100))
    
    # Resource usage
    tokens_used = Column(Integer, default=0)
    api_calls_made = Column(Integer, default=0)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    
    # Quality metrics
    confidence_score = Column(Float)
    quality_score = Column(Float)
    
    # Relationships
    query = relationship("Query", back_populates="agent_executions")
    parent_execution = relationship("AgentExecution", remote_side="AgentExecution.id", backref="child_executions")
    
    # Indexes
    __table_args__ = (
        Index("idx_agent_exec_query_order", "query_id", "execution_order"),
        Index("idx_agent_exec_type_status", "agent_type", "status"),
        Index("idx_agent_exec_timing", "started_at", "completed_at"),
    )


class QueryResponse(BaseModel):
    """Model for storing query responses and their metadata."""
    
    __tablename__ = "query_responses"
    
    # Response identification
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False, index=True)
    response_version = Column(Integer, default=1)  # For iterative refinement
    
    # Response content
    answer_text = Column(Text, nullable=False)
    explanation = Column(Text)  # Detailed explanation of the answer
    summary = Column(Text)  # Brief summary
    
    # Response metadata
    response_type = Column(String(50), index=True)  # direct, synthesized, uncertain, multi_path
    confidence_score = Column(Float)
    uncertainty_level = Column(String(20))  # low, medium, high
    
    # Sources and evidence
    primary_sources = Column(JSON, default=list)  # Main clause IDs used
    supporting_sources = Column(JSON, default=list)  # Additional supporting evidence
    contradicting_sources = Column(JSON, default=list)  # Contradictory information found
    
    # Quality metrics
    completeness_score = Column(Float)  # How complete is the answer
    accuracy_score = Column(Float)  # Estimated accuracy
    relevance_score = Column(Float)  # How relevant to the original query
    
    # Alternative responses
    alternative_interpretations = Column(JSON, default=list)  # Other possible answers
    assumptions_made = Column(JSON, default=list)  # Assumptions in the response
    limitations = Column(JSON, default=list)  # Known limitations of the response
    
    # Follow-up suggestions
    suggested_clarifications = Column(JSON, default=list)  # Questions to clarify the query
    related_topics = Column(JSON, default=list)  # Related topics user might explore
    
    # Processing metadata
    generated_at = Column(DateTime, default=datetime.utcnow)
    generation_time_ms = Column(Integer)
    total_tokens_used = Column(Integer)
    
    # Response status
    status = Column(String(50), default="final", index=True)  # draft, reviewed, final, superseded
    reviewed_by = Column(String(100))  # Who reviewed the response
    review_notes = Column(Text)
    
    # Relationships
    query = relationship("Query", back_populates="responses")
    feedback_entries = relationship("ResponseFeedback", back_populates="response", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_response_query_version", "query_id", "response_version"),
        Index("idx_response_confidence", "confidence_score", "status"),
        Index("idx_response_generated", "generated_at", "status"),
    )


class ResponseFeedback(BaseModel):
    """Model for storing user feedback on responses."""
    
    __tablename__ = "response_feedback"
    
    # Feedback identification
    response_id = Column(UUID(as_uuid=True), ForeignKey("query_responses.id"), nullable=False, index=True)
    user_id = Column(String(100), index=True)
    feedback_type = Column(String(50), nullable=False, index=True)  # rating, correction, suggestion, etc.
    
    # Feedback content
    rating = Column(Integer)  # 1-5 scale
    feedback_text = Column(Text)
    specific_issues = Column(JSON, default=list)  # Specific problems identified
    
    # Feedback categories
    accuracy_rating = Column(Integer)  # Accuracy-specific rating
    completeness_rating = Column(Integer)  # Completeness-specific rating
    clarity_rating = Column(Integer)  # Clarity-specific rating
    usefulness_rating = Column(Integer)  # Usefulness-specific rating
    
    # Correction data
    suggested_correction = Column(Text)
    corrected_sources = Column(JSON, default=list)
    
    # Feedback metadata
    feedback_source = Column(String(50), default="user")  # user, expert, system
    verification_status = Column(String(50), default="unverified")  # unverified, verified, disputed
    
    # Processing status
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)
    action_taken = Column(String(100))  # Action taken based on feedback
    
    # Relationships
    response = relationship("QueryResponse", back_populates="feedback_entries")
    
    # Indexes
    __table_args__ = (
        Index("idx_feedback_response_type", "response_id", "feedback_type"),
        Index("idx_feedback_rating", "rating", "feedback_type"),
        Index("idx_feedback_processed", "processed", "created_at"),
    )
