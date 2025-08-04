"""
Database models for clause storage and management.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from .base import BaseModel


class Document(BaseModel):
    """Document model for storing policy documents."""
    
    __tablename__ = "documents"
    
    # Basic document information
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    document_type = Column(String(100), nullable=False, index=True)  # policy, regulation, contract, etc.
    source_url = Column(String(1000))
    version = Column(String(50), default="1.0")
    language = Column(String(10), default="en", index=True)
    
    # Document metadata
    author = Column(String(200))
    organization = Column(String(200), index=True)
    publication_date = Column(DateTime)
    effective_date = Column(DateTime)
    expiration_date = Column(DateTime)
    
    # Document status
    status = Column(String(50), default="active", index=True)  # active, draft, archived, superseded
    confidence_score = Column(Float, default=1.0)  # Overall document trust score
    
    # Document content
    raw_content = Column(Text)  # Original document text
    processed_content = Column(Text)  # Cleaned and processed text
    
    # Metadata and tags
    metadata = Column(JSON, default=dict)
    tags = Column(ARRAY(String), default=list)
    
    # Relationships
    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_type_status", "document_type", "status"),
        Index("idx_document_org_type", "organization", "document_type"),
        Index("idx_document_dates", "publication_date", "effective_date"),
    )


class Clause(BaseModel):
    """Clause model for storing individual clauses from documents."""
    
    __tablename__ = "clauses"
    
    # Basic clause information
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    clause_number = Column(String(50), index=True)  # e.g., "3.2.1", "Article 5"
    title = Column(String(500))
    content = Column(Text, nullable=False)
    
    # Clause classification
    clause_type = Column(String(100), index=True)  # coverage, exclusion, procedure, definition, etc.
    category = Column(String(100), index=True)  # medical, dental, vision, etc.
    subcategory = Column(String(100), index=True)
    
    # Position in document
    section = Column(String(200))  # Section name or number
    page_number = Column(Integer)
    paragraph_number = Column(Integer)
    start_position = Column(Integer)  # Character position in document
    end_position = Column(Integer)
    
    # Content analysis
    word_count = Column(Integer)
    complexity_score = Column(Float)  # Readability/complexity score
    ambiguity_score = Column(Float)  # Measure of ambiguity in language
    
    # Semantic information
    keywords = Column(ARRAY(String), default=list)
    entities = Column(JSON, default=dict)  # Named entities extracted
    summary = Column(Text)  # AI-generated summary
    
    # Relationships and references
    parent_clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id"), index=True)
    related_clause_ids = Column(ARRAY(UUID), default=list)
    
    # Processing metadata
    processing_version = Column(String(20), default="1.0")
    last_processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Confidence and quality scores
    confidence_score = Column(Float, default=1.0)
    quality_score = Column(Float, default=1.0)
    
    # Relationships
    document = relationship("Document", back_populates="clauses")
    parent_clause = relationship("Clause", remote_side="Clause.id", backref="child_clauses")
    embeddings = relationship("ClauseEmbedding", back_populates="clause", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_clause_document_type", "document_id", "clause_type"),
        Index("idx_clause_category", "category", "subcategory"),
        Index("idx_clause_content_text", "content"),  # Full-text search index
        Index("idx_clause_scores", "confidence_score", "quality_score"),
    )


class ClauseEmbedding(BaseModel):
    """Model for storing clause embeddings for vector search."""
    
    __tablename__ = "clause_embeddings"
    
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id"), nullable=False, index=True)
    embedding_model = Column(String(100), nullable=False, index=True)
    embedding_version = Column(String(20), default="1.0")
    
    # Embedding data (stored as JSON array for flexibility)
    embedding_vector = Column(JSON, nullable=False)  # Array of float values
    embedding_dimension = Column(Integer, nullable=False)
    
    # Chunk information (for long clauses that are split)
    chunk_index = Column(Integer, default=0)
    chunk_text = Column(Text)  # The text that was embedded
    
    # Processing metadata
    created_with_model = Column(String(100))  # Model used for embedding
    processing_time_ms = Column(Integer)  # Time taken to generate embedding
    
    # Relationships
    clause = relationship("Clause", back_populates="embeddings")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("clause_id", "embedding_model", "chunk_index", name="uq_clause_embedding"),
        Index("idx_embedding_model_version", "embedding_model", "embedding_version"),
    )


class ClauseRelationship(BaseModel):
    """Model for storing relationships between clauses."""
    
    __tablename__ = "clause_relationships"
    
    source_clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id"), nullable=False, index=True)
    target_clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id"), nullable=False, index=True)
    
    relationship_type = Column(String(50), nullable=False, index=True)
    # Types: references, contradicts, supports, requires, excludes, implies, etc.
    
    confidence_score = Column(Float, default=1.0)
    strength_score = Column(Float, default=1.0)  # Strength of the relationship
    
    # Context information
    context = Column(Text)  # Description of the relationship
    evidence = Column(JSON, default=dict)  # Evidence supporting the relationship
    
    # Automatic detection metadata
    detected_by = Column(String(100))  # Agent or method that detected this relationship
    detection_confidence = Column(Float, default=1.0)
    manually_verified = Column(Boolean, default=False)
    
    # Relationships
    source_clause = relationship("Clause", foreign_keys=[source_clause_id])
    target_clause = relationship("Clause", foreign_keys=[target_clause_id])
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("source_clause_id", "target_clause_id", "relationship_type", name="uq_clause_relationship"),
        Index("idx_relationship_type_confidence", "relationship_type", "confidence_score"),
    )


class ClauseVersion(BaseModel):
    """Model for tracking clause version history."""
    
    __tablename__ = "clause_versions"
    
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id"), nullable=False, index=True)
    version_number = Column(String(20), nullable=False)
    
    # Versioned content
    content = Column(Text, nullable=False)
    title = Column(String(500))
    metadata = Column(JSON, default=dict)
    
    # Change information
    change_type = Column(String(50))  # created, updated, deleted, restored
    change_description = Column(Text)
    changed_by = Column(String(200))  # User or system that made the change
    change_reason = Column(String(500))
    
    # Previous version reference
    previous_version_id = Column(UUID(as_uuid=True), ForeignKey("clause_versions.id"))
    
    # Relationships
    clause = relationship("Clause")
    previous_version = relationship("ClauseVersion", remote_side="ClauseVersion.id")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("clause_id", "version_number", name="uq_clause_version"),
        Index("idx_clause_version_change", "clause_id", "change_type", "created_at"),
    )
