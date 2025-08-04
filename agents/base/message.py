"""
Inter-agent messaging system for communication between agents.
Provides structured message passing with type safety and validation.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    
    # Control messages
    INITIALIZE = "initialize"
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    
    # Data messages
    QUERY = "query"
    RESPONSE = "response"
    DATA = "data"
    RESULT = "result"
    
    # Coordination messages
    REQUEST = "request"
    ACKNOWLEDGE = "acknowledge"
    ERROR = "error"
    STATUS = "status"
    
    # Meta-agent messages
    CRITIQUE = "critique"
    REFINEMENT = "refinement"
    CONSENSUS = "consensus"
    FEEDBACK = "feedback"


class MessagePriority(str, Enum):
    """Priority levels for message processing."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessage(BaseModel):
    """Base message class for inter-agent communication."""
    
    # Message identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Routing information
    from_agent_id: str
    from_agent_type: str
    to_agent_id: Optional[str] = None  # None for broadcast
    to_agent_type: Optional[str] = None
    
    # Message content
    content: Dict[str, Any] = Field(default_factory=dict)
    payload: Optional[Any] = None
    
    # Message metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Conversation tracking
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    
    # Processing metadata
    processed: bool = False
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @validator('expires_at')
    def validate_expiration(cls, v, values):
        if v and v <= values.get('created_at', datetime.utcnow()):
            raise ValueError("Expiration time must be in the future")
        return v
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries
    
    def mark_processed(self, success: bool = True, error: Optional[str] = None):
        """Mark message as processed."""
        self.processed = True
        self.processed_at = datetime.utcnow()
        if not success and error:
            self.error_message = error
    
    def create_response(
        self,
        response_type: MessageType,
        content: Dict[str, Any],
        payload: Optional[Any] = None
    ) -> "AgentMessage":
        """Create a response message to this message."""
        return AgentMessage(
            type=response_type,
            from_agent_id=self.to_agent_id or "unknown",
            from_agent_type=self.to_agent_type or "unknown",
            to_agent_id=self.from_agent_id,
            to_agent_type=self.from_agent_type,
            content=content,
            payload=payload,
            conversation_id=self.conversation_id,
            parent_message_id=self.id
        )


class QueryMessage(AgentMessage):
    """Specialized message for query processing."""
    
    type: Literal[MessageType.QUERY] = MessageType.QUERY
    query_text: str = Field(..., min_length=1, max_length=10000)
    query_context: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ResponseMessage(AgentMessage):
    """Specialized message for agent responses."""
    
    type: Literal[MessageType.RESPONSE] = MessageType.RESPONSE
    answer: str = Field(..., min_length=1)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(AgentMessage):
    """Specialized message for error reporting."""
    
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    error_code: str
    error_message: str
    error_details: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True


class CritiqueMessage(AgentMessage):
    """Specialized message for meta-agent critiques."""
    
    type: Literal[MessageType.CRITIQUE] = MessageType.CRITIQUE
    target_response: str  # The response being critiqued
    critique_points: List[Dict[str, Any]]  # Specific critique points
    suggested_improvements: List[str]
    confidence_assessment: Dict[str, float]
    alternative_interpretations: List[str] = Field(default_factory=list)


class MessageBus:
    """Message bus for routing messages between agents."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 10000
    
    def subscribe(self, agent_type: str, handler: callable):
        """Subscribe an agent to receive messages."""
        if agent_type not in self.subscribers:
            self.subscribers[agent_type] = []
        self.subscribers[agent_type].append(handler)
        logger.info("Agent subscribed to message bus", agent_type=agent_type)
    
    def unsubscribe(self, agent_type: str, handler: callable):
        """Unsubscribe an agent from messages."""
        if agent_type in self.subscribers:
            try:
                self.subscribers[agent_type].remove(handler)
                logger.info("Agent unsubscribed from message bus", agent_type=agent_type)
            except ValueError:
                logger.warning("Handler not found for unsubscription", agent_type=agent_type)
    
    async def publish(self, message: AgentMessage) -> bool:
        """Publish a message to interested subscribers."""
        try:
            # Add to history
            self._add_to_history(message)
            
            # Check if message is expired
            if message.is_expired():
                logger.warning("Attempting to publish expired message", message_id=message.id)
                return False
            
            # Determine target subscribers
            targets = []
            if message.to_agent_type:
                # Direct message to specific agent type
                if message.to_agent_type in self.subscribers:
                    targets = self.subscribers[message.to_agent_type]
            else:
                # Broadcast to all subscribers
                for handlers in self.subscribers.values():
                    targets.extend(handlers)
            
            if not targets:
                logger.warning(
                    "No subscribers found for message",
                    message_id=message.id,
                    target_type=message.to_agent_type
                )
                return False
            
            # Deliver message to all targets
            success_count = 0
            for handler in targets:
                try:
                    await handler(message)
                    success_count += 1
                except Exception as e:
                    logger.error(
                        "Error delivering message to handler",
                        message_id=message.id,
                        handler=str(handler),
                        error=str(e),
                        exc_info=True
                    )
            
            logger.info(
                "Message published",
                message_id=message.id,
                message_type=message.type,
                delivered_to=success_count,
                total_targets=len(targets)
            )
            
            return success_count > 0
            
        except Exception as e:
            logger.error(
                "Error publishing message",
                message_id=message.id,
                error=str(e),
                exc_info=True
            )
            return False
    
    def _add_to_history(self, message: AgentMessage):
        """Add message to history with size management."""
        self.message_history.append(message)
        
        # Trim history if too large
        if len(self.message_history) > self.max_history_size:
            # Remove oldest messages
            self.message_history = self.message_history[-self.max_history_size:]
    
    def get_conversation_history(self, conversation_id: str) -> List[AgentMessage]:
        """Get all messages for a specific conversation."""
        return [
            msg for msg in self.message_history
            if msg.conversation_id == conversation_id
        ]
    
    def get_agent_messages(
        self,
        agent_type: str,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get recent messages for a specific agent type."""
        messages = [
            msg for msg in self.message_history
            if msg.from_agent_type == agent_type or msg.to_agent_type == agent_type
        ]
        return messages[-limit:]


# Global message bus instance
message_bus = MessageBus()
