"""
Abstract base agent class providing common functionality for all agents.
Defines the interface and shared behavior for the multi-agent system.
"""
import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable
from enum import Enum

from pydantic import BaseModel
import structlog

from .message import AgentMessage, MessageType, MessagePriority, message_bus
from core.exceptions import AgentError, AgentTimeoutError, AgentCommunicationError
from core.config import settings

logger = structlog.get_logger(__name__)


class AgentState(str, Enum):
    """Possible states for an agent."""
    
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    STOPPED = "stopped"


class AgentCapability(BaseModel):
    """Description of an agent's capability."""
    
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    confidence_level: float = 1.0
    estimated_processing_time: Optional[float] = None  # seconds


class AgentMetrics(BaseModel):
    """Runtime metrics for an agent."""
    
    messages_processed: int = 0
    messages_sent: int = 0
    processing_time_total: float = 0.0
    processing_time_average: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_activity: Optional[datetime] = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type or self.__class__.__name__
        self.config = config or {}
        
        # Agent state
        self.state = AgentState.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        
        # Capabilities
        self.capabilities: List[AgentCapability] = []
        
        # Metrics
        self.metrics = AgentMetrics()
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # Dependencies
        self.dependencies: Set[str] = set()
        self.dependents: Set[str] = set()
        
        # Callbacks
        self.on_message_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        self.on_state_change_callbacks: List[Callable] = []
        
        logger.info(
            "Agent created",
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )
    
    async def initialize(self) -> bool:
        """Initialize the agent and prepare for processing."""
        try:
            logger.info("Initializing agent", agent_id=self.agent_id, agent_type=self.agent_type)
            
            # Set state
            await self._set_state(AgentState.INITIALIZING)
            
            # Subscribe to message bus
            message_bus.subscribe(self.agent_type, self._handle_message)
            
            # Initialize capabilities
            self.capabilities = await self._initialize_capabilities()
            
            # Perform agent-specific initialization
            await self._initialize()
            
            # Start message processing
            self.processing_task = asyncio.create_task(self._message_processing_loop())
            
            # Set state to idle
            await self._set_state(AgentState.IDLE)
            
            logger.info(
                "Agent initialized successfully",
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                capabilities=len(self.capabilities)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Agent initialization failed",
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                error=str(e),
                exc_info=True
            )
            await self._set_state(AgentState.ERROR)
            return False
    
    async def shutdown(self):
        """Shutdown the agent gracefully."""
        logger.info("Shutting down agent", agent_id=self.agent_id, agent_type=self.agent_type)
        
        try:
            # Set stop event
            self.stop_event.set()
            
            # Cancel processing task
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # Unsubscribe from message bus
            message_bus.unsubscribe(self.agent_type, self._handle_message)
            
            # Perform agent-specific cleanup
            await self._cleanup()
            
            # Set final state
            await self._set_state(AgentState.STOPPED)
            
            logger.info("Agent shutdown complete", agent_id=self.agent_id)
            
        except Exception as e:
            logger.error(
                "Error during agent shutdown",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a single message and return response if applicable."""
        start_time = datetime.utcnow()
        
        try:
            # Update metrics
            self.metrics.messages_processed += 1
            self.metrics.last_activity = start_time
            
            # Set state to busy
            await self._set_state(AgentState.BUSY)
            
            # Validate message
            if message.is_expired():
                logger.warning(
                    "Received expired message",
                    agent_id=self.agent_id,
                    message_id=message.id
                )
                return None
            
            # Check if we can handle this message type
            if not await self._can_handle_message(message):
                logger.debug(
                    "Agent cannot handle message type",
                    agent_id=self.agent_id,
                    message_type=message.type
                )
                return None
            
            logger.debug(
                "Processing message",
                agent_id=self.agent_id,
                message_id=message.id,
                message_type=message.type
            )
            
            # Process the message (implemented by subclasses)
            response = await self._process_message(message)
            
            # Update success metrics
            self.metrics.success_count += 1
            
            # Mark message as processed
            message.mark_processed(success=True)
            
            # Trigger callbacks
            for callback in self.on_message_callbacks:
                try:
                    await callback(message, response)
                except Exception as e:
                    logger.error(
                        "Error in message callback",
                        agent_id=self.agent_id,
                        error=str(e)
                    )
            
            return response
            
        except Exception as e:
            # Update error metrics
            self.metrics.error_count += 1
            
            # Mark message as failed
            message.mark_processed(success=False, error=str(e))
            
            # Set error state
            await self._set_state(AgentState.ERROR)
            
            logger.error(
                "Error processing message",
                agent_id=self.agent_id,
                message_id=message.id,
                error=str(e),
                exc_info=True
            )
            
            # Trigger error callbacks
            for callback in self.on_error_callbacks:
                try:
                    await callback(e, message)
                except Exception as callback_error:
                    logger.error(
                        "Error in error callback",
                        agent_id=self.agent_id,
                        callback_error=str(callback_error)
                    )
            
            raise AgentError(
                f"Failed to process message: {str(e)}",
                self.agent_type,
                self.agent_id
            )
            
        finally:
            # Update timing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.processing_time_total += processing_time
            
            if self.metrics.messages_processed > 0:
                self.metrics.processing_time_average = (
                    self.metrics.processing_time_total / self.metrics.messages_processed
                )
            
            # Return to idle state if not in error
            if self.state != AgentState.ERROR:
                await self._set_state(AgentState.IDLE)
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the message bus."""
        try:
            # Set sender information
            message.from_agent_id = self.agent_id
            message.from_agent_type = self.agent_type
            
            # Publish message
            success = await message_bus.publish(message)
            
            if success:
                self.metrics.messages_sent += 1
                logger.debug(
                    "Message sent",
                    agent_id=self.agent_id,
                    message_id=message.id,
                    message_type=message.type,
                    target_type=message.to_agent_type
                )
            else:
                logger.warning(
                    "Failed to send message",
                    agent_id=self.agent_id,
                    message_id=message.id
                )
            
            return success
            
        except Exception as e:
            logger.error(
                "Error sending message",
                agent_id=self.agent_id,
                message_id=message.id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def request_response(
        self,
        target_agent_type: str,
        message: AgentMessage,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send a request message and wait for a response."""
        # Create a response future
        response_future = asyncio.Future()
        conversation_id = str(uuid.uuid4())
        
        # Set conversation ID
        message.conversation_id = conversation_id
        
        # Set up temporary callback for response
        async def response_callback(received_message: AgentMessage, _):
            if (received_message.conversation_id == conversation_id and
                received_message.parent_message_id == message.id):
                if not response_future.done():
                    response_future.set_result(received_message)
        
        self.on_message_callbacks.append(response_callback)
        
        try:
            # Send the request
            message.to_agent_type = target_agent_type
            success = await self.send_message(message)
            
            if not success:
                raise AgentCommunicationError(
                    self.agent_type,
                    target_agent_type,
                    "Failed to send request message"
                )
            
            # Wait for response
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                raise AgentTimeoutError(
                    target_agent_type,
                    timeout,
                    message.to_agent_id
                )
            
        finally:
            # Remove callback
            if response_callback in self.on_message_callbacks:
                self.on_message_callbacks.remove(response_callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "capabilities": [cap.dict() for cap in self.capabilities],
            "metrics": self.metrics.dict(),
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents)
        }
    
    def add_dependency(self, agent_type: str):
        """Add a dependency on another agent type."""
        self.dependencies.add(agent_type)
    
    def add_dependent(self, agent_type: str):
        """Add a dependent agent type."""
        self.dependents.add(agent_type)
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def _initialize(self):
        """Agent-specific initialization logic."""
        pass
    
    @abstractmethod
    async def _cleanup(self):
        """Agent-specific cleanup logic."""
        pass
    
    @abstractmethod
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a message and return response if applicable."""
        pass
    
    @abstractmethod
    async def _initialize_capabilities(self) -> List[AgentCapability]:
        """Initialize and return agent capabilities."""
        pass
    
    # Private methods
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message from message bus."""
        # Filter messages for this agent
        if (message.to_agent_id and message.to_agent_id != self.agent_id or
            message.to_agent_type and message.to_agent_type != self.agent_type):
            return
        
        # Add to processing queue
        await self.message_queue.put(message)
    
    async def _message_processing_loop(self):
        """Main message processing loop."""
        while not self.stop_event.is_set():
            try:
                # Wait for next message with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0  # Check stop event every second
                    )
                except asyncio.TimeoutError:
                    # Update heartbeat and continue
                    self.last_heartbeat = datetime.utcnow()
                    continue
                
                # Process the message
                response = await self.process_message(message)
                
                # Send response if generated
                if response:
                    await self.send_message(response)
                
            except asyncio.CancelledError:
                logger.info("Message processing loop cancelled", agent_id=self.agent_id)
                break
            except Exception as e:
                logger.error(
                    "Error in message processing loop",
                    agent_id=self.agent_id,
                    error=str(e),
                    exc_info=True
                )
                # Continue processing despite errors
                continue
    
    async def _can_handle_message(self, message: AgentMessage) -> bool:
        """Check if this agent can handle the given message type."""
        # Default implementation - can handle all message types
        # Subclasses can override for specific filtering
        return True
    
    async def _set_state(self, new_state: AgentState):
        """Set agent state and trigger callbacks."""
        old_state = self.state
        self.state = new_state
        
        logger.debug(
            "Agent state changed",
            agent_id=self.agent_id,
            old_state=old_state,
            new_state=new_state
        )
        
        # Trigger state change callbacks
        for callback in self.on_state_change_callbacks:
            try:
                await callback(old_state, new_state)
            except Exception as e:
                logger.error(
                    "Error in state change callback",
                    agent_id=self.agent_id,
                    error=str(e)
                )
