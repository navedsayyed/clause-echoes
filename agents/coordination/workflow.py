"""
Workflow Orchestration - Coordinates multi-agent workflows and manages execution flow.
Handles complex agent interactions, dependency management, and workflow execution.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid

import structlog
from pydantic import BaseModel, Field

from agents.base.message import AgentMessage, MessageType, message_bus
from agents.base.agent import BaseAgent, AgentState
from core.exceptions import AgentError, AgentTimeoutError
from core.config import settings

logger = structlog.get_logger(__name__)


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Status of individual tasks."""
    
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowTask(BaseModel):
    """Individual task within a workflow."""
    
    task_id: str
    agent_type: str
    task_type: str
    
    # Task configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30
    retry_count: int = 3
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # Task IDs this depends on
    blocks: List[str] = Field(default_factory=list)      # Task IDs this blocks
    
    # Execution state
    status: TaskStatus = TaskStatus.WAITING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Execution metadata
    attempts: int = 0
    execution_time: float = 0.0
    assigned_agent: Optional[str] = None


class WorkflowDefinition(BaseModel):
    """Definition of a complete workflow."""
    
    workflow_id: str
    name: str
    description: str
    version: str = "1.0"
    
    # Tasks and dependencies
    tasks: List[WorkflowTask]
    
    # Workflow configuration
    max_execution_time: int = 300  # seconds
    allow_parallel_execution: bool = True
    fail_fast: bool = False  # Stop on first failure
    
    # Input/output
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)


class WorkflowExecution(BaseModel):
    """Runtime execution state of a workflow."""
    
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    
    # Execution context
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Task execution tracking
    task_executions: Dict[str, WorkflowTask] = Field(default_factory=dict)
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_phase: str = "initialization"
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Callbacks
    on_progress_callbacks: List[Callable] = Field(default_factory=list)
    on_completion_callbacks: List[Callable] = Field(default_factory=list)


class WorkflowOrchestrator:
    """Orchestrates complex multi-agent workflows."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Workflow storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        
        # Agent registry
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Execution control
        self.max_concurrent_workflows = self.config.get('max_concurrent', 10)
        self.default_timeout = self.config.get('default_timeout', 30)
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        
        # Task queue and scheduler
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.scheduler_task: Optional[asyncio.Task] = None
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Workflow Orchestrator initialized")
    
    async def initialize(self):
        """Initialize the workflow orchestrator."""
        # Start the task scheduler
        self.scheduler_task = asyncio.create_task(self._task_scheduler_loop())
        
        # Load predefined workflows
        await self._load_predefined_workflows()
        
        logger.info("Workflow orchestrator initialized successfully")
    
    async def cleanup(self):
        """Cleanup orchestrator resources."""
        # Cancel scheduler
        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active executions
        for execution_id, task in self.execution_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Workflow orchestrator cleanup complete")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.registered_agents[agent.agent_type] = agent
        self.agent_capabilities[agent.agent_type] = [cap.name for cap in agent.capabilities]
        
        logger.info(
            "Agent registered with orchestrator",
            agent_type=agent.agent_type,
            capabilities=len(agent.capabilities)
        )
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition."""
        self.workflow_definitions[workflow.workflow_id] = workflow
        
        logger.info(
            "Workflow registered",
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            tasks=len(workflow.tasks)
        )
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute a workflow asynchronously."""
        
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if len(self.active_executions) >= self.max_concurrent_workflows:
            raise RuntimeError("Maximum concurrent workflows reached")
        
        # Create execution instance
        execution = WorkflowExecution(
            execution_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            input_data=input_data,
            context=context or {}
        )
        
        # Initialize task executions from workflow definition
        workflow_def = self.workflow_definitions[workflow_id]
        execution.task_executions = {
            task.task_id: task.copy() for task in workflow_def.tasks
        }
        
        # Register execution
        self.active_executions[execution.execution_id] = execution
        
        # Start execution task
        execution_task = asyncio.create_task(
            self._execute_workflow_internal(execution)
        )
        self.execution_tasks[execution.execution_id] = execution_task
        
        logger.info(
            "Workflow execution started",
            execution_id=execution.execution_id,
            workflow_id=workflow_id
        )
        
        return execution
    
    async def _execute_workflow_internal(self, execution: WorkflowExecution):
        """Internal workflow execution logic."""
        workflow_def = self.workflow_definitions[execution.workflow_id]
        
        try:
            # Update execution status
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.utcnow()
            execution.current_phase = "execution"
            
            # Execute tasks based on dependencies
            await self._execute_task_dag(execution, workflow_def)
            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            execution.progress_percentage = 100.0
            execution.current_phase = "completed"
            
            logger.info(
                "Workflow execution completed",
                execution_id=execution.execution_id,
                execution_time=execution.execution_time
            )
            
        except Exception as e:
            # Mark as failed
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.error_details = {
                'exception_type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
            execution.current_phase = "failed"
            
            logger.error(
                "Workflow execution failed",
                execution_id=execution.execution_id,
                error=str(e),
                exc_info=True
            )
            
        finally:
            # Trigger completion callbacks
            for callback in execution.on_completion_callbacks:
                try:
                    await callback(execution)
                except Exception as e:
                    logger.warning("Completion callback failed", error=str(e))
            
            # Cleanup
            if execution.execution_id in self.execution_tasks:
                del self.execution_tasks[execution.execution_id]
    
    async def _execute_task_dag(self, execution: WorkflowExecution, workflow_def: WorkflowDefinition):
        """Execute tasks according to their dependency DAG."""
        
        # Build dependency graph
        task_graph = self._build_task_graph(execution.task_executions)
        
        # Find tasks ready to execute (no dependencies)
        ready_tasks = self._find_ready_tasks(execution.task_executions)
        
        while ready_tasks or any(task.status == TaskStatus.RUNNING for task in execution.task_executions.values()):
            
            # Start ready tasks
            task_futures = []
            for task_id in ready_tasks:
                if task_id not in execution.task_executions:
                    continue
                    
                task = execution.task_executions[task_id]
                if task.status == TaskStatus.READY:
                    task_future = asyncio.create_task(
                        self._execute_single_task(execution, task)
                    )
                    task_futures.append((task_id, task_future))
            
            # Wait for at least one task to complete
            if task_futures:
                done, pending = await asyncio.wait(
                    [future for _, future in task_futures],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0  # Check periodically
                )
                
                # Process completed tasks
                for task_id, future in task_futures:
                    if future in done:
                        try:
                            await future  # Get result or raise exception
                            execution.completed_tasks.append(task_id)
                        except Exception as e:
                            execution.failed_tasks.append(task_id)
                            if workflow_def.fail_fast:
                                raise
            
            # Update progress
            total_tasks = len(execution.task_executions)
            completed_tasks = len(execution.completed_tasks) + len(execution.failed_tasks)
            execution.progress_percentage = (completed_tasks / total_tasks) * 100
            
            # Find newly ready tasks
            ready_tasks = self._find_ready_tasks(execution.task_executions)
            
            # Check for deadlock or completion
            running_tasks = [t for t in execution.task_executions.values() if t.status == TaskStatus.RUNNING]
            if not ready_tasks and not running_tasks:
                # All done or deadlocked
                break
            
            # Brief pause to prevent tight loop
            await asyncio.sleep(0.1)
    
    async def _execute_single_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        task.attempts += 1
        
        try:
            # Find appropriate agent
            if task.agent_type not in self.registered_agents:
                raise AgentError(f"Agent type {task.agent_type} not available", task.agent_type)
            
            agent = self.registered_agents[task.agent_type]
            task.assigned_agent = agent.agent_id
            
            # Prepare task message
            task_message = AgentMessage(
                type=MessageType.REQUEST,
                from_agent_id="workflow_orchestrator",
                from_agent_type="WorkflowOrchestrator",
                to_agent_type=task.agent_type,
                content={
                    'request_type': task.task_type,
                    'parameters': task.parameters,
                    'execution_context': execution.context,
                    'task_id': task.task_id
                },
                conversation_id=execution.execution_id
            )
            
            # Execute task with timeout
            response = await asyncio.wait_for(
                agent.request_response(
                    task.agent_type,
                    task_message,
                    timeout=task.timeout_seconds
                ),
                timeout=task.timeout_seconds
            )
            
            # Process response
            if response and response.content.get('success', False):
                task.status = TaskStatus.COMPLETED
                task.result = response.content
                task.completed_at = datetime.utcnow()
                task.execution_time = (task.completed_at - task.started_at).total_seconds()
                
                logger.info(
                    "Task completed successfully",
                    task_id=task.task_id,
                    execution_time=task.execution_time
                )
                
            else:
                error_msg = response.content.get('error', 'Unknown error') if response else 'No response received'
                raise AgentError(f"Task failed: {error_msg}", task.agent_type)
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error_message = f"Task timed out after {task.timeout_seconds} seconds"
            
            if task.attempts < task.retry_count:
                logger.warning(
                    "Task timed out, will retry",
                    task_id=task.task_id,
                    attempt=task.attempts
                )
                task.status = TaskStatus.READY  # Will be retried
            else:
                logger.error("Task failed after max retries", task_id=task.task_id)
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            if task.attempts < task.retry_count:
                logger.warning(
                    "Task failed, will retry",
                    task_id=task.task_id,
                    attempt=task.attempts,
                    error=str(e)
                )
                task.status = TaskStatus.READY  # Will be retried
            else:
                logger.error(
                    "Task failed after max retries",
                    task_id=task.task_id,
                    error=str(e)
                )
    
    def _build_task_graph(self, tasks: Dict[str, WorkflowTask]) -> Dict[str, Set[str]]:
        """Build task dependency graph."""
        graph = {}
        
        for task_id, task in tasks.items():
            graph[task_id] = set(task.depends_on)
        
        return graph
    
    def _find_ready_tasks(self, tasks: Dict[str, WorkflowTask]) -> List[str]:
        """Find tasks that are ready to execute."""
        ready = []
        
        for task_id, task in tasks.items():
            if task.status in [TaskStatus.WAITING, TaskStatus.READY]:
                # Check if all dependencies are completed
                dependencies_met = all(
                    tasks[dep_id].status == TaskStatus.COMPLETED 
                    for dep_id in task.depends_on
                    if dep_id in tasks
                )
                
                if dependencies_met:
                    task.status = TaskStatus.READY
                    ready.append(task_id)
        
        return ready
    
    async def _task_scheduler_loop(self):
        """Main task scheduler loop."""
        while True:
            try:
                # Process any queued tasks
                try:
                    task_item = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    # Process task item
                    await self._process_queued_task(task_item)
                except asyncio.TimeoutError:
                    # No tasks queued, continue
                    pass
                
                # Cleanup completed executions
                await self._cleanup_completed_executions()
                
                # Brief pause
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                logger.info("Task scheduler cancelled")
                break
            except Exception as e:
                logger.error("Error in task scheduler", error=str(e), exc_info=True)
                await asyncio.sleep(5)  # Back off on error
    
    async def _process_queued_task(self, task_item: Dict[str, Any]):
        """Process a queued task."""
        # Implementation for processing queued tasks
        pass
    
    async def _cleanup_completed_executions(self):
        """Cleanup old completed executions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)  # Keep for 1 hour
        
        completed_executions = [
            exec_id for exec_id, execution in self.active_executions.items()
            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
            and execution.completed_at
            and execution.completed_at < cutoff_time
        ]
        
        for exec_id in completed_executions:
            del self.active_executions[exec_id]
            logger.debug("Cleaned up completed execution", execution_id=exec_id)
    
    async def _load_predefined_workflows(self):
        """Load predefined workflow templates."""
        # Comprehensive Query Processing Workflow
        query_processing_workflow = WorkflowDefinition(
            workflow_id="comprehensive_query_processing",
            name="Comprehensive Query Processing",
            description="Complete pipeline for processing user queries with self-critique and consensus",
            tasks=[
                WorkflowTask(
                    task_id="parse_query",
                    agent_type="QueryParserAgent",
                    task_type="parse_query",
                    parameters={"enable_entity_extraction": True, "enable_intent_detection": True}
                ),
                WorkflowTask(
                    task_id="retrieve_clauses",
                    agent_type="ClauseRetrieverAgent",
                    task_type="search_clauses",
                    depends_on=["parse_query"],
                    parameters={"search_type": "hybrid", "max_results": 20}
                ),
                WorkflowTask(
                    task_id="check_consistency",
                    agent_type="ConsistencyCheckerAgent",
                    task_type="check_contradictions",
                    depends_on=["retrieve_clauses"],
                    parameters={"analysis_level": "practical"}
                ),
                WorkflowTask(
                    task_id="evaluate_logic",
                    agent_type="LogicEvaluatorAgent",
                    task_type="evaluate_logic",
                    depends_on=["retrieve_clauses"],
                    parameters={"reasoning_type": "deductive"}
                ),
                WorkflowTask(
                    task_id="synthesize_response",
                    agent_type="SynthesisAgent",
                    task_type="synthesize_response",
                    depends_on=["retrieve_clauses", "check_consistency", "evaluate_logic"],
                    parameters={"strategy": "hierarchical"}
                ),
                WorkflowTask(
                    task_id="self_critique",
                    agent_type="SelfCriticAgent",
                    task_type="critique_response",
                    depends_on=["synthesize_response"],
                    parameters={"critique_types": ["accuracy", "completeness", "clarity"]}
                ),
                WorkflowTask(
                    task_id="analyze_uncertainty",
                    agent_type="UncertaintyAnalyzerAgent",
                    task_type="analyze_uncertainty",
                    depends_on=["synthesize_response"],
                    parameters={"enable_sensitivity_analysis": True}
                ),
                WorkflowTask(
                    task_id="detect_assumptions",
                    agent_type="AssumptionDetectorAgent",
                    task_type="detect_assumptions",
                    depends_on=["synthesize_response"],
                    parameters={"sensitivity": 0.7}
                ),
                WorkflowTask(
                    task_id="build_consensus",
                    agent_type="ConsensusEngineAgent",
                    task_type="build_consensus",
                    depends_on=["self_critique", "analyze_uncertainty", "detect_assumptions"],
                    parameters={"method": "weighted_voting"}
                )
            ],
            max_execution_time=300,
            allow_parallel_execution=True,
            fail_fast=False
        )
        
        self.register_workflow(query_processing_workflow)
        
        logger.info("Predefined workflows loaded successfully")
    
    # Public API methods
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get the status of a workflow execution."""
        return self.active_executions.get(execution_id)
    
    def list_active_executions(self) -> List[str]:
        """List all active execution IDs."""
        return list(self.active_executions.keys())
    
    def list_available_workflows(self) -> List[str]:
        """List all available workflow IDs."""
        return list(self.workflow_definitions.keys())
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        
        # Cancel execution task if running
        if execution_id in self.execution_tasks:
            task = self.execution_tasks[execution_id]
            if not task.done():
                task.cancel()
        
        logger.info("Workflow execution cancelled", execution_id=execution_id)
        return True
