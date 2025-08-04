"""
Main System Orchestrator - Coordinates the entire multi-agent ecosystem.
Manages agent initialization, workflow execution, and system-wide coordination.
"""
import asyncio
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import structlog

from agents.base.agent import BaseAgent
from agents.base.message import message_bus, AgentMessage, MessageType
from agents.coordination.workflow import WorkflowOrchestrator
from agents.micro_agents.query_parser import QueryParserAgent
from agents.micro_agents.clause_retriever import ClauseRetrieverAgent
from agents.micro_agents.consistency_checker import ConsistencyCheckerAgent
from agents.micro_agents.logic_evaluator import LogicEvaluatorAgent
from agents.micro_agents.synthesis_agent import SynthesisAgent
from agents.micro_agents.hypothesis_generator import HypothesisGeneratorAgent
from agents.micro_agents.curriculum_agent import CurriculumAgent
from agents.meta_agents.self_critic import SelfCriticAgent
from agents.meta_agents.uncertainty_analyzer import UncertaintyAnalyzerAgent
from agents.meta_agents.assumption_detector import AssumptionDetectorAgent
from agents.meta_agents.consensus_engine import ConsensusEngineAgent
from core.exceptions import AgentError
from core.config import settings

logger = structlog.get_logger(__name__)


class SystemStatus:
    """System status tracking."""
    
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class AgentOrchestrator:
    """Main orchestrator for the Clause Echoes multi-agent system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # System state
        self.status = SystemStatus.INITIALIZING
        self.initialized_at: Optional[datetime] = None
        self.agents: Dict[str, BaseAgent] = {}
        
        # Core components
        self.workflow_orchestrator: Optional[WorkflowOrchestrator] = None
        
        # System metrics
        self.total_queries_processed = 0
        self.average_response_time = 0.0
        self.system_uptime = 0.0
        
        logger.info("Agent Orchestrator created")
    
    async def initialize(self):
        """Initialize the complete multi-agent system."""
        try:
            logger.info("Initializing Clause Echoes multi-agent system")
            
            # Initialize workflow orchestrator first
            self.workflow_orchestrator = WorkflowOrchestrator(
                self.config.get('workflow_orchestrator', {})
            )
            await self.workflow_orchestrator.initialize()
            
            # Initialize all agents
            await self._initialize_micro_agents()
            await self._initialize_meta_agents()
            
            # Register agents with workflow orchestrator
            for agent in self.agents.values():
                self.workflow_orchestrator.register_agent(agent)
            
            # System ready
            self.status = SystemStatus.READY
            self.initialized_at = datetime.utcnow()
            
            logger.info(
                "Clause Echoes system initialized successfully",
                agents_count=len(self.agents),
                agent_types=list(self.agents.keys())
            )
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            logger.error(
                "Failed to initialize Clause Echoes system",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _initialize_micro_agents(self):
        """Initialize all micro-agents."""
        logger.info("Initializing micro-agents")
        
        # Query Parser Agent
        query_parser = QueryParserAgent(self.config.get('query_parser', {}))
        await query_parser.initialize()
        self.agents['QueryParserAgent'] = query_parser
        
        # Clause Retriever Agent  
        clause_retriever = ClauseRetrieverAgent(self.config.get('clause_retriever', {}))
        await clause_retriever.initialize()
        self.agents['ClauseRetrieverAgent'] = clause_retriever
        
        # Consistency Checker Agent
        consistency_checker = ConsistencyCheckerAgent(self.config.get('consistency_checker', {}))
        await consistency_checker.initialize()
        self.agents['ConsistencyCheckerAgent'] = consistency_checker
        
        # Logic Evaluator Agent
        logic_evaluator = LogicEvaluatorAgent(self.config.get('logic_evaluator', {}))
        await logic_evaluator.initialize()
        self.agents['LogicEvaluatorAgent'] = logic_evaluator
        
        # Synthesis Agent
        synthesis_agent = SynthesisAgent(self.config.get('synthesis_agent', {}))
        await synthesis_agent.initialize()
        self.agents['SynthesisAgent'] = synthesis_agent
        
        # Hypothesis Generator Agent
        hypothesis_generator = HypothesisGeneratorAgent(self.config.get('hypothesis_generator', {}))
        await hypothesis_generator.initialize()
        self.agents['HypothesisGeneratorAgent'] = hypothesis_generator
        
        # Curriculum Agent
        curriculum_agent = CurriculumAgent(self.config.get('curriculum_agent', {}))
        await curriculum_agent.initialize()
        self.agents['CurriculumAgent'] = curriculum_agent
        
        logger.info(f"Initialized {len(self.agents)} micro-agents")
    
    async def _initialize_meta_agents(self):
        """Initialize all meta-agents."""
        logger.info("Initializing meta-agents")
        
        initial_count = len(self.agents)
        
        # Self-Critic Meta-Agent
        self_critic = SelfCriticAgent(self.config.get('self_critic', {}))
        await self_critic.initialize()
        self.agents['SelfCriticAgent'] = self_critic
        
        # Uncertainty Analyzer Meta-Agent
        uncertainty_analyzer = UncertaintyAnalyzerAgent(self.config.get('uncertainty_analyzer', {}))
        await uncertainty_analyzer.initialize()
        self.agents['UncertaintyAnalyzerAgent'] = uncertainty_analyzer
        
        # Assumption Detector Meta-Agent
        assumption_detector = AssumptionDetectorAgent(self.config.get('assumption_detector', {}))
        await assumption_detector.initialize()
        self.agents['AssumptionDetectorAgent'] = assumption_detector
        
        # Consensus Engine Meta-Agent
        consensus_engine = ConsensusEngineAgent(self.config.get('consensus_engine', {}))
        await consensus_engine.initialize()
        self.agents['ConsensusEngineAgent'] = consensus_engine
        
        meta_agents_count = len(self.agents) - initial_count
        logger.info(f"Initialized {meta_agents_count} meta-agents")
    
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        workflow_id: str = "comprehensive_query_processing"
    ) -> Dict[str, Any]:
        """Process a user query through the complete multi-agent pipeline."""
        
        if self.status != SystemStatus.READY:
            raise RuntimeError(f"System not ready. Current status: {self.status}")
        
        start_time = datetime.utcnow()
        self.status = SystemStatus.RUNNING
        
        try:
            logger.info(
                "Processing user query",
                query_length=len(query),
                workflow_id=workflow_id
            )
            
            # Prepare workflow input
            workflow_input = {
                'query': query,
                'context': context or {},
                'timestamp': start_time.isoformat(),
                'request_id': f"query_{int(start_time.timestamp())}"
            }
            
            # Execute workflow
            execution = await self.workflow_orchestrator.execute_workflow(
                workflow_id, workflow_input, context
            )
            
            # Wait for completion
            while execution.status in ['pending', 'running']:
                await asyncio.sleep(0.5)
                # In production, would have timeout handling
            
            # Extract results
            if execution.status == 'completed':
                result = await self._extract_final_result(execution)
                
                # Update metrics
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                self._update_metrics(processing_time)
                
                logger.info(
                    "Query processed successfully",
                    processing_time=processing_time,
                    execution_id=execution.execution_id
                )
                
                return {
                    'success': True,
                    'result': result,
                    'execution_id': execution.execution_id,
                    'processing_time': processing_time,
                    'timestamp': start_time.isoformat()
                }
                
            else:
                logger.error(
                    "Query processing failed",
                    execution_id=execution.execution_id,
                    error=execution.error_message
                )
                
                return {
                    'success': False,
                    'error': execution.error_message,
                    'execution_id': execution.execution_id,
                    'timestamp': start_time.isoformat()
                }
        
        except Exception as e:
            logger.error(
                "Error during query processing",
                error=str(e),
                exc_info=True
            )
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': start_time.isoformat()
            }
        
        finally:
            self.status = SystemStatus.READY
    
    async def _extract_final_result(self, execution) -> Dict[str, Any]:
        """Extract and structure the final result from workflow execution."""
        
        # Get results from key tasks
        synthesis_result = None
        critique_result = None
        uncertainty_result = None
        assumption_result = None
        consensus_result = None
        
        for task_id, task in execution.task_executions.items():
            if task.status == 'completed' and task.result:
                if task.task_type == 'synthesize_response':
                    synthesis_result = task.result.get('synthesis_result', {})
                elif task.task_type == 'critique_response':
                    critique_result = task.result.get('critique_result', {})
                elif task.task_type == 'analyze_uncertainty':
                    uncertainty_result = task.result.get('uncertainty_analysis', {})
                elif task.task_type == 'detect_assumptions':
                    assumption_result = task.result.get('assumption_analysis', {})
                elif task.task_type == 'build_consensus':
                    consensus_result = task.result.get('consensus_result', {})
        
        # Structure final response
        final_result = {
            'answer': synthesis_result.get('answer', 'No answer generated') if synthesis_result else 'No answer generated',
            'explanation': synthesis_result.get('explanation', '') if synthesis_result else '',
            'confidence': synthesis_result.get('confidence_score', 0.0) if synthesis_result else 0.0,
            'sources': synthesis_result.get('sources_used', []) if synthesis_result else [],
            
            # Meta-analysis results
            'critique': {
                'quality_score': critique_result.get('overall_quality_score', 0.0) if critique_result else 0.0,
                'strengths': critique_result.get('strengths', []) if critique_result else [],
                'weaknesses': critique_result.get('weaknesses', []) if critique_result else [],
                'suggested_revision': critique_result.get('suggested_revision', '') if critique_result else ''
            },
            
            'uncertainty': {
                'confidence_breakdown': uncertainty_result.get('confidence_breakdown', {}) if uncertainty_result else {},
                'risk_level': uncertainty_result.get('risk_level', 'unknown') if uncertainty_result else 'unknown',
                'uncertainty_factors': uncertainty_result.get('uncertainty_factors', []) if uncertainty_result else []
            },
            
            'assumptions': {
                'detected_assumptions': assumption_result.get('detected_assumptions', []) if assumption_result else [],
                'assumption_burden': assumption_result.get('assumption_burden', 0.0) if assumption_result else 0.0,
                'robustness_score': assumption_result.get('robustness_score', 0.0) if assumption_result else 0.0
            },
            
            'consensus': {
                'consensus_reached': consensus_result.get('consensus_reached', False) if consensus_result else False,
                'consensus_position': consensus_result.get('consensus_position', '') if consensus_result else '',
                'consensus_confidence': consensus_result.get('consensus_confidence', 0.0) if consensus_result else 0.0
            },
            
            # Processing metadata
            'workflow_execution': {
                'completed_tasks': len(execution.completed_tasks),
                'total_tasks': len(execution.task_executions),
                'execution_time': execution.execution_time,
                'progress_percentage': execution.progress_percentage
            }
        }
        
        return final_result
    
    def _update_metrics(self, processing_time: float):
        """Update system performance metrics."""
        self.total_queries_processed += 1
        
        # Update average response time using moving average
        if self.total_queries_processed == 1:
            self.average_response_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_response_time = (
                alpha * processing_time + 
                (1 - alpha) * self.average_response_time
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        agent_status = {}
        for agent_type, agent in self.agents.items():
            try:
                status = agent.get_status()
                agent_status[agent_type] = {
                    'state': status['state'],
                    'capabilities': len(status['capabilities']),
                    'last_heartbeat': status['last_heartbeat']
                }
            except Exception as e:
                agent_status[agent_type] = {
                    'state': 'error',
                    'error': str(e)
                }
        
        uptime = 0.0
        if self.initialized_at:
            uptime = (datetime.utcnow() - self.initialized_at).total_seconds()
        
        return {
            'system_status': self.status,
            'uptime_seconds': uptime,
            'initialized_at': self.initialized_at.isoformat() if self.initialized_at else None,
            'total_agents': len(self.agents),
            'agent_status': agent_status,
            'performance_metrics': {
                'total_queries_processed': self.total_queries_processed,
                'average_response_time': self.average_response_time
            },
            'active_workflows': len(self.workflow_orchestrator.active_executions) if self.workflow_orchestrator else 0
        }
    
    async def cleanup(self):
        """Cleanup all system resources."""
        logger.info("Shutting down Clause Echoes system")
        
        self.status = SystemStatus.SHUTTING_DOWN
        
        try:
            # Shutdown workflow orchestrator
            if self.workflow_orchestrator:
                await self.workflow_orchestrator.cleanup()
            
            # Shutdown all agents
            shutdown_tasks = []
            for agent in self.agents.values():
                shutdown_tasks.append(agent.shutdown())
            
            # Wait for all agents to shutdown
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            self.status = SystemStatus.STOPPED
            
            logger.info("Clause Echoes system shutdown complete")
            
        except Exception as e:
            logger.error(
                "Error during system shutdown",
                error=str(e),
                exc_info=True
            )
            self.status = SystemStatus.ERROR
    
    # Additional utility methods
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get a specific agent by type."""
        return self.agents.get(agent_type)
    
    def list_available_workflows(self) -> List[str]:
        """List available workflow templates."""
        if self.workflow_orchestrator:
            return self.workflow_orchestrator.list_available_workflows()
        return []
    
    async def execute_custom_workflow(
        self, workflow_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a custom workflow."""
        if not self.workflow_orchestrator:
            raise RuntimeError("Workflow orchestrator not initialized")
        
        execution = await self.workflow_orchestrator.execute_workflow(workflow_id, input_data)
        
        # Wait for completion (in production, would be asynchronous)
        while execution.status in ['pending', 'running']:
            await asyncio.sleep(0.5)
        
        return {
            'execution_id': execution.execution_id,
            'status': execution.status,
            'result': execution.output_data,
            'execution_time': execution.execution_time
        }
