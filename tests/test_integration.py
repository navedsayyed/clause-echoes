"""
Integration tests for the complete Clause Echoes multi-agent system.
Tests end-to-end workflows and agent interactions.
"""
import pytest
import asyncio
from datetime import datetime

from core.orchestrator import AgentOrchestrator
from core.config import settings


@pytest.fixture
async def orchestrator():
    """Create and initialize system orchestrator for testing."""
    config = {
        'query_parser': {'enable_deep_analysis': True},
        'synthesis_agent': {'enable_alternatives': True},
        'self_critic': {'enable_revision': True}
    }
    
    orchestrator = AgentOrchestrator(config)
    await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.cleanup()


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_query_processing(self, orchestrator):
        """Test complete query processing workflow."""
        
        # Test query
        query = "What are the coverage requirements for emergency surgery procedures?"
        context = {
            'user_type': 'policy_holder',
            'domain': 'medical_insurance'
        }
        
        # Process query
        result = await orchestrator.process_query(query, context)
        
        # Verify successful processing
        assert result['success'] is True
        assert 'result' in result
        assert result['result']['answer']  # Non-empty answer
        assert result['result']['confidence'] > 0.0
        
        # Verify meta-analysis components
        assert 'critique' in result['result']
        assert 'uncertainty' in result['result']
        assert 'assumptions' in result['result']
        assert 'consensus' in result['result']
        
        # Verify processing metadata
        assert result['processing_time'] > 0
        assert result['execution_id']
    
    @pytest.mark.asyncio
    async def test_self_critique_improvement(self, orchestrator):
        """Test that self-critique improves response quality."""
        
        query = "Does our policy cover experimental treatments?"
        
        # Process with self-critique enabled
        result_with_critique = await orchestrator.process_query(query)
        
        # Verify critique was performed
        assert result_with_critique['success'] is True
        critique = result_with_critique['result']['critique']
        assert critique['quality_score'] > 0
        assert len(critique['strengths']) > 0 or len(critique['weaknesses']) > 0
        
        # If revision was suggested, it should be different from original
        if critique['suggested_revision']:
            assert critique['suggested_revision'] != result_with_critique['result']['answer']
    
    @pytest.mark.asyncio
    async def test_uncertainty_quantification(self, orchestrator):
        """Test uncertainty analysis and confidence scoring."""
        
        # Ambiguous query that should have high uncertainty
        ambiguous_query = "What about coverage for that procedure we discussed?"
        
        result = await orchestrator.process_query(ambiguous_query)
        
        # Should still process but with appropriate uncertainty
        assert result['success'] is True
        
        uncertainty = result['result']['uncertainty']
        assert uncertainty['risk_level'] in ['low', 'medium', 'high', 'critical']
        assert len(uncertainty['uncertainty_factors']) > 0
        
        # Should detect the ambiguity
        assumptions = result['result']['assumptions']
        assert assumptions['assumption_burden'] > 0.3  # High assumption burden for ambiguous query
    
    @pytest.mark.asyncio
    async def test_consensus_building(self, orchestrator):
        """Test multi-source consensus building."""
        
        query = "What is the standard deductible amount for primary care visits?"
        
        result = await orchestrator.process_query(query)
        
        assert result['success'] is True
        
        consensus = result['result']['consensus']
        # Should attempt consensus even with single query
        assert 'consensus_reached' in consensus
        assert 'consensus_confidence' in consensus
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestrator):
        """Test system error handling and recovery."""
        
        # Test with invalid input
        result = await orchestrator.process_query("")  # Empty query
        
        # Should handle gracefully
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, orchestrator):
        """Test hypothesis generation for policy gaps."""
        
        query = "What happens if someone needs surgery but it's not explicitly covered?"
        
        result = await orchestrator.process_query(query)
        
        # Should generate response even for edge cases
        assert result['success'] is True
        assert result['result']['answer']
        
        # Should identify this as requiring assumptions/hypotheses
        assumptions = result['result']['assumptions']
        assert len(assumptions['detected_assumptions']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, orchestrator):
        """Test system performance tracking."""
        
        # Process multiple queries
        queries = [
            "What is covered under basic medical?",
            "How do I file a claim?",
            "What are the network restrictions?"
        ]
        
        for query in queries:
            result = await orchestrator.process_query(query)
            assert result['success'] is True
        
        # Check system metrics
        status = orchestrator.get_system_status()
        assert status['performance_metrics']['total_queries_processed'] == len(queries)
        assert status['performance_metrics']['average_response_time'] > 0


class TestAgentInteractions:
    """Test specific agent interactions and workflows."""
    
    @pytest.mark.asyncio
    async def test_query_parser_to_retriever_flow(self, orchestrator):
        """Test query parser to clause retriever workflow."""
        
        query_parser = orchestrator.get_agent('QueryParserAgent')
        clause_retriever = orchestrator.get_agent('ClauseRetrieverAgent')
        
        assert query_parser is not None
        assert clause_retriever is not None
        
        # Test agents are properly initialized
        assert query_parser.state == 'idle'
        assert clause_retriever.state == 'idle'
    
    @pytest.mark.asyncio
    async def test_meta_agent_coordination(self, orchestrator):
        """Test meta-agent coordination and feedback loops."""
        
        # Get meta-agents
        self_critic = orchestrator.get_agent('SelfCriticAgent')
        uncertainty_analyzer = orchestrator.get_agent('UncertaintyAnalyzerAgent')
        assumption_detector = orchestrator.get_agent('AssumptionDetectorAgent')
        consensus_engine = orchestrator.get_agent('ConsensusEngineAgent')
        
        # Verify all meta-agents are initialized
        meta_agents = [self_critic, uncertainty_analyzer, assumption_detector, consensus_engine]
        for agent in meta_agents:
            assert agent is not None
            assert agent.state in ['idle', 'busy']  # Should be ready for work


if __name__ == "__main__":
    # Run basic integration test
    async def run_basic_test():
        print("🚀 Starting Clause Echoes Integration Test")
        
        orchestrator = AgentOrchestrator()
        await orchestrator.initialize()
        
        try:
            print("✅ System initialized successfully")
            
            # Test basic query processing
            result = await orchestrator.process_query(
                "What are the requirements for pre-authorization of medical procedures?",
                {"domain": "healthcare", "user_type": "provider"}
            )
            
            if result['success']:
                print("✅ Query processed successfully")
                print(f"📊 Processing time: {result['processing_time']:.2f}s")
                print(f"🎯 Confidence: {result['result']['confidence']:.2f}")
                print(f"📝 Answer length: {len(result['result']['answer'])} characters")
                
                # Display key metrics
                critique = result['result']['critique']
                uncertainty = result['result']['uncertainty']
                assumptions = result['result']['assumptions']
                
                print(f"🔍 Quality score: {critique['quality_score']:.2f}")
                print(f"⚠️ Risk level: {uncertainty['risk_level']}")
                print(f"🤔 Assumptions detected: {len(assumptions['detected_assumptions'])}")
                
                print("\n🎉 Integration test completed successfully!")
                
            else:
                print(f"❌ Query processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            
        finally:
            await orchestrator.cleanup()
            print("🧹 System cleanup completed")
    
    # Run the test
    asyncio.run(run_basic_test())
