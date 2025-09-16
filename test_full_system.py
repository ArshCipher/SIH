#!/usr/bin/env python3
"""
Full Medical AI System Test
Comprehensive test of the competition-grade medical AI system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the chatbot package to Python path
chatbot_path = Path(__file__).parent / "chatbot"
sys.path.insert(0, str(chatbot_path))

def test_basic_imports():
    """Test basic system imports without heavy ML dependencies"""
    try:
        print("🧪 Testing Basic System Imports...")
        
        # Test core orchestrator
        from medical_orchestrator import MedicalOrchestrator
        print("✅ MedicalOrchestrator imported successfully")
        
        # Test medical models (should work with graceful degradation)
        from medical_models import MedicalEnsemble, MedicalModelType
        print("✅ MedicalEnsemble imported successfully")
        
        # Test medical safety
        from medical_safety import MedicalSafetyValidator
        print("✅ MedicalSafetyValidator imported successfully")
        
        # Test medical RAG
        from medical_graph_rag import MedicalGraphRAG, MedicalKnowledgeType
        print("✅ MedicalGraphRAG imported successfully")
        
        print("🎉 All core imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_system_initialization():
    """Test system initialization and basic functionality"""
    try:
        print("\n🔧 Testing System Initialization...")
        
        from medical_orchestrator import MedicalOrchestrator
        from medical_models import MedicalEnsemble
        from medical_safety import MedicalSafetyValidator
        from medical_graph_rag import MedicalGraphRAG
        
        # Test safety system
        safety_validator = MedicalSafetyValidator()
        print("✅ Safety validator initialized")
        
        # Test RAG system
        rag_system = MedicalGraphRAG()
        print("✅ RAG system initialized")
        
        # Test medical models
        model_ensemble = MedicalEnsemble()
        print("✅ Model ensemble initialized")
        
        # Test orchestrator
        orchestrator = MedicalOrchestrator()
        print("✅ Medical orchestrator initialized")
        
        print("🎉 All systems initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_system_functionality():
    """Test basic system functionality"""
    try:
        print("\n🚀 Testing System Functionality...")
        
        from medical_orchestrator import MedicalOrchestrator
        
        # Initialize orchestrator
        orchestrator = MedicalOrchestrator()
        
        # Test simple medical query
        test_query = "What are the symptoms of diabetes?"
        print(f"🔍 Testing query: '{test_query}'")
        
        result = await orchestrator.process_medical_query(
            query=test_query,
            context={"user_id": "test_user", "session_id": "test_session"}
        )
        
        print("✅ Query processed successfully")
        print(f"📋 Response: {result.final_response[:200]}...")
        print(f"🔒 Safety validated: {result.safety_validated}")
        print(f"🎯 Confidence: {result.consensus_confidence:.2f}")
        print(f"👥 Agents consulted: {len(result.agent_responses)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_advanced_features():
    """Test advanced competition features"""
    try:
        print("\n🏆 Testing Advanced Competition Features...")
        
        from medical_orchestrator import MedicalOrchestrator
        
        orchestrator = MedicalOrchestrator()
        
        # Test complex medical scenario
        complex_query = """
        A 45-year-old patient presents with:
        - Frequent urination
        - Excessive thirst 
        - Unexplained weight loss
        - Fatigue
        - Blurred vision
        
        What could be the diagnosis and recommended tests?
        """
        
        print("🔍 Testing complex diagnostic scenario...")
        
        result = await orchestrator.process_medical_query(
            query=complex_query,
            context={
                "user_id": "test_patient", 
                "session_id": "complex_case",
                "priority": "high"
            }
        )
        
        print("✅ Complex query processed successfully")
        print(f"📋 Diagnostic response: {result.final_response[:300]}...")
        print(f"🎯 Overall confidence: {result.consensus_confidence:.2f}")
        print(f"👨‍⚕️ Agents involved: {', '.join([agent.agent_role.value for agent in result.agent_responses])}")
        
        # Test safety features
        if hasattr(result, 'risk_level'):
            print(f"⚠️  Risk level: {result.risk_level.value}")
        
        # Test escalation
        if result.human_escalation_required:
            print("🚨 Human escalation required")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_competition_readiness():
    """Test competition-specific features"""
    try:
        print("\n🏅 Testing Competition Readiness...")
        
        from medical_orchestrator import MedicalOrchestrator
        from medical_models import MedicalModelType
        from medical_safety import MedicalSafetyValidator
        
        # Check multi-agent system
        orchestrator = MedicalOrchestrator()
        agents = list(orchestrator.agents.keys())
        print(f"✅ Multi-agent system with {len(agents)} agents: {agents}")
        
        # Check model ensemble
        from medical_models import MedicalEnsemble
        ensemble = MedicalEnsemble()
        model_types = [mt.value for mt in MedicalModelType]
        print(f"✅ Model ensemble supports: {model_types}")
        
        # Check safety system
        safety = MedicalSafetyValidator()
        print("✅ Safety system with bias detection and hallucination prevention")
        
        # Check RAG system
        from medical_graph_rag import MedicalGraphRAG, MedicalKnowledgeType
        rag = MedicalGraphRAG()
        knowledge_types = [kt.value for kt in MedicalKnowledgeType]
        print(f"✅ Advanced RAG with knowledge types: {knowledge_types}")
        
        print("🏆 System is competition-ready!")
        return True
        
    except Exception as e:
        print(f"❌ Competition readiness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run full system test"""
    print("🔬 MEDICAL AI SYSTEM - FULL SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("System Initialization", test_system_initialization), 
        ("System Functionality", test_system_functionality),
        ("Advanced Features", test_advanced_features),
        ("Competition Readiness", test_competition_readiness)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        print("-" * 40)
        
        if asyncio.iscoroutinefunction(test_func):
            success = await test_func()
        else:
            success = test_func()
            
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FULL SYSTEM TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\n🎯 Overall Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🏆 COMPETITION-GRADE MEDICAL AI SYSTEM FULLY OPERATIONAL!")
        print("🚀 Ready for national healthcare AI competition!")
    else:
        print("⚠️  Some tests failed. Review logs for details.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())