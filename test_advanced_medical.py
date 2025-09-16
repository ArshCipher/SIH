#!/usr/bin/env python3
"""
Test script for advanced medical AI responses
"""

import asyncio
import sys
import os

# Add the chatbot directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chatbot'))

from medical_orchestrator import MedicalOrchestrator

async def test_advanced_medical_query():
    """Test advanced medical query handling"""
    
    print("🧬 Testing Advanced Medical AI System")
    print("=" * 50)
    
    # Initialize medical orchestrator
    orchestrator = MedicalOrchestrator()
    
    # Test queries of increasing complexity
    test_queries = [
        "What is the molecular mechanism of Type 2 diabetes insulin resistance?",
        "Explain the oncogenic pathways in lung adenocarcinoma and targeted therapies",
        "How do ACE inhibitors work at the cellular level for hypertension?", 
        "What are the latest advances in CAR-T cell therapy for hematologic malignancies?",
        "Describe the pathophysiology of heart failure with preserved ejection fraction"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔬 Test Query {i}:")
        print(f"Query: {query}")
        print("-" * 80)
        
        try:
            # Process the query through all medical models
            result = await orchestrator.process_medical_query(query, {})
            
            print(f"✅ Response Generated Successfully!")
            print(f"📊 Confidence: {result.consensus_confidence:.2f}")
            print(f"🏥 Medical Models Used: {len(result.agent_responses)} agents")
            print(f"🛡️ Safety Validated: {result.safety_validated}")
            print(f"⚠️ Risk Level: {result.risk_level.value}")
            print(f"🚨 Human Escalation: {result.human_escalation_required}")
            
            # Show the comprehensive response
            print(f"\n📋 Medical Analysis:")
            print(result.final_response[:500] + "..." if len(result.final_response) > 500 else result.final_response)
            
            # Show which models contributed
            print(f"\n🤖 Contributing Agents:")
            for agent_response in result.agent_responses:
                print(f"  • {agent_response.agent_role.value}: {agent_response.confidence:.2f}")
                
            # Show explanation
            if hasattr(result, 'explanation') and result.explanation:
                print(f"\n🧠 AI Explanation:")
                print(result.explanation[:300] + "..." if len(result.explanation) > 300 else result.explanation)
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_advanced_medical_query())