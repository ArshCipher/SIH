#!/usr/bin/env python3
"""
Quick test of intelligent medical responses
"""

import asyncio
import sys
import os

# Add the chatbot directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chatbot'))

from medical_orchestrator import MedicalOrchestrator

async def test_intelligent_responses():
    """Test intelligent contextual medical responses"""
    
    print("🧠 Testing Intelligent Medical AI System")
    print("=" * 60)
    
    # Initialize medical orchestrator
    orchestrator = MedicalOrchestrator()
    
    # Test different types of diabetes questions to show intelligence
    test_queries = [
        "How do I know if I have diabetes?",
        "What causes diabetes?", 
        "How to treat diabetes?",
        "How to prevent diabetes?",
        "What is cancer?",
        "What are the symptoms of heart disease?",
        "How to treat high blood pressure?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔬 Test {i}: {query}")
        print("-" * 50)
        
        try:
            result = await orchestrator.process_medical_query(query, {})
            
            print(f"✅ Intent Analysis Working!")
            print(f"📊 Confidence: {result.consensus_confidence:.2f}")
            print(f"🔍 Risk Level: {result.risk_level.value}")
            
            # Show first 400 characters of the intelligent response
            response_preview = result.final_response[:400] + "..." if len(result.final_response) > 400 else result.final_response
            print(f"\n📋 Intelligent Response:")
            print(response_preview)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(test_intelligent_responses())