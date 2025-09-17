#!/usr/bin/env python3
"""
Quick test for improved Hindi fever query
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.medical_orchestrator import MedicalOrchestrator

async def test_improved_fever():
    """Test improved fever detection"""
    print("🧪 Testing Improved Fever Detection")
    print("=" * 50)
    
    orchestrator = MedicalOrchestrator()
    
    test_queries = [
        "मुझे बुखार है",  # Hindi: I have fever
        "I have fever",
        "I have high temperature",
        "मुझे सिरदर्द है",  # Hindi: I have headache
        "I have headache and fever"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        
        # Detect language
        detected_lang = orchestrator._detect_language(query)
        print(f"   Language: {detected_lang}")
        
        # Translate if needed
        translated = await orchestrator._translate_query_if_needed(query, detected_lang)
        print(f"   Translated: {translated}")
        
        # Process the query
        context = {"user_id": "test_user"}
        try:
            result = await orchestrator.process_medical_query(query, context)
            print(f"   ✅ Confidence: {result.consensus_confidence:.2f}")
            print(f"   Risk Level: {result.risk_level}")
            
            # Check if response mentions relevant diseases
            response = result.final_response.lower()
            fever_related = any(word in response for word in ['fever', 'temperature', 'malaria', 'dengue', 'flu', 'infection'])
            
            if fever_related:
                print(f"   ✅ Response mentions fever-related conditions")
            else:
                print(f"   ⚠️ Response might not be specific enough")
                
            # Show first 150 characters of response
            print(f"   Response: {result.final_response[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_improved_fever())