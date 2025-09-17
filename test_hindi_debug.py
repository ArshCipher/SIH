#!/usr/bin/env python3
"""
Quick test for Hindi translation in web interface
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.medical_orchestrator import MedicalOrchestrator

async def test_hindi_query():
    """Test Hindi query processing"""
    print("🧪 Testing Hindi Query Processing")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = MedicalOrchestrator()
    
    # Test Hindi query exactly as received from web
    hindi_query = "मुझे बुखार है"
    print(f"Original Hindi Query: {hindi_query}")
    
    # Test language detection
    detected_lang = orchestrator._detect_language(hindi_query)
    print(f"Detected Language: {detected_lang}")
    
    # Test translation
    translated = await orchestrator._translate_query_if_needed(hindi_query, detected_lang)
    print(f"Translated Query: {translated}")
    
    # Test full processing
    context = {
        "user_id": "test_user",
        "language": "hi",
        "country": "IN",
        "platform": "web"
    }
    
    print("\n🔍 Processing full query...")
    try:
        result = await orchestrator.process_medical_query(hindi_query, context)
        print(f"✅ Processing successful!")
        print(f"Confidence: {result.consensus_confidence:.2f}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Response (first 200 chars): {result.final_response[:200]}...")
        
        # Check if response contains fever-related content
        response_lower = result.final_response.lower()
        fever_keywords = ['fever', 'temperature', 'बुखार', 'pyrexia', 'hyperthermia']
        
        found_keywords = [keyword for keyword in fever_keywords if keyword in response_lower]
        if found_keywords:
            print(f"✅ Response contains fever-related content: {found_keywords}")
        else:
            print("❌ Response doesn't seem to address fever")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hindi_query())