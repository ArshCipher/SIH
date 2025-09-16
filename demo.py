#!/usr/bin/env python3
"""
Public Health Chatbot Demo Script
Demonstrates the core functionality of the health chatbot
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.core import HealthChatbot
from chatbot.database import DatabaseManager
from chatbot.translation import TranslationService
from chatbot.health_data import HealthDataService

async def demo_chatbot():
    """Demonstrate chatbot functionality"""
    print("🏥 Public Health Chatbot Demo")
    print("=" * 50)
    
    # Initialize services
    print("🔧 Initializing services...")
    db_manager = DatabaseManager()
    translation_service = TranslationService()
    health_data_service = HealthDataService()
    
    chatbot = HealthChatbot(
        db_manager=db_manager,
        translation_service=translation_service,
        health_data_service=health_data_service
    )
    
    print("✅ Services initialized successfully!")
    print()
    
    # Demo conversations
    demo_conversations = [
        {
            "message": "Hello, I want to know about COVID-19 symptoms",
            "language": "en",
            "description": "Disease symptoms query"
        },
        {
            "message": "What vaccines do I need as an adult?",
            "language": "en",
            "description": "Vaccination schedule query"
        },
        {
            "message": "I need emergency help",
            "language": "en",
            "description": "Emergency help request"
        },
        {
            "message": "How can I prevent malaria?",
            "language": "en",
            "description": "Prevention tips query"
        },
        {
            "message": "नमस्ते, मुझे बुखार है",
            "language": "hi",
            "description": "Hindi language query"
        }
    ]
    
    for i, conv in enumerate(demo_conversations, 1):
        print(f"💬 Demo {i}: {conv['description']}")
        print(f"   User: {conv['message']}")
        
        try:
            response = await chatbot.process_message(
                message=conv['message'],
                user_id=f"demo_user_{i}",
                language=conv['language'],
                platform="demo"
            )
            
            print(f"   Bot: {response['response'][:100]}...")
            print(f"   Confidence: {response['confidence']:.2f}")
            print(f"   Language: {response['language']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            print()
    
    # Demo analytics
    print("📊 Analytics Demo")
    print("-" * 30)
    
    try:
        analytics = await db_manager.get_analytics_data()
        print(f"Total conversations: {analytics.get('total_conversations', 0)}")
        print(f"Platform distribution: {analytics.get('platform_distribution', {})}")
        print(f"Language distribution: {analytics.get('language_distribution', {})}")
        print()
        
    except Exception as e:
        print(f"❌ Analytics error: {str(e)}")
        print()
    
    # Demo health data
    print("🏥 Health Data Demo")
    print("-" * 30)
    
    try:
        # Test disease info
        disease_info = await health_data_service.get_disease_info("COVID-19")
        if disease_info:
            print(f"COVID-19 symptoms: {disease_info['symptoms'][:100]}...")
        
        # Test vaccination schedule
        schedule = await health_data_service.get_vaccination_schedule("adult")
        if schedule:
            print(f"Adult vaccines: {len(schedule)} vaccines available")
        
        print()
        
    except Exception as e:
        print(f"❌ Health data error: {str(e)}")
        print()
    
    # Demo translation
    print("🌐 Translation Demo")
    print("-" * 30)
    
    try:
        # Test language detection
        detected = await translation_service.detect_language("Hello, how are you?")
        print(f"Detected language: {detected}")
        
        # Test translation
        translated = await translation_service.translate_text("Hello", "hi")
        print(f"Translated 'Hello' to Hindi: {translated}")
        
        print()
        
    except Exception as e:
        print(f"❌ Translation error: {str(e)}")
        print()
    
    print("🎉 Demo completed successfully!")
    print()
    print("To start the full application:")
    print("1. Configure your API credentials in .env file")
    print("2. Run: ./start.sh")
    print("3. Visit: http://localhost:8000/docs for API documentation")
    print("4. Visit: http://localhost:8000/analytics/dashboard/html for analytics")

if __name__ == "__main__":
    try:
        asyncio.run(demo_chatbot())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        sys.exit(1)
