import pytest
import pytest_asyncio
import asyncio
from chatbot.core import HealthChatbot
from chatbot.database import DatabaseManager
from chatbot.translation import TranslationService
from chatbot.health_data import HealthDataService

@pytest.fixture
def chatbot_event_loop():
    # Provide an event loop for non-async fixtures if needed
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def chatbot():
    """Create chatbot instance for testing"""
    db_manager = DatabaseManager()
    translation_service = TranslationService()
    health_data_service = HealthDataService()
    
    bot = HealthChatbot(
        db_manager=db_manager,
        translation_service=translation_service,
        health_data_service=health_data_service
    )
    return bot

@pytest.mark.asyncio
async def test_chatbot_greeting(chatbot):
    """Test chatbot greeting response"""
    response = await chatbot.process_message(
        message="Hello",
        user_id="test_user",
        language="en",
        platform="web"
    )
    
    assert response is not None
    assert "response" in response
    assert len(response["response"]) > 0

@pytest.mark.asyncio
async def test_disease_symptoms_query(chatbot):
    """Test disease symptoms query"""
    response = await chatbot.process_message(
        message="What are the symptoms of COVID-19?",
        user_id="test_user",
        language="en",
        platform="web"
    )
    
    assert response is not None
    assert "response" in response
    assert "symptoms" in response["response"].lower() or "covid" in response["response"].lower()

@pytest.mark.asyncio
async def test_vaccination_query(chatbot):
    """Test vaccination schedule query"""
    response = await chatbot.process_message(
        message="What vaccines do I need as an adult?",
        user_id="test_user",
        language="en",
        platform="web"
    )
    
    assert response is not None
    assert "response" in response
    assert len(response["response"]) > 0

@pytest.mark.asyncio
async def test_emergency_query(chatbot):
    """Test emergency help query"""
    response = await chatbot.process_message(
        message="I need emergency help",
        user_id="test_user",
        language="en",
        platform="web"
    )
    
    assert response is not None
    assert "response" in response
    assert "emergency" in response["response"].lower() or "108" in response["response"]

@pytest.mark.asyncio
async def test_multilingual_support(chatbot):
    """Test multilingual support"""
    response = await chatbot.process_message(
        message="नमस्ते",
        user_id="test_user",
        language="hi",
        platform="web"
    )
    
    assert response is not None
    assert "response" in response
    assert response["language"] == "hi"

@pytest.mark.asyncio
async def test_translation_service():
    """Test translation service"""
    translation_service = TranslationService()
    
    # Test language detection
    detected_lang = await translation_service.detect_language("Hello, how are you?")
    assert detected_lang == "en"
    
    # Test translation
    translated = await translation_service.translate_text("Hello", "hi")
    assert translated is not None
    assert len(translated) > 0

@pytest.mark.asyncio
async def test_health_data_service():
    """Test health data service"""
    health_service = HealthDataService()
    
    # Test disease info
    disease_info = await health_service.get_disease_info("COVID-19")
    assert disease_info is not None
    assert "symptoms" in disease_info
    
    # Test vaccination schedule
    schedule = await health_service.get_vaccination_schedule("adult")
    assert schedule is not None
    assert len(schedule) > 0

@pytest.mark.asyncio
async def test_database_manager():
    """Test database manager"""
    db_manager = DatabaseManager()
    
    # Test analytics data
    analytics = await db_manager.get_analytics_data()
    assert analytics is not None
    assert "total_conversations" in analytics

if __name__ == "__main__":
    pytest.main([__file__])
