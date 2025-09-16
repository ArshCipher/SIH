from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Public Health Chatbot API",
    description="AI-driven chatbot for disease awareness and preventive healthcare",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: str
    language: str = "en"
    platform: str = "web"  # web, whatsapp, sms

class ChatResponse(BaseModel):
    response: str
    confidence: float
    suggested_actions: list = []
    language: str = "en"

class HealthQuery(BaseModel):
    query: str
    user_id: str
    location: str = None
    age_group: str = None

# Advanced medical (optional) request models
class MedicalQuery(BaseModel):
    text: str
    user_id: str
    language: str = "en"
    country: str = "IN"

class MedicalRAGQuery(BaseModel):
    query: str
    top_k: int = 5

# Import modules
from chatbot.core import HealthChatbot
from chatbot.database import DatabaseManager
from chatbot.translation import TranslationService
from chatbot.integrations import WhatsAppService
from chatbot.health_data import HealthDataService
from chatbot.alerts import AlertService, SMSService
from chatbot.analytics import AnalyticsDashboard

# Optional advanced medical system imports (graceful fallback)
try:
    from chatbot.medical_orchestrator import MedicalOrchestrator
except Exception:
    MedicalOrchestrator = None  # type: ignore

try:
    from chatbot.medical_graph_rag import MedicalGraphRAG
except Exception:
    MedicalGraphRAG = None  # type: ignore

# Initialize services
db_manager = DatabaseManager()
translation_service = TranslationService()
health_data_service = HealthDataService()
whatsapp_service = WhatsAppService()
sms_service = SMSService()
alert_service = AlertService()
analytics_dashboard = AnalyticsDashboard(db_manager)

# Initialize advanced systems if available
medical_orchestrator = None
medical_rag = None
try:
    if MedicalOrchestrator is not None:
        medical_orchestrator = MedicalOrchestrator()
except Exception as e:
    logger.warning(f"MedicalOrchestrator unavailable: {e}")

try:
    if MedicalGraphRAG is not None:
        medical_rag = MedicalGraphRAG()
except Exception as e:
    logger.warning(f"MedicalGraphRAG unavailable: {e}")

# Initialize chatbot
chatbot = HealthChatbot(
    db_manager=db_manager,
    translation_service=translation_service,
    health_data_service=health_data_service
)

@app.get("/")
async def root():
    return {"message": "Public Health Chatbot API is running - Visit /frontend for the chat interface"}

@app.get("/frontend", response_class=HTMLResponse)
async def chat_frontend():
    """Serve the ChatGPT-like frontend interface"""
    try:
        with open("templates/chat.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <html><body>
        <h1>Chat Frontend Not Found</h1>
        <p>The chat.html template is missing. Please ensure templates/chat.html exists.</p>
        <p><a href="/docs">Go to API Documentation</a></p>
        </body></html>
        """)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint for processing user messages"""
    try:
        response = await chatbot.process_message(
            message=message.message,
            user_id=message.user_id,
            language=message.language,
            platform=message.platform
        )
        return response
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/health-query")
async def health_query_endpoint(query: HealthQuery):
    """Specific health-related queries endpoint"""
    try:
        result = await health_data_service.process_health_query(
            query=query.query,
            user_id=query.user_id,
            location=query.location,
            age_group=query.age_group
        )
        return result
    except Exception as e:
        logger.error(f"Error processing health query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/vaccination-schedule/{age_group}")
async def get_vaccination_schedule(age_group: str):
    """Get vaccination schedule for specific age group"""
    try:
        schedule = await health_data_service.get_vaccination_schedule(age_group)
        return schedule
    except Exception as e:
        logger.error(f"Error fetching vaccination schedule: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/disease-info/{disease_name}")
async def get_disease_info(disease_name: str, language: str = "en"):
    """Get information about a specific disease"""
    try:
        info = await health_data_service.get_disease_info(disease_name, language)
        return info
    except Exception as e:
        logger.error(f"Error fetching disease info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/outbreak-alerts")
async def get_outbreak_alerts(location: str = None):
    """Get current outbreak alerts"""
    try:
        alerts = await alert_service.get_outbreak_alerts(location)
        return alerts
    except Exception as e:
        logger.error(f"Error fetching outbreak alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """WhatsApp webhook endpoint"""
    try:
        data = await request.json()
        response = await whatsapp_service.process_webhook(data)
        return response
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/sms/webhook")
async def sms_webhook(request: Request):
    """SMS webhook endpoint"""
    try:
        data = await request.json()
        response = await sms_service.process_webhook(data)
        return response
    except Exception as e:
        logger.error(f"Error processing SMS webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Advanced medical endpoints (optional)
@app.post("/medical/answer")
async def medical_answer_endpoint(payload: MedicalQuery):
    try:
        if medical_orchestrator is None:
            raise HTTPException(status_code=503, detail="Medical Orchestrator not available")
        context = {
            "user_id": payload.user_id,
            "language": payload.language,
            "country": payload.country,
            "platform": "api"
        }
        consensus = await medical_orchestrator.process_medical_query(payload.text, context)
        return {
            "response": consensus.final_response,
            "confidence": consensus.consensus_confidence,
            "risk_level": consensus.risk_level.value,
            "safety_validated": consensus.safety_validated,
            "human_escalation_required": consensus.human_escalation_required,
            "disclaimer": consensus.medical_disclaimer,
            "session_id": consensus.session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /medical/answer: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/medical/retrieve")
async def medical_retrieve_endpoint(payload: MedicalRAGQuery):
    try:
        if medical_rag is None:
            raise HTTPException(status_code=503, detail="Medical Graph RAG not available")
        result = await medical_rag.retrieve_medical_knowledge(payload.query, top_k=payload.top_k)
        return {
            "content": result.content,
            "relevance_score": result.relevance_score,
            "confidence": result.confidence,
            "retrieval_method": result.retrieval_method,
            "sources": [
                {"title": d.title, "source": d.source.value, "credibility": d.credibility_score}
                for d in result.source_documents
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /medical/retrieve: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data"""
    try:
        analytics = await analytics_dashboard.get_dashboard_data()
        return analytics
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/dashboard/html", response_class=HTMLResponse)
async def get_analytics_dashboard_html():
    """Get analytics dashboard as HTML"""
    try:
        data = await analytics_dashboard.get_dashboard_data()
        html = analytics_dashboard.generate_dashboard_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Error generating analytics HTML: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/effectiveness")
async def get_effectiveness_metrics():
    """Get chatbot effectiveness metrics"""
    try:
        metrics = await analytics_dashboard.get_effectiveness_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error fetching effectiveness metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
