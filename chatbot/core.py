import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import re
import os

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    import spacy
except ImportError:
    spacy = None

try:
    from googletrans import Translator
except ImportError:
    Translator = None

logger = logging.getLogger(__name__)

# Optional advanced medical system imports (graceful fallback)
try:
    from chatbot.medical_orchestrator import MedicalOrchestrator
    from chatbot.medical_safety import MedicalSafetyValidator
    ADVANCED_MEDICAL_AVAILABLE = True
except Exception:
    MedicalOrchestrator = None  # type: ignore
    MedicalSafetyValidator = None  # type: ignore
    ADVANCED_MEDICAL_AVAILABLE = False

class HealthChatbot:
    """Main chatbot class for processing health-related queries"""
    
    def __init__(self, db_manager, translation_service, health_data_service):
        self.db_manager = db_manager
        self.translation_service = translation_service
        self.health_data_service = health_data_service
        
        # Advanced medical system (multi-agent orchestrator + safety)
        self.medical_orchestrator = None
        self.safety_validator = None
        if ADVANCED_MEDICAL_AVAILABLE:
            try:
                self.medical_orchestrator = MedicalOrchestrator()
                self.safety_validator = MedicalSafetyValidator()
                logger.info("Advanced Medical Orchestrator initialized")
            except Exception as e:
                logger.warning(f"Advanced medical system unavailable, using fallback: {e}")
        
        # Initialize NLP models
        self.intent_classifier = None
        self.entity_extractor = None
        self.translator = Translator() if Translator else None
        
        # Load models
        self._load_models()
        
        # Intent mappings
        self.intent_mappings = {
            "disease_symptoms": self._handle_disease_symptoms,
            "prevention_tips": self._handle_prevention_tips,
            "vaccination_schedule": self._handle_vaccination_schedule,
            "emergency_help": self._handle_emergency_help,
            "general_health": self._handle_general_health,
            "outbreak_info": self._handle_outbreak_info,
            "medication_info": self._handle_medication_info,
            "doctor_referral": self._handle_doctor_referral
        }
        
        # Confidence threshold
        self.confidence_threshold = float(os.getenv("CHATBOT_CONFIDENCE_THRESHOLD", "0.7"))
    
    def _load_models(self):
        """Load NLP models for intent classification and entity extraction"""
        try:
            # Load intent classification model if transformers is available
            if pipeline:
                try:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model="microsoft/DialoGPT-medium",
                        return_all_scores=True
                    )
                except Exception as e:
                    logger.warning(f"Could not load transformers model: {str(e)}")
                    self.intent_classifier = None
            
            # Load spaCy model for entity extraction
            if spacy:
                try:
                    self.entity_extractor = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy English model not found. Using fallback.")
                    self.entity_extractor = None
                except Exception as e:
                    logger.warning(f"Could not load spaCy model: {str(e)}")
                    self.entity_extractor = None
                
            logger.info("NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {str(e)}")
            # Fallback to simple keyword-based classification
            self.intent_classifier = None
            self.entity_extractor = None
    
    async def process_message(self, message: str, user_id: str, language: str = "en", platform: str = "web") -> Dict:
        """Process incoming message and generate response (advanced orchestrator + fallback)"""
        try:
            # Translate message to English if needed
            translated_message = message
            if language != "en" and self.translation_service:
                try:
                    translated_message = await self.translation_service.translate_text(message, "en")
                except Exception:
                    translated_message = message

            # Prefer advanced medical orchestrator when available
            if self.medical_orchestrator is not None:
                context = {
                    "user_id": user_id,
                    "platform": platform,
                    "language": language,
                }
                consensus = await self.medical_orchestrator.process_medical_query(translated_message, context)

                response_text = consensus.final_response
                # Append disclaimer from orchestrator
                if consensus.medical_disclaimer and consensus.medical_disclaimer.strip() not in response_text:
                    response_text = f"{response_text}\n\n{consensus.medical_disclaimer.strip()}"

                # Translate back
                if language != "en" and self.translation_service:
                    try:
                        response_text = await self.translation_service.translate_text(response_text, language)
                    except Exception:
                        pass

                result = {
                    "response": response_text,
                    "confidence": float(max(0.0, min(1.0, consensus.consensus_confidence))),
                    "suggested_actions": [
                        "Consult a qualified healthcare provider",
                        "Call emergency services for urgent symptoms"
                    ],
                    "language": language
                }

                # Log conversation
                try:
                    await self.db_manager.log_conversation(
                        user_id=user_id,
                        message=message,
                        intent="medical_consensus",
                        confidence=result["confidence"],
                        entities=[],
                        platform=platform,
                        language=language,
                        response=result["response"]
                    )
                except Exception:
                    pass

                return result

            # Legacy fallback path
            intent, confidence = await self._classify_intent(translated_message)
            entities = await self._extract_entities(translated_message)

            await self.db_manager.log_conversation(
                user_id=user_id,
                message=message,
                intent=intent,
                confidence=confidence,
                entities=entities,
                platform=platform,
                language=language
            )

            if confidence >= self.confidence_threshold and intent in self.intent_mappings:
                response_data = await self.intent_mappings[intent](
                    message=translated_message,
                    entities=entities,
                    user_id=user_id,
                    language=language
                )
            else:
                response_data = await self._handle_fallback(
                    message=translated_message,
                    user_id=user_id,
                    language=language
                )

            if language != "en" and self.translation_service:
                try:
                    response_data["response"] = await self.translation_service.translate_text(
                        response_data["response"], language
                    )
                except Exception:
                    pass

            response_data["confidence"] = confidence
            response_data["language"] = language
            return response_data

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. Please try again or contact a healthcare professional for urgent matters.",
                "confidence": 0.0,
                "suggested_actions": ["Contact nearest health center", "Call emergency helpline"],
                "language": language
            }
    
    async def _classify_intent(self, message: str) -> Tuple[str, float]:
        """Classify the intent of the user message"""
        try:
            if self.intent_classifier:
                results = self.intent_classifier(message)
                # Process results and map to our intents
                # This is a simplified version - in production, you'd train a custom model
                pass
            
            # Fallback to keyword-based classification
            message_lower = message.lower()
            
            # Disease symptoms keywords
            if any(keyword in message_lower for keyword in ["symptom", "pain", "fever", "cough", "headache", "rash"]):
                return "disease_symptoms", 0.8
            
            # Prevention keywords
            if any(keyword in message_lower for keyword in ["prevent", "avoid", "protection", "hygiene", "wash"]):
                return "prevention_tips", 0.8
            
            # Vaccination keywords
            if any(keyword in message_lower for keyword in ["vaccine", "vaccination", "immunization", "shot"]):
                return "vaccination_schedule", 0.8
            
            # Emergency keywords
            if any(keyword in message_lower for keyword in ["emergency", "urgent", "help", "ambulance", "hospital"]):
                return "emergency_help", 0.9
            
            # Outbreak keywords
            if any(keyword in message_lower for keyword in ["outbreak", "epidemic", "pandemic", "alert"]):
                return "outbreak_info", 0.8
            
            # Medication keywords
            if any(keyword in message_lower for keyword in ["medicine", "drug", "medication", "prescription"]):
                return "medication_info", 0.7
            
            # Doctor referral keywords
            if any(keyword in message_lower for keyword in ["doctor", "physician", "specialist", "referral"]):
                return "doctor_referral", 0.7
            
            # Default to general health
            return "general_health", 0.6
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return "general_health", 0.5
    
    async def _extract_entities(self, message: str) -> List[Dict]:
        """Extract entities from the message"""
        entities = []
        
        try:
            if self.entity_extractor:
                doc = self.entity_extractor(message)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            
            # Extract health-related entities using regex patterns
            health_patterns = {
                "disease": r"\b(?:covid|covid-19|malaria|dengue|chikungunya|typhoid|hepatitis|diabetes|hypertension|asthma)\b",
                "symptom": r"\b(?:fever|cough|headache|nausea|vomiting|diarrhea|rash|pain|fatigue|weakness)\b",
                "age": r"\b(?:infant|child|adult|elderly|senior|baby|toddler)\b",
                "location": r"\b(?:village|city|district|state|rural|urban)\b"
            }
            
            for entity_type, pattern in health_patterns.items():
                matches = re.finditer(pattern, message.lower())
                for match in matches:
                    entities.append({
                        "text": match.group(),
                        "label": entity_type,
                        "start": match.start(),
                        "end": match.end()
                    })
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
        
        return entities
    
    async def _handle_disease_symptoms(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle queries about disease symptoms"""
        # Extract disease name from entities
        diseases = [e["text"] for e in entities if e["label"] == "disease"]
        
        if diseases:
            disease_info = await self.health_data_service.get_disease_info(diseases[0], language)
            return {
                "response": f"Here's information about {diseases[0]} symptoms:\n\n{disease_info.get('symptoms', 'Symptoms not available')}",
                "suggested_actions": ["Consult a doctor if symptoms persist", "Monitor symptoms closely"]
            }
        else:
            return {
                "response": "I can help you with information about disease symptoms. Could you please specify which disease you're asking about?",
                "suggested_actions": ["Specify the disease name", "Describe your symptoms"]
            }
    
    async def _handle_prevention_tips(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle queries about prevention tips"""
        diseases = [e["text"] for e in entities if e["label"] == "disease"]
        
        if diseases:
            disease_info = await self.health_data_service.get_disease_info(diseases[0], language)
            prevention_tips = disease_info.get('prevention', 'Prevention tips not available')
        else:
            prevention_tips = await self.health_data_service.get_general_prevention_tips(language)
        
        return {
            "response": f"Here are some prevention tips:\n\n{prevention_tips}",
            "suggested_actions": ["Follow hygiene practices", "Get vaccinated", "Maintain healthy lifestyle"]
        }
    
    async def _handle_vaccination_schedule(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle queries about vaccination schedules"""
        age_groups = [e["text"] for e in entities if e["label"] == "age"]
        
        if age_groups:
            age_group = age_groups[0]
        else:
            age_group = "adult"  # Default
        
        schedule = await self.health_data_service.get_vaccination_schedule(age_group)
        
        return {
            "response": f"Here's the vaccination schedule for {age_group}s:\n\n{schedule}",
            "suggested_actions": ["Schedule vaccination appointment", "Contact nearest health center"]
        }
    
    async def _handle_emergency_help(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle emergency help requests"""
        return {
            "response": "🚨 EMERGENCY ALERT 🚨\n\nFor immediate medical assistance:\n• Call 108 (Emergency Ambulance)\n• Contact nearest hospital\n• Visit emergency room immediately\n\nStay calm and seek professional help right away.",
            "suggested_actions": ["Call emergency helpline 108", "Visit nearest hospital", "Contact family doctor"]
        }
    
    async def _handle_general_health(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle general health queries"""
        general_tips = await self.health_data_service.get_general_health_tips(language)
        
        return {
            "response": f"Here are some general health tips:\n\n{general_tips}\n\nHow can I help you with more specific health information?",
            "suggested_actions": ["Ask about specific diseases", "Get vaccination information", "Learn prevention tips"]
        }
    
    async def _handle_outbreak_info(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle outbreak information queries"""
        alerts = await self.alert_service.get_outbreak_alerts()
        
        if alerts:
            alert_text = "\n".join([f"• {alert['disease']} in {alert['location']}" for alert in alerts])
            return {
                "response": f"Current outbreak alerts:\n\n{alert_text}\n\nStay informed and follow preventive measures.",
                "suggested_actions": ["Follow preventive measures", "Get vaccinated", "Avoid affected areas"]
            }
        else:
            return {
                "response": "No current outbreak alerts. Continue following general health guidelines.",
                "suggested_actions": ["Maintain hygiene", "Get regular checkups", "Stay updated"]
            }
    
    async def _handle_medication_info(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle medication information queries"""
        return {
            "response": "I can provide general information about medications, but please consult a qualified healthcare professional for specific medical advice and prescriptions.",
            "suggested_actions": ["Consult a doctor", "Visit pharmacy", "Check with healthcare provider"]
        }
    
    async def _handle_doctor_referral(self, message: str, entities: List[Dict], user_id: str, language: str) -> Dict:
        """Handle doctor referral requests"""
        return {
            "response": "I can help you find healthcare providers in your area. Please specify your location and the type of specialist you need.",
            "suggested_actions": ["Specify your location", "Mention specialist type", "Contact nearest health center"]
        }
    
    async def _handle_fallback(self, message: str, user_id: str, language: str) -> Dict:
        """Handle cases where intent classification fails"""
        return {
            "response": "I understand you have a health-related question. Could you please rephrase your question or be more specific? I can help with:\n• Disease symptoms and information\n• Prevention tips\n• Vaccination schedules\n• Emergency guidance\n• General health advice",
            "suggested_actions": ["Rephrase your question", "Ask about specific health topics", "Contact healthcare professional"]
        }
