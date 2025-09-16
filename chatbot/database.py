import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

logger = logging.getLogger(__name__)

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    response = Column(Text)
    intent = Column(String)
    confidence = Column(Float)
    entities = Column(Text)  # JSON string
    platform = Column(String)
    language = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    phone_number = Column(String)
    language_preference = Column(String, default="en")
    location = Column(String)
    age_group = Column(String)
    vaccination_status = Column(Text)  # JSON string
    health_conditions = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DiseaseInfo(Base):
    __tablename__ = "disease_info"
    
    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String, index=True)
    symptoms = Column(Text)
    prevention = Column(Text)
    treatment = Column(Text)
    severity = Column(String)
    contagious = Column(Boolean)
    language = Column(String, default="en")
    last_updated = Column(DateTime, default=datetime.utcnow)

class VaccinationSchedule(Base):
    __tablename__ = "vaccination_schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    age_group = Column(String, index=True)
    vaccine_name = Column(String)
    recommended_age = Column(String)
    dosage = Column(String)
    frequency = Column(String)
    side_effects = Column(Text)
    language = Column(String, default="en")

class OutbreakAlert(Base):
    __tablename__ = "outbreak_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String, index=True)
    location = Column(String)
    severity = Column(String)
    cases_count = Column(Integer)
    alert_level = Column(String)
    prevention_measures = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

class DatabaseManager:
    """Database manager for handling all data operations"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./health_chatbot.db")
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        # Initialize with sample data
        self._initialize_sample_data()
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def _initialize_sample_data(self):
        """Initialize database with sample health data"""
        session = self.get_session()
        
        try:
            # Check if data already exists
            if session.query(DiseaseInfo).count() > 0:
                return
            
            # Sample disease information
            diseases = [
                {
                    "disease_name": "COVID-19",
                    "symptoms": "Fever, cough, shortness of breath, fatigue, body aches, loss of taste or smell",
                    "prevention": "Wear masks, maintain social distancing, wash hands frequently, get vaccinated",
                    "treatment": "Rest, stay hydrated, monitor symptoms, seek medical help if severe",
                    "severity": "moderate",
                    "contagious": True
                },
                {
                    "disease_name": "Malaria",
                    "symptoms": "Fever, chills, headache, nausea, vomiting, muscle pain",
                    "prevention": "Use mosquito nets, insect repellent, avoid stagnant water, take preventive medication",
                    "treatment": "Antimalarial medication, rest, hydration",
                    "severity": "high",
                    "contagious": False
                },
                {
                    "disease_name": "Dengue",
                    "symptoms": "High fever, severe headache, pain behind eyes, muscle and joint pain, rash",
                    "prevention": "Eliminate mosquito breeding sites, use repellent, wear protective clothing",
                    "treatment": "Rest, hydration, pain relief medication, avoid aspirin",
                    "severity": "moderate",
                    "contagious": False
                },
                {
                    "disease_name": "Diabetes",
                    "symptoms": "Increased thirst, frequent urination, extreme fatigue, blurred vision, slow healing",
                    "prevention": "Maintain healthy weight, exercise regularly, eat balanced diet, avoid smoking",
                    "treatment": "Blood sugar monitoring, medication, diet control, regular exercise",
                    "severity": "chronic",
                    "contagious": False
                }
            ]
            
            for disease_data in diseases:
                disease = DiseaseInfo(**disease_data)
                session.add(disease)
            
            # Sample vaccination schedules
            vaccinations = [
                {
                    "age_group": "infant",
                    "vaccine_name": "BCG",
                    "recommended_age": "At birth",
                    "dosage": "Single dose",
                    "frequency": "Once",
                    "side_effects": "Mild fever, swelling at injection site"
                },
                {
                    "age_group": "infant",
                    "vaccine_name": "Hepatitis B",
                    "recommended_age": "At birth, 6 weeks, 10 weeks, 14 weeks",
                    "dosage": "4 doses",
                    "frequency": "As per schedule",
                    "side_effects": "Mild fever, soreness at injection site"
                },
                {
                    "age_group": "child",
                    "vaccine_name": "DPT",
                    "recommended_age": "6 weeks, 10 weeks, 14 weeks, 16-24 months",
                    "dosage": "4 doses",
                    "frequency": "Primary series + booster",
                    "side_effects": "Fever, swelling, redness at injection site"
                },
                {
                    "age_group": "adult",
                    "vaccine_name": "COVID-19",
                    "recommended_age": "18+ years",
                    "dosage": "2 doses (primary), 1 booster",
                    "frequency": "As recommended",
                    "side_effects": "Mild fever, fatigue, muscle pain, headache"
                }
            ]
            
            for vaccine_data in vaccinations:
                vaccine = VaccinationSchedule(**vaccine_data)
                session.add(vaccine)
            
            session.commit()
            logger.info("Sample data initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sample data: {str(e)}")
            session.rollback()
        finally:
            session.close()
    
    async def log_conversation(self, user_id: str, message: str, intent: str, 
                             confidence: float, entities: List[Dict], 
                             platform: str, language: str, response: str = None):
        """Log conversation data"""
        session = self.get_session()
        
        try:
            conversation = Conversation(
                user_id=user_id,
                message=message,
                response=response,
                intent=intent,
                confidence=confidence,
                entities=json.dumps(entities),
                platform=platform,
                language=language
            )
            session.add(conversation)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error logging conversation: {str(e)}")
            session.rollback()
        finally:
            session.close()
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by user ID"""
        session = self.get_session()
        
        try:
            profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                return {
                    "user_id": profile.user_id,
                    "phone_number": profile.phone_number,
                    "language_preference": profile.language_preference,
                    "location": profile.location,
                    "age_group": profile.age_group,
                    "vaccination_status": json.loads(profile.vaccination_status) if profile.vaccination_status else {},
                    "health_conditions": json.loads(profile.health_conditions) if profile.health_conditions else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return None
        finally:
            session.close()
    
    async def update_user_profile(self, user_id: str, profile_data: Dict):
        """Update or create user profile"""
        session = self.get_session()
        
        try:
            profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            
            if profile:
                # Update existing profile
                for key, value in profile_data.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                profile.updated_at = datetime.utcnow()
            else:
                # Create new profile
                profile_data["user_id"] = user_id
                profile = UserProfile(**profile_data)
                session.add(profile)
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            session.rollback()
        finally:
            session.close()
    
    async def get_disease_info(self, disease_name: str, language: str = "en") -> Optional[Dict]:
        """Get disease information from database"""
        session = self.get_session()
        
        try:
            disease = session.query(DiseaseInfo).filter(
                DiseaseInfo.disease_name.ilike(f"%{disease_name}%"),
                DiseaseInfo.language == language
            ).first()
            
            if disease:
                return {
                    "disease_name": disease.disease_name,
                    "symptoms": disease.symptoms,
                    "prevention": disease.prevention,
                    "treatment": disease.treatment,
                    "severity": disease.severity,
                    "contagious": disease.contagious
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting disease info: {str(e)}")
            return None
        finally:
            session.close()
    
    async def get_vaccination_schedule(self, age_group: str, language: str = "en") -> List[Dict]:
        """Get vaccination schedule for age group"""
        session = self.get_session()
        
        try:
            vaccines = session.query(VaccinationSchedule).filter(
                VaccinationSchedule.age_group == age_group,
                VaccinationSchedule.language == language
            ).all()
            
            return [
                {
                    "vaccine_name": v.vaccine_name,
                    "recommended_age": v.recommended_age,
                    "dosage": v.dosage,
                    "frequency": v.frequency,
                    "side_effects": v.side_effects
                }
                for v in vaccines
            ]
            
        except Exception as e:
            logger.error(f"Error getting vaccination schedule: {str(e)}")
            return []
        finally:
            session.close()
    
    async def get_outbreak_alerts(self, location: str = None) -> List[Dict]:
        """Get current outbreak alerts"""
        session = self.get_session()
        
        try:
            query = session.query(OutbreakAlert).filter(
                OutbreakAlert.expires_at > datetime.utcnow()
            )
            
            if location:
                query = query.filter(OutbreakAlert.location.ilike(f"%{location}%"))
            
            alerts = query.all()
            
            return [
                {
                    "disease_name": a.disease_name,
                    "location": a.location,
                    "severity": a.severity,
                    "cases_count": a.cases_count,
                    "alert_level": a.alert_level,
                    "prevention_measures": a.prevention_measures,
                    "created_at": a.created_at.isoformat()
                }
                for a in alerts
            ]
            
        except Exception as e:
            logger.error(f"Error getting outbreak alerts: {str(e)}")
            return []
        finally:
            session.close()
    
    async def add_outbreak_alert(self, alert_data: Dict):
        """Add new outbreak alert"""
        session = self.get_session()
        
        try:
            alert = OutbreakAlert(**alert_data)
            session.add(alert)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error adding outbreak alert: {str(e)}")
            session.rollback()
        finally:
            session.close()
    
    async def get_analytics_data(self) -> Dict:
        """Get analytics data for dashboard"""
        session = self.get_session()
        
        try:
            # Total conversations
            total_conversations = session.query(Conversation).count()
            
            # Conversations by platform
            platform_stats = {}
            platforms = session.query(Conversation.platform).distinct().all()
            for platform in platforms:
                count = session.query(Conversation).filter(Conversation.platform == platform[0]).count()
                platform_stats[platform[0]] = count
            
            # Conversations by language
            language_stats = {}
            languages = session.query(Conversation.language).distinct().all()
            for language in languages:
                count = session.query(Conversation).filter(Conversation.language == language[0]).count()
                language_stats[language[0]] = count
            
            # Intent distribution
            intent_stats = {}
            intents = session.query(Conversation.intent).distinct().all()
            for intent in intents:
                count = session.query(Conversation).filter(Conversation.intent == intent[0]).count()
                intent_stats[intent[0]] = count
            
            # Recent activity (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_conversations = session.query(Conversation).filter(
                Conversation.timestamp >= week_ago
            ).count()
            
            return {
                "total_conversations": total_conversations,
                "platform_distribution": platform_stats,
                "language_distribution": language_stats,
                "intent_distribution": intent_stats,
                "recent_activity": recent_conversations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {str(e)}")
            return {}
        finally:
            session.close()
