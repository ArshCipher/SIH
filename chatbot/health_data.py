import logging
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class HealthDataService:
    """Service for managing health data and government database integration"""
    
    def __init__(self):
        # Optional health API configuration - will use fallback data if not available
        self.health_api_base_url = os.getenv("HEALTH_API_BASE_URL", "")
        self.health_api_key = os.getenv("HEALTH_API_KEY", "")
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Sample health data (in production, this would come from APIs)
        self.sample_disease_data = {
            "COVID-19": {
                "symptoms": "Fever, cough, shortness of breath, fatigue, body aches, loss of taste or smell, headache, sore throat",
                "prevention": "Wear masks in public, maintain social distancing, wash hands frequently, get vaccinated, avoid crowded places",
                "treatment": "Rest, stay hydrated, monitor symptoms, seek medical help if severe, follow doctor's advice",
                "severity": "moderate",
                "contagious": True,
                "incubation_period": "2-14 days",
                "transmission": "Respiratory droplets, close contact"
            },
            "Malaria": {
                "symptoms": "High fever, chills, headache, nausea, vomiting, muscle pain, fatigue, sweating",
                "prevention": "Use mosquito nets, insect repellent, eliminate standing water, take preventive medication, wear protective clothing",
                "treatment": "Antimalarial medication, rest, hydration, monitor symptoms closely",
                "severity": "high",
                "contagious": False,
                "incubation_period": "7-30 days",
                "transmission": "Mosquito bite"
            },
            "Dengue": {
                "symptoms": "High fever, severe headache, pain behind eyes, muscle and joint pain, rash, nausea, vomiting",
                "prevention": "Eliminate mosquito breeding sites, use repellent, wear protective clothing, avoid peak mosquito hours",
                "treatment": "Rest, hydration, pain relief medication (avoid aspirin), monitor platelet count",
                "severity": "moderate",
                "contagious": False,
                "incubation_period": "3-14 days",
                "transmission": "Mosquito bite"
            },
            "Diabetes": {
                "symptoms": "Increased thirst, frequent urination, extreme fatigue, blurred vision, slow healing wounds, unexplained weight loss",
                "prevention": "Maintain healthy weight, exercise regularly, eat balanced diet, avoid smoking, regular checkups",
                "treatment": "Blood sugar monitoring, medication, diet control, regular exercise, lifestyle changes",
                "severity": "chronic",
                "contagious": False,
                "incubation_period": "N/A",
                "transmission": "Not contagious"
            },
            "Hypertension": {
                "symptoms": "Often asymptomatic, severe headaches, chest pain, dizziness, shortness of breath, nosebleeds",
                "prevention": "Reduce salt intake, exercise regularly, maintain healthy weight, limit alcohol, avoid smoking",
                "treatment": "Medication, lifestyle changes, regular monitoring, stress management",
                "severity": "chronic",
                "contagious": False,
                "incubation_period": "N/A",
                "transmission": "Not contagious"
            }
        }
        
        self.sample_vaccination_schedules = {
            "infant": [
                {
                    "vaccine_name": "BCG",
                    "recommended_age": "At birth",
                    "dosage": "Single dose",
                    "frequency": "Once",
                    "side_effects": "Mild fever, swelling at injection site",
                    "importance": "Protects against tuberculosis"
                },
                {
                    "vaccine_name": "Hepatitis B",
                    "recommended_age": "At birth, 6 weeks, 10 weeks, 14 weeks",
                    "dosage": "4 doses",
                    "frequency": "As per schedule",
                    "side_effects": "Mild fever, soreness at injection site",
                    "importance": "Protects against hepatitis B"
                },
                {
                    "vaccine_name": "OPV",
                    "recommended_age": "6 weeks, 10 weeks, 14 weeks, 16-24 months",
                    "dosage": "4 doses",
                    "frequency": "Primary series + booster",
                    "side_effects": "Rare allergic reactions",
                    "importance": "Protects against polio"
                }
            ],
            "child": [
                {
                    "vaccine_name": "DPT",
                    "recommended_age": "6 weeks, 10 weeks, 14 weeks, 16-24 months",
                    "dosage": "4 doses",
                    "frequency": "Primary series + booster",
                    "side_effects": "Fever, swelling, redness at injection site",
                    "importance": "Protects against diphtheria, pertussis, tetanus"
                },
                {
                    "vaccine_name": "MMR",
                    "recommended_age": "9-12 months, 15-18 months",
                    "dosage": "2 doses",
                    "frequency": "Primary + booster",
                    "side_effects": "Mild fever, rash, joint pain",
                    "importance": "Protects against measles, mumps, rubella"
                },
                {
                    "vaccine_name": "Chickenpox",
                    "recommended_age": "12-15 months",
                    "dosage": "2 doses",
                    "frequency": "Primary + booster",
                    "side_effects": "Mild fever, rash at injection site",
                    "importance": "Protects against chickenpox"
                }
            ],
            "adult": [
                {
                    "vaccine_name": "COVID-19",
                    "recommended_age": "18+ years",
                    "dosage": "2 doses (primary), 1 booster",
                    "frequency": "As recommended by health authorities",
                    "side_effects": "Mild fever, fatigue, muscle pain, headache",
                    "importance": "Protects against COVID-19"
                },
                {
                    "vaccine_name": "Influenza",
                    "recommended_age": "6+ months (annually)",
                    "dosage": "Annual dose",
                    "frequency": "Yearly",
                    "side_effects": "Mild fever, soreness at injection site",
                    "importance": "Protects against seasonal flu"
                },
                {
                    "vaccine_name": "Tetanus",
                    "recommended_age": "Every 10 years",
                    "dosage": "Single dose",
                    "frequency": "Every 10 years",
                    "side_effects": "Mild fever, soreness at injection site",
                    "importance": "Protects against tetanus"
                }
            ]
        }
    
    async def get_disease_info(self, disease_name: str, language: str = "en") -> Optional[Dict]:
        """Get comprehensive disease information"""
        try:
            # Check cache first
            cache_key = f"disease_{disease_name}_{language}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Try to get from external API first
            api_data = await self._fetch_disease_from_api(disease_name)
            if api_data:
                self.cache[cache_key] = (api_data, datetime.now().timestamp())
                return api_data
            
            # Fallback to sample data
            disease_key = self._find_disease_key(disease_name)
            if disease_key and disease_key in self.sample_disease_data:
                disease_info = self.sample_disease_data[disease_key].copy()
                disease_info["disease_name"] = disease_key
                
                # Cache the result
                self.cache[cache_key] = (disease_info, datetime.now().timestamp())
                return disease_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting disease info: {str(e)}")
            return None
    
    async def _fetch_disease_from_api(self, disease_name: str) -> Optional[Dict]:
        """Fetch disease information from external health API (optional)"""
        try:
            # Only try API if both URL and key are provided
            if not self.health_api_base_url or not self.health_api_key:
                logger.info("Health API credentials not configured, using fallback data")
                return None
            
            headers = {
                "Authorization": f"Bearer {self.health_api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "disease": disease_name,
                "format": "json"
            }
            
            response = requests.get(
                f"{self.health_api_base_url}/diseases",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from health API: {str(e)}")
            return None
    
    def _find_disease_key(self, disease_name: str) -> Optional[str]:
        """Find disease key from sample data"""
        disease_lower = disease_name.lower()
        
        for key in self.sample_disease_data.keys():
            if disease_lower in key.lower() or key.lower() in disease_lower:
                return key
        
        return None
    
    async def get_vaccination_schedule(self, age_group: str, language: str = "en") -> List[Dict]:
        """Get vaccination schedule for specific age group"""
        try:
            cache_key = f"vaccination_{age_group}_{language}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Try to get from external API
            api_data = await self._fetch_vaccination_from_api(age_group)
            if api_data:
                self.cache[cache_key] = (api_data, datetime.now().timestamp())
                return api_data
            
            # Fallback to sample data
            if age_group in self.sample_vaccination_schedules:
                schedule = self.sample_vaccination_schedules[age_group]
                self.cache[cache_key] = (schedule, datetime.now().timestamp())
                return schedule
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting vaccination schedule: {str(e)}")
            return []
    
    async def _fetch_vaccination_from_api(self, age_group: str) -> Optional[List[Dict]]:
        """Fetch vaccination schedule from external API (optional)"""
        try:
            # Only try API if both URL and key are provided
            if not self.health_api_base_url or not self.health_api_key:
                logger.info("Health API credentials not configured, using fallback data")
                return None
            
            headers = {
                "Authorization": f"Bearer {self.health_api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "age_group": age_group,
                "format": "json"
            }
            
            response = requests.get(
                f"{self.health_api_base_url}/vaccinations",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching vaccination from API: {str(e)}")
            return None
    
    async def get_general_prevention_tips(self, language: str = "en") -> str:
        """Get general prevention tips"""
        tips = {
            "en": """General Health Prevention Tips:

1. **Personal Hygiene**
   • Wash hands frequently with soap and water
   • Use hand sanitizer when soap is not available
   • Cover mouth and nose when coughing or sneezing

2. **Healthy Lifestyle**
   • Eat a balanced diet with fruits and vegetables
   • Exercise regularly (at least 30 minutes daily)
   • Get adequate sleep (7-9 hours for adults)

3. **Environmental Hygiene**
   • Keep living spaces clean and well-ventilated
   • Dispose of waste properly
   • Avoid stagnant water to prevent mosquito breeding

4. **Regular Health Checkups**
   • Visit healthcare providers regularly
   • Get recommended vaccinations
   • Monitor blood pressure and blood sugar levels

5. **Mental Health**
   • Practice stress management techniques
   • Stay connected with family and friends
   • Seek help when feeling overwhelmed""",
            
            "hi": """सामान्य स्वास्थ्य रोकथाम सुझाव:

1. **व्यक्तिगत स्वच्छता**
   • साबुन और पानी से बार-बार हाथ धोएं
   • जब साबुन उपलब्ध न हो तो हैंड सैनिटाइजर का उपयोग करें
   • खांसते या छींकते समय मुंह और नाक को ढकें

2. **स्वस्थ जीवनशैली**
   • फल और सब्जियों के साथ संतुलित आहार लें
   • नियमित व्यायाम करें (प्रतिदिन कम से कम 30 मिनट)
   • पर्याप्त नींद लें (वयस्कों के लिए 7-9 घंटे)

3. **पर्यावरणीय स्वच्छता**
   • रहने की जगह को साफ और हवादार रखें
   • कचरे का उचित निपटान करें
   • मच्छरों के प्रजनन को रोकने के लिए स्थिर पानी से बचें

4. **नियमित स्वास्थ्य जांच**
   • नियमित रूप से स्वास्थ्य सेवा प्रदाताओं से मिलें
   • अनुशंसित टीकाकरण करवाएं
   • रक्तचाप और रक्त शर्करा के स्तर की निगरानी करें

5. **मानसिक स्वास्थ्य**
   • तनाव प्रबंधन तकनीकों का अभ्यास करें
   • परिवार और दोस्तों के साथ जुड़े रहें
   • अभिभूत महसूस करने पर मदद लें"""
        }
        
        return tips.get(language, tips["en"])
    
    async def get_general_health_tips(self, language: str = "en") -> str:
        """Get general health tips"""
        tips = {
            "en": """General Health Tips:

1. **Stay Hydrated**: Drink at least 8 glasses of water daily
2. **Balanced Diet**: Include fruits, vegetables, whole grains, and lean proteins
3. **Regular Exercise**: Aim for 150 minutes of moderate activity per week
4. **Adequate Sleep**: Get 7-9 hours of quality sleep each night
5. **Stress Management**: Practice meditation, deep breathing, or yoga
6. **Regular Checkups**: Visit your doctor for preventive care
7. **Avoid Smoking**: Quit smoking and avoid secondhand smoke
8. **Limit Alcohol**: Drink alcohol in moderation or avoid it
9. **Sun Protection**: Use sunscreen and protective clothing
10. **Mental Health**: Take care of your emotional well-being""",
            
            "hi": """सामान्य स्वास्थ्य सुझाव:

1. **हाइड्रेटेड रहें**: प्रतिदिन कम से कम 8 गिलास पानी पिएं
2. **संतुलित आहार**: फल, सब्जियां, साबुत अनाज और दुबले प्रोटीन शामिल करें
3. **नियमित व्यायाम**: प्रति सप्ताह 150 मिनट मध्यम गतिविधि का लक्ष्य रखें
4. **पर्याप्त नींद**: हर रात 7-9 घंटे गुणवत्तापूर्ण नींद लें
5. **तनाव प्रबंधन**: ध्यान, गहरी सांस लेने या योग का अभ्यास करें
6. **नियमित जांच**: निवारक देखभाल के लिए अपने डॉक्टर से मिलें
7. **धूम्रपान से बचें**: धूम्रपान छोड़ें और सेकेंडहैंड धुएं से बचें
8. **शराब सीमित करें**: शराब कम मात्रा में पिएं या बचें
9. **सूर्य संरक्षण**: सनस्क्रीन और सुरक्षात्मक कपड़े का उपयोग करें
10. **मानसिक स्वास्थ्य**: अपनी भावनात्मक भलाई का ध्यान रखें"""
        }
        
        return tips.get(language, tips["en"])
    
    async def process_health_query(self, query: str, user_id: str, location: str = None, age_group: str = None) -> Dict:
        """Process complex health queries"""
        try:
            # Analyze the query to determine what information is needed
            query_lower = query.lower()
            
            # Extract relevant information based on query
            if "symptom" in query_lower:
                return await self._process_symptom_query(query, user_id, location, age_group)
            elif "prevention" in query_lower:
                return await self._process_prevention_query(query, user_id, location, age_group)
            elif "treatment" in query_lower:
                return await self._process_treatment_query(query, user_id, location, age_group)
            elif "vaccine" in query_lower or "vaccination" in query_lower:
                return await self._process_vaccination_query(query, user_id, location, age_group)
            else:
                return await self._process_general_query(query, user_id, location, age_group)
                
        except Exception as e:
            logger.error(f"Error processing health query: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your health query. Please try rephrasing your question or contact a healthcare professional.",
                "confidence": 0.0,
                "suggested_actions": ["Rephrase your question", "Contact healthcare provider", "Visit nearest health center"]
            }
    
    async def _process_symptom_query(self, query: str, user_id: str, location: str, age_group: str) -> Dict:
        """Process symptom-related queries"""
        # This would involve more sophisticated NLP to extract symptoms
        # and match them with potential diseases
        return {
            "response": "I can help you understand symptoms. Please describe your symptoms in detail, and I'll provide relevant information. Remember, this is not a substitute for professional medical advice.",
            "confidence": 0.7,
            "suggested_actions": ["Describe symptoms in detail", "Consult a doctor", "Monitor symptoms closely"]
        }
    
    async def _process_prevention_query(self, query: str, user_id: str, location: str, age_group: str) -> Dict:
        """Process prevention-related queries"""
        prevention_tips = await self.get_general_prevention_tips()
        return {
            "response": f"Here are some general prevention tips:\n\n{prevention_tips}",
            "confidence": 0.8,
            "suggested_actions": ["Follow prevention guidelines", "Get vaccinated", "Maintain hygiene"]
        }
    
    async def _process_treatment_query(self, query: str, user_id: str, location: str, age_group: str) -> Dict:
        """Process treatment-related queries"""
        return {
            "response": "I can provide general information about treatments, but please consult a qualified healthcare professional for specific medical advice and treatment plans.",
            "confidence": 0.6,
            "suggested_actions": ["Consult a doctor", "Visit healthcare provider", "Follow medical advice"]
        }
    
    async def _process_vaccination_query(self, query: str, user_id: str, location: str, age_group: str) -> Dict:
        """Process vaccination-related queries"""
        if age_group:
            schedule = await self.get_vaccination_schedule(age_group)
            if schedule:
                schedule_text = "\n".join([
                    f"• {v['vaccine_name']}: {v['recommended_age']} - {v['importance']}"
                    for v in schedule
                ])
                return {
                    "response": f"Vaccination schedule for {age_group}s:\n\n{schedule_text}",
                    "confidence": 0.9,
                    "suggested_actions": ["Schedule vaccination", "Contact health center", "Follow vaccination schedule"]
                }
        
        return {
            "response": "I can help you with vaccination information. Please specify your age group (infant, child, adult) for a personalized vaccination schedule.",
            "confidence": 0.7,
            "suggested_actions": ["Specify age group", "Contact health center", "Get vaccination schedule"]
        }
    
    async def _process_general_query(self, query: str, user_id: str, location: str, age_group: str) -> Dict:
        """Process general health queries"""
        health_tips = await self.get_general_health_tips()
        return {
            "response": f"Here are some general health tips:\n\n{health_tips}\n\nHow can I help you with more specific health information?",
            "confidence": 0.6,
            "suggested_actions": ["Ask specific health questions", "Get disease information", "Learn about prevention"]
        }
