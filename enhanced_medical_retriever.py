"""
Enhanced Medical Knowledge Retrieval System
Integrates with comprehensive 500+ disease database for superior medical chatbot responses
Now enhanced with real-time medical APIs for the latest information
"""

import sqlite3
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import re

# Real-time medical APIs import
try:
    from immediate_medical_apis import ImmediateMedicalAPIs
    IMMEDIATE_APIS_AVAILABLE = True
except ImportError:
    IMMEDIATE_APIS_AVAILABLE = False

# Optional imports with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DiseaseMatch:
    """Disease matching result with confidence scores"""
    disease_name: str
    disease_name_hindi: str
    disease_category: str
    confidence_score: float
    symptom_match_score: float
    severity: str
    contagious: bool
    symptoms: str
    treatment: str
    prevention: str
    complications: str
    early_symptoms: str
    severe_symptoms: str
    affected_states: str
    prevalence_india: float

@dataclass
class MedicalResponse:
    """Comprehensive medical response structure"""
    primary_response: str
    disease_matches: List[DiseaseMatch]
    confidence_level: float
    safety_warnings: List[str]
    recommended_actions: List[str]
    similar_diseases: List[str]
    prevention_tips: List[str]
    when_to_seek_help: str
    data_sources: List[str]

class EnhancedMedicalKnowledgeRetriever:
    """Enhanced medical knowledge retrieval system"""
    
    def __init__(self, db_path: str = "comprehensive_medical_database.db"):
        self.db_path = db_path
        self.vectorizer = None
        self.disease_vectors = None
        self.diseases_data = []
        self.sentence_model = None
        
        # Verify comprehensive database exists, fallback if needed
        self._verify_database()
        
        # Initialize components
        self._initialize_vectorizer()
        self._load_disease_data()
        self._create_disease_vectors()
        
        # Initialize immediate medical APIs
        self.immediate_apis = None
        if IMMEDIATE_APIS_AVAILABLE:
            try:
                self.immediate_apis = ImmediateMedicalAPIs()
                logger.info("Immediate medical APIs initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize immediate APIs: {e}")
    
    def _verify_database(self):
        """Verify comprehensive database exists and has data"""
        try:
            import os
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM diseases")
                count = cursor.fetchone()[0]
                if count > 50:  # Comprehensive database should have many diseases
                    logger.info(f"Using comprehensive database with {count} diseases")
                    conn.close()
                    return
                conn.close()
            
            # Fallback to basic database
            fallback_paths = ["enhanced_medical_database.db", "health_chatbot.db"]
            for fallback_path in fallback_paths:
                if os.path.exists(fallback_path):
                    self.db_path = fallback_path
                    logger.warning(f"Falling back to database: {fallback_path}")
                    return
            
            logger.warning("No suitable database found")
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
        
    def _initialize_vectorizer(self):
        """Initialize text vectorization components"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use medical-specific sentence transformer if available
                try:
                    self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    logger.info("Loaded SentenceTransformer model")
                except:
                    logger.warning("Failed to load SentenceTransformer, falling back to TF-IDF")
                    
            if SKLEARN_AVAILABLE and self.sentence_model is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    analyzer='word'
                )
                logger.info("Initialized TF-IDF vectorizer")
                
        except Exception as e:
            logger.error(f"Error initializing vectorizer: {e}")
    
    def _load_disease_data(self):
        """Load comprehensive disease data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check which schema is available
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'diseases' in tables:
                # Use comprehensive database schema
                cursor.execute("""
                SELECT name, category, icd10_code, symptoms, treatment, 
                       prevention, prevalence_india, severity, contagious, 
                       emergency_indicators, complications
                FROM diseases
                ORDER BY name
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    disease_data = {
                        'disease_name': row[0],
                        'disease_name_hindi': '',  # Add Hindi names later
                        'disease_category': row[1] or '',
                        'icd_10_code': row[2] or '',
                        'symptoms': row[3] or '',
                        'treatment': row[4] or '',
                        'prevention': row[5] or '',
                        'prevalence_india': row[6] or '',
                        'severity': row[7] or 'moderate',
                        'contagious': bool(row[8]) if row[8] is not None else False,
                        'emergency_indicators': row[9] or '',
                        'complications': row[10] or '',
                        'early_symptoms': '',
                        'severe_symptoms': '',
                        'causative_agent': '',
                        'disease_type': row[1] or '',
                        'transmission_mode': '',
                        'affected_states': '',
                        'data_source': 'comprehensive_database',
                        'confidence_score': 0.95
                    }
                    self.diseases_data.append(disease_data)
                    
            elif 'enhanced_disease_info' in tables:
                # Use enhanced database schema (fallback)
                cursor.execute("""
                SELECT disease_name, disease_name_hindi, disease_category, icd_10_code,
                       symptoms, early_symptoms, severe_symptoms, causative_agent,
                       disease_type, transmission_mode, prevention, treatment,
                       prevalence_india, severity, contagious, complications,
                       affected_states, data_source, confidence_score
                FROM enhanced_disease_info
                ORDER BY prevalence_india DESC
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    disease_data = {
                        'disease_name': row[0],
                        'disease_name_hindi': row[1] or '',
                        'disease_category': row[2] or '',
                        'icd_10_code': row[3] or '',
                        'symptoms': row[4] or '',
                        'early_symptoms': row[5] or '',
                        'severe_symptoms': row[6] or '',
                        'causative_agent': row[7] or '',
                        'disease_type': row[8] or '',
                        'transmission_mode': row[9] or '',
                        'prevention': row[10] or '',
                        'treatment': row[11] or '',
                        'prevalence_india': row[12] or '',
                        'severity': row[13] or 'moderate',
                        'contagious': bool(row[14]) if row[14] is not None else False,
                        'complications': row[15] or '',
                        'affected_states': row[16] or '',
                        'data_source': row[17] or 'enhanced_database',
                        'confidence_score': float(row[18]) if row[18] else 0.8
                    }
                    self.diseases_data.append(disease_data)
            
            else:
                # Basic fallback
                logger.warning("No comprehensive disease tables found, using basic schema")
                cursor.execute("SELECT * FROM disease_info LIMIT 10")
                rows = cursor.fetchall()
                # Basic processing for fallback...
            
            conn.close()
            logger.info(f"Loaded {len(self.diseases_data)} diseases from database")
            
        except Exception as e:
            logger.error(f"Error loading disease data: {e}")
            self.diseases_data = []
    
    def _create_disease_vectors(self):
        """Create vector representations of disease information"""
        if not self.diseases_data:
            return
            
        try:
            # Combine all relevant text fields for each disease
            disease_texts = []
            for disease in self.diseases_data:
                combined_text = f"{disease['disease_name']} {disease['symptoms']} {disease['early_symptoms']} {disease['severe_symptoms']} {disease['disease_category']} {disease['causative_agent']} {disease['complications']}"
                disease_texts.append(combined_text)
            
            if self.sentence_model:
                # Use sentence transformers
                self.disease_vectors = self.sentence_model.encode(disease_texts)
                logger.info("Created disease vectors using SentenceTransformer")
            elif self.vectorizer:
                # Use TF-IDF
                self.disease_vectors = self.vectorizer.fit_transform(disease_texts)
                logger.info("Created disease vectors using TF-IDF")
            else:
                logger.warning("No vectorizer available, using simple text matching")
                
        except Exception as e:
            logger.error(f"Error creating disease vectors: {e}")
    
    def search_diseases_by_symptoms(self, query: str, top_k: int = 5) -> List[DiseaseMatch]:
        """Search diseases based on symptoms with advanced matching"""
        if not self.diseases_data:
            return []
        
        matches = []
        query_lower = query.lower()
        
        try:
            if self.sentence_model and self.disease_vectors is not None:
                # Vector-based similarity search
                query_vector = self.sentence_model.encode([query])
                similarities = cosine_similarity(query_vector, self.disease_vectors)[0]
                
                # Get top matches
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering
                
                for idx in top_indices:
                    if similarities[idx] > 0.05:  # Lowered similarity threshold
                        disease = self.diseases_data[idx]
                        symptom_match_score = self._calculate_symptom_match(query_lower, disease)
                        
                        if symptom_match_score > 0.1:  # Lowered symptom match threshold
                            match = DiseaseMatch(
                                disease_name=disease['disease_name'],
                                disease_name_hindi=disease['disease_name_hindi'],
                                disease_category=disease['disease_category'],
                                confidence_score=float(similarities[idx]),
                                symptom_match_score=symptom_match_score,
                                severity=disease['severity'],
                                contagious=disease['contagious'],
                                symptoms=disease['symptoms'],
                                treatment=disease['treatment'],
                                prevention=disease['prevention'],
                                complications=disease['complications'],
                                early_symptoms=disease['early_symptoms'],
                                severe_symptoms=disease['severe_symptoms'],
                                affected_states=disease['affected_states'],
                                prevalence_india=disease['prevalence_india']
                            )
                            matches.append(match)
            
            elif self.vectorizer and self.disease_vectors is not None:
                # TF-IDF based search
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.disease_vectors)[0]
                
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]
                
                for idx in top_indices:
                    if similarities[idx] > 0.05:
                        disease = self.diseases_data[idx]
                        symptom_match_score = self._calculate_symptom_match(query_lower, disease)
                        
                        if symptom_match_score > 0.2:
                            match = DiseaseMatch(
                                disease_name=disease['disease_name'],
                                disease_name_hindi=disease['disease_name_hindi'],
                                disease_category=disease['disease_category'],
                                confidence_score=float(similarities[idx]),
                                symptom_match_score=symptom_match_score,
                                severity=disease['severity'],
                                contagious=disease['contagious'],
                                symptoms=disease['symptoms'],
                                treatment=disease['treatment'],
                                prevention=disease['prevention'],
                                complications=disease['complications'],
                                early_symptoms=disease['early_symptoms'],
                                severe_symptoms=disease['severe_symptoms'],
                                affected_states=disease['affected_states'],
                                prevalence_india=disease['prevalence_india']
                            )
                            matches.append(match)
            
            else:
                # Fallback to simple text matching
                matches = self._simple_text_search(query_lower, top_k)
            
            # Sort by combined score and return top_k
            matches.sort(key=lambda x: (x.symptom_match_score + x.confidence_score) / 2, reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error in disease search: {e}")
            return self._simple_text_search(query_lower, top_k)
    
    def _calculate_symptom_match(self, query: str, disease: Dict) -> float:
        """Calculate how well query symptoms match disease symptoms"""
        try:
            # Extract keywords from query
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            
            # Get all symptom text from disease
            all_symptoms = f"{disease['symptoms']} {disease['early_symptoms']} {disease['severe_symptoms']}".lower()
            disease_keywords = set(re.findall(r'\b\w+\b', all_symptoms))
            
            # Enhanced symptom matching for common medical terms
            common_symptoms = {
                'fever': ['fever', 'high temperature', 'pyrexia', 'temperature'],
                'headache': ['headache', 'head pain', 'cephalgia'],
                'cough': ['cough', 'coughing', 'persistent cough'],
                'pain': ['pain', 'ache', 'aching', 'hurt', 'hurting'],
                'fatigue': ['fatigue', 'tired', 'tiredness', 'weakness', 'weak'],
                'nausea': ['nausea', 'feeling sick', 'queasy'],
                'vomiting': ['vomiting', 'throwing up', 'emesis'],
                'diarrhea': ['diarrhea', 'loose stools', 'watery stools'],
                'stomach': ['stomach', 'abdominal', 'belly', 'gastric'],
                'chest': ['chest', 'thoracic', 'breast'],
                'breathing': ['breathing', 'breath', 'respiratory', 'dyspnea']
            }
            
            # Calculate base Jaccard similarity
            intersection = len(query_keywords.intersection(disease_keywords))
            union = len(query_keywords.union(disease_keywords))
            
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # Enhanced matching for common symptoms
            enhanced_score = 0.0
            for word in query_keywords:
                # Direct match
                if word in disease_keywords:
                    enhanced_score += 0.3
                
                # Synonym matching
                for symptom, synonyms in common_symptoms.items():
                    if word in synonyms:
                        for synonym in synonyms:
                            if synonym in all_symptoms:
                                enhanced_score += 0.4
                                break
            
            # Boost score for exact symptom phrases
            exact_matches = 0
            for symptom_phrase in query.split(','):
                symptom_phrase = symptom_phrase.strip()
                if len(symptom_phrase) > 3 and symptom_phrase in all_symptoms:
                    exact_matches += 1
            
            exact_match_boost = min(exact_matches * 0.2, 0.5)
            
            # Combine scores with weights
            final_score = (jaccard_score * 0.3) + (enhanced_score * 0.6) + exact_match_boost
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating symptom match: {e}")
            return 0.0
    
    def _simple_text_search(self, query: str, top_k: int) -> List[DiseaseMatch]:
        """Fallback simple text-based search"""
        matches = []
        
        for disease in self.diseases_data:
            # Simple keyword matching
            all_text = f"{disease['disease_name']} {disease['symptoms']} {disease['early_symptoms']} {disease['severe_symptoms']}".lower()
            
            score = 0.0
            for word in query.split():
                if len(word) > 2 and word in all_text:
                    score += 1
            
            if score > 0:
                symptom_match_score = self._calculate_symptom_match(query, disease)
                
                match = DiseaseMatch(
                    disease_name=disease['disease_name'],
                    disease_name_hindi=disease['disease_name_hindi'],
                    disease_category=disease['disease_category'],
                    confidence_score=score / len(query.split()),
                    symptom_match_score=symptom_match_score,
                    severity=disease['severity'],
                    contagious=disease['contagious'],
                    symptoms=disease['symptoms'],
                    treatment=disease['treatment'],
                    prevention=disease['prevention'],
                    complications=disease['complications'],
                    early_symptoms=disease['early_symptoms'],
                    severe_symptoms=disease['severe_symptoms'],
                    affected_states=disease['affected_states'],
                    prevalence_india=disease['prevalence_india']
                )
                matches.append(match)
        
        matches.sort(key=lambda x: x.symptom_match_score + x.confidence_score, reverse=True)
        return matches[:top_k]
    
    def get_disease_by_name(self, disease_name: str) -> Optional[DiseaseMatch]:
        """Get specific disease information by name"""
        for disease in self.diseases_data:
            if (disease['disease_name'].lower() == disease_name.lower() or 
                disease['disease_name_hindi'] == disease_name):
                
                return DiseaseMatch(
                    disease_name=disease['disease_name'],
                    disease_name_hindi=disease['disease_name_hindi'],
                    disease_category=disease['disease_category'],
                    confidence_score=1.0,
                    symptom_match_score=1.0,
                    severity=disease['severity'],
                    contagious=disease['contagious'],
                    symptoms=disease['symptoms'],
                    treatment=disease['treatment'],
                    prevention=disease['prevention'],
                    complications=disease['complications'],
                    early_symptoms=disease['early_symptoms'],
                    severe_symptoms=disease['severe_symptoms'],
                    affected_states=disease['affected_states'],
                    prevalence_india=disease['prevalence_india']
                )
        return None
    
    def generate_comprehensive_response(self, query: str) -> MedicalResponse:
        """Generate comprehensive medical response with safety considerations"""
        # Search for matching diseases
        disease_matches = self.search_diseases_by_symptoms(query, top_k=3)
        
        if not disease_matches:
            return MedicalResponse(
                primary_response="I couldn't find specific matches for your symptoms. Please consult a healthcare professional for proper diagnosis.",
                disease_matches=[],
                confidence_level=0.0,
                safety_warnings=["Always consult a qualified healthcare professional for medical advice"],
                recommended_actions=["See a doctor for proper diagnosis", "Monitor symptoms carefully"],
                similar_diseases=[],
                prevention_tips=[],
                when_to_seek_help="Seek immediate medical attention if symptoms worsen or persist",
                data_sources=[]
            )
        
        # Generate primary response
        top_match = disease_matches[0]
        primary_response = self._generate_primary_response(top_match, query)
        
        # Calculate overall confidence
        avg_confidence = sum(match.confidence_score + match.symptom_match_score for match in disease_matches) / (2 * len(disease_matches))
        
        # Generate safety warnings
        safety_warnings = self._generate_safety_warnings(disease_matches)
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(disease_matches)
        
        # Find similar diseases
        similar_diseases = self._find_similar_diseases(top_match)
        
        # Generate prevention tips
        prevention_tips = self._extract_prevention_tips(disease_matches)
        
        # Determine when to seek help
        when_to_seek_help = self._generate_seek_help_advice(disease_matches)
        
        # Collect data sources
        data_sources = list(set(disease['data_source'] for disease in self.diseases_data 
                                if disease['disease_name'] in [match.disease_name for match in disease_matches]))
        
        response = MedicalResponse(
            primary_response=primary_response,
            disease_matches=disease_matches,
            confidence_level=avg_confidence,
            safety_warnings=safety_warnings,
            recommended_actions=recommended_actions,
            similar_diseases=similar_diseases,
            prevention_tips=prevention_tips,
            when_to_seek_help=when_to_seek_help,
            data_sources=data_sources
        )
        
        # Enhance with live medical data
        response = self.enhance_response_with_live_data(response, query)
        
        return response
    
    def _generate_primary_response(self, top_match: DiseaseMatch, query: str) -> str:
        """Generate the primary response text"""
        confidence_text = "Based on the symptoms you described" if top_match.symptom_match_score > 0.5 else "Your symptoms might be related to"
        
        response = f"{confidence_text}, this could indicate **{top_match.disease_name}** ({top_match.disease_category}).\n\n"
        
        if top_match.early_symptoms:
            response += f"**Early symptoms typically include:** {top_match.early_symptoms}\n\n"
        
        if top_match.treatment:
            response += f"**Common treatment approaches:** {top_match.treatment}\n\n"
        
        if top_match.contagious:
            response += "⚠️ **Note:** This condition is contagious. Take appropriate precautions.\n\n"
        
        if top_match.severity == "high":
            response += "🚨 **Important:** This is considered a serious condition. Seek medical attention promptly.\n\n"
        
        response += "**Disclaimer:** This information is for educational purposes only. Please consult a healthcare professional for proper diagnosis and treatment."
        
        return response
    
    def _generate_safety_warnings(self, matches: List[DiseaseMatch]) -> List[str]:
        """Generate safety warnings based on disease matches"""
        warnings = ["This information is for educational purposes only - not a substitute for professional medical advice"]
        
        for match in matches:
            if match.severity == "high":
                warnings.append(f"{match.disease_name} can be serious - seek immediate medical attention")
            
            if match.contagious:
                warnings.append(f"{match.disease_name} is contagious - take isolation precautions")
            
            if "emergency" in match.severe_symptoms.lower():
                warnings.append("Some symptoms require emergency medical care")
        
        return list(set(warnings))
    
    def _generate_recommendations(self, matches: List[DiseaseMatch]) -> List[str]:
        """Generate recommended actions"""
        recommendations = ["Consult a healthcare professional for proper diagnosis"]
        
        for match in matches:
            if match.severity == "high":
                recommendations.append("Seek urgent medical attention")
            elif match.severity == "moderate":
                recommendations.append("Schedule an appointment with your doctor")
            
            if match.contagious:
                recommendations.append("Isolate yourself to prevent transmission")
            
            if "rest" in match.treatment.lower():
                recommendations.append("Get adequate rest and sleep")
            
            if "hydration" in match.treatment.lower() or "fluid" in match.treatment.lower():
                recommendations.append("Stay well hydrated")
        
        return list(set(recommendations))
    
    def _find_similar_diseases(self, top_match: DiseaseMatch) -> List[str]:
        """Find diseases with similar symptoms"""
        similar = []
        
        for disease in self.diseases_data:
            if (disease['disease_name'] != top_match.disease_name and 
                disease['disease_category'] == top_match.disease_category):
                similar.append(disease['disease_name'])
        
        return similar[:3]
    
    def _extract_prevention_tips(self, matches: List[DiseaseMatch]) -> List[str]:
        """Extract prevention tips from disease matches"""
        tips = []
        
        for match in matches:
            if match.prevention:
                # Split prevention text into individual tips
                prevention_parts = match.prevention.split(',')
                for part in prevention_parts:
                    tip = part.strip()
                    if len(tip) > 10:
                        tips.append(tip)
        
        return list(set(tips))[:5]
    
    def _generate_seek_help_advice(self, matches: List[DiseaseMatch]) -> str:
        """Generate advice on when to seek medical help"""
        has_high_severity = any(match.severity == "high" for match in matches)
        has_contagious = any(match.contagious for match in matches)
        
        if has_high_severity:
            return "Seek immediate medical attention, especially if symptoms worsen or you experience severe manifestations."
        elif has_contagious:
            return "Consult a healthcare provider soon to confirm diagnosis and prevent transmission to others."
        else:
            return "Schedule an appointment with your healthcare provider for proper evaluation and treatment."

    def enhance_response_with_live_data(self, response: MedicalResponse, query: str) -> MedicalResponse:
        """Enhance response with real-time medical API data"""
        if not self.immediate_apis:
            return response
            
        try:
            enhanced_sources = list(response.data_sources)
            enhanced_recommendations = list(response.recommended_actions)
            
            # Extract potential drug names from query or disease matches
            drug_keywords = self._extract_drug_keywords_from_response(query, response)
            
            # Get FDA drug information
            for drug in drug_keywords[:2]:  # Limit to prevent overload
                fda_data = self.immediate_apis.get_fda_drug_info(drug)
                if fda_data.get("status") == "success" and fda_data.get("results"):
                    enhanced_sources.append(f"FDA Database - {drug}")
                    enhanced_recommendations.append(f"Current FDA data available for {drug} - verify with healthcare provider")
            
            # Get latest medical research
            mesh_data = self.immediate_apis.get_mesh_disease_info(query)
            if mesh_data.get("status") == "success":
                enhanced_sources.append("PubMed/MeSH Medical Research Database")
                enhanced_recommendations.append("Latest medical research available - discuss with healthcare provider")
            
            # Get drug interaction data
            if drug_keywords:
                rxnorm_data = self.immediate_apis.get_rxnorm_drug_info(drug_keywords[0])
                if rxnorm_data.get("status") == "success":
                    enhanced_sources.append("RxNorm Drug Database")
                    enhanced_recommendations.append("Current drug interaction data available - review with pharmacist")
            
            # Update response with enhanced data
            response.data_sources = enhanced_sources
            response.recommended_actions = enhanced_recommendations
            
            # Add note about real-time data
            if len(enhanced_sources) > len(response.data_sources):
                response.primary_response += "\n\n*Enhanced with real-time medical database information*"
                
        except Exception as e:
            logger.warning(f"Failed to enhance response with live data: {e}")
            
        return response
    
    def _extract_drug_keywords_from_response(self, query: str, response: MedicalResponse) -> List[str]:
        """Extract potential drug names from query and response"""
        drug_keywords = []
        
        # Simple drug name extraction - can be enhanced with medical NER
        drug_indicators = ['medication', 'drug', 'pill', 'medicine', 'tablet', 'treatment']
        
        # Check query
        words = query.lower().split()
        for i, word in enumerate(words):
            if any(indicator in word for indicator in drug_indicators):
                # Look for following words that might be drug names
                for j in range(i+1, min(i+3, len(words))):
                    if words[j].isalpha() and len(words[j]) > 3:
                        drug_keywords.append(words[j])
        
        # Check treatment information in disease matches
        for match in response.disease_matches:
            treatment_words = match.treatment.lower().split()
            for word in treatment_words:
                if word.isalpha() and len(word) > 5 and word not in ['treatment', 'medication', 'therapy']:
                    drug_keywords.append(word)
        
        return list(set(drug_keywords))[:3]  # Return unique drugs, max 3


# Usage example and testing
if __name__ == "__main__":
    # Initialize the enhanced retriever
    retriever = EnhancedMedicalKnowledgeRetriever()
    
    # Test queries
    test_queries = [
        "I have fever, cough and difficulty breathing",
        "Joint pain and morning stiffness",
        "Severe headache and sensitivity to light",
        "Persistent fatigue and weight loss",
        "Chest pain and shortness of breath"
    ]
    
    print("Enhanced Medical Knowledge Retriever Test Results")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        response = retriever.generate_comprehensive_response(query)
        
        print(f"Confidence Level: {response.confidence_level:.2f}")
        print(f"Primary Response: {response.primary_response[:200]}...")
        
        if response.disease_matches:
            print(f"Top Disease Match: {response.disease_matches[0].disease_name}")
            print(f"Match Score: {response.disease_matches[0].symptom_match_score:.2f}")
        
        print(f"Safety Warnings: {len(response.safety_warnings)}")
        print(f"Recommendations: {len(response.recommended_actions)}")
        print("="* 60)