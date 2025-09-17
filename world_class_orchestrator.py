"""
Enhanced Medical Orchestrator with World-Class Database Integration
Combines our local database with UMLS, SNOMED CT, WHO, and PubMed APIs
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from dataclasses import dataclass

# Import our existing components
from chatbot.medical_orchestrator import MedicalOrchestrator
from chatbot.enhanced_medical_retriever import EnhancedMedicalRetriever
from chatbot.medical_safety import MedicalSafetyValidator

# Import world-class integration
from world_class_medical_integration import (
    EnhancedMedicalKnowledgeIntegrator, 
    DiseaseProfile,
    MedicalConcept
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMedicalResponse:
    """Enhanced response with world-class database backing"""
    response: str
    confidence: float
    sources: List[str]
    umls_cui: Optional[str] = None
    snomed_id: Optional[str] = None
    icd11_code: Optional[str] = None
    evidence_level: str = "Medium"
    latest_research: List[Dict[str, Any]] = None
    safety_warnings: List[str] = None
    local_db_used: bool = True
    api_sources_used: List[str] = None

class WorldClassMedicalOrchestrator:
    """
    Medical orchestrator with world-class database integration
    Combines speed of local database with comprehensiveness of global APIs
    """
    
    def __init__(self, umls_api_key: str = None, use_local_first: bool = True):
        # Initialize existing components
        self.local_orchestrator = MedicalOrchestrator()
        self.local_retriever = EnhancedMedicalRetriever()
        self.safety_validator = MedicalSafetyValidator()
        
        # Initialize world-class integration
        self.world_integrator = EnhancedMedicalKnowledgeIntegrator(umls_api_key)
        
        # Configuration
        self.use_local_first = use_local_first
        self.api_timeout = 10.0  # seconds
        self.max_retries = 2
        
        # Performance tracking
        self.performance_stats = {
            "local_hits": 0,
            "api_calls": 0,
            "hybrid_responses": 0,
            "total_queries": 0
        }
        
    async def process_medical_query(self, query: str, user_context: Dict[str, Any] = None) -> EnhancedMedicalResponse:
        """
        Process medical query with world-class database integration
        Strategy: Local first for speed, API fallback for comprehensiveness
        """
        self.performance_stats["total_queries"] += 1
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing medical query: {query}")
            
            # Phase 1: Try local database first (for speed)
            local_response = None
            if self.use_local_first:
                local_response = await self._try_local_database(query, user_context)
                
                # If local response is confident enough, use it
                if local_response and local_response.confidence >= 0.8:
                    self.performance_stats["local_hits"] += 1
                    logger.info(f"High-confidence local response: {local_response.confidence}")
                    return local_response
            
            # Phase 2: Enhance with world-class APIs
            enhanced_response = await self._enhance_with_world_apis(query, local_response, user_context)
            
            if enhanced_response:
                self.performance_stats["api_calls"] += 1
                if local_response:
                    self.performance_stats["hybrid_responses"] += 1
                    
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Enhanced response generated in {processing_time:.2f}s")
                return enhanced_response
            
            # Phase 3: Fallback to local if APIs fail
            if local_response:
                logger.warning("API enhancement failed, using local response")
                return local_response
            
            # Phase 4: Last resort - basic response
            return EnhancedMedicalResponse(
                response="I apologize, but I'm having trouble accessing medical databases right now. Please consult a healthcare professional for medical advice.",
                confidence=0.1,
                sources=["System Message"],
                safety_warnings=["Unable to access medical databases - seek professional medical advice"]
            )
            
        except Exception as e:
            logger.error(f"Error processing medical query: {e}")
            return self._create_error_response()
    
    async def _try_local_database(self, query: str, user_context: Dict[str, Any]) -> Optional[EnhancedMedicalResponse]:
        """Try to answer using local database first"""
        try:
            # Use existing local orchestrator
            local_result = await asyncio.wait_for(
                self.local_orchestrator.process_medical_query(query, user_context or {}),
                timeout=2.0  # Quick local lookup
            )
            
            # Convert to enhanced response format
            if local_result and hasattr(local_result, 'response'):
                return EnhancedMedicalResponse(
                    response=local_result.response,
                    confidence=getattr(local_result, 'confidence', 0.7),
                    sources=["Local Database"],
                    local_db_used=True,
                    api_sources_used=[]
                )
                
        except asyncio.TimeoutError:
            logger.warning("Local database timeout")
        except Exception as e:
            logger.error(f"Local database error: {e}")
            
        return None
    
    async def _enhance_with_world_apis(self, query: str, local_response: Optional[EnhancedMedicalResponse], 
                                     user_context: Dict[str, Any]) -> Optional[EnhancedMedicalResponse]:
        """Enhance response using world-class medical APIs"""
        try:
            # Extract disease/condition from query
            disease_entities = self._extract_medical_entities(query)
            
            if not disease_entities:
                logger.info("No medical entities found in query")
                return local_response
            
            # Get comprehensive profiles from world-class databases
            world_profiles = []
            api_sources_used = []
            
            for entity in disease_entities[:3]:  # Limit to top 3 entities
                try:
                    profile = await asyncio.wait_for(
                        self.world_integrator.get_comprehensive_disease_profile(entity),
                        timeout=self.api_timeout
                    )
                    
                    if profile:
                        world_profiles.append(profile)
                        api_sources_used.extend(profile.sources)
                        
                except asyncio.TimeoutError:
                    logger.warning(f"API timeout for entity: {entity}")
                except Exception as e:
                    logger.error(f"API error for {entity}: {e}")
            
            if not world_profiles:
                logger.warning("No world-class profiles obtained")
                return local_response
            
            # Combine local and world-class data
            enhanced_response = self._combine_responses(query, local_response, world_profiles, api_sources_used)
            
            # Add safety validation
            enhanced_response = await self._add_safety_validation(enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing with world APIs: {e}")
            return local_response
    
    def _extract_medical_entities(self, query: str) -> List[str]:
        """Extract medical entities from query (simplified NER)"""
        # This is a simplified version - in production, use BioBERT NER
        common_diseases = [
            "diabetes", "hypertension", "tuberculosis", "malaria", "covid", "fever", 
            "headache", "asthma", "arthritis", "cancer", "heart disease", "stroke"
        ]
        
        query_lower = query.lower()
        entities = []
        
        for disease in common_diseases:
            if disease in query_lower:
                entities.append(disease.title())
        
        # Also try to extract from local retriever if available
        try:
            local_entities = self.local_retriever.extract_medical_entities(query)
            entities.extend(local_entities)
        except:
            pass
        
        return list(set(entities))  # Remove duplicates
    
    def _combine_responses(self, query: str, local_response: Optional[EnhancedMedicalResponse], 
                          world_profiles: List[DiseaseProfile], api_sources_used: List[str]) -> EnhancedMedicalResponse:
        """Combine local database response with world-class API data"""
        
        # Start with world-class data as the foundation
        primary_profile = world_profiles[0] if world_profiles else None
        
        if primary_profile:
            # Generate comprehensive response using world-class data
            response_parts = []
            
            # Add authoritative definition
            if primary_profile.definitions:
                best_definition = self._get_best_definition(primary_profile.definitions)
                response_parts.append(f"**Medical Definition**: {best_definition}")
            
            # Add symptoms if available
            if primary_profile.symptoms:
                symptoms_text = ", ".join(primary_profile.symptoms[:5])  # Top 5
                response_parts.append(f"**Common Symptoms**: {symptoms_text}")
            
            # Add treatments if available
            if primary_profile.treatments:
                treatments_text = ", ".join(primary_profile.treatments[:3])  # Top 3
                response_parts.append(f"**Treatment Options**: {treatments_text}")
            
            # Add prevalence data
            if primary_profile.prevalence:
                prevalence_text = self._format_prevalence(primary_profile.prevalence)
                response_parts.append(f"**Prevalence**: {prevalence_text}")
            
            # Combine with local response if it adds value
            if local_response and local_response.confidence > 0.6:
                response_parts.append(f"\n**Additional Context**: {local_response.response}")
            
            # Add data sources
            all_sources = list(set(api_sources_used + (["Local Database"] if local_response else [])))
            response_parts.append(f"\n**Sources**: {', '.join(all_sources)}")
            
            combined_response = "\n\n".join(response_parts)
            
            return EnhancedMedicalResponse(
                response=combined_response,
                confidence=0.9,  # High confidence with world-class sources
                sources=all_sources,
                umls_cui=primary_profile.umls_cui,
                snomed_id=primary_profile.snomed_id,
                icd11_code=primary_profile.icd11_code,
                evidence_level=primary_profile.evidence_level,
                local_db_used=local_response is not None,
                api_sources_used=api_sources_used
            )
        
        # Fallback to local response if world APIs failed
        return local_response or self._create_basic_response(query)
    
    def _get_best_definition(self, definitions: Dict[str, str]) -> str:
        """Get the best definition from multiple sources"""
        # Priority order for medical definitions
        priority_sources = ["UMLS", "SNOMED CT", "WHO", "PubMed"]
        
        for source in priority_sources:
            if source in definitions and definitions[source]:
                return definitions[source]
        
        # Return any available definition
        return next(iter(definitions.values())) if definitions else ""
    
    def _format_prevalence(self, prevalence: Dict[str, float]) -> str:
        """Format prevalence data for display"""
        if "India" in prevalence:
            return f"India: {prevalence['India']:.2f}%"
        elif prevalence:
            country, rate = next(iter(prevalence.items()))
            return f"{country}: {rate:.2f}%"
        return "Data not available"
    
    async def _add_safety_validation(self, response: EnhancedMedicalResponse) -> EnhancedMedicalResponse:
        """Add safety validation to the response"""
        try:
            # Use existing safety validator
            safety_result = await asyncio.wait_for(
                self.safety_validator.validate_medical_response(response.response),
                timeout=2.0
            )
            
            if hasattr(safety_result, 'warnings'):
                response.safety_warnings = safety_result.warnings
            
            # Add general medical disclaimer
            if not response.safety_warnings:
                response.safety_warnings = []
            
            response.safety_warnings.append(
                "This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice."
            )
            
        except Exception as e:
            logger.error(f"Safety validation error: {e}")
            response.safety_warnings = ["Please consult a healthcare professional for medical advice."]
        
        return response
    
    def _create_basic_response(self, query: str) -> EnhancedMedicalResponse:
        """Create basic response when everything else fails"""
        return EnhancedMedicalResponse(
            response=f"I understand you're asking about medical topics related to '{query}'. While I don't have specific information available right now, I recommend consulting with a healthcare professional who can provide personalized medical advice based on your specific situation.",
            confidence=0.3,
            sources=["General Medical Guidance"],
            safety_warnings=["Always consult a qualified healthcare professional for medical advice."]
        )
    
    def _create_error_response(self) -> EnhancedMedicalResponse:
        """Create error response"""
        return EnhancedMedicalResponse(
            response="I'm experiencing technical difficulties accessing medical databases. Please consult a healthcare professional for medical advice.",
            confidence=0.1,
            sources=["System Error"],
            safety_warnings=["System error - seek professional medical advice"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self.performance_stats["total_queries"]
        if total == 0:
            return self.performance_stats
        
        return {
            **self.performance_stats,
            "local_hit_rate": self.performance_stats["local_hits"] / total,
            "api_usage_rate": self.performance_stats["api_calls"] / total,
            "hybrid_rate": self.performance_stats["hybrid_responses"] / total
        }

# Integration with existing Flask app
async def demonstrate_world_class_integration():
    """Demonstrate the enhanced orchestrator"""
    
    print("🌐 World-Class Medical AI Orchestrator")
    print("=" * 50)
    
    # Initialize (you can add UMLS API key for full functionality)
    orchestrator = WorldClassMedicalOrchestrator(
        umls_api_key=os.getenv("UMLS_API_KEY"),  # Add your key to .env
        use_local_first=True
    )
    
    # Test queries
    test_queries = [
        "What is diabetes and how is it treated?",
        "I have persistent headaches and fever, what could this be?",
        "Tell me about tuberculosis symptoms and prevention",
        "What are the latest treatments for COVID-19?",
        "How common is malaria in India?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        response = await orchestrator.process_medical_query(query)
        
        print(f"📋 Response:")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Sources: {', '.join(response.sources)}")
        if response.umls_cui:
            print(f"   UMLS CUI: {response.umls_cui}")
        if response.snomed_id:
            print(f"   SNOMED ID: {response.snomed_id}")
        print(f"   Evidence Level: {response.evidence_level}")
        print(f"   Local DB Used: {response.local_db_used}")
        if response.api_sources_used:
            print(f"   API Sources: {', '.join(response.api_sources_used)}")
        
        # Show snippet of response
        response_snippet = response.response[:200] + "..." if len(response.response) > 200 else response.response
        print(f"   Response: {response_snippet}")
        
        if response.safety_warnings:
            print(f"   ⚠️  Safety: {response.safety_warnings[0]}")
    
    # Show performance stats
    print(f"\n📊 Performance Statistics:")
    stats = orchestrator.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demonstrate_world_class_integration())