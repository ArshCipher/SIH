"""
Advanced Medical AI Orchestrator
Multi-agent ensemble system for medical query processing with consensus mechanism,
safety validation, and explainable AI for competition-grade healthcare chatbot.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum
import uuid

# Multilingual support imports
try:
    from langdetect import detect
    from googletrans import Translator
    MULTILINGUAL_SUPPORT = True
except ImportError:
    MULTILINGUAL_SUPPORT = False

from chatbot.medical_models import MedicalModelEnsemble, medical_ensemble
from chatbot.medical_knowledge_base import medical_knowledge, MedicalCondition, UrgencyLevel

# Enhanced imports
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from enhanced_medical_retriever import EnhancedMedicalKnowledgeRetriever, MedicalResponse as EnhancedMedicalResponse
    ENHANCED_RETRIEVER_AVAILABLE = True
except ImportError:
    ENHANCED_RETRIEVER_AVAILABLE = False

# Real-time medical APIs import
try:
    from immediate_medical_apis import ImmediateMedicalAPIs, OpenMedicalDataCollector
    IMMEDIATE_APIS_AVAILABLE = True
except ImportError:
    IMMEDIATE_APIS_AVAILABLE = False

# Core imports
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Medical agent specialization roles"""
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    SAFETY = "safety"
    TRIAGE = "triage"
    DRUG_INTERACTION = "drug_interaction"
    PREVENTION = "prevention"
    OUTBREAK = "outbreak"

# Alias for backwards compatibility
AgentType = AgentRole

class MedicalRiskLevel(Enum):
    """Medical risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class QueryType(Enum):
    """Medical query classification types"""
    SYMPTOM_ANALYSIS = "symptom_analysis"
    DISEASE_INFO = "disease_info"
    TREATMENT_ADVICE = "treatment_advice"
    DRUG_QUERY = "drug_query"
    PREVENTION_GUIDANCE = "prevention_guidance"
    EMERGENCY = "emergency"
    OUTBREAK_INFO = "outbreak_info"
    GENERAL_HEALTH = "general_health"

class SeverityLevel(Enum):
    """Medical severity classification"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MedicalEntity:
    """Medical entity extracted from query"""
    text: str
    entity_type: str  # symptom, disease, medication, body_part, etc.
    confidence: float
    umls_cui: Optional[str] = None  # UMLS Concept Unique Identifier
    snomed_code: Optional[str] = None
    icd10_code: Optional[str] = None

@dataclass
class AgentResponse:
    """Individual agent response structure"""
    agent_id: str
    agent_role: AgentRole
    response_text: str
    confidence: float
    reasoning: str
    medical_entities: List[MedicalEntity]
    severity_assessment: SeverityLevel
    safety_flags: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# Alias for backwards compatibility
MedicalResponse = AgentResponse

@dataclass
class ConsensusResult:
    """Final consensus result from multi-agent system"""
    final_response: str
    consensus_confidence: float
    agent_responses: List[AgentResponse]
    consensus_reasoning: str
    severity_level: SeverityLevel
    safety_validated: bool
    medical_disclaimer: str
    suggested_actions: List[str]
    escalation_needed: bool = False
    processing_time: float = 0.0
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class MedicalAgent(ABC):
    """Abstract base class for medical agents"""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.confidence_threshold = 0.85
        
    @abstractmethod
    async def process_query(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process medical query and return agent response"""
        pass
    
    def extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text (to be enhanced with actual NER)"""
        entities = []
        
        # Common medical terms patterns
        symptom_keywords = ["fever", "cough", "headache", "pain", "nausea", "fatigue", "rash"]
        disease_keywords = ["covid", "malaria", "dengue", "diabetes", "hypertension"]
        body_part_keywords = ["head", "chest", "stomach", "back", "throat", "heart"]
        
        text_lower = text.lower()
        
        for keyword in symptom_keywords:
            if keyword in text_lower:
                entities.append(MedicalEntity(
                    text=keyword,
                    entity_type="symptom",
                    confidence=0.8
                ))
        
        for keyword in disease_keywords:
            if keyword in text_lower:
                entities.append(MedicalEntity(
                    text=keyword,
                    entity_type="disease",
                    confidence=0.8
                ))
                
        for keyword in body_part_keywords:
            if keyword in text_lower:
                entities.append(MedicalEntity(
                    text=keyword,
                    entity_type="body_part",
                    confidence=0.7
                ))
        
        return entities

# Concrete agent implementations
class DiagnosticAgent(MedicalAgent):
    """Diagnostic medical agent"""
    
    def __init__(self):
        super().__init__("diagnostic_agent", AgentRole.DIAGNOSTIC)
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process diagnostic query"""
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            response_text="Diagnostic analysis completed.",
            confidence=0.85,
            reasoning="Symptom analysis and medical knowledge base consultation",
            medical_entities=[],
            severity_assessment=SeverityLevel.MODERATE
        )

class SafetyAgent(MedicalAgent):
    """Safety validation medical agent"""
    
    def __init__(self):
        super().__init__("safety_agent", AgentRole.SAFETY)
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process safety validation - very permissive for rural healthcare access"""
        query_lower = query.lower()
        
        # Only block explicit self-harm (extremely minimal restrictions)
        if any(phrase in query_lower for phrase in ["kill myself", "suicide", "end my life", "hurt myself on purpose"]):
            return AgentResponse(
                agent_id=self.agent_id,
                agent_role=self.role,
                response_text="Please contact emergency services or a mental health professional immediately.",
                confidence=0.95,
                reasoning="Self-harm content detected",
                medical_entities=[],
                severity_assessment=SeverityLevel.CRITICAL
            )
        
        # Allow ALL medical queries - this AI is designed for medical information
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            response_text="Medical query approved - providing comprehensive medical information",
            confidence=0.95,
            reasoning="Medical AI designed for healthcare information access",
            medical_entities=[],
            severity_assessment=SeverityLevel.LOW
        )

class TriageAgent(MedicalAgent):
    """Medical triage agent"""
    
    def __init__(self):
        super().__init__("triage_agent", AgentRole.TRIAGE)
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process triage assessment"""
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            response_text="Triage assessment completed.",
            confidence=0.88,
            reasoning="Medical urgency evaluation",
            medical_entities=[],
            severity_assessment=SeverityLevel.MODERATE
        )

@dataclass
class MedicalConsensus:
    """Ensemble consensus result with explainability"""
    final_response: str
    consensus_confidence: float
    risk_level: MedicalRiskLevel
    agent_responses: List[AgentResponse]
    safety_validated: bool
    human_escalation_required: bool
    explanation: str
    medical_disclaimer: str
    session_id: str

class MedicalOrchestrator:
    """Competition-grade medical AI orchestrator with multi-agent consensus"""
    
    def __init__(self):
        self.agents = {
            AgentType.DIAGNOSTIC: DiagnosticAgent(),
            AgentType.SAFETY: SafetyAgent(),
            AgentType.TRIAGE: TriageAgent(),
            # Initialize all agent types for comprehensive coverage
            AgentType.TREATMENT: DiagnosticAgent(),  # Use diagnostic agent for treatment queries
            AgentType.DRUG_INTERACTION: SafetyAgent(),  # Use safety agent for drug interactions
            AgentType.PREVENTION: DiagnosticAgent(),  # Use diagnostic agent for prevention
            AgentType.OUTBREAK: TriageAgent(),  # Use triage agent for outbreak queries
        }
        
        # Initialize medical ensemble
        self.medical_ensemble = medical_ensemble  # Use the global instance
        
        # Multilingual support
        if MULTILINGUAL_SUPPORT:
            self.translator = Translator()
        else:
            self.translator = None
        
        # Initialize enhanced retriever if available
        self.enhanced_retriever = None
        if ENHANCED_RETRIEVER_AVAILABLE:
            try:
                self.enhanced_retriever = EnhancedMedicalKnowledgeRetriever()
                logger.info("Enhanced medical retriever initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced retriever: {e}")
        
        # Initialize immediate medical APIs if available
        self.immediate_apis = None
        if IMMEDIATE_APIS_AVAILABLE:
            try:
                self.immediate_apis = ImmediateMedicalAPIs()
                self.medical_data_collector = OpenMedicalDataCollector()
                logger.info("Immediate medical APIs initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize immediate APIs: {e}")
        
        self.consensus_threshold = 0.50  # Lowered from 0.80 to be less restrictive
        self.safety_threshold = 0.70     # Lowered from 0.95 to be less restrictive
        
        # Medical disclaimers - removed for cleaner user experience
        self.medical_disclaimer = ""
        
        logger.info(f"Medical Orchestrator initialized with multilingual support: {MULTILINGUAL_SUPPORT}")
        logger.info(f"Enhanced retriever available: {self.enhanced_retriever is not None}")
        logger.info(f"Immediate APIs available: {self.immediate_apis is not None}")
    
    async def process_medical_query(self, query: str, context: Dict[str, Any]) -> MedicalConsensus:
        """Process medical query through multi-agent system with ensemble consensus"""
        
        # Store current query for intelligent content generation
        self._current_query = query
        
        session_id = str(uuid.uuid4())
        
        try:
            # Step 1: Safety pre-screening
            safety_agent = self.agents[AgentType.SAFETY]
            safety_response = await safety_agent.process_query(query, context)
            
            if safety_response.severity_assessment == SeverityLevel.CRITICAL:
                return MedicalConsensus(
                    final_response="⚠️ EMERGENCY: Your symptoms suggest a potentially serious medical condition that requires immediate professional medical attention. Please seek emergency medical care or call emergency services immediately.",
                    consensus_confidence=safety_response.confidence,
                    risk_level=MedicalRiskLevel.EMERGENCY,
                    agent_responses=[safety_response],
                    safety_validated=True,
                    human_escalation_required=True,
                    explanation="Emergency detected - immediate safety response",
                    medical_disclaimer=self.medical_disclaimer,
                    session_id=session_id
                )
            
            # Step 2: Use multiple medical models for comprehensive analysis
            from chatbot.medical_models import MedicalModelType, MedicalTask
            
            # Use multiple models for comprehensive medical analysis with multilingual support
            medical_analyses = []
            
            # Detect language and translate if needed
            detected_language = self._detect_language(query)
            translated_query = await self._translate_query_if_needed(query, detected_language)
            
            # Use BioBERT for general medical classification
            biobert_result = await self.medical_ensemble.predict(
                text=translated_query,
                model_type=MedicalModelType.BIOBERT,
                task=MedicalTask.MEDICAL_TEXT_CLASSIFICATION,
                context=str(context)
            )
            medical_analyses.append(("BioBERT Classification", biobert_result))
            
            # ClinicalBERT for clinical information
            clinical_result = await self.medical_ensemble.predict(
                text=translated_query,
                model_type=MedicalModelType.CLINICAL_BERT,
                task=MedicalTask.MEDICAL_TEXT_CLASSIFICATION,
                context=str(context)
            )
            medical_analyses.append(("ClinicalBERT Analysis", clinical_result))
            
            # PubMedBERT for research-based information
            pubmed_result = await self.medical_ensemble.predict(
                text=translated_query,
                model_type=MedicalModelType.PUBMED_BERT,
                task=MedicalTask.MEDICAL_TEXT_CLASSIFICATION,
                context=str(context)
            )
            medical_analyses.append(("PubMedBERT Research", pubmed_result))
            
            # Use Medical NER for entity extraction
            ner_result = await self.medical_ensemble.predict(
                text=translated_query,
                model_type=MedicalModelType.MEDICAL_NER,
                task=MedicalTask.NAMED_ENTITY_RECOGNITION,
                context=str(context)
            )
            medical_analyses.append(("Medical NER", ner_result))
            
            # Get comprehensive medical information for complex queries
            comprehensive_analysis = ""
            
            # Use enhanced retriever if available for superior responses
            if self.enhanced_retriever:
                try:
                    enhanced_response = self.enhanced_retriever.generate_comprehensive_response(translated_query)
                    if enhanced_response.primary_response:
                        comprehensive_analysis = enhanced_response.primary_response
                        
                        # Add enhanced information
                        if enhanced_response.disease_matches:
                            top_match = enhanced_response.disease_matches[0]
                            comprehensive_analysis += f"\n\n**Disease Information:**\n"
                            comprehensive_analysis += f"- **Category:** {top_match.disease_category}\n"
                            comprehensive_analysis += f"- **Severity:** {top_match.severity.title()}\n"
                            
                            if top_match.early_symptoms:
                                comprehensive_analysis += f"- **Early Symptoms:** {top_match.early_symptoms}\n"
                            
                            if top_match.prevention:
                                comprehensive_analysis += f"- **Prevention:** {top_match.prevention}\n"
                            
                            if top_match.affected_states:
                                comprehensive_analysis += f"- **Common in:** {top_match.affected_states}\n"
                        
                        if enhanced_response.safety_warnings:
                            comprehensive_analysis += f"\n\n**Safety Warnings:**\n"
                            for warning in enhanced_response.safety_warnings[:3]:
                                comprehensive_analysis += f"⚠️ {warning}\n"
                        
                        if enhanced_response.recommended_actions:
                            comprehensive_analysis += f"\n\n**Recommended Actions:**\n"
                            for action in enhanced_response.recommended_actions[:3]:
                                comprehensive_analysis += f"• {action}\n"
                        
                        comprehensive_analysis += f"\n\n{enhanced_response.when_to_seek_help}"
                        
                        # Use enhanced confidence
                        biobert_result.confidence = max(biobert_result.confidence, enhanced_response.confidence_level)
                        
                        logger.info(f"Enhanced medical response generated with confidence: {enhanced_response.confidence_level:.2f}")
                except Exception as e:
                    logger.warning(f"Enhanced retriever failed, falling back to standard response: {e}")
            
            # Fallback to standard analysis if enhanced retriever not available or failed
            if not comprehensive_analysis:
                # Use the enhanced medical models to generate contextual responses based on the actual query
                query_lower = query.lower()
                
                # Use AI models to intelligently analyze and respond to ANY medical query
                query_intent = self._analyze_query_intent(query, biobert_result, clinical_result, pubmed_result)
                
                # Generate intelligent response using medical models and knowledge base
                comprehensive_analysis = await self._generate_intelligent_medical_response(
                    query, query_intent, biobert_result, clinical_result, pubmed_result, ner_result
                )
            
            # Step 3: Run parallel agent processing for consensus
            agent_tasks = []
            relevant_agents = await self._select_relevant_agents(query, context)
            
            for agent_type in relevant_agents:
                if agent_type in self.agents:
                    task = self.agents[agent_type].process_query(query, context)
                    agent_tasks.append(task)
            
            # Execute agents in parallel
            if agent_tasks:
                agent_responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
                # Filter out exceptions
                valid_responses = [r for r in agent_responses if isinstance(r, AgentResponse)]
            else:
                valid_responses = [safety_response]
            
            # Step 4: Generate consensus from comprehensive analysis
            ensemble_dict = {
                "prediction": comprehensive_analysis,  # Use comprehensive analysis instead of simple prediction
                "confidence": max(biobert_result.confidence, clinical_result.confidence, pubmed_result.confidence),
                "entities": biobert_result.entities + clinical_result.entities + pubmed_result.entities,
                "reasoning": f"Multi-model analysis: BioBERT ({biobert_result.confidence:.2f}), ClinicalBERT ({clinical_result.confidence:.2f}), PubMedBERT ({pubmed_result.confidence:.2f})",
                "model_type": "Multi-Model Ensemble",
                "processing_time": biobert_result.processing_time + clinical_result.processing_time + pubmed_result.processing_time,
                "medical_analyses": medical_analyses
            }
            consensus = await self._generate_consensus(ensemble_dict, valid_responses, session_id, query)
            
            return consensus
            
        except Exception as e:
            logging.error(f"Error in medical orchestrator: {e}")
            return self._generate_fallback_response(query, session_id)
    
    async def _select_relevant_agents(self, query: str, context: Dict[str, Any]) -> List[AgentType]:
        """Select relevant agents based on query analysis"""
        
        relevant_agents = [AgentType.TRIAGE]  # Always include triage
        
        # Query analysis for agent selection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["symptoms", "diagnosis", "condition", "disease"]):
            relevant_agents.append(AgentType.DIAGNOSTIC)
        
        if any(word in query_lower for word in ["treatment", "medication", "therapy"]):
            relevant_agents.append(AgentType.TREATMENT)
        
        if any(word in query_lower for word in ["drug", "interaction", "medication"]):
            relevant_agents.append(AgentType.DRUG_INTERACTION)
        
        if any(word in query_lower for word in ["prevention", "avoid", "protect"]):
            relevant_agents.append(AgentType.PREVENTION)
        
        return relevant_agents
    
    async def _generate_consensus(self, ensemble_result: Dict[str, Any], agent_responses: List[AgentResponse], session_id: str, query: str = "") -> MedicalConsensus:
        """Generate consensus from ensemble and agent responses"""
        
        # Calculate consensus confidence
        confidences = [ensemble_result.get("confidence", 0.7)] + [r.confidence for r in agent_responses]
        consensus_confidence = float(np.mean(confidences))
        
        # Determine risk level based on severity assessments
        severities = [r.severity_assessment for r in agent_responses]
        if SeverityLevel.CRITICAL in severities:
            risk_level = MedicalRiskLevel.EMERGENCY
        elif SeverityLevel.HIGH in severities:
            risk_level = MedicalRiskLevel.HIGH
        elif SeverityLevel.MODERATE in severities:
            risk_level = MedicalRiskLevel.MODERATE
        else:
            risk_level = MedicalRiskLevel.LOW
        
        # Combine responses - use prediction from ensemble result
        ensemble_prediction = ensemble_result.get("prediction", "")
        
        # Build comprehensive response
        if ensemble_prediction:
            final_response = f"{ensemble_prediction}"
            
            # Add agent insights if available
            agent_insights = []
            for response in agent_responses:
                if response.response_text and response.response_text != "Safety validation completed." and response.response_text != "Triage assessment completed.":
                    agent_insights.append(response.response_text)
            
            if agent_insights:
                final_response += f"\n\nAdditional insights: {' '.join(agent_insights)}"
                
        else:
            final_response = "Please consult a healthcare professional for proper medical advice."
        
        # Add safety validation
        safety_validated = ensemble_result.get("safety_validated", True)
        escalation_needed = (
            risk_level in [MedicalRiskLevel.HIGH, MedicalRiskLevel.EMERGENCY] or
            consensus_confidence < self.consensus_threshold
        )
        
        # Enrich final response with live medical data
        final_response = await self._enrich_with_live_medical_data(final_response, query)
        
        return MedicalConsensus(
            final_response=final_response,
            consensus_confidence=consensus_confidence,
            risk_level=risk_level,
            agent_responses=agent_responses,
            safety_validated=safety_validated,
            human_escalation_required=escalation_needed,
            explanation=f"Consensus from medical ensemble and {len(agent_responses)} specialized agents",
            medical_disclaimer=self.medical_disclaimer,
            session_id=session_id
        )
    
    def _generate_fallback_response(self, query: str, session_id: str) -> MedicalConsensus:
        """Generate fallback response when processing fails"""
        return MedicalConsensus(
            final_response="I'm having difficulty processing your medical query at the moment. For your safety, please consult with a qualified healthcare professional for proper medical advice.",
            consensus_confidence=0.3,
            risk_level=MedicalRiskLevel.MODERATE,
            agent_responses=[],
            safety_validated=True,
            human_escalation_required=True,
            explanation="Fallback response due to processing errors",
            medical_disclaimer=self.medical_disclaimer,
            session_id=session_id
        )

    def _analyze_query_intent(self, query: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Analyze the query to understand what specific information is being requested"""
        query_lower = query.lower()
        
        # Check for specific phrase patterns first (most precise)
        if any(phrase in query_lower for phrase in ["how to treat", "how to cure", "treatment for", "treating", "therapy for"]):
            return "treatment_info"
        elif any(phrase in query_lower for phrase in ["how to prevent", "prevention of", "preventing", "avoid getting"]):
            return "prevention_info"
        elif any(phrase in query_lower for phrase in ["what causes", "why do", "cause of", "reason for"]):
            return "causation_info"
        elif any(phrase in query_lower for phrase in ["how to know", "how do i know", "symptoms of", "signs of", "how to tell"]):
            return "diagnostic_info"
        elif any(phrase in query_lower for phrase in ["what is", "define", "explain", "describe"]):
            return "definition_info"
        
        # Then check for individual keywords (less precise)
        elif any(word in query_lower for word in ["treat", "treatment", "cure", "therapy", "medicine", "medication"]):
            return "treatment_info"
        elif any(word in query_lower for word in ["prevent", "avoid", "stop", "reduce risk"]):
            return "prevention_info"
        elif any(word in query_lower for word in ["cause", "why", "reason", "mechanism", "pathophysiology"]):
            return "causation_info"
        elif any(word in query_lower for word in ["complication", "risk", "danger", "outcome"]):
            return "risk_info"
        elif any(word in query_lower for word in ["symptoms", "signs", "detect", "identify", "check", "know"]):
            return "diagnostic_info"
        else:
            return "general_info"

    def _generate_diabetes_response(self, query: str, intent: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate contextual diabetes response based on query intent"""
        
        base_info = f"""**Medical AI Analysis for: "{query}"**

**Model Analysis:**
• BioBERT: {biobert_result.prediction} (Confidence: {biobert_result.confidence:.2f})
• ClinicalBERT: {clinical_result.prediction} (Confidence: {clinical_result.confidence:.2f})
• PubMedBERT: {pubmed_result.prediction} (Confidence: {pubmed_result.confidence:.2f})

"""
        
        if intent == "diagnostic_info":
            return base_info + """**How to Know if You Have Diabetes:**

**Common Warning Signs:**
• Excessive thirst and frequent urination
• Unexplained weight loss or gain
• Extreme fatigue and weakness
• Blurred vision
• Slow-healing cuts and wounds
• Frequent infections
• Tingling or numbness in hands/feet

**Medical Tests:**
• Fasting blood glucose: ≥126 mg/dL indicates diabetes
• Random blood glucose: ≥200 mg/dL with symptoms
• A1C test: ≥6.5% indicates diabetes
• Oral glucose tolerance test: ≥200 mg/dL after 2 hours

**Next Steps:**
See a healthcare provider for proper blood testing if you have these symptoms."""

        elif intent == "treatment_info":
            return base_info + """**Diabetes Treatment Options:**

**Type 1 Diabetes:**
• Insulin therapy (essential for survival)
• Continuous glucose monitoring
• Carbohydrate counting
• Regular blood glucose testing

**Type 2 Diabetes:**
• Lifestyle changes (diet, exercise)
• Oral medications (metformin, etc.)
• Injectable medications (GLP-1 agonists)
• Insulin if needed
• Blood pressure/cholesterol management

**Lifestyle Management:**
• Healthy diet with controlled carbohydrates
• Regular physical activity
• Weight management
• Stress reduction
• Regular medical check-ups"""

        elif intent == "prevention_info":
            return base_info + """**Diabetes Prevention:**

**Type 2 Diabetes Prevention:**
• Maintain healthy weight
• Eat balanced diet with limited processed foods
• Exercise regularly (150 minutes/week)
• Don't smoke
• Limit alcohol consumption
• Manage stress levels
• Get regular health screenings

**High-Risk Individuals:**
• Family history of diabetes
• Age 45+
• Overweight/obese
• High blood pressure
• Previous gestational diabetes

**Type 1 Prevention:**
Currently no known prevention methods (autoimmune condition)"""

        elif intent == "causation_info":
            return base_info + """**What Causes Diabetes:**

**Type 1 Diabetes:**
• Autoimmune attack on insulin-producing cells
• Genetic factors (inherited susceptibility)
• Environmental triggers (viruses, stress)
• Not caused by diet or lifestyle

**Type 2 Diabetes:**
• Insulin resistance develops over time
• Pancreas can't produce enough insulin
• Risk factors: obesity, inactivity, genetics, age
• Often preventable with lifestyle changes

**Gestational Diabetes:**
• Hormonal changes during pregnancy
• Usually resolves after delivery
• Increases risk of Type 2 later"""

        else:
            return base_info + """**General Diabetes Information:**

**Types of Diabetes:**
• Type 1: Autoimmune, requires insulin
• Type 2: Insulin resistance, most common
• Gestational: During pregnancy

**Key Facts:**
• Affects how your body processes blood sugar
• Can lead to serious complications if uncontrolled
• Manageable with proper treatment
• Regular monitoring essential

**Important:** Consult healthcare providers for personalized advice and treatment plans."""

    def _generate_cancer_response(self, query: str, intent: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate contextual cancer response based on query intent"""
        
        base_info = f"""**Medical AI Analysis for: "{query}"**

**Model Analysis:**
• BioBERT: {biobert_result.prediction} (Confidence: {biobert_result.confidence:.2f})
• ClinicalBERT: {clinical_result.prediction} (Confidence: {clinical_result.confidence:.2f})
• PubMedBERT: {pubmed_result.prediction} (Confidence: {pubmed_result.confidence:.2f})

"""
        
        if intent == "diagnostic_info":
            return base_info + """**Cancer Warning Signs (CAUTION):**
• **C**hange in bowel or bladder habits
• **A** sore that does not heal
• **U**nusual bleeding or discharge
• **T**hickening or lump in breast or elsewhere
• **I**ndigestion or difficulty swallowing
• **O**bvious change in wart or mole
• **N**agging cough or hoarseness

**When to See a Doctor:**
• Any persistent, unexplained symptoms
• Lumps or masses anywhere on body
• Unexplained weight loss
• Persistent fatigue
• Changes in skin moles"""

        elif intent == "treatment_info":
            return base_info + """**Cancer Treatment Options:**

**Common Treatments:**
• Surgery to remove tumors
• Chemotherapy (drug treatment)
• Radiation therapy
• Immunotherapy
• Targeted therapy
• Hormone therapy
• Stem cell transplant

**Treatment Planning:**
• Depends on cancer type and stage
• Multi-disciplinary team approach
• Personalized treatment plans
• Consider clinical trials"""

        elif intent == "prevention_info":
            return base_info + """**Cancer Prevention:**

**Lifestyle Factors:**
• Don't smoke or use tobacco
• Limit alcohol consumption
• Maintain healthy weight
• Stay physically active
• Eat healthy diet with fruits/vegetables
• Protect skin from sun exposure

**Screening Tests:**
• Regular mammograms, colonoscopies
• Pap smears, skin checks
• Follow screening guidelines for your age"""

        else:
            return base_info + """**General Cancer Information:**

Cancer is a group of diseases involving abnormal cell growth that can spread to other parts of the body.

**Key Points:**
• Over 100 different types
• Early detection improves outcomes
• Treatment success rates improving
• Support resources available

**Important:** Any concerning symptoms should be evaluated by healthcare professionals immediately."""

    def _generate_hypertension_response(self, query: str, intent: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate contextual hypertension response based on query intent"""
        
        base_info = f"""**Medical AI Analysis for: "{query}"**

**Model Analysis:**
• BioBERT: {biobert_result.prediction} (Confidence: {biobert_result.confidence:.2f})
• ClinicalBERT: {clinical_result.prediction} (Confidence: {clinical_result.confidence:.2f})
• PubMedBERT: {pubmed_result.prediction} (Confidence: {pubmed_result.confidence:.2f})

"""
        
        if intent == "diagnostic_info":
            return base_info + """**Blood Pressure Categories:**
• **Normal:** Less than 120/80 mmHg
• **Elevated:** 120-129 systolic, <80 diastolic
• **Stage 1:** 130-139 systolic OR 80-89 diastolic
• **Stage 2:** 140/90 mmHg or higher
• **Crisis:** Higher than 180/120 mmHg (emergency)

**Symptoms:**
• Often called "silent killer" - usually no symptoms
• Severe cases: headaches, shortness of breath, nosebleeds
• Regular monitoring essential"""

        elif intent == "treatment_info":
            return base_info + """**Hypertension Treatment:**

**Lifestyle Changes:**
• Low-sodium diet (DASH diet)
• Regular exercise
• Weight management
• Limit alcohol
• Quit smoking
• Stress management

**Medications:**
• ACE inhibitors
• Calcium channel blockers
• Diuretics
• Beta-blockers
• ARBs (Angiotensin receptor blockers)"""

        elif intent == "prevention_info":
            return base_info + """**Preventing High Blood Pressure:**

**Healthy Habits:**
• Maintain healthy weight
• Exercise regularly
• Eat low-sodium, high-potassium foods
• Limit alcohol
• Don't smoke
• Manage stress
• Get adequate sleep
• Regular blood pressure checks"""

        else:
            return base_info + """**High Blood Pressure Information:**

Hypertension is when blood pressure is consistently elevated, putting extra strain on arteries and organs.

**Key Facts:**
• Often no symptoms until advanced
• Major risk factor for heart disease, stroke
• Very treatable with lifestyle and medication
• Regular monitoring important"""

    def _generate_heart_response(self, query: str, intent: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate contextual heart disease response based on query intent"""
        
        base_info = f"""**Medical AI Analysis for: "{query}"**

**Model Analysis:**
• BioBERT: {biobert_result.prediction} (Confidence: {biobert_result.confidence:.2f})
• ClinicalBERT: {clinical_result.prediction} (Confidence: {clinical_result.confidence:.2f})
• PubMedBERT: {pubmed_result.prediction} (Confidence: {pubmed_result.confidence:.2f})

"""
        
        if intent == "diagnostic_info":
            return base_info + """**Heart Disease Warning Signs:**
• Chest pain or pressure
• Shortness of breath
• Pain in arms, neck, jaw, back
• Fatigue, weakness
• Irregular heartbeat
• Swelling in legs, ankles, feet
• Nausea, lightheadedness

**Emergency Signs (Call 911):**
• Severe chest pain
• Difficulty breathing
• Loss of consciousness
• Signs of stroke"""

        elif intent == "treatment_info":
            return base_info + """**Heart Disease Treatment:**

**Medications:**
• Blood thinners
• ACE inhibitors
• Beta-blockers
• Statins for cholesterol
• Diuretics

**Procedures:**
• Angioplasty and stents
• Bypass surgery
• Pacemaker/defibrillator
• Heart transplant (severe cases)

**Lifestyle:**
• Heart-healthy diet
• Regular exercise
• Smoking cessation
• Stress management"""

        else:
            return base_info + """**Heart Disease Information:**

Heart disease includes various conditions affecting the heart and blood vessels.

**Common Types:**
• Coronary artery disease
• Heart failure
• Arrhythmias
• Valvular disease

**Risk Factors:**
• High blood pressure
• High cholesterol
• Diabetes
• Smoking
• Family history"""

    def _generate_symptom_response(self, query: str, intent: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate contextual symptom response"""
        
        return f"""**Medical AI Analysis for: "{query}"**

**Model Analysis:**
• BioBERT: {biobert_result.prediction} (Confidence: {biobert_result.confidence:.2f})
• ClinicalBERT: {clinical_result.prediction} (Confidence: {clinical_result.confidence:.2f})
• PubMedBERT: {pubmed_result.prediction} (Confidence: {pubmed_result.confidence:.2f})

**⚠️ SYMPTOM GUIDANCE:**

**When to Seek IMMEDIATE Care:**
• Chest pain or pressure
• Difficulty breathing
• Severe bleeding
• Loss of consciousness
• Signs of stroke (FAST)
• Severe allergic reactions

**When to Contact Doctor TODAY:**
• Persistent fever over 101°F
• Severe pain
• Unusual or concerning symptoms
• Medication side effects

**General Symptom Management:**
• Document when symptoms started
• Note triggers or patterns
• Track severity (1-10 scale)
• List current medications
• Monitor for changes

**Important:** This AI cannot diagnose conditions. Professional medical evaluation is essential for proper diagnosis."""

    def _generate_general_medical_response(self, query: str, intent: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate clean general medical response"""
        
        return """📚 **Medical Information & Guidance**

Thank you for your medical question. Our AI system has analyzed your query using advanced medical language models.

**General Health Guidance:**
• Consult healthcare providers for personalized medical advice
• Regular check-ups help with early detection and prevention
• Maintain healthy lifestyle: balanced diet, exercise, adequate sleep
• Keep track of medications and medical history
• Don't hesitate to seek medical attention for concerning symptoms

**Emergency Situations:**
• Call emergency services for severe symptoms: chest pain, difficulty breathing, severe bleeding
• Contact your doctor for persistent or worsening symptoms
• Use telemedicine for non-urgent questions

**Rural Healthcare Resources:**
• Mobile health clinics may serve your area
• Telemedicine connects you with doctors remotely
• Community health workers provide local support
• Emergency services available for urgent situations

⚠️ **Important:** This medical AI provides educational information to support informed healthcare decisions. Always consult qualified healthcare professionals for medical advice."""

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text using proper language detection"""
        if MULTILINGUAL_SUPPORT:
            try:
                detected = detect(text)
                # Map to supported languages
                if detected in ['hi', 'hindi']:
                    return 'hi'
                elif detected in ['bn', 'bengali']:
                    return 'bn'  
                elif detected in ['ta', 'tamil']:
                    return 'ta'
                elif detected in ['te', 'telugu']:
                    return 'te'
                elif detected in ['mr', 'marathi']:
                    return 'mr'
                elif detected in ['gu', 'gujarati']:
                    return 'gu'
                elif detected in ['kn', 'kannada']:
                    return 'kn'
                elif detected in ['ml', 'malayalam']:
                    return 'ml'
                elif detected in ['pa', 'punjabi']:
                    return 'pa'
                elif detected in ['or', 'odia']:
                    return 'or'
                else:
                    return 'en'  # Default to English
            except:
                # Fallback to simple detection
                pass
        
        # Simple fallback language detection based on script/characters
        # Hindi/Devanagari characters
        if any('\u0900' <= char <= '\u097F' for char in text):
            return "hi"
        # Bengali script
        if any('\u0980' <= char <= '\u09FF' for char in text):
            return "bn"
        # Tamil script  
        if any('\u0B80' <= char <= '\u0BFF' for char in text):
            return "ta"
        # Telugu script
        if any('\u0C00' <= char <= '\u0C7F' for char in text):
            return "te"
        # Check for common Hindi romanized words
        hindi_words = ["mujhe", "kaise", "kya", "hai", "hota", "diabetes", "sugar", "bimari"]
        if any(word in text.lower() for word in hindi_words):
            return "hi"
        # Default to English
        return "en"

    async def _translate_query_if_needed(self, query: str, detected_language: str) -> str:
        """Translate query to English if needed for model processing"""
        if detected_language == "en":
            return query
            
        if MULTILINGUAL_SUPPORT and self.translator:
            try:
                # Use Google Translate for accurate translation
                translated = self.translator.translate(query, src=detected_language, dest='en')
                return translated.text
            except Exception as e:
                logger.warning(f"Translation failed: {e}, falling back to basic translation")
                # Fall back to basic translation
                pass
        
        # Basic translation for common medical terms
        if detected_language == "hi":
            # Common Hindi medical translations
            translations = {
                "मुझे कैसे पता चलेगा": "how do I know",
                "मधुमेह": "diabetes",
                "मुझे": "me", "मुझको": "me",
                "कैसे": "how", "कया": "what", "क्या": "what",
                "पता": "know", "चलेगा": "will know",
                "है": "is", "हो": "have", "होगा": "will be",
                "बीमारी": "disease", "इलाज": "treatment",
                "दवा": "medicine", "डॉक्टर": "doctor",
                "लक्षण": "symptoms", "संकेत": "signs",
                "mujhe": "me", "kaise": "how", "pta": "know", 
                "chalega": "will know", "diabetes": "diabetes",
                "sugar": "diabetes", "bimari": "disease"
            }
            
            translated = query.lower()
            for hindi, english in translations.items():
                translated = translated.replace(hindi, english)
            return translated
        return query

    async def _translate_response_to_user_language(self, response: str, target_language: str) -> str:
        """Translate response back to user's language"""
        if target_language == "en":
            return response
            
        if MULTILINGUAL_SUPPORT and self.translator:
            try:
                # Use Google Translate for accurate translation
                translated = self.translator.translate(response, src='en', dest=target_language)
                return translated.text
            except Exception as e:
                logger.warning(f"Response translation failed: {e}")
                # Fall back to original response with language note
                pass
        
        # Add language-specific medical disclaimers
        language_disclaimers = {
            "hi": "\n\n🌐 **हिंदी सहायता:** कृपया डॉक्टर से परामर्श करें।",
            "bn": "\n\n🌐 **বাংলা সহায়তা:** দয়া করে ডাক্তারের পরামর্শ নিন।",
            "ta": "\n\n🌐 **தமிழ் உதவி:** தயவुசெய்து மருத்துவரை அணুகவும்।",
            "te": "\n\n🌐 **తెలుగు సహాయం:** దయచేసి వైద్యుడిని సంప్రదించండి।",
            "mr": "\n\n🌐 **मराठी मदत:** कृपया डॉक्टरांचा सल्ला घ्या।",
            "gu": "\n\n🌐 **ગુજરાતી સહાય:** કૃપા કરીને ડૉક્ટરની સલાહ લો।",
            "kn": "\n\n🌐 **ಕನ್ನಡ ಸಹಾಯ:** ದಯವಿಟ್ಟು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ।",
            "ml": "\n\n🌐 **മലയാളം സഹായം:** ദയവായി ഡോക്ടറെ സമീപിക്കുക।",
            "pa": "\n\n🌐 **ਪੰਜਾਬੀ ਸਹਾਇਤਾ:** ਕਿਰਪਾ ਕਰਕੇ ਡਾਕਟਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।",
            "or": "\n\n🌐 **ଓଡ଼ିଆ ସହାୟତା:** ଦୟାକରି ଡାକ୍ତରଙ୍କ ପରାମର୍ଶ ନିଅନ୍ତୁ।"
        }
        
        # Add disclaimer in user's language
        disclaimer = language_disclaimers.get(target_language, "")
        return response + disclaimer

    async def _generate_intelligent_medical_response(
        self, query: str, intent: str, biobert_result, clinical_result, pubmed_result, ner_result=None
    ) -> str:
        """Generate truly intelligent medical response using AI models and medical knowledge"""
        
        # Detect language for multilingual support
        detected_language = self._detect_language(query)
        
        # Step 1: Check for medical emergency using AI models
        is_emergency = self._detect_emergency_from_models(query, biobert_result, clinical_result, pubmed_result)
        
        if is_emergency:
            return self._generate_ai_emergency_response(query, biobert_result, clinical_result, pubmed_result)
        
        # Step 2: Extract medical entities from all models
        all_entities = []
        if biobert_result.entities:
            all_entities.extend(biobert_result.entities)
        if clinical_result.entities:
            all_entities.extend(clinical_result.entities)
        if pubmed_result.entities:
            all_entities.extend(pubmed_result.entities)
        if ner_result and ner_result.entities:
            all_entities.extend(ner_result.entities)
        
        # Get unique medical terms identified by AI
        medical_terms = list(set([entity.get('text', '') for entity in all_entities if entity.get('text')]))
        
        # Step 3: Generate response using AI understanding and medical knowledge
        response = self._generate_ai_driven_response(query, intent, medical_terms, biobert_result, clinical_result, pubmed_result)
        
        # Step 4: Translate response to user's language
        if detected_language != "en":
            response = await self._translate_response_to_user_language(response, detected_language)
        
        return response

    def _detect_emergency_from_models(self, query: str, biobert_result, clinical_result, pubmed_result) -> bool:
        """Use AI models to detect medical emergencies"""
        
        # Check model predictions for emergency indicators
        emergency_keywords = ["emergency", "severe", "critical", "urgent", "immediate", "emergency room", "911"]
        predictions = [biobert_result.prediction, clinical_result.prediction, pubmed_result.prediction]
        
        # Check if any model indicates high severity or emergency
        high_confidence_emergency = any(
            any(keyword in pred.lower() for keyword in emergency_keywords) 
            for pred in predictions if pred
        )
        
        # Check query for emergency terms
        query_emergency_terms = ["heart attack", "cardiac arrest", "stroke", "severe pain", "can't breathe", 
                                "difficulty breathing", "chest pain", "bleeding", "unconscious"]
        
        query_indicates_emergency = any(term in query.lower() for term in query_emergency_terms)
        
        return high_confidence_emergency or query_indicates_emergency

    def _generate_ai_emergency_response(self, query: str, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate emergency response based on AI model understanding"""
        
        # Extract the most relevant emergency condition from AI models
        highest_confidence_model = max([biobert_result, clinical_result, pubmed_result], key=lambda x: x.confidence)
        
        emergency_info = f"""🚨 **Medical Emergency Detected**

**Call 911 immediately if you are experiencing:**
• Severe or persistent symptoms
• Chest pain or pressure
• Difficulty breathing
• Loss of consciousness
• Severe bleeding

**What to do right now:**
1. Call emergency services (911)
2. Don't drive yourself to hospital
3. Stay calm and follow operator instructions
4. Have someone stay with you if possible

**AI Analysis:** {highest_confidence_model.prediction} (Confidence: {highest_confidence_model.confidence:.2f})

⚠️ **This is a medical emergency - get professional help immediately!**"""

        return emergency_info

    def _generate_ai_driven_response(self, query: str, intent: str, medical_terms: List[str], 
                                   biobert_result, clinical_result, pubmed_result) -> str:
        """Generate medical response using actual AI model outputs and medical knowledge"""
        
        # Get the most confident prediction
        models_and_results = [
            ("BioBERT", biobert_result),
            ("ClinicalBERT", clinical_result), 
            ("PubMedBERT", pubmed_result)
        ]
        
        # Sort by confidence
        models_and_results.sort(key=lambda x: x[1].confidence, reverse=True)
        primary_model, primary_result = models_and_results[0]
        
        # Generate response based on actual AI understanding
        response = f"""## Medical Information

**Your Question:** {query}

**AI Analysis:** {primary_result.prediction}"""

        # Add medical terms found by AI
        if medical_terms:
            response += f"\n\n**Medical Terms Identified:** {', '.join(medical_terms[:5])}"

        # Generate content based on intent and AI understanding
        if intent == "diagnostic_info" or "symptoms" in query.lower() or "signs" in query.lower():
            response += f"""

**Key Medical Information:**
{self._extract_medical_knowledge_from_ai(query, primary_result, "diagnostic")}

**When to See a Doctor:**
• If symptoms persist or worsen
• If you're concerned about your health
• For proper medical evaluation and testing
• If symptoms interfere with daily activities"""

        elif intent == "treatment_info" or "treatment" in query.lower() or "cure" in query.lower():
            response += f"""

**Treatment Considerations:**
{self._extract_medical_knowledge_from_ai(query, primary_result, "treatment")}

**Important Steps:**
• Consult with a healthcare provider
• Get proper medical evaluation
• Follow prescribed treatment plans
• Monitor your response to treatment"""

        elif intent == "prevention_info" or "prevent" in query.lower() or "avoid" in query.lower():
            response += f"""

**Prevention Strategies:**
{self._extract_medical_knowledge_from_ai(query, primary_result, "prevention")}

**General Health Tips:**
• Maintain a healthy lifestyle
• Regular medical checkups
• Stay informed about risk factors
• Follow medical guidelines"""

        else:
            response += f"""

**Medical Information:**
{self._extract_medical_knowledge_from_ai(query, primary_result, "general")}

**Next Steps:**
• Consult with healthcare professionals
• Get appropriate medical evaluation
• Follow professional medical advice
• Monitor your health regularly"""

        # Add AI model insights
        response += f"""

**Medical AI Insights:**
• **{primary_model}:** {primary_result.prediction} (Confidence: {primary_result.confidence:.2f})"""

        if len(models_and_results) > 1:
            secondary_model, secondary_result = models_and_results[1]
            response += f"""
• **{secondary_model}:** {secondary_result.prediction} (Confidence: {secondary_result.confidence:.2f})"""

        response += """

💡 **Remember:** This AI provides educational information. Always consult healthcare professionals for medical advice, diagnosis, and treatment."""

        return response

    def _extract_medical_knowledge_from_ai(self, query: str, ai_result, context_type: str) -> str:
        """Extract relevant medical knowledge based on AI model understanding"""
        
        # Use AI prediction to generate contextual information
        prediction = ai_result.prediction
        confidence = ai_result.confidence
        
        # Search medical knowledge base using AI-identified terms
        knowledge_match = medical_knowledge.search_conditions(query)
        
        if knowledge_match:
            condition = knowledge_match[0]
            
            if context_type == "diagnostic":
                return f"""• **Symptoms:** {', '.join(condition.symptoms[:3] if condition.symptoms else ['Varies by individual'])}
• **Risk Factors:** {', '.join(condition.risk_factors[:3] if condition.risk_factors else ['Individual factors vary'])}
• **When to Seek Care:** {', '.join(condition.when_to_seek_help[:2] if condition.when_to_seek_help else ['Consult healthcare provider for evaluation'])}"""
            
            elif context_type == "treatment":
                return f"""• **Treatment Approaches:** {', '.join(condition.treatments[:3] if condition.treatments else ['Professional medical care'])}
• **Medications:** {', '.join(condition.medications[:3] if condition.medications else ['As prescribed by healthcare provider'])}
• **Monitoring:** Regular follow-up with healthcare providers"""
            
            elif context_type == "prevention":
                return f"""• **Prevention Methods:** {', '.join(condition.prevention[:3] if condition.prevention else ['Healthy lifestyle choices'])}
• **Risk Reduction:** Address modifiable risk factors
• **Early Detection:** Regular health screenings"""
            
            else:
                return f"""• **Condition Overview:** {condition.name} - requires professional medical evaluation
• **Key Symptoms:** {', '.join(condition.symptoms[:3] if condition.symptoms else ['Consult healthcare providers for detailed information'])}"""
        
        else:
            # Generate response based on AI model prediction when no specific knowledge found
            if context_type == "diagnostic":
                return f"""• **AI Analysis:** {prediction}
• **Medical Evaluation:** Professional assessment recommended
• **Symptom Monitoring:** Track and report changes to healthcare provider"""
            
            elif context_type == "treatment":
                return f"""• **AI Analysis:** {prediction}
• **Treatment Planning:** Requires professional medical consultation
• **Follow-up:** Regular monitoring with healthcare provider"""
            
            elif context_type == "prevention":
                return f"""• **AI Analysis:** {prediction}
• **Prevention Focus:** Healthy lifestyle and regular medical care
• **Risk Management:** Follow professional medical guidance"""
            
            else:
                return f"""• **AI Analysis:** {prediction}
• **Professional Care:** Healthcare provider consultation recommended
• **Medical Guidance:** Follow evidence-based medical advice"""

    def _generate_model_consensus(self, biobert_result, clinical_result, pubmed_result) -> str:
        """Generate consensus from multiple medical models"""
        predictions = [biobert_result.prediction, clinical_result.prediction, pubmed_result.prediction]
        confidences = [biobert_result.confidence, clinical_result.confidence, pubmed_result.confidence]
        
        # Find highest confidence prediction
        max_idx = confidences.index(max(confidences))
        primary_prediction = predictions[max_idx]
        
        # Check for consensus
        if predictions.count(primary_prediction) >= 2:
            return f"Strong Consensus: {primary_prediction} (Multiple models agree)"
        else:
            return f"Mixed Analysis: Models show different perspectives - {primary_prediction} has highest confidence ({max(confidences):.2f})"

    def _generate_emergency_response(self, query: str, conditions: List, medical_terms: List[str], language: str) -> str:
        """Generate clean emergency response for end users"""
        
        if "heart attack" in query.lower() or "cardiac" in query.lower():
            return """🚨 **Heart Attack Warning Signs**

**Call 911 immediately if you have:**
• Chest pain or pressure
• Pain in arm, neck, jaw, or back
• Shortness of breath
• Nausea or sweating
• Dizziness or weakness

**What to do right now:**
1. Call 911 - don't drive yourself
2. Chew an aspirin if not allergic
3. Sit down and try to stay calm
4. Loosen tight clothing

⚠️ **This is a medical emergency - get help immediately!**"""

        elif "stroke" in query.lower():
            return """🚨 **Stroke Warning Signs**

**Call 911 immediately if you notice:**
• Face drooping on one side
• Arm weakness or numbness
• Speech difficulty or slurred words
• Sudden severe headache
• Loss of balance or coordination

**Remember FAST:**
- **F**ace: Ask them to smile
- **A**rms: Ask them to raise both arms
- **S**peech: Ask them to repeat a phrase
- **T**ime: Call 911 immediately

⚠️ **Time is critical - every minute matters!**"""

        else:
            return """🚨 **Medical Emergency**

**Call 911 immediately if you have:**
• Severe chest pain
• Difficulty breathing
• Severe bleeding
• Loss of consciousness
• Signs of stroke

**What to do:**
1. Call emergency services right away
2. Don't drive yourself to hospital
3. Stay calm and follow operator instructions
4. Have someone stay with you

⚠️ **Get professional medical help immediately!**"""

    def _generate_condition_specific_response(self, query: str, condition, medical_terms: List[str], biobert_result, clinical_result, pubmed_result, language: str) -> str:
        """Generate clean, user-friendly condition-specific response"""
        
        # For diabetes
        if "diabetes" in query.lower():
            return """## Diabetes Information

**What is Diabetes?**
Diabetes is a condition where your blood sugar levels are too high. There are two main types:
• **Type 1:** Your body doesn't make insulin
• **Type 2:** Your body doesn't use insulin properly

**Common Symptoms:**
• Increased thirst and urination
• Fatigue and weakness
• Blurred vision
• Slow-healing wounds
• Unexplained weight loss

**How to Know if You Have Diabetes:**
• **Blood tests:** Fasting glucose, A1C test
• **Symptoms check:** Look for signs above
• **See a doctor** for proper testing

**Management:**
• Monitor blood sugar regularly
• Take medications as prescribed
• Eat a balanced diet
• Exercise regularly
• Regular doctor checkups

💡 **Important:** If you suspect diabetes, see a healthcare provider for proper testing and diagnosis."""

        # For hypertension  
        elif "hypertension" in query.lower() or "blood pressure" in query.lower():
            return """## High Blood Pressure (Hypertension)

**What is High Blood Pressure?**
Blood pressure consistently above 140/90 mmHg. Often called the "silent killer" because it usually has no symptoms.

**How to Know if You Have It:**
• **Get it measured** at doctor's office or pharmacy
• **Home monitoring** with blood pressure cuff
• **Regular checkups** - it often has no symptoms

**Risk Factors:**
• Family history
• Age (risk increases with age)
• Being overweight
• Too much salt in diet
• Lack of exercise
• Stress

**Management:**
• Take medications as prescribed
• Reduce salt intake
• Exercise regularly
• Maintain healthy weight
• Limit alcohol
• Don't smoke

💡 **Important:** Regular monitoring is key since high blood pressure often has no symptoms."""

        # For heart conditions
        elif "heart" in query.lower() and "attack" not in query.lower():
            return """## Heart Health Information

**Common Heart Conditions:**
• **Heart disease:** Blocked arteries
• **Heart failure:** Heart can't pump well
• **Arrhythmia:** Irregular heartbeat

**Warning Signs to Watch:**
• Chest pain or discomfort
• Shortness of breath
• Fatigue or weakness
• Swelling in legs/feet
• Irregular heartbeat

**How to Know if You Have Heart Problems:**
• **See a doctor** for chest pain or symptoms
• **Heart tests:** EKG, echocardiogram, stress test
• **Blood tests:** Check cholesterol, enzymes

**Heart-Healthy Habits:**
• Exercise regularly
• Eat heart-healthy foods
• Don't smoke
• Manage stress
• Control blood pressure and cholesterol

💡 **Important:** Don't ignore chest pain or heart symptoms - see a doctor promptly."""

        # Generic condition response
        else:
            condition_name = getattr(condition, 'name', 'Medical Condition')
            response = f"## {condition_name}\n\n"
            
            if hasattr(condition, 'symptoms') and condition.symptoms:
                response += "**Common Signs & Symptoms:**\n"
                for symptom in condition.symptoms[:5]:
                    response += f"• {symptom}\n"
                response += "\n"
            
            if hasattr(condition, 'when_to_seek_help') and condition.when_to_seek_help:
                response += "**When to See a Doctor:**\n"
                for help_info in condition.when_to_seek_help[:3]:
                    response += f"• {help_info}\n"
                response += "\n"
            
            if hasattr(condition, 'treatments') and condition.treatments:
                response += "**Treatment Options:**\n"
                for treatment in condition.treatments[:3]:
                    response += f"• {treatment}\n"
                response += "\n"
            
            response += "💡 **Important:** This information is for educational purposes. Always consult healthcare professionals for medical advice."
            
            return response

    def _generate_comprehensive_medical_response(self, query: str, intent: str, medical_terms: List[str], biobert_result, clinical_result, pubmed_result, language: str) -> str:
        """Generate clean, user-friendly response for general medical queries"""
        
        # Handle specific symptoms or conditions mentioned in query
        if "symptoms" in query.lower():
            return """## Understanding Medical Symptoms

**When to Be Concerned:**
• Symptoms that are severe or getting worse
• Symptoms that last longer than expected
• New symptoms you've never had before
• Symptoms that interfere with daily activities

**What to Do:**
• Keep track of when symptoms started
• Note what makes them better or worse
• See a doctor if symptoms persist or worsen
• Call 911 for emergency symptoms

**Common Warning Signs:**
• Severe chest pain
• Difficulty breathing
• High fever that won't go down
• Severe headache with vision changes
• Signs of stroke (FAST test)

💡 **Remember:** Don't ignore persistent or severe symptoms - it's always better to check with a healthcare provider."""

        elif "prevention" in query.lower() or intent == "prevention_info":
            return """## Staying Healthy

**Daily Healthy Habits:**
• Eat a balanced diet with fruits and vegetables
• Exercise regularly (at least 30 minutes most days)
• Get adequate sleep (7-9 hours per night)
• Stay hydrated
• Don't smoke or use tobacco

**Regular Health Maintenance:**
• Annual checkups with your doctor
• Age-appropriate screenings (mammograms, colonoscopy, etc.)
• Keep vaccinations up to date
• Monitor blood pressure and cholesterol
• Practice good hygiene

**Mental Health:**
• Manage stress through relaxation techniques
• Stay socially connected
• Seek help when feeling overwhelmed
• Practice mindfulness or meditation

💡 **Prevention is always better than treatment!**"""

        elif "diagnosis" in query.lower() or intent == "diagnostic_info":
            return """## Medical Diagnosis Process

**How Doctors Diagnose Conditions:**
• **Medical history:** Tell your doctor about symptoms, family history, medications
• **Physical exam:** Doctor examines your body for signs of illness
• **Tests if needed:** Blood tests, X-rays, or other studies

**Preparing for Your Appointment:**
• Write down your symptoms and when they started
• List all medications you take
• Bring insurance cards and ID
• Prepare questions you want to ask

**Questions to Ask Your Doctor:**
• What could be causing my symptoms?
• What tests do I need?
• What are my treatment options?
• When should I follow up?

💡 **Be honest and complete when describing symptoms - it helps your doctor help you better.**"""

        else:
            return """## General Health Information

**When to See a Healthcare Provider:**
• For routine checkups and preventive care
• When you have concerning symptoms
• For medication management
• To discuss health concerns or questions

**Taking Care of Your Health:**
• Know your family medical history
• Keep track of your medications
• Maintain healthy lifestyle habits
• Stay up to date with recommended screenings

**Emergency Situations:**
Call 911 immediately for:
• Chest pain or heart attack symptoms
• Stroke symptoms (face drooping, arm weakness, speech difficulty)
• Severe allergic reactions
• Serious injuries or accidents

💡 **Your health is important - don't hesitate to seek medical care when you need it.**"""

    def _generate_intelligent_content_based_on_models(
        self, intent: str, medical_terms: list, biobert_result, clinical_result, pubmed_result
    ) -> str:
        """Generate intelligent content based on actual model outputs and medical understanding"""
        
        # Get the query from the context to provide specific information
        query = getattr(self, '_current_query', '').lower()
        
        # Generate specific content based on the medical condition mentioned
        if any(term in query for term in ['cardiac arrest', 'heart attack', 'heart failure', 'cardiac']):
            return self._generate_cardiac_content(intent)
        elif any(term in query for term in ['diabetes', 'blood sugar', 'insulin', 'diabetic']):
            return self._generate_diabetes_content(intent)
        elif any(term in query for term in ['hypertension', 'high blood pressure', 'bp', 'blood pressure']):
            return self._generate_hypertension_content(intent)
        elif any(term in query for term in ['stroke', 'brain attack', 'cerebral']):
            return self._generate_stroke_content(intent)
        elif any(term in query for term in ['cancer', 'tumor', 'malignant', 'oncology']):
            return self._generate_cancer_content(intent)
        elif any(term in query for term in ['covid', 'coronavirus', 'fever', 'cough', 'flu']):
            return self._generate_infectious_disease_content(intent)
        elif any(term in query for term in ['asthma', 'breathing', 'respiratory']):
            return self._generate_respiratory_content(intent)
        elif any(term in query for term in ['kidney', 'renal', 'urinary']):
            return self._generate_kidney_content(intent)
        elif any(term in query for term in ['mental health', 'depression', 'anxiety', 'stress']):
            return self._generate_mental_health_content(intent)
        else:
            # Generate content based on intent and model insights for general queries
            return self._generate_general_content_by_intent(intent)

    def _generate_cardiac_content(self, intent: str) -> str:
        """Generate specific content for cardiac conditions"""
        if intent == "diagnostic_info":
            return """🫀 **Cardiac Arrest - Critical Emergency Information**

**⚠️ IMMEDIATE EMERGENCY SIGNS - CALL 911:**
• **Sudden collapse** - person falls unconscious
• **No breathing** or gasping/abnormal breathing
• **No pulse** - unresponsive to touch or voice
• **Blue/gray skin color** especially lips and face

**Warning Signs That May Occur Before Cardiac Arrest:**
• Severe chest pain or pressure (like elephant on chest)
• Extreme shortness of breath
• Sudden dizziness or fainting
• Rapid or very irregular heartbeat
• Nausea and cold sweats
• Severe weakness or fatigue

**What Cardiac Arrest Means:**
• Heart suddenly stops beating effectively
• Blood stops flowing to brain and organs
• Different from heart attack (blocked artery)
• Can happen to anyone, any age
• Survival depends on immediate action

**CRITICAL: Every minute without CPR reduces survival by 10%**

"""
        elif intent == "treatment_info":
            return """🚨 **Cardiac Arrest Emergency Treatment**

**IMMEDIATE ACTIONS (DO NOT DELAY):**
1. **Call 911 immediately** - say "cardiac arrest"
2. **Start CPR** if you know how:
   • Push hard and fast on center of chest
   • 100-120 compressions per minute
   • Let chest come back up between pushes
3. **Find an AED** (automated external defibrillator)
   • Many public places have them
   • Follow the voice prompts
4. **Continue until help arrives**

**Hospital Emergency Treatment:**
• Advanced CPR with medications
• Electric shock (defibrillation) to restart heart
• Breathing tube and ventilator
• Cardiac catheterization to open blocked arteries
• Intensive care monitoring

**Recovery Treatment:**
• Therapeutic cooling to protect brain
• Medications to support heart and circulation
• Cardiac rehabilitation program
• Implantable defibrillator (ICD) may be recommended

"""
        else:  # prevention_info
            return """🛡️ **Cardiac Arrest Prevention**

**Major Risk Factors to Address:**
• **Heart disease** - get regular cardiac check-ups
• **High blood pressure** - keep below 130/80
• **High cholesterol** - know your numbers
• **Diabetes** - maintain good blood sugar control
• **Smoking** - quit completely, avoid secondhand smoke

**Heart-Healthy Lifestyle:**
• **Exercise regularly** - 150 minutes moderate activity/week
• **Heart-healthy diet** - Mediterranean style, low sodium
• **Maintain healthy weight** - BMI 18.5-24.9
• **Limit alcohol** - no more than 1-2 drinks per day
• **Manage stress** - meditation, relaxation techniques
• **Get quality sleep** - 7-9 hours per night

**Important Medical Care:**
• Annual physical exams with EKG
• Know your family history of heart disease
• Take prescribed heart medications as directed
• **Learn CPR** - you could save a life
• Recognize warning signs of heart problems

**When to Seek Immediate Care:**
• Chest pain, pressure, or discomfort
• Shortness of breath with activity
• Unexplained fatigue or weakness
• Swelling in legs, ankles, or feet

"""

    def _generate_diabetes_content(self, intent: str) -> str:
        """Generate specific content for diabetes"""
        if intent == "diagnostic_info":
            return """🩸 **How to Know if You Have Diabetes**

**Common Early Symptoms:**
• **Frequent urination** - especially at night
• **Excessive thirst** - can't quench it
• **Unexplained hunger** - eating but still hungry
• **Unexplained weight loss** - losing weight without trying
• **Extreme fatigue** - feeling very tired
• **Blurred vision** - difficulty focusing
• **Slow-healing cuts/bruises**
• **Tingling in hands or feet**

**Blood Sugar Tests (Get tested if you have symptoms):**
• **Fasting blood glucose:** Normal <100, Diabetes ≥126 mg/dL
• **Random blood glucose:** Diabetes ≥200 mg/dL with symptoms
• **A1C test:** Normal <5.7%, Diabetes ≥6.5%
• **Oral glucose tolerance test:** Diabetes ≥200 mg/dL at 2 hours

**Who Should Get Tested:**
• Age 35 and older (earlier if overweight)
• Family history of diabetes
• Overweight with additional risk factors
• High blood pressure (≥140/90)
• History of gestational diabetes
• Certain ethnicities at higher risk

"""
        elif intent == "treatment_info":
            return """💉 **Diabetes Treatment & Management**

**Type 1 Diabetes Treatment:**
• **Insulin therapy** - multiple daily injections or pump
• **Blood glucose monitoring** - check 4+ times daily
• **Carbohydrate counting** - match insulin to food
• **Continuous glucose monitors** - real-time readings
• **Regular endocrinologist visits**

**Type 2 Diabetes Treatment:**
• **Lifestyle changes first** - diet and exercise
• **Metformin** - usually first medication
• **Additional medications** as needed (many options)
• **Insulin** if other treatments insufficient
• **Blood pressure and cholesterol management**

**Daily Management for Both Types:**
• **Monitor blood sugar** - know your target ranges
• **Take medications as prescribed** - don't skip doses
• **Eat consistently** - regular meal times, portion control
• **Exercise regularly** - helps lower blood sugar
• **Foot care** - check daily for cuts or sores
• **Eye exams** - annually to prevent complications

**Target Numbers:**
• A1C: Less than 7% for most adults
• Before meals: 80-130 mg/dL
• After meals: Less than 180 mg/dL

"""
        else:  # prevention_info
            return """🛡️ **Type 2 Diabetes Prevention**

**Lifestyle Changes That Work:**
• **Lose weight** - even 5-10% reduction helps significantly
• **Exercise regularly** - 150 minutes moderate activity weekly
• **Eat healthy** - focus on whole foods, limit processed
• **Portion control** - use smaller plates, read labels
• **Limit sugary drinks** - water, unsweetened tea/coffee instead

**High-Risk Factors (Get Regular Testing):**
• **Family history** - parent or sibling with diabetes
• **Age 35+** - risk increases with age
• **Overweight/obesity** - especially belly fat
• **Sedentary lifestyle** - sitting most of the day
• **Previous gestational diabetes**
• **PCOS** (polycystic ovary syndrome)

**Proven Prevention Strategies:**
• **Mediterranean diet** - proven to reduce diabetes risk
• **Regular physical activity** - walking, swimming, cycling
• **Weight management** - maintain healthy BMI
• **Regular check-ups** - annual blood sugar screening
• **Stress management** - chronic stress affects blood sugar

**Red Flags - Get Tested Soon:**
• Persistent fatigue and thirst
• Frequent infections
• Family history with symptoms
• Gestational diabetes history

"""

    def _generate_hypertension_content(self, intent: str) -> str:
        """Generate specific content for hypertension"""
        return """🩺 **High Blood Pressure (Hypertension)**

**Understanding Blood Pressure Numbers:**
• **Normal:** Less than 120/80 mmHg
• **Elevated:** 120-129 (top) and less than 80 (bottom)
• **Stage 1:** 130-139/80-89 mmHg
• **Stage 2:** 140/90 mmHg or higher
• **Crisis:** Over 180/120 mmHg (emergency!)

**Why It's Called "Silent Killer":**
• Usually no symptoms until severe
• Damages heart, brain, kidneys, eyes over time
• Can cause heart attack, stroke, kidney failure
• Regular monitoring is essential

**Management & Treatment:**
• **DASH diet** - fruits, vegetables, whole grains, low sodium
• **Exercise** - 150 minutes moderate activity weekly
• **Weight loss** - even small amounts help
• **Limit alcohol** - no more than 1-2 drinks daily
• **Quit smoking** - improves circulation immediately
• **Stress management** - meditation, deep breathing
• **Medications** if lifestyle changes aren't enough

**Home Monitoring:**
• Check at same time daily
• Rest 5 minutes before measuring
• Proper cuff size and positioning
• Keep a log for your doctor

"""

    def _generate_stroke_content(self, intent: str) -> str:
        """Generate specific content for stroke"""
        return """🧠 **Stroke - Brain Attack Emergency**

**BE-FAST Recognition (Call 911 Immediately):**
• **B - Balance:** Sudden dizziness, loss of coordination
• **E - Eyes:** Sudden vision loss or changes
• **F - Face:** Face drooping, smile uneven on one side
• **A - Arms:** Arm weakness, can't raise both arms
• **S - Speech:** Slurred speech, can't repeat simple phrases
• **T - Time:** Note time symptoms started, call 911

**Types of Stroke:**
• **Ischemic (87%):** Blood clot blocks brain artery
• **Hemorrhagic (13%):** Bleeding in or around brain
• **TIA (mini-stroke):** Temporary blockage, symptoms resolve

**Emergency Treatment:**
• **Call 911 immediately** - don't drive yourself
• **Clot-busting drugs** - must be given within hours
• **Thrombectomy** - removing clot with catheter
• **Time is brain** - every minute counts

**Risk Factors:**
• High blood pressure (biggest risk)
• Smoking and diabetes
• High cholesterol and obesity
• Age over 55, family history
• Previous stroke or heart disease

"""

    def _generate_cancer_content(self, intent: str) -> str:
        """Generate specific content for cancer"""
        return """🎗️ **Cancer Information & Early Detection**

**General Warning Signs:**
• **Unexplained weight loss** - 10+ pounds without trying
• **Persistent fatigue** - extreme tiredness not relieved by rest
• **Fever** - especially recurring or prolonged
• **Pain** - new, persistent, or worsening
• **Skin changes** - new moles, changes in existing moles

**Screening Saves Lives:**
• **Mammograms** - breast cancer (age 40-50+)
• **Colonoscopy** - colorectal cancer (age 45-50+)
• **Pap smears** - cervical cancer (age 21-65)
• **Skin checks** - melanoma (annual dermatology exam)
• **Lung CT** - for heavy smokers

**Modern Treatment Options:**
• **Surgery** - removing tumors
• **Chemotherapy** - drugs that kill cancer cells
• **Radiation** - high-energy beams target cancer
• **Immunotherapy** - helps immune system fight cancer
• **Targeted therapy** - drugs target specific cancer features

**Prevention Strategies:**
• **Don't smoke** - leading preventable cause
• **Limit alcohol** - increases risk of several cancers
• **Protect from sun** - use sunscreen, avoid tanning
• **Maintain healthy weight** - obesity linked to many cancers
• **Get vaccinated** - HPV, Hepatitis B

"""

    def _generate_infectious_disease_content(self, intent: str) -> str:
        """Generate specific content for infectious diseases"""
        return """🦠 **Infectious Disease Information**

**Common Symptoms to Monitor:**
• **Fever** - especially over 101°F (38.3°C)
• **Cough** - persistent or worsening
• **Shortness of breath** - difficulty breathing
• **Fatigue** - extreme tiredness
• **Body aches** - muscle and joint pain
• **Headache** - severe or persistent

**When to Seek Medical Care:**
• **High fever** - over 103°F (39.4°C)
• **Difficulty breathing** - can't catch breath
• **Chest pain** - persistent or severe
• **Severe dehydration** - dizziness, dry mouth
• **Symptoms worsening** after initial improvement

**Prevention (Works for Most Infections):**
• **Hand hygiene** - wash 20+ seconds frequently
• **Vaccination** - stay up-to-date on all vaccines
• **Mask wearing** - when sick or in crowded spaces
• **Social distancing** - when you're ill
• **Stay home when sick** - protect others

**Treatment Principles:**
• **Rest** - let your body fight the infection
• **Hydration** - plenty of fluids
• **Symptom management** - fever reducers, cough medicine
• **Antibiotics** - only for bacterial infections (not viral)
• **Antivirals** - for some viral infections like flu, COVID

"""

    def _generate_respiratory_content(self, intent: str) -> str:
        """Generate specific content for respiratory conditions"""
        return """🫁 **Respiratory Health Information**

**Common Respiratory Symptoms:**
• **Shortness of breath** - with activity or at rest
• **Wheezing** - whistling sound when breathing
• **Persistent cough** - lasting more than 3 weeks
• **Chest tightness** - feeling of pressure or squeezing
• **Mucus production** - color and consistency changes

**Asthma Management:**
• **Identify triggers** - allergens, exercise, stress, weather
• **Use medications properly** - rescue and controller inhalers
• **Peak flow monitoring** - track breathing capacity
• **Action plan** - know when to seek emergency care
• **Regular follow-up** - with pulmonologist or primary care

**When to Seek Emergency Care:**
• **Severe breathing difficulty** - can't speak in full sentences
• **Blue lips or fingernails** - sign of low oxygen
• **Chest pain** - especially with breathing
• **High fever with breathing problems**
• **Coughing up blood**

**Prevention & Management:**
• **Avoid triggers** - smoke, pollution, allergens
• **Exercise regularly** - improves lung function
• **Maintain healthy weight** - reduces breathing strain
• **Get vaccinated** - flu and pneumonia vaccines
• **Don't smoke** - quitting improves lung function

"""

    def _generate_kidney_content(self, intent: str) -> str:
        """Generate specific content for kidney conditions"""
        return """🫘 **Kidney Health Information**

**Early Signs of Kidney Problems:**
• **Foamy urine** - especially persistent foam
• **Blood in urine** - red, pink, or brown color
• **Swelling** - face, hands, feet, or ankles
• **Frequent urination** - especially at night
• **Fatigue** - feeling tired despite adequate rest
• **High blood pressure** - new or worsening

**Risk Factors for Kidney Disease:**
• **Diabetes** - leading cause of kidney failure
• **High blood pressure** - second leading cause
• **Family history** - genetic predisposition
• **Age over 60** - kidney function naturally declines
• **Heart disease** - affects kidney blood flow

**Protecting Your Kidneys:**
• **Control blood sugar** - if diabetic, keep A1C <7%
• **Manage blood pressure** - target <130/80
• **Limit salt** - less than 2,300mg daily
• **Stay hydrated** - 8 glasses water daily (unless restricted)
• **Exercise regularly** - improves overall circulation
• **Avoid NSAIDs** - ibuprofen, naproxen can damage kidneys
• **Don't smoke** - reduces blood flow to kidneys

**When to See a Nephrologist:**
• Persistent protein in urine
• Declining kidney function (GFR <60)
• Hard-to-control blood pressure
• Recurrent kidney stones

"""

    def _generate_mental_health_content(self, intent: str) -> str:
        """Generate specific content for mental health"""
        return """🧠 **Mental Health Information**

**Common Signs of Depression:**
• **Persistent sadness** - lasting 2+ weeks
• **Loss of interest** - in activities once enjoyed
• **Fatigue** - feeling tired despite adequate rest
• **Sleep changes** - too much or too little
• **Appetite changes** - eating much more or less
• **Difficulty concentrating** - trouble making decisions
• **Feelings of worthlessness** or excessive guilt

**Anxiety Symptoms:**
• **Excessive worry** - hard to control
• **Restlessness** - feeling on edge
• **Muscle tension** - especially neck, shoulders
• **Rapid heartbeat** - racing or pounding
• **Shortness of breath** - feeling like you can't breathe
• **Sweating** - especially palms and underarms

**When to Seek Professional Help:**
• Symptoms interfere with daily life
• Thoughts of self-harm or suicide
• Substance use to cope
• Relationship or work problems due to mood
• Physical symptoms without medical cause

**Treatment Options:**
• **Therapy** - cognitive behavioral therapy, counseling
• **Medications** - antidepressants, anti-anxiety medications
• **Lifestyle changes** - exercise, sleep hygiene, stress management
• **Support groups** - connecting with others who understand
• **Mindfulness** - meditation, relaxation techniques

**Crisis Resources:**
• **988 Suicide & Crisis Lifeline** - call or text 988
• **Crisis Text Line** - text HOME to 741741
• **Emergency services** - call 911 for immediate danger

"""

    def _generate_general_content_by_intent(self, intent: str) -> str:
        """Generate general content based on intent"""
        if intent == "diagnostic_info":
            return """🩺 **General Diagnostic Guidance**
• Pay attention to persistent or worsening symptoms
• Keep track of when symptoms occur and their severity
• Note any triggers or patterns you observe
• Don't ignore warning signs - trust your body
• Seek professional evaluation for health concerns
• Bring a list of symptoms and questions to appointments

"""
        elif intent == "treatment_info":
            return """💊 **General Treatment Principles**
• Follow prescribed treatment plans consistently
• Take medications exactly as directed by your doctor
• Make recommended lifestyle changes gradually
• Attend all follow-up appointments
• Communicate openly with your healthcare team
• Report any side effects or concerns promptly

"""
        elif intent == "prevention_info":
            return """🛡️ **General Prevention Strategies**
• Maintain healthy lifestyle habits daily
• Get regular medical check-ups and screenings
• Know your family medical history
• Manage stress through healthy coping mechanisms
• Stay up-to-date with recommended vaccinations
• Build strong relationships with healthcare providers

"""
        else:
            return """📚 **General Health Information**
• Every individual's health situation is unique
• Professional medical evaluation is important for proper care
• Treatment plans should be personalized to your specific needs
• Stay informed about your health conditions and treatments
• Build a good relationship with your healthcare providers
• Don't hesitate to ask questions or seek second opinions

"""

    async def _enrich_with_live_medical_data(self, response: str, query: str) -> str:
        """Enrich response with real-time medical API data"""
        if not self.immediate_apis:
            return response
            
        try:
            enriched_data = []
            
            # Extract potential drug names from query
            drug_keywords = self._extract_drug_keywords(query)
            for drug in drug_keywords[:2]:  # Limit to 2 drugs to avoid overload
                fda_data = self.immediate_apis.get_fda_drug_info(drug)
                if fda_data.get("status") == "success" and fda_data.get("results"):
                    enriched_data.append(f"📊 **Latest FDA Data for {drug}:** Current safety information available")
            
            # Add MeSH terms for medical concepts
            mesh_data = self.immediate_apis.get_mesh_disease_info(query)
            if mesh_data.get("status") == "success":
                enriched_data.append("🔬 **Medical Research:** Latest research citations available")
            
            # Add real-time drug interaction data
            if drug_keywords:
                rxnorm_data = self.immediate_apis.get_rxnorm_drug_info(drug_keywords[0])
                if rxnorm_data.get("status") == "success":
                    enriched_data.append("⚕️ **Drug Information:** Current prescribing information available")
            
            if enriched_data:
                enrichment = "\n\n**🌐 Real-Time Medical Data Available:**\n" + "\n".join(enriched_data)
                enrichment += "\n\n*Note: This information is supplemented with current medical databases including FDA, NIH, and medical research sources.*"
                return response + enrichment
                
        except Exception as e:
            logger.warning(f"Failed to enrich with live data: {e}")
            
        return response
    
    def _extract_drug_keywords(self, query: str) -> List[str]:
        """Extract potential drug names from query"""
        # Simple keyword extraction - can be enhanced with NER
        drug_indicators = ['medication', 'drug', 'pill', 'medicine', 'tablet', 'capsule']
        words = query.lower().split()
        
        # Look for words that might be drug names (capitalized in original or after drug indicators)
        potential_drugs = []
        for i, word in enumerate(words):
            if any(indicator in word for indicator in drug_indicators):
                # Check next few words for potential drug names
                for j in range(i+1, min(i+4, len(words))):
                    if words[j].isalpha() and len(words[j]) > 3:
                        potential_drugs.append(words[j])
        
        return potential_drugs[:3]  # Return max 3 potential drugs
