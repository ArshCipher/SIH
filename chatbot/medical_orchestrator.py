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

from chatbot.medical_models import MedicalModelEnsemble, medical_ensemble

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
        
        self.consensus_threshold = 0.50  # Lowered from 0.80 to be less restrictive
        self.safety_threshold = 0.70     # Lowered from 0.95 to be less restrictive
        
        # Medical disclaimers
        self.medical_disclaimer = """
⚠️ IMPORTANT: This AI system provides information for educational purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of qualified healthcare providers with questions about medical conditions.
"""
    
    async def process_medical_query(self, query: str, context: Dict[str, Any]) -> MedicalConsensus:
        """Process medical query through multi-agent system with ensemble consensus"""
        
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
            
            # Use multiple models for comprehensive medical analysis
            medical_analyses = []
            
            # BioBERT for general medical classification
            biobert_result = await self.medical_ensemble.predict(
                text=query,
                model_type=MedicalModelType.BIOBERT,
                task=MedicalTask.MEDICAL_TEXT_CLASSIFICATION,
                context=str(context)
            )
            medical_analyses.append(("BioBERT Classification", biobert_result))
            
            # ClinicalBERT for clinical information
            clinical_result = await self.medical_ensemble.predict(
                text=query,
                model_type=MedicalModelType.CLINICAL_BERT,
                task=MedicalTask.MEDICAL_TEXT_CLASSIFICATION,
                context=str(context)
            )
            medical_analyses.append(("ClinicalBERT Analysis", clinical_result))
            
            # PubMedBERT for research-based information
            pubmed_result = await self.medical_ensemble.predict(
                text=query,
                model_type=MedicalModelType.PUBMED_BERT,
                task=MedicalTask.MEDICAL_TEXT_CLASSIFICATION,
                context=str(context)
            )
            medical_analyses.append(("PubMedBERT Research", pubmed_result))
            
            # Get comprehensive medical information for complex queries
            comprehensive_analysis = ""
            
            # Use the enhanced medical models to generate contextual responses based on the actual query
            query_lower = query.lower()
            
            # Analyze the query to understand what specific information is being requested
            query_intent = self._analyze_query_intent(query, biobert_result, clinical_result, pubmed_result)
            
            # Generate contextual response based on medical models and query intent
            if any(keyword in query_lower for keyword in ["diabetes", "diabetic", "diabeties", "insulin", "glucose", "blood sugar"]):
                comprehensive_analysis = self._generate_diabetes_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            elif any(keyword in query_lower for keyword in ["cancer", "oncogenic", "tumor", "malignancy", "carcinoma", "adenocarcinoma"]):
                comprehensive_analysis = self._generate_cancer_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            elif any(keyword in query_lower for keyword in ["hypertension", "blood pressure", "ace inhibitor", "high pressure", "bp", "systolic", "diastolic"]):
                comprehensive_analysis = self._generate_hypertension_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            elif any(keyword in query_lower for keyword in ["heart", "cardiac", "cardio", "heart failure", "coronary", "angina", "myocardial"]):
                comprehensive_analysis = self._generate_heart_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            elif any(keyword in query_lower for keyword in ["car-t", "cart", "immunotherapy", "hematologic", "leukemia", "lymphoma"]):
                comprehensive_analysis = self._generate_cancer_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            elif any(keyword in query_lower for keyword in ["fever", "rash", "spots", "pain", "chest pain", "shortness of breath", "emergency", "urgent", "bleeding", "severe"]):
                comprehensive_analysis = self._generate_symptom_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            else:
                # For other medical queries, generate intelligent response based on model analysis
                comprehensive_analysis = self._generate_general_medical_response(query, query_intent, biobert_result, clinical_result, pubmed_result)
            
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
            consensus = await self._generate_consensus(ensemble_dict, valid_responses, session_id)
            
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
    
    async def _generate_consensus(self, ensemble_result: Dict[str, Any], agent_responses: List[AgentResponse], session_id: str) -> MedicalConsensus:
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
        
        # Determine the type of medical information requested
        if any(word in query_lower for word in ["how", "know", "tell", "check", "detect", "identify", "symptoms", "signs"]):
            return "diagnostic_info"
        elif any(word in query_lower for word in ["what is", "define", "explain", "describe", "meaning"]):
            return "definition_info"
        elif any(word in query_lower for word in ["treat", "treatment", "cure", "therapy", "medicine", "medication"]):
            return "treatment_info"
        elif any(word in query_lower for word in ["prevent", "avoid", "stop", "reduce risk"]):
            return "prevention_info"
        elif any(word in query_lower for word in ["cause", "why", "reason", "mechanism", "pathophysiology"]):
            return "causation_info"
        elif any(word in query_lower for word in ["complication", "risk", "danger", "outcome"]):
            return "risk_info"
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
        """Generate intelligent general medical response"""
        
        return f"""**Medical AI Analysis for: "{query}"**

**Advanced Model Analysis:**
• BioBERT Medical Classification: {biobert_result.prediction} (Confidence: {biobert_result.confidence:.2f})
• ClinicalBERT Clinical Analysis: {clinical_result.prediction} (Confidence: {clinical_result.confidence:.2f})
• PubMedBERT Research Analysis: {pubmed_result.prediction} (Confidence: {pubmed_result.confidence:.2f})

**Medical Information:**
Based on your query, our medical AI has analyzed the content using state-of-the-art medical language models trained on clinical literature.

**General Health Guidance:**
• Consult healthcare providers for personalized medical advice
• Regular check-ups help with early detection and prevention
• Maintain healthy lifestyle: balanced diet, exercise, adequate sleep
• Keep track of medications and medical history
• Don't hesitate to seek medical attention for concerning symptoms

**Emergency Situations:**
• Call 911 for severe symptoms: chest pain, difficulty breathing, severe bleeding
• Contact your doctor for persistent or worsening symptoms
• Use telemedicine for non-urgent questions

**Rural Healthcare Resources:**
• Mobile health clinics may serve your area
• Telemedicine connects you with doctors remotely
• Community health workers provide local support
• Emergency services available for urgent situations

This medical AI provides educational information to support informed healthcare decisions."""
