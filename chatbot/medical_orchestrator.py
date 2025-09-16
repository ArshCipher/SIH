"""
Competition-Grade Medical AI Orchestrator
Multi-Agent Medical System with Advanced Safety and Consensus Mechanisms
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class MedicalRiskLevel(Enum):
    """Medical risk classification levels"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AgentType(Enum):
    """Specialized medical agent types"""
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    SAFETY = "safety"
    TRIAGE = "triage"
    DRUG_INTERACTION = "drug_interaction"
    PREVENTION = "prevention"
    OUTBREAK = "outbreak"
    MENTAL_HEALTH = "mental_health"

@dataclass
class MedicalResponse:
    """Structured medical response with safety metadata"""
    agent_type: AgentType
    content: str
    confidence: float
    risk_level: MedicalRiskLevel
    medical_entities: List[str]
    sources: List[str]
    safety_flags: List[str]
    requires_human_review: bool
    explanation: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class MedicalConsensus:
    """Ensemble consensus result with explainability"""
    final_response: str
    consensus_confidence: float
    risk_level: MedicalRiskLevel
    agent_responses: List[MedicalResponse]
    safety_validated: bool
    human_escalation_required: bool
    explanation: str
    medical_disclaimer: str
    session_id: str

class MedicalAgent:
    """Base class for specialized medical agents"""
    
    def __init__(self, agent_type: AgentType, confidence_threshold: float = 0.85):
        self.agent_type = agent_type
        self.confidence_threshold = confidence_threshold
        self.name = f"MedicalAgent_{agent_type.value}"
        
    async def process_query(self, query: str, context: Dict[str, Any]) -> MedicalResponse:
        """Process medical query - to be implemented by specialized agents"""
        raise NotImplementedError("Specialized agents must implement process_query")
    
    async def validate_safety(self, response: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate response safety - base implementation"""
        safety_flags = []
        
        # Basic safety checks
        dangerous_keywords = [
            "self-medicate", "ignore doctor", "skip medication", "dangerous", 
            "overdose", "suicide", "self-harm"
        ]
        
        for keyword in dangerous_keywords:
            if keyword.lower() in response.lower():
                safety_flags.append(f"Contains dangerous keyword: {keyword}")
        
        is_safe = len(safety_flags) == 0
        return is_safe, safety_flags

class DiagnosticAgent(MedicalAgent):
    """Specialized agent for medical diagnosis assistance"""
    
    def __init__(self):
        super().__init__(AgentType.DIAGNOSTIC, confidence_threshold=0.90)
        
    async def process_query(self, query: str, context: Dict[str, Any]) -> MedicalResponse:
        """Process diagnostic queries with high safety standards"""
        
        # Extract symptoms from query
        symptoms = await self._extract_symptoms(query)
        
        # Get potential conditions (simplified for demo - would use medical models)
        conditions = await self._analyze_symptoms(symptoms)
        
        # Generate response with appropriate disclaimers
        response_content = await self._generate_diagnostic_response(symptoms, conditions)
        
        # Determine risk level
        risk_level = await self._assess_risk_level(symptoms, conditions)
        
        # Safety validation
        is_safe, safety_flags = await self.validate_safety(response_content, context)
        
        return MedicalResponse(
            agent_type=self.agent_type,
            content=response_content,
            confidence=0.85,  # Would be calculated by medical models
            risk_level=risk_level,
            medical_entities=symptoms + conditions,
            sources=["Clinical Guidelines", "Medical Literature"],
            safety_flags=safety_flags,
            requires_human_review=risk_level in [MedicalRiskLevel.HIGH, MedicalRiskLevel.CRITICAL, MedicalRiskLevel.EMERGENCY],
            explanation="Diagnostic assessment based on symptom analysis and medical knowledge base"
        )
    
    async def _extract_symptoms(self, query: str) -> List[str]:
        """Extract medical symptoms from query"""
        # Simplified implementation - would use medical NER models
        symptom_keywords = {
            "fever", "headache", "cough", "nausea", "vomiting", "diarrhea", 
            "chest pain", "shortness of breath", "fatigue", "dizziness"
        }
        
        found_symptoms = []
        query_lower = query.lower()
        for symptom in symptom_keywords:
            if symptom in query_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    async def _analyze_symptoms(self, symptoms: List[str]) -> List[str]:
        """Analyze symptoms to suggest potential conditions"""
        # Simplified medical logic - would use advanced medical reasoning
        conditions = []
        
        if "fever" in symptoms and "cough" in symptoms:
            conditions.extend(["Upper Respiratory Infection", "Influenza"])
        if "chest pain" in symptoms and "shortness of breath" in symptoms:
            conditions.extend(["Possible Cardiac Event", "Pulmonary Embolism"])
        if "headache" in symptoms and "fever" in symptoms:
            conditions.extend(["Viral Infection", "Meningitis"])
            
        return conditions
    
    async def _generate_diagnostic_response(self, symptoms: List[str], conditions: List[str]) -> str:
        """Generate diagnostic response with medical disclaimers"""
        if not symptoms:
            return "I need more specific information about your symptoms to provide helpful guidance. Please describe what you're experiencing."
        
        response = f"Based on the symptoms you've described ({', '.join(symptoms)}), "
        
        if conditions:
            response += f"some possible conditions to consider include: {', '.join(conditions)}. "
        
        response += """

⚠️ IMPORTANT MEDICAL DISCLAIMER:
This is AI-generated information for educational purposes only. It is NOT a medical diagnosis or substitute for professional medical advice. 

🏥 SEEK IMMEDIATE MEDICAL ATTENTION if you experience:
• Severe chest pain or difficulty breathing
• Signs of stroke (sudden weakness, confusion, speech problems)
• Severe allergic reactions
• Any life-threatening symptoms

👨‍⚕️ Please consult a qualified healthcare provider for proper diagnosis and treatment."""
        
        return response
    
    async def _assess_risk_level(self, symptoms: List[str], conditions: List[str]) -> MedicalRiskLevel:
        """Assess medical risk level based on symptoms and conditions"""
        
        emergency_symptoms = {"chest pain", "shortness of breath", "severe headache"}
        high_risk_conditions = {"cardiac event", "pulmonary embolism", "meningitis"}
        
        if any(symptom in emergency_symptoms for symptom in symptoms):
            return MedicalRiskLevel.EMERGENCY
            
        if any(condition.lower() in " ".join(high_risk_conditions) for condition in conditions):
            return MedicalRiskLevel.HIGH
            
        if len(symptoms) > 3:
            return MedicalRiskLevel.MODERATE
            
        return MedicalRiskLevel.LOW

class SafetyAgent(MedicalAgent):
    """Specialized agent for medical safety validation"""
    
    def __init__(self):
        super().__init__(AgentType.SAFETY, confidence_threshold=0.95)
        
    async def process_query(self, query: str, context: Dict[str, Any]) -> MedicalResponse:
        """Process safety validation for medical responses"""
        
        # Analyze query for safety concerns
        safety_assessment = await self._comprehensive_safety_check(query, context)
        
        response_content = await self._generate_safety_response(safety_assessment)
        
        return MedicalResponse(
            agent_type=self.agent_type,
            content=response_content,
            confidence=safety_assessment["confidence"],
            risk_level=safety_assessment["risk_level"],
            medical_entities=safety_assessment["flagged_entities"],
            sources=["Medical Safety Guidelines", "FDA Safety Database"],
            safety_flags=safety_assessment["safety_flags"],
            requires_human_review=safety_assessment["requires_review"],
            explanation="Comprehensive medical safety validation"
        )
    
    async def _comprehensive_safety_check(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive safety validation"""
        
        safety_flags = []
        flagged_entities = []
        risk_level = MedicalRiskLevel.LOW
        
        # Check for dangerous medication combinations
        if "medication" in query.lower() or "drug" in query.lower():
            drug_safety = await self._check_drug_safety(query)
            safety_flags.extend(drug_safety["flags"])
            flagged_entities.extend(drug_safety["entities"])
        
        # Check for self-harm indicators
        self_harm_indicators = ["suicide", "self-harm", "hurt myself", "end it all"]
        if any(indicator in query.lower() for indicator in self_harm_indicators):
            safety_flags.append("CRITICAL: Self-harm indicators detected")
            risk_level = MedicalRiskLevel.EMERGENCY
        
        # Check for medical misinformation patterns
        misinformation_patterns = [
            "vaccines cause autism", "covid is fake", "bleach cure", 
            "essential oils cure cancer"
        ]
        if any(pattern in query.lower() for pattern in misinformation_patterns):
            safety_flags.append("Medical misinformation detected")
            risk_level = MedicalRiskLevel.HIGH
        
        return {
            "confidence": 0.95,
            "risk_level": risk_level,
            "safety_flags": safety_flags,
            "flagged_entities": flagged_entities,
            "requires_review": len(safety_flags) > 0 or risk_level in [MedicalRiskLevel.HIGH, MedicalRiskLevel.CRITICAL, MedicalRiskLevel.EMERGENCY]
        }
    
    async def _check_drug_safety(self, query: str) -> Dict[str, Any]:
        """Check for drug safety issues"""
        # Simplified implementation - would integrate with drug interaction databases
        return {
            "flags": [],
            "entities": []
        }
    
    async def _generate_safety_response(self, assessment: Dict[str, Any]) -> str:
        """Generate safety-focused response"""
        if assessment["risk_level"] == MedicalRiskLevel.EMERGENCY:
            return """🚨 MEDICAL EMERGENCY DETECTED 🚨

If you or someone you know is having thoughts of self-harm:
• Call emergency services: 911 (US), 108 (India)
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741

You are not alone. Professional help is available 24/7."""

        elif assessment["safety_flags"]:
            return f"""⚠️ SAFETY CONCERN IDENTIFIED

I've detected potential safety issues in your query. For your safety:
• Please consult a qualified healthcare professional
• Do not attempt self-treatment for serious conditions
• Verify all medical information with authoritative sources

Safety flags: {', '.join(assessment['safety_flags'])}"""
        
        return "Safety validation completed. No immediate concerns identified."

class TriageAgent(MedicalAgent):
    """Specialized agent for medical triage and urgency assessment"""
    
    def __init__(self):
        super().__init__(AgentType.TRIAGE, confidence_threshold=0.88)
        
    async def process_query(self, query: str, context: Dict[str, Any]) -> MedicalResponse:
        """Process medical triage assessment"""
        
        triage_assessment = await self._assess_urgency(query)
        response_content = await self._generate_triage_response(triage_assessment)
        
        return MedicalResponse(
            agent_type=self.agent_type,
            content=response_content,
            confidence=triage_assessment["confidence"],
            risk_level=triage_assessment["urgency_level"],
            medical_entities=triage_assessment["symptoms"],
            sources=["Emergency Medicine Guidelines", "Triage Protocols"],
            safety_flags=triage_assessment["alerts"],
            requires_human_review=triage_assessment["urgency_level"] in [MedicalRiskLevel.HIGH, MedicalRiskLevel.EMERGENCY],
            explanation="Medical triage assessment based on symptom urgency"
        )
    
    async def _assess_urgency(self, query: str) -> Dict[str, Any]:
        """Assess medical urgency level"""
        
        # Emergency indicators
        emergency_keywords = [
            "chest pain", "can't breathe", "unconscious", "seizure", 
            "severe bleeding", "stroke", "heart attack", "overdose", "emergency"
        ]
        
        # High priority indicators  
        high_priority = [
            "severe pain", "high fever", "persistent vomiting", 
            "difficulty swallowing", "severe headache"
        ]
        
        # Moderate priority
        moderate_priority = [
            "fever", "cough", "headache", "nausea", "fatigue"
        ]
        
        query_lower = query.lower()
        symptoms = []
        urgency_level = MedicalRiskLevel.LOW
        alerts = []
        
        for keyword in emergency_keywords:
            if keyword in query_lower:
                urgency_level = MedicalRiskLevel.EMERGENCY
                symptoms.append(keyword)
                alerts.append(f"Emergency symptom detected: {keyword}")
        
        if urgency_level != MedicalRiskLevel.EMERGENCY:
            for keyword in high_priority:
                if keyword in query_lower:
                    urgency_level = MedicalRiskLevel.HIGH
                    symptoms.append(keyword)
            
            if urgency_level not in [MedicalRiskLevel.HIGH, MedicalRiskLevel.EMERGENCY]:
                for keyword in moderate_priority:
                    if keyword in query_lower:
                        urgency_level = MedicalRiskLevel.MODERATE
                        symptoms.append(keyword)
        
        return {
            "confidence": 0.88,
            "urgency_level": urgency_level,
            "symptoms": symptoms,
            "alerts": alerts
        }
    
    async def _generate_triage_response(self, assessment: Dict[str, Any]) -> str:
        """Generate triage-appropriate response"""
        
        if assessment["urgency_level"] == MedicalRiskLevel.EMERGENCY:
            return """🚨 MEDICAL EMERGENCY - SEEK IMMEDIATE HELP 🚨

Call emergency services NOW:
• US: 911
• India: 108
• UK: 999

Do not delay seeking emergency medical care."""
        
        elif assessment["urgency_level"] == MedicalRiskLevel.HIGH:
            return """⚠️ HIGH PRIORITY MEDICAL CONCERN

Recommend seeking medical attention within 2-4 hours:
• Visit urgent care or emergency room
• Contact your healthcare provider immediately
• Do not wait if symptoms worsen"""
        
        elif assessment["urgency_level"] == MedicalRiskLevel.MODERATE:
            return """📋 MODERATE PRIORITY

Consider scheduling medical consultation within 24-48 hours:
• Contact your primary care physician
• Monitor symptoms closely
• Seek immediate care if symptoms worsen"""
        
        return """ℹ️ LOW PRIORITY

Your symptoms appear to be non-urgent:
• Continue monitoring symptoms
• Consider scheduling routine consultation if symptoms persist
• Practice appropriate self-care measures"""

class MedicalOrchestrator:
    """Competition-grade medical AI orchestrator with multi-agent consensus"""
    
    def __init__(self):
        self.agents = {
            AgentType.DIAGNOSTIC: DiagnosticAgent(),
            AgentType.SAFETY: SafetyAgent(),
            AgentType.TRIAGE: TriageAgent(),
            # Additional agents would be initialized here
        }
        
        self.consensus_threshold = 0.80
        self.safety_threshold = 0.95
        
        # Medical disclaimers
        self.medical_disclaimer = """
⚠️ IMPORTANT: This AI system provides information for educational purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of qualified healthcare providers with questions about medical conditions.
"""
    
    async def process_medical_query(self, query: str, context: Dict[str, Any]) -> MedicalConsensus:
        """Process medical query through multi-agent system with consensus"""
        
        session_id = str(uuid.uuid4())
        
        # Step 1: Safety pre-screening
        safety_agent = self.agents[AgentType.SAFETY]
        safety_response = await safety_agent.process_query(query, context)
        
        if safety_response.risk_level == MedicalRiskLevel.EMERGENCY:
            return MedicalConsensus(
                final_response=safety_response.content,
                consensus_confidence=safety_response.confidence,
                risk_level=safety_response.risk_level,
                agent_responses=[safety_response],
                safety_validated=True,
                human_escalation_required=True,
                explanation="Emergency detected - immediate safety response",
                medical_disclaimer=self.medical_disclaimer,
                session_id=session_id
            )
        
        # Step 2: Parallel agent processing
        agent_tasks = []
        relevant_agents = await self._select_relevant_agents(query, context)
        
        for agent_type in relevant_agents:
            if agent_type in self.agents:
                task = self.agents[agent_type].process_query(query, context)
                agent_tasks.append(task)
        
        # Execute agents in parallel
        agent_responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Filter out exceptions and invalid responses
        valid_responses = [
            response for response in agent_responses 
            if isinstance(response, MedicalResponse)
        ]
        
        # Step 3: Consensus building
        consensus = await self._build_consensus(valid_responses, safety_response)
        
        # Step 4: Final safety validation
        final_safety_check = await self._final_safety_validation(consensus)
        
        return MedicalConsensus(
            final_response=consensus["final_response"],
            consensus_confidence=consensus["confidence"],
            risk_level=consensus["risk_level"],
            agent_responses=valid_responses,
            safety_validated=final_safety_check["is_safe"],
            human_escalation_required=final_safety_check["requires_escalation"],
            explanation=consensus["explanation"],
            medical_disclaimer=self.medical_disclaimer,
            session_id=session_id
        )
    
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
    
    async def _build_consensus(self, responses: List[MedicalResponse], safety_response: MedicalResponse) -> Dict[str, Any]:
        """Build consensus from multiple agent responses"""
        
        if not responses:
            return {
                "final_response": "I need more information to provide a helpful medical response.",
                "confidence": 0.0,
                "risk_level": MedicalRiskLevel.LOW,
                "explanation": "No valid agent responses received"
            }
        
        # Weighted voting based on confidence and agent type
        weights = {
            AgentType.SAFETY: 0.3,
            AgentType.TRIAGE: 0.25,
            AgentType.DIAGNOSTIC: 0.2,
            AgentType.TREATMENT: 0.15,
            AgentType.DRUG_INTERACTION: 0.1
        }
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        
        for response in responses:
            weight = weights.get(response.agent_type, 0.1)
            weighted_confidence += response.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            consensus_confidence = weighted_confidence / total_weight
        else:
            consensus_confidence = 0.5
        
        # Determine highest risk level
        risk_levels = [response.risk_level for response in responses]
        risk_priorities = {
            MedicalRiskLevel.EMERGENCY: 5,
            MedicalRiskLevel.CRITICAL: 4,
            MedicalRiskLevel.HIGH: 3,
            MedicalRiskLevel.MODERATE: 2,
            MedicalRiskLevel.LOW: 1
        }
        
        highest_risk = max(risk_levels, key=lambda x: risk_priorities[x])
        
        # Combine responses intelligently
        final_response = await self._combine_responses(responses, highest_risk)
        
        return {
            "final_response": final_response,
            "confidence": consensus_confidence,
            "risk_level": highest_risk,
            "explanation": f"Consensus built from {len(responses)} specialized medical agents"
        }
    
    async def _combine_responses(self, responses: List[MedicalResponse], risk_level: MedicalRiskLevel) -> str:
        """Intelligently combine multiple agent responses"""
        
        # Prioritize by risk level and agent type
        sorted_responses = sorted(
            responses, 
            key=lambda x: (
                {"emergency": 5, "critical": 4, "high": 3, "moderate": 2, "low": 1}[x.risk_level.value],
                x.confidence
            ),
            reverse=True
        )
        
        combined_response = ""
        
        # Start with highest priority response
        if sorted_responses:
            combined_response = sorted_responses[0].content
        
        # Add relevant information from other agents
        for response in sorted_responses[1:]:
            if response.agent_type == AgentType.SAFETY and response.safety_flags:
                combined_response += f"\n\n⚠️ Safety Alert: {', '.join(response.safety_flags)}"
        
        return combined_response
    
    async def _final_safety_validation(self, consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Final safety validation of consensus response"""
        
        requires_escalation = (
            consensus["risk_level"] in [MedicalRiskLevel.HIGH, MedicalRiskLevel.CRITICAL, MedicalRiskLevel.EMERGENCY] or
            consensus["confidence"] < self.consensus_threshold
        )
        
        return {
            "is_safe": consensus["confidence"] >= self.safety_threshold,
            "requires_escalation": requires_escalation
        }
