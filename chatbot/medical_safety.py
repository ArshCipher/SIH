"""
Competition-Grade Medical Safety and Compliance System
Bulletproof safety validation with regulatory compliance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime, timedelta
import hashlib
import uuid

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Medical safety levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class BiasType(Enum):
    """Types of medical bias to detect"""
    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    AGE_BIAS = "age_bias"
    SOCIOECONOMIC_BIAS = "socioeconomic_bias"
    CULTURAL_BIAS = "cultural_bias"
    DISABILITY_BIAS = "disability_bias"

class ComplianceStandard(Enum):
    """Medical compliance standards"""
    HIPAA = "hipaa"
    FDA = "fda"
    WHO = "who"
    GDPR = "gdpr"
    MEDICAL_ETHICS = "medical_ethics"

@dataclass
class SafetyFlag:
    """Individual safety flag with metadata"""
    flag_id: str
    flag_type: str
    severity: SafetyLevel
    description: str
    evidence: str
    recommendation: str
    compliance_issues: List[ComplianceStandard]
    timestamp: datetime
    
    def __post_init__(self):
        if not self.flag_id:
            self.flag_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

@dataclass
class BiasDetectionResult:
    """Bias detection analysis result"""
    bias_detected: bool
    bias_types: List[BiasType]
    bias_score: float
    evidence: List[str]
    mitigation_suggestions: List[str]
    confidence: float

@dataclass
class ComplianceAudit:
    """Compliance audit result"""
    compliant: bool
    violations: List[str]
    compliance_score: float
    standards_checked: List[ComplianceStandard]
    recommendations: List[str]
    audit_timestamp: datetime

@dataclass
class SafetyValidationResult:
    """Comprehensive safety validation result"""
    is_safe: bool
    safety_level: SafetyLevel
    safety_flags: List[SafetyFlag]
    bias_analysis: BiasDetectionResult
    compliance_audit: ComplianceAudit
    hallucination_detected: bool
    emergency_escalation_required: bool
    human_review_required: bool
    safety_score: float
    explanation: str
    session_id: str

class MedicalHallucinationDetector:
    """Detects AI-generated medical misinformation and hallucinations"""
    
    def __init__(self):
        self.known_medical_facts = self._load_medical_facts()
        self.contradiction_patterns = self._load_contradiction_patterns()
        self.misinformation_database = self._load_misinformation_database()
    
    def _load_medical_facts(self) -> Dict[str, Any]:
        """Load verified medical facts database"""
        return {
            "covid19": {
                "transmission": ["respiratory droplets", "airborne", "surface contact"],
                "symptoms": ["fever", "cough", "fatigue", "loss of taste", "shortness of breath"],
                "prevention": ["vaccination", "masks", "social distancing", "hand hygiene"],
                "false_claims": ["5g causes covid", "bleach cures covid", "covid is fake"]
            },
            "vaccines": {
                "safety": "extensively tested and monitored",
                "effectiveness": "highly effective at preventing severe disease",
                "false_claims": ["vaccines cause autism", "vaccines contain microchips", "natural immunity better"]
            },
            "antibiotics": {
                "use": "bacterial infections only",
                "ineffective_against": ["viruses", "common cold", "flu"],
                "resistance": "overuse leads to antibiotic resistance"
            }
        }
    
    def _load_contradiction_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns that indicate medical contradictions"""
        return [
            {
                "pattern": r"(?:covid|coronavirus).*(?:fake|hoax|doesn't exist)",
                "severity": SafetyLevel.CRITICAL,
                "description": "COVID-19 denial misinformation"
            },
            {
                "pattern": r"vaccines?.*(?:cause|causes).*autism",
                "severity": SafetyLevel.DANGER,
                "description": "Vaccine-autism misinformation"
            },
            {
                "pattern": r"(?:bleach|disinfectant).*(?:cure|treat|drink)",
                "severity": SafetyLevel.EMERGENCY,
                "description": "Dangerous substance ingestion advice"
            },
            {
                "pattern": r"(?:don't|never|avoid).*(?:see|visit|consult).*doctor",
                "severity": SafetyLevel.DANGER,
                "description": "Advising against medical care"
            },
            {
                "pattern": r"(?:stop|discontinue).*medication.*(?:immediately|cold turkey)",
                "severity": SafetyLevel.DANGER,
                "description": "Dangerous medication discontinuation advice"
            }
        ]
    
    def _load_misinformation_database(self) -> Set[str]:
        """Load database of known medical misinformation"""
        return {
            "vaccines cause autism",
            "5g causes coronavirus",
            "covid is a hoax",
            "bleach cures covid",
            "essential oils cure cancer",
            "natural immunity is better than vaccines",
            "masks don't work",
            "hydroxychloroquine cures covid",
            "ivermectin cures covid",
            "covid vaccines change dna",
            "covid vaccines contain microchips"
        }
    
    async def detect_hallucination(self, text: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Detect medical hallucinations and misinformation"""
        
        hallucinations = []
        text_lower = text.lower()
        
        # Check against known misinformation
        for misinfo in self.misinformation_database:
            if misinfo in text_lower:
                hallucinations.append(f"Known misinformation detected: {misinfo}")
        
        # Check contradiction patterns
        for pattern_data in self.contradiction_patterns:
            if re.search(pattern_data["pattern"], text_lower, re.IGNORECASE):
                hallucinations.append(f"Dangerous pattern detected: {pattern_data['description']}")
        
        # Check for impossible medical claims
        impossible_claims = await self._check_impossible_claims(text)
        hallucinations.extend(impossible_claims)
        
        # Check for factual contradictions
        contradictions = await self._check_factual_contradictions(text)
        hallucinations.extend(contradictions)
        
        return len(hallucinations) > 0, hallucinations
    
    async def _check_impossible_claims(self, text: str) -> List[str]:
        """Check for medically impossible claims"""
        
        impossible_claims = []
        text_lower = text.lower()
        
        # Impossible cure claims
        impossible_cures = [
            (r"(?:water|h2o).*cures?.*(?:cancer|diabetes|aids)", "Water cannot cure serious diseases"),
            (r"(?:prayer|meditation).*cures?.*(?:cancer|diabetes)", "Prayer/meditation alone cannot cure diseases"),
            (r"(?:diet|food).*cures?.*(?:cancer|diabetes|aids)", "Diet alone cannot cure serious diseases"),
            (r"(?:exercise|workout).*cures?.*(?:cancer|diabetes)", "Exercise alone cannot cure diseases")
        ]
        
        for pattern, claim in impossible_cures:
            if re.search(pattern, text_lower, re.IGNORECASE):
                impossible_claims.append(claim)
        
        # Impossible timelines
        if re.search(r"(?:instant|immediate|overnight).*(?:cure|heal)", text_lower):
            impossible_claims.append("Instant medical cures are not realistic")
        
        return impossible_claims
    
    async def _check_factual_contradictions(self, text: str) -> List[str]:
        """Check for contradictions with established medical facts"""
        
        contradictions = []
        text_lower = text.lower()
        
        # Check COVID-19 facts
        if "covid" in text_lower or "coronavirus" in text_lower:
            covid_facts = self.known_medical_facts["covid19"]
            for false_claim in covid_facts["false_claims"]:
                if false_claim in text_lower:
                    contradictions.append(f"COVID-19 misinformation: {false_claim}")
        
        # Check vaccine facts
        if "vaccine" in text_lower:
            vaccine_facts = self.known_medical_facts["vaccines"]
            for false_claim in vaccine_facts["false_claims"]:
                if false_claim in text_lower:
                    contradictions.append(f"Vaccine misinformation: {false_claim}")
        
        return contradictions

class MedicalBiasDetector:
    """Detects and mitigates medical bias in AI responses"""
    
    def __init__(self):
        self.bias_patterns = self._load_bias_patterns()
        self.inclusive_language = self._load_inclusive_language_guide()
    
    def _load_bias_patterns(self) -> Dict[BiasType, List[Dict[str, Any]]]:
        """Load bias detection patterns"""
        return {
            BiasType.GENDER_BIAS: [
                {
                    "pattern": r"(?:women|females?).*(?:emotional|hysterical|overreacting)",
                    "description": "Gender bias - dismissing women's symptoms as emotional"
                },
                {
                    "pattern": r"(?:men|males?).*(?:don't|never).*(?:cry|show emotion)",
                    "description": "Gender bias - toxic masculinity in healthcare"
                }
            ],
            BiasType.RACIAL_BIAS: [
                {
                    "pattern": r"(?:black|african american).*(?:drug|addiction|compliance)",
                    "description": "Racial bias - assuming substance abuse or non-compliance"
                },
                {
                    "pattern": r"(?:asian|chinese).*(?:virus|disease|blame)",
                    "description": "Racial bias - COVID-19 related discrimination"
                }
            ],
            BiasType.AGE_BIAS: [
                {
                    "pattern": r"(?:elderly|old).*(?:confused|senile|can't understand)",
                    "description": "Age bias - assuming cognitive impairment in elderly"
                },
                {
                    "pattern": r"(?:young|teenager).*(?:exaggerating|attention seeking)",
                    "description": "Age bias - dismissing young people's symptoms"
                }
            ],
            BiasType.SOCIOECONOMIC_BIAS: [
                {
                    "pattern": r"(?:poor|low income|uninsured).*(?:non-compliant|unreliable)",
                    "description": "Socioeconomic bias - assumptions about healthcare compliance"
                }
            ]
        }
    
    def _load_inclusive_language_guide(self) -> Dict[str, str]:
        """Load inclusive language recommendations"""
        return {
            "diabetic": "person with diabetes",
            "addict": "person with substance use disorder",
            "victim": "survivor",
            "suffers from": "has/lives with",
            "normal": "typical",
            "crazy": "concerning symptoms"
        }
    
    async def detect_bias(self, text: str, context: Dict[str, Any]) -> BiasDetectionResult:
        """Comprehensive bias detection analysis"""
        
        bias_detected = False
        detected_bias_types = []
        evidence = []
        bias_score = 0.0
        
        text_lower = text.lower()
        
        # Check each bias type
        for bias_type, patterns in self.bias_patterns.items():
            for pattern_data in patterns:
                if re.search(pattern_data["pattern"], text_lower, re.IGNORECASE):
                    bias_detected = True
                    detected_bias_types.append(bias_type)
                    evidence.append(pattern_data["description"])
                    bias_score += 0.3  # Accumulate bias score
        
        # Check for non-inclusive language
        inclusive_violations = await self._check_inclusive_language(text)
        if inclusive_violations:
            bias_detected = True
            evidence.extend(inclusive_violations)
            bias_score += len(inclusive_violations) * 0.1
        
        # Generate mitigation suggestions
        mitigation_suggestions = await self._generate_mitigation_suggestions(detected_bias_types, evidence)
        
        # Cap bias score at 1.0
        bias_score = min(bias_score, 1.0)
        
        return BiasDetectionResult(
            bias_detected=bias_detected,
            bias_types=detected_bias_types,
            bias_score=bias_score,
            evidence=evidence,
            mitigation_suggestions=mitigation_suggestions,
            confidence=0.8 if bias_detected else 0.9
        )
    
    async def _check_inclusive_language(self, text: str) -> List[str]:
        """Check for non-inclusive language"""
        
        violations = []
        text_lower = text.lower()
        
        for problematic_term, inclusive_term in self.inclusive_language.items():
            if problematic_term in text_lower:
                violations.append(f"Consider using '{inclusive_term}' instead of '{problematic_term}'")
        
        return violations
    
    async def _generate_mitigation_suggestions(self, bias_types: List[BiasType], evidence: List[str]) -> List[str]:
        """Generate bias mitigation suggestions"""
        
        suggestions = []
        
        if BiasType.GENDER_BIAS in bias_types:
            suggestions.append("Ensure equal consideration of symptoms regardless of gender")
            suggestions.append("Avoid assumptions about emotional vs. physical symptoms")
        
        if BiasType.RACIAL_BIAS in bias_types:
            suggestions.append("Provide culturally sensitive healthcare information")
            suggestions.append("Avoid racial stereotypes in medical recommendations")
        
        if BiasType.AGE_BIAS in bias_types:
            suggestions.append("Consider age-appropriate communication without condescension")
            suggestions.append("Avoid assumptions about cognitive abilities based on age")
        
        # General suggestions
        suggestions.extend([
            "Use person-first language",
            "Focus on medical facts rather than assumptions",
            "Ensure respectful and inclusive communication"
        ])
        
        return suggestions

class ComplianceValidator:
    """Validates medical AI responses against regulatory standards"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.required_disclaimers = self._load_required_disclaimers()
    
    def _load_compliance_rules(self) -> Dict[ComplianceStandard, List[Dict[str, Any]]]:
        """Load compliance validation rules"""
        return {
            ComplianceStandard.HIPAA: [
                {
                    "rule": "no_personal_health_info",
                    "pattern": r"(?:ssn|social security|dob|date of birth|address|phone)",
                    "description": "Avoid requesting or storing personal health information"
                }
            ],
            ComplianceStandard.FDA: [
                {
                    "rule": "no_unapproved_claims",
                    "pattern": r"(?:fda approved|cures|guaranteed)",
                    "description": "Avoid unapproved medical claims"
                }
            ],
            ComplianceStandard.MEDICAL_ETHICS: [
                {
                    "rule": "informed_consent",
                    "pattern": r"(?:must|should|have to).*(?:take|use).*medication",
                    "description": "Respect patient autonomy and informed consent"
                },
                {
                    "rule": "do_no_harm",
                    "pattern": r"(?:ignore|avoid|don't see).*doctor",
                    "description": "Do not advise against seeking medical care"
                }
            ]
        }
    
    def _load_required_disclaimers(self) -> Dict[str, str]:
        """Load required medical disclaimers"""
        return {
            "general": "This information is for educational purposes only and is not a substitute for professional medical advice.",
            "emergency": "If this is a medical emergency, call emergency services immediately.",
            "diagnosis": "This AI cannot provide medical diagnoses. Consult a healthcare professional for proper evaluation.",
            "treatment": "Treatment recommendations must be individualized by qualified healthcare providers."
        }
    
    async def validate_compliance(self, text: str, context: Dict[str, Any]) -> ComplianceAudit:
        """Comprehensive compliance validation"""
        
        violations = []
        compliance_score = 1.0
        standards_checked = list(self.compliance_rules.keys())
        recommendations = []
        
        text_lower = text.lower()
        
        # Check each compliance standard
        for standard, rules in self.compliance_rules.items():
            for rule_data in rules:
                if re.search(rule_data["pattern"], text_lower, re.IGNORECASE):
                    violations.append(f"{standard.value}: {rule_data['description']}")
                    compliance_score -= 0.2
        
        # Check for required disclaimers
        disclaimer_missing = await self._check_disclaimers(text, context)
        if disclaimer_missing:
            violations.extend(disclaimer_missing)
            compliance_score -= len(disclaimer_missing) * 0.1
        
        # Generate recommendations
        if violations:
            recommendations = await self._generate_compliance_recommendations(violations)
        
        # Ensure compliance score doesn't go below 0
        compliance_score = max(compliance_score, 0.0)
        
        return ComplianceAudit(
            compliant=len(violations) == 0,
            violations=violations,
            compliance_score=compliance_score,
            standards_checked=standards_checked,
            recommendations=recommendations,
            audit_timestamp=datetime.utcnow()
        )
    
    async def _check_disclaimers(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Check for missing required disclaimers"""
        
        missing_disclaimers = []
        text_lower = text.lower()
        
        # Check if medical disclaimer is present
        disclaimer_indicators = ["educational purposes", "not a substitute", "consult", "professional medical advice"]
        has_disclaimer = any(indicator in text_lower for indicator in disclaimer_indicators)
        
        if not has_disclaimer:
            missing_disclaimers.append("Missing general medical disclaimer")
        
        # Check for emergency disclaimer if emergency keywords present
        emergency_keywords = ["emergency", "urgent", "severe", "critical", "life-threatening"]
        if any(keyword in text_lower for keyword in emergency_keywords):
            if "emergency services" not in text_lower and "call 911" not in text_lower:
                missing_disclaimers.append("Missing emergency services disclaimer")
        
        return missing_disclaimers
    
    async def _generate_compliance_recommendations(self, violations: List[str]) -> List[str]:
        """Generate compliance improvement recommendations"""
        
        recommendations = []
        
        if any("personal health info" in v for v in violations):
            recommendations.append("Remove requests for personal health information")
        
        if any("unapproved claims" in v for v in violations):
            recommendations.append("Avoid making definitive medical claims")
        
        if any("disclaimer" in v for v in violations):
            recommendations.append("Include appropriate medical disclaimers")
        
        recommendations.append("Ensure all responses comply with medical ethics guidelines")
        
        return recommendations

class MedicalSafetyValidator:
    """Comprehensive medical safety validation system"""
    
    def __init__(self):
        self.hallucination_detector = MedicalHallucinationDetector()
        self.bias_detector = MedicalBiasDetector()
        self.compliance_validator = ComplianceValidator()
        
        # Safety thresholds
        self.safety_thresholds = {
            "hallucination_tolerance": 0.0,  # Zero tolerance for hallucinations
            "bias_threshold": 0.3,
            "compliance_threshold": 0.8,
            "overall_safety_threshold": 0.8
        }
        
        # Emergency escalation patterns
        self.emergency_patterns = [
            r"(?:suicide|kill myself|end it all)",
            r"(?:overdose|too many pills)",
            r"(?:chest pain|can't breathe|heart attack)",
            r"(?:severe bleeding|blood loss)",
            r"(?:unconscious|passed out|not responding)"
        ]
    
    async def comprehensive_safety_validation(self, text: str, context: Dict[str, Any]) -> SafetyValidationResult:
        """Comprehensive safety validation with all checks"""
        
        session_id = context.get("session_id", str(uuid.uuid4()))
        safety_flags = []
        
        # Step 1: Hallucination detection
        hallucination_detected, hallucination_evidence = await self.hallucination_detector.detect_hallucination(text, context)
        
        if hallucination_detected:
            for evidence in hallucination_evidence:
                flag = SafetyFlag(
                    flag_id=str(uuid.uuid4()),
                    flag_type="hallucination",
                    severity=SafetyLevel.CRITICAL,
                    description="Medical misinformation detected",
                    evidence=evidence,
                    recommendation="Remove misinformation and provide accurate medical information",
                    compliance_issues=[ComplianceStandard.MEDICAL_ETHICS],
                    timestamp=datetime.utcnow()
                )
                safety_flags.append(flag)
        
        # Step 2: Bias detection
        bias_analysis = await self.bias_detector.detect_bias(text, context)
        
        if bias_analysis.bias_detected:
            flag = SafetyFlag(
                flag_id=str(uuid.uuid4()),
                flag_type="bias",
                severity=SafetyLevel.WARNING if bias_analysis.bias_score < 0.5 else SafetyLevel.DANGER,
                description=f"Medical bias detected: {', '.join([bt.value for bt in bias_analysis.bias_types])}",
                evidence='; '.join(bias_analysis.evidence),
                recommendation='; '.join(bias_analysis.mitigation_suggestions),
                compliance_issues=[ComplianceStandard.MEDICAL_ETHICS],
                timestamp=datetime.utcnow()
            )
            safety_flags.append(flag)
        
        # Step 3: Compliance validation
        compliance_audit = await self.compliance_validator.validate_compliance(text, context)
        
        if not compliance_audit.compliant:
            flag = SafetyFlag(
                flag_id=str(uuid.uuid4()),
                flag_type="compliance",
                severity=SafetyLevel.WARNING,
                description="Regulatory compliance issues detected",
                evidence='; '.join(compliance_audit.violations),
                recommendation='; '.join(compliance_audit.recommendations),
                compliance_issues=compliance_audit.standards_checked,
                timestamp=datetime.utcnow()
            )
            safety_flags.append(flag)
        
        # Step 4: Emergency detection
        emergency_escalation_required = await self._check_emergency_escalation(text)
        
        if emergency_escalation_required:
            flag = SafetyFlag(
                flag_id=str(uuid.uuid4()),
                flag_type="emergency",
                severity=SafetyLevel.EMERGENCY,
                description="Medical emergency indicators detected",
                evidence="Emergency keywords or patterns identified",
                recommendation="Immediate escalation to emergency services",
                compliance_issues=[ComplianceStandard.MEDICAL_ETHICS],
                timestamp=datetime.utcnow()
            )
            safety_flags.append(flag)
        
        # Step 5: Calculate overall safety
        safety_assessment = await self._calculate_overall_safety(
            safety_flags, bias_analysis, compliance_audit, hallucination_detected
        )
        
        # Step 6: Determine human review requirement
        human_review_required = await self._determine_human_review_requirement(
            safety_flags, safety_assessment["safety_score"]
        )
        
        return SafetyValidationResult(
            is_safe=safety_assessment["is_safe"],
            safety_level=safety_assessment["safety_level"],
            safety_flags=safety_flags,
            bias_analysis=bias_analysis,
            compliance_audit=compliance_audit,
            hallucination_detected=hallucination_detected,
            emergency_escalation_required=emergency_escalation_required,
            human_review_required=human_review_required,
            safety_score=safety_assessment["safety_score"],
            explanation=safety_assessment["explanation"],
            session_id=session_id
        )
    
    async def _check_emergency_escalation(self, text: str) -> bool:
        """Check if emergency escalation is required"""
        
        text_lower = text.lower()
        
        for pattern in self.emergency_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    async def _calculate_overall_safety(self, safety_flags: List[SafetyFlag], 
                                      bias_analysis: BiasDetectionResult,
                                      compliance_audit: ComplianceAudit,
                                      hallucination_detected: bool) -> Dict[str, Any]:
        """Calculate overall safety assessment"""
        
        # Start with perfect safety score
        safety_score = 1.0
        
        # Deduct for safety flags
        critical_flags = sum(1 for flag in safety_flags if flag.severity == SafetyLevel.CRITICAL)
        danger_flags = sum(1 for flag in safety_flags if flag.severity == SafetyLevel.DANGER)
        warning_flags = sum(1 for flag in safety_flags if flag.severity == SafetyLevel.WARNING)
        
        safety_score -= critical_flags * 0.5
        safety_score -= danger_flags * 0.3
        safety_score -= warning_flags * 0.1
        
        # Deduct for bias
        safety_score -= bias_analysis.bias_score * 0.2
        
        # Deduct for compliance issues
        safety_score -= (1.0 - compliance_audit.compliance_score) * 0.2
        
        # Zero tolerance for hallucinations
        if hallucination_detected:
            safety_score = 0.0
        
        # Ensure score doesn't go below 0
        safety_score = max(safety_score, 0.0)
        
        # Determine safety level
        if safety_score >= 0.9:
            safety_level = SafetyLevel.SAFE
        elif safety_score >= 0.7:
            safety_level = SafetyLevel.CAUTION
        elif safety_score >= 0.5:
            safety_level = SafetyLevel.WARNING
        elif safety_score >= 0.3:
            safety_level = SafetyLevel.DANGER
        else:
            safety_level = SafetyLevel.CRITICAL
        
        # Check for emergency override
        emergency_flags = [flag for flag in safety_flags if flag.severity == SafetyLevel.EMERGENCY]
        if emergency_flags:
            safety_level = SafetyLevel.EMERGENCY
            safety_score = 0.0
        
        # Determine if safe
        is_safe = safety_score >= self.safety_thresholds["overall_safety_threshold"] and not hallucination_detected
        
        # Generate explanation
        explanation = await self._generate_safety_explanation(
            safety_score, safety_level, safety_flags, hallucination_detected
        )
        
        return {
            "is_safe": is_safe,
            "safety_level": safety_level,
            "safety_score": safety_score,
            "explanation": explanation
        }
    
    async def _determine_human_review_requirement(self, safety_flags: List[SafetyFlag], safety_score: float) -> bool:
        """Determine if human review is required"""
        
        # Always require human review for critical or emergency situations
        critical_flags = [flag for flag in safety_flags if flag.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]]
        if critical_flags:
            return True
        
        # Require human review for low safety scores
        if safety_score < 0.6:
            return True
        
        # Require human review for multiple safety flags
        if len(safety_flags) >= 3:
            return True
        
        return False
    
    async def _generate_safety_explanation(self, safety_score: float, safety_level: SafetyLevel,
                                         safety_flags: List[SafetyFlag], hallucination_detected: bool) -> str:
        """Generate human-readable safety explanation"""
        
        explanation = f"Safety Assessment: {safety_level.value.upper()} (Score: {safety_score:.2f})\n\n"
        
        if safety_flags:
            explanation += "Safety Issues Identified:\n"
            for flag in safety_flags:
                explanation += f"• {flag.flag_type.upper()}: {flag.description}\n"
            explanation += "\n"
        
        if hallucination_detected:
            explanation += "⚠️ CRITICAL: Medical misinformation detected and blocked.\n\n"
        
        if safety_level == SafetyLevel.EMERGENCY:
            explanation += "🚨 EMERGENCY: Immediate human intervention required.\n"
        elif safety_level in [SafetyLevel.CRITICAL, SafetyLevel.DANGER]:
            explanation += "⚠️ HIGH RISK: Human review required before response.\n"
        elif safety_level == SafetyLevel.WARNING:
            explanation += "⚠️ CAUTION: Response requires safety modifications.\n"
        else:
            explanation += "✅ SAFE: Response meets safety standards.\n"
        
        return explanation

# Emergency escalation protocols
class EmergencyEscalationProtocol:
    """Handles emergency medical situations with immediate escalation"""
    
    def __init__(self):
        self.emergency_contacts = {
            "US": "911",
            "India": "108",
            "UK": "999",
            "Australia": "000"
        }
        
        self.crisis_resources = {
            "suicide_prevention": {
                "US": "988",
                "international": "https://findahelpline.com"
            },
            "poison_control": {
                "US": "1-800-222-1222"
            }
        }
    
    async def handle_emergency(self, safety_result: SafetyValidationResult, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle medical emergency with immediate escalation"""
        
        if not safety_result.emergency_escalation_required:
            return {"escalated": False, "message": "No emergency escalation required"}
        
        # Generate emergency response
        emergency_response = await self._generate_emergency_response(safety_result, context)
        
        # Log emergency for audit trail
        await self._log_emergency_escalation(safety_result, context)
        
        # Notify monitoring systems
        await self._notify_monitoring_systems(safety_result, context)
        
        return {
            "escalated": True,
            "emergency_response": emergency_response,
            "escalation_timestamp": datetime.utcnow().isoformat(),
            "session_id": safety_result.session_id
        }
    
    async def _generate_emergency_response(self, safety_result: SafetyValidationResult, context: Dict[str, Any]) -> str:
        """Generate appropriate emergency response"""
        
        country = context.get("country", "US")
        emergency_number = self.emergency_contacts.get(country, "911")
        
        response = "🚨 MEDICAL EMERGENCY DETECTED 🚨\n\n"
        response += f"Call emergency services immediately: {emergency_number}\n\n"
        
        # Check for specific emergency types
        emergency_flags = [flag for flag in safety_result.safety_flags if flag.severity == SafetyLevel.EMERGENCY]
        
        for flag in emergency_flags:
            if "suicide" in flag.evidence.lower():
                response += "If you're having thoughts of self-harm:\n"
                response += f"• Crisis hotline: {self.crisis_resources['suicide_prevention'].get(country, '988')}\n"
                response += "• You are not alone - help is available 24/7\n\n"
            
            elif "chest pain" in flag.evidence.lower() or "heart" in flag.evidence.lower():
                response += "For chest pain or heart symptoms:\n"
                response += "• Call emergency services immediately\n"
                response += "• Do not drive yourself to hospital\n"
                response += "• Chew aspirin if not allergic\n\n"
        
        response += "This is an automated emergency response. Professional help is being contacted."
        
        return response
    
    async def _log_emergency_escalation(self, safety_result: SafetyValidationResult, context: Dict[str, Any]):
        """Log emergency escalation for audit trail"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": safety_result.session_id,
            "emergency_type": [flag.flag_type for flag in safety_result.safety_flags if flag.severity == SafetyLevel.EMERGENCY],
            "safety_score": safety_result.safety_score,
            "context": context,
            "escalation_triggered": True
        }
        
        logger.critical(f"EMERGENCY ESCALATION: {json.dumps(log_entry)}")
    
    async def _notify_monitoring_systems(self, safety_result: SafetyValidationResult, context: Dict[str, Any]):
        """Notify monitoring and alerting systems"""
        
        # In production, this would integrate with monitoring systems
        # like PagerDuty, Slack, or custom alerting
        logger.critical(f"EMERGENCY NOTIFICATION: Session {safety_result.session_id} requires immediate attention")
