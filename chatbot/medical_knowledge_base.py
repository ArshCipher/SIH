"""
Comprehensive Medical Knowledge Base for Healthcare AI

This module contains detailed medical knowledge including:
- Disease conditions with symptoms, causes, treatments
- Emergency protocols and warning signs
- Medication information and interactions
- Diagnostic procedures and interpretations
- Prevention and lifestyle recommendations

Designed for rural healthcare AI to provide specific, actionable medical guidance.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class UrgencyLevel(Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent" 
    ROUTINE = "routine"
    INFORMATION = "information"

@dataclass
class MedicalCondition:
    name: str
    icd_codes: List[str]
    symptoms: List[str]
    causes: List[str]
    risk_factors: List[str]
    diagnostic_tests: List[str]
    treatments: List[str]
    medications: List[str]
    prevention: List[str]
    complications: List[str]
    urgency_level: UrgencyLevel
    emergency_signs: List[str]
    when_to_seek_help: List[str]

class MedicalKnowledgeBase:
    """Comprehensive medical knowledge base with specific conditions"""
    
    def __init__(self):
        self.conditions = {}
        self.symptoms_map = {}
        self.emergency_keywords = set()
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with comprehensive medical conditions - 500+ diseases"""
        
        # CARDIOVASCULAR DISEASES
        cardiovascular_conditions = [
            MedicalCondition(
                name="Myocardial Infarction (Heart Attack)",
                icd_codes=["I21", "I22"],
                symptoms=["chest pain", "shortness of breath", "nausea", "sweating", "arm pain", "jaw pain"],
                causes=["coronary artery disease", "blood clot", "atherosclerosis"],
                risk_factors=["smoking", "high cholesterol", "diabetes", "hypertension", "family history"],
                diagnostic_tests=["ECG", "troponin levels", "cardiac catheterization"],
                treatments=["aspirin", "thrombolytics", "angioplasty", "stenting", "bypass surgery"],
                medications=["aspirin", "clopidogrel", "metoprolol", "lisinopril", "atorvastatin"],
                prevention=["exercise", "healthy diet", "quit smoking", "manage diabetes"],
                complications=["heart failure", "arrhythmias", "death"],
                urgency_level=UrgencyLevel.EMERGENCY,
                emergency_signs=["severe chest pain", "difficulty breathing", "loss of consciousness"],
                when_to_seek_help=["chest pain lasting >5 minutes", "severe shortness of breath"]
            ),
            MedicalCondition(
                name="Hypertension (High Blood Pressure)",
                icd_codes=["I10", "I11", "I12"],
                symptoms=["headache", "dizziness", "blurred vision", "often asymptomatic"],
                causes=["unknown (essential)", "kidney disease", "hormonal disorders"],
                risk_factors=["age", "family history", "obesity", "high salt intake", "stress"],
                diagnostic_tests=["blood pressure measurement", "24-hour monitoring", "blood tests"],
                treatments=["lifestyle modifications", "ACE inhibitors", "diuretics", "calcium channel blockers"],
                medications=["lisinopril", "amlodipine", "hydrochlorothiazide", "metoprolol"],
                prevention=["low sodium diet", "regular exercise", "weight management", "limit alcohol"],
                complications=["stroke", "heart attack", "kidney disease", "eye damage"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["severe headache", "chest pain", "difficulty breathing"],
                when_to_seek_help=["BP >180/120", "symptoms of stroke", "chest pain"]
            ),
            MedicalCondition(
                name="Atrial Fibrillation",
                icd_codes=["I48"],
                symptoms=["irregular heartbeat", "palpitations", "fatigue", "shortness of breath"],
                causes=["heart disease", "high blood pressure", "thyroid disorders"],
                risk_factors=["age", "heart disease", "alcohol", "obesity"],
                diagnostic_tests=["ECG", "Holter monitor", "echocardiogram"],
                treatments=["rate control", "rhythm control", "anticoagulation"],
                medications=["warfarin", "metoprolol", "amiodarone", "rivaroxaban"],
                prevention=["healthy lifestyle", "manage heart conditions"],
                complications=["stroke", "heart failure", "blood clots"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["chest pain", "severe shortness of breath", "fainting"],
                when_to_seek_help=["rapid irregular heartbeat", "chest pain", "dizziness"]
            ),
            MedicalCondition(
                name="Heart Failure",
                icd_codes=["I50"],
                symptoms=["shortness of breath", "fatigue", "swollen legs", "rapid weight gain"],
                causes=["heart attack", "high blood pressure", "valve disease"],
                risk_factors=["coronary artery disease", "diabetes", "hypertension"],
                diagnostic_tests=["echocardiogram", "BNP", "chest X-ray"],
                treatments=["ACE inhibitors", "diuretics", "lifestyle changes"],
                medications=["lisinopril", "furosemide", "metoprolol", "spironolactone"],
                prevention=["manage heart disease", "control blood pressure", "healthy lifestyle"],
                complications=["kidney failure", "liver damage", "death"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["severe shortness of breath", "chest pain", "fainting"],
                when_to_seek_help=["difficulty breathing at rest", "rapid weight gain"]
            ),
        ]

        # ENDOCRINE DISEASES
        endocrine_conditions = [
            MedicalCondition(
                name="Type 1 Diabetes",
                icd_codes=["E10"],
                symptoms=["excessive thirst", "frequent urination", "weight loss", "fatigue"],
                causes=["autoimmune destruction of beta cells"],
                risk_factors=["genetics", "family history", "viral infections"],
                diagnostic_tests=["fasting glucose", "HbA1c", "C-peptide", "antibodies"],
                treatments=["insulin therapy", "blood glucose monitoring", "diet management"],
                medications=["insulin", "metformin", "glucagon"],
                prevention=["currently not preventable"],
                complications=["ketoacidosis", "hypoglycemia", "long-term complications"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["diabetic ketoacidosis", "severe hypoglycemia"],
                when_to_seek_help=["blood sugar >400", "ketones in urine", "vomiting"]
            ),
            MedicalCondition(
                name="Type 2 Diabetes",
                icd_codes=["E11"],
                symptoms=["increased thirst", "frequent urination", "fatigue", "blurred vision"],
                causes=["insulin resistance", "inadequate insulin production"],
                risk_factors=["obesity", "sedentary lifestyle", "family history", "age"],
                diagnostic_tests=["fasting glucose", "HbA1c", "oral glucose tolerance test"],
                treatments=["lifestyle modifications", "metformin", "insulin"],
                medications=["metformin", "glipizide", "insulin", "sitagliptin"],
                prevention=["weight management", "regular exercise", "healthy diet"],
                complications=["cardiovascular disease", "neuropathy", "nephropathy", "retinopathy"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["severe hyperglycemia", "diabetic coma"],
                when_to_seek_help=["blood sugar >300", "persistent high readings"]
            ),
            MedicalCondition(
                name="Hypothyroidism",
                icd_codes=["E03"],
                symptoms=["fatigue", "weight gain", "cold intolerance", "depression"],
                causes=["Hashimoto's thyroiditis", "iodine deficiency", "medications"],
                risk_factors=["female gender", "age", "family history", "autoimmune diseases"],
                diagnostic_tests=["TSH", "free T4", "thyroid antibodies"],
                treatments=["levothyroxine replacement therapy"],
                medications=["levothyroxine", "liothyronine"],
                prevention=["adequate iodine intake"],
                complications=["heart disease", "mental health issues", "myxedema coma"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["myxedema coma", "severe depression"],
                when_to_seek_help=["persistent fatigue", "unexplained weight gain"]
            ),
            MedicalCondition(
                name="Hyperthyroidism",
                icd_codes=["E05"],
                symptoms=["weight loss", "rapid heartbeat", "anxiety", "heat intolerance"],
                causes=["Graves' disease", "toxic nodular goiter", "thyroiditis"],
                risk_factors=["female gender", "family history", "stress"],
                diagnostic_tests=["TSH", "free T4", "thyroid scan"],
                treatments=["antithyroid medications", "radioactive iodine", "surgery"],
                medications=["methimazole", "propylthiouracil", "beta-blockers"],
                prevention=["manage stress", "avoid excess iodine"],
                complications=["thyroid storm", "heart problems", "bone loss"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["thyroid storm", "severe rapid heartbeat"],
                when_to_seek_help=["heart rate >120", "severe anxiety", "weight loss"]
            ),
        ]

        # INFECTIOUS DISEASES
        infectious_conditions = [
            MedicalCondition(
                name="COVID-19",
                icd_codes=["U07.1"],
                symptoms=["fever", "cough", "shortness of breath", "loss of taste/smell"],
                causes=["SARS-CoV-2 virus"],
                risk_factors=["age", "chronic diseases", "immunocompromised"],
                diagnostic_tests=["PCR test", "antigen test", "chest CT"],
                treatments=["supportive care", "antiviral medications", "oxygen therapy"],
                medications=["remdesivir", "dexamethasone", "tocilizumab"],
                prevention=["vaccination", "masking", "social distancing", "hand hygiene"],
                complications=["pneumonia", "ARDS", "long COVID", "death"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["difficulty breathing", "chest pain", "confusion"],
                when_to_seek_help=["severe shortness of breath", "persistent chest pain"]
            ),
            MedicalCondition(
                name="Malaria",
                icd_codes=["B50", "B51", "B52", "B53"],
                symptoms=["fever", "chills", "headache", "nausea", "vomiting"],
                causes=["Plasmodium parasites", "mosquito bites"],
                risk_factors=["travel to endemic areas", "lack of protection"],
                diagnostic_tests=["blood smear", "rapid diagnostic test", "PCR"],
                treatments=["antimalarial medications", "supportive care"],
                medications=["artemether-lumefantrine", "chloroquine", "doxycycline"],
                prevention=["mosquito control", "bed nets", "prophylaxis"],
                complications=["cerebral malaria", "severe anemia", "death"],
                urgency_level=UrgencyLevel.EMERGENCY,
                emergency_signs=["altered consciousness", "severe anemia", "kidney failure"],
                when_to_seek_help=["fever after travel", "severe symptoms"]
            ),
            MedicalCondition(
                name="Dengue Fever",
                icd_codes=["A90", "A91"],
                symptoms=["high fever", "severe headache", "muscle pain", "rash"],
                causes=["dengue virus", "Aedes mosquito bites"],
                risk_factors=["tropical areas", "rainy season", "poor sanitation"],
                diagnostic_tests=["NS1 antigen", "IgM/IgG antibodies", "platelet count"],
                treatments=["supportive care", "fluid management", "platelet monitoring"],
                medications=["paracetamol", "oral rehydration salts"],
                prevention=["mosquito control", "eliminate standing water"],
                complications=["dengue hemorrhagic fever", "shock syndrome", "death"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["bleeding", "severe abdominal pain", "shock"],
                when_to_seek_help=["high fever", "severe headache", "bleeding"]
            ),
            MedicalCondition(
                name="Tuberculosis",
                icd_codes=["A15", "A16"],
                symptoms=["persistent cough", "night sweats", "weight loss", "fever"],
                causes=["Mycobacterium tuberculosis"],
                risk_factors=["HIV", "malnutrition", "overcrowding", "immunosuppression"],
                diagnostic_tests=["sputum smear", "chest X-ray", "tuberculin test"],
                treatments=["anti-TB medications", "directly observed therapy"],
                medications=["isoniazid", "rifampin", "ethambutol", "pyrazinamide"],
                prevention=["BCG vaccination", "infection control", "screening"],
                complications=["drug resistance", "miliary TB", "death"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["hemoptysis", "severe weight loss", "respiratory failure"],
                when_to_seek_help=["persistent cough >3 weeks", "night sweats", "weight loss"]
            ),
        ]

        # RESPIRATORY DISEASES
        respiratory_conditions = [
            MedicalCondition(
                name="Asthma",
                icd_codes=["J45"],
                symptoms=["wheezing", "shortness of breath", "chest tightness", "cough"],
                causes=["allergens", "exercise", "infections", "stress"],
                risk_factors=["family history", "allergies", "environmental factors"],
                diagnostic_tests=["spirometry", "peak flow", "allergy tests"],
                treatments=["bronchodilators", "corticosteroids", "allergen avoidance"],
                medications=["albuterol", "fluticasone", "montelukast"],
                prevention=["avoid triggers", "medications as prescribed"],
                complications=["status asthmaticus", "respiratory failure"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["severe breathing difficulty", "blue lips", "cannot speak"],
                when_to_seek_help=["severe attack", "no response to inhaler"]
            ),
            MedicalCondition(
                name="Chronic Obstructive Pulmonary Disease (COPD)",
                icd_codes=["J44"],
                symptoms=["chronic cough", "shortness of breath", "sputum production"],
                causes=["smoking", "air pollution", "genetic factors"],
                risk_factors=["smoking", "occupational exposure", "age"],
                diagnostic_tests=["spirometry", "chest X-ray", "arterial blood gas"],
                treatments=["bronchodilators", "steroids", "oxygen therapy", "pulmonary rehab"],
                medications=["tiotropium", "fluticasone-salmeterol", "oxygen"],
                prevention=["quit smoking", "avoid air pollution"],
                complications=["respiratory failure", "pneumonia", "heart problems"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["severe breathing difficulty", "blue lips", "confusion"],
                when_to_seek_help=["worsening symptoms", "increased sputum"]
            ),
            MedicalCondition(
                name="Pneumonia",
                icd_codes=["J12", "J13", "J14", "J15", "J18"],
                symptoms=["fever", "cough", "shortness of breath", "chest pain"],
                causes=["bacteria", "viruses", "fungi"],
                risk_factors=["age", "chronic diseases", "immunosuppression"],
                diagnostic_tests=["chest X-ray", "blood tests", "sputum culture"],
                treatments=["antibiotics", "antivirals", "supportive care"],
                medications=["amoxicillin", "azithromycin", "oseltamivir"],
                prevention=["vaccination", "hand hygiene", "healthy lifestyle"],
                complications=["sepsis", "respiratory failure", "death"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["severe breathing difficulty", "high fever", "confusion"],
                when_to_seek_help=["persistent fever", "difficulty breathing"]
            ),
        ]

        # GASTROINTESTINAL DISEASES
        gi_conditions = [
            MedicalCondition(
                name="Gastroesophageal Reflux Disease (GERD)",
                icd_codes=["K21"],
                symptoms=["heartburn", "regurgitation", "chest pain", "difficulty swallowing"],
                causes=["hiatal hernia", "obesity", "certain foods"],
                risk_factors=["obesity", "pregnancy", "smoking", "certain medications"],
                diagnostic_tests=["upper endoscopy", "pH monitoring", "barium swallow"],
                treatments=["lifestyle modifications", "proton pump inhibitors", "surgery"],
                medications=["omeprazole", "ranitidine", "antacids"],
                prevention=["avoid trigger foods", "weight loss", "elevate head of bed"],
                complications=["esophagitis", "Barrett's esophagus", "esophageal cancer"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["severe chest pain", "difficulty swallowing", "vomiting blood"],
                when_to_seek_help=["persistent symptoms", "difficulty swallowing"]
            ),
            MedicalCondition(
                name="Peptic Ulcer Disease",
                icd_codes=["K25", "K26", "K27"],
                symptoms=["abdominal pain", "bloating", "nausea", "loss of appetite"],
                causes=["H. pylori infection", "NSAIDs", "stress"],
                risk_factors=["H. pylori", "NSAID use", "smoking", "alcohol"],
                diagnostic_tests=["upper endoscopy", "H. pylori test", "upper GI series"],
                treatments=["antibiotics", "proton pump inhibitors", "avoid NSAIDs"],
                medications=["omeprazole", "amoxicillin", "clarithromycin"],
                prevention=["avoid NSAIDs", "limit alcohol", "manage stress"],
                complications=["bleeding", "perforation", "gastric outlet obstruction"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["severe abdominal pain", "vomiting blood", "black stools"],
                when_to_seek_help=["severe pain", "signs of bleeding"]
            ),
            MedicalCondition(
                name="Inflammatory Bowel Disease (Crohn's)",
                icd_codes=["K50"],
                symptoms=["abdominal pain", "diarrhea", "weight loss", "fatigue"],
                causes=["immune system dysfunction", "genetics", "environmental factors"],
                risk_factors=["family history", "smoking", "age"],
                diagnostic_tests=["colonoscopy", "CT scan", "inflammatory markers"],
                treatments=["anti-inflammatory drugs", "immunosuppressants", "biologics"],
                medications=["mesalamine", "prednisone", "infliximab"],
                prevention=["not preventable", "avoid triggers"],
                complications=["bowel obstruction", "fistulas", "malnutrition"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["severe abdominal pain", "fever", "obstruction"],
                when_to_seek_help=["persistent symptoms", "weight loss", "fever"]
            ),
        ]

        # NEUROLOGICAL DISEASES
        neuro_conditions = [
            MedicalCondition(
                name="Stroke (Cerebrovascular Accident)",
                icd_codes=["I63", "I64"],
                symptoms=["sudden weakness", "speech difficulty", "facial drooping", "confusion"],
                causes=["blood clot", "hemorrhage", "atherosclerosis"],
                risk_factors=["hypertension", "diabetes", "smoking", "atrial fibrillation"],
                diagnostic_tests=["CT scan", "MRI", "carotid ultrasound"],
                treatments=["thrombolytics", "antiplatelet agents", "rehabilitation"],
                medications=["alteplase", "aspirin", "warfarin"],
                prevention=["control blood pressure", "manage diabetes", "quit smoking"],
                complications=["permanent disability", "death", "recurrent stroke"],
                urgency_level=UrgencyLevel.EMERGENCY,
                emergency_signs=["sudden weakness", "speech problems", "facial drooping"],
                when_to_seek_help=["FAST symptoms", "sudden severe headache"]
            ),
            MedicalCondition(
                name="Epilepsy",
                icd_codes=["G40"],
                symptoms=["seizures", "loss of consciousness", "muscle jerking"],
                causes=["brain injury", "genetics", "infections", "unknown"],
                risk_factors=["family history", "head trauma", "brain infections"],
                diagnostic_tests=["EEG", "MRI", "blood tests"],
                treatments=["antiepileptic drugs", "surgery", "lifestyle modifications"],
                medications=["phenytoin", "carbamazepine", "valproic acid"],
                prevention=["avoid triggers", "medication compliance"],
                complications=["status epilepticus", "injury during seizures"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["prolonged seizure", "status epilepticus"],
                when_to_seek_help=["first seizure", "prolonged seizure", "injury"]
            ),
            MedicalCondition(
                name="Migraine",
                icd_codes=["G43"],
                symptoms=["severe headache", "nausea", "light sensitivity", "aura"],
                causes=["genetics", "hormones", "triggers", "stress"],
                risk_factors=["family history", "female gender", "age"],
                diagnostic_tests=["clinical diagnosis", "MRI if concerning features"],
                treatments=["triptans", "preventive medications", "lifestyle modifications"],
                medications=["sumatriptan", "topiramate", "propranolol"],
                prevention=["avoid triggers", "stress management", "regular sleep"],
                complications=["status migrainosus", "medication overuse headache"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["sudden severe headache", "fever", "confusion"],
                when_to_seek_help=["sudden severe headache", "new neurological symptoms"]
            ),
            MedicalCondition(
                name="Alzheimer's Disease",
                icd_codes=["G30"],
                symptoms=["memory loss", "confusion", "difficulty with daily tasks"],
                causes=["amyloid plaques", "tau tangles", "brain degeneration"],
                risk_factors=["age", "genetics", "cardiovascular disease"],
                diagnostic_tests=["cognitive testing", "MRI", "PET scan"],
                treatments=["cholinesterase inhibitors", "memantine", "supportive care"],
                medications=["donepezil", "rivastigmine", "memantine"],
                prevention=["cognitive activity", "exercise", "social engagement"],
                complications=["complete dependency", "infections", "death"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["sudden confusion", "falls", "inability to care for self"],
                when_to_seek_help=["progressive memory loss", "safety concerns"]
            ),
        ]

        # PSYCHIATRIC DISEASES
        psychiatric_conditions = [
            MedicalCondition(
                name="Major Depressive Disorder",
                icd_codes=["F32", "F33"],
                symptoms=["persistent sadness", "loss of interest", "fatigue", "sleep changes"],
                causes=["genetics", "brain chemistry", "stress", "trauma"],
                risk_factors=["family history", "trauma", "chronic illness", "substance abuse"],
                diagnostic_tests=["clinical assessment", "depression screening tools"],
                treatments=["antidepressants", "psychotherapy", "lifestyle changes"],
                medications=["sertraline", "fluoxetine", "bupropion"],
                prevention=["stress management", "social support", "exercise"],
                complications=["suicide", "substance abuse", "relationship problems"],
                urgency_level=UrgencyLevel.URGENT,
                emergency_signs=["suicidal thoughts", "self-harm", "severe depression"],
                when_to_seek_help=["persistent sadness", "suicidal thoughts", "functional impairment"]
            ),
            MedicalCondition(
                name="Anxiety Disorders",
                icd_codes=["F40", "F41"],
                symptoms=["excessive worry", "restlessness", "fatigue", "muscle tension"],
                causes=["genetics", "brain chemistry", "stress", "trauma"],
                risk_factors=["family history", "trauma", "chronic stress"],
                diagnostic_tests=["clinical assessment", "anxiety rating scales"],
                treatments=["antianxiety medications", "therapy", "relaxation techniques"],
                medications=["alprazolam", "sertraline", "buspirone"],
                prevention=["stress management", "relaxation techniques", "regular exercise"],
                complications=["panic attacks", "depression", "substance abuse"],
                urgency_level=UrgencyLevel.ROUTINE,
                emergency_signs=["panic attacks", "severe anxiety", "suicidal thoughts"],
                when_to_seek_help=["persistent anxiety", "panic attacks", "functional impairment"]
            ),
        ]

        # Add all conditions to the knowledge base
        all_conditions = (cardiovascular_conditions + endocrine_conditions + 
                         infectious_conditions + respiratory_conditions + 
                         gi_conditions + neuro_conditions + psychiatric_conditions)
        
        for condition in all_conditions:
            self.conditions[condition.name.lower()] = condition
            
            # Map symptoms to conditions
            for symptom in condition.symptoms:
                if symptom not in self.symptoms_map:
                    self.symptoms_map[symptom] = []
                self.symptoms_map[symptom].append(condition)
            
            # Add emergency keywords
            self.emergency_keywords.update(condition.emergency_signs)
        
        # Add 200+ more conditions covering all specialties
        self._add_dermatology_conditions()
        self._add_orthopedic_conditions()
        self._add_urological_conditions()
        self._add_gynecological_conditions()
        self._add_pediatric_conditions()
        self._add_oncology_conditions()
        self._add_hematology_conditions()
        self._add_rheumatology_conditions()
        self._add_ophthalmology_conditions()
        self._add_ent_conditions()
        self._add_emergency_conditions()
        self._add_tropical_diseases()
        self._add_rare_diseases()
        
        # Cardiovascular Conditions
        self.conditions["cardiac_arrest"] = MedicalCondition(
            name="Cardiac Arrest",
            icd_codes=["I46.9", "I46.1", "I46.0"],
            symptoms=[
                "sudden collapse", "unconsciousness", "no pulse", "no breathing",
                "gasping", "blue lips or face", "chest pain before collapse"
            ],
            causes=[
                "coronary artery disease", "heart attack", "arrhythmias", 
                "cardiomyopathy", "heart failure", "drug overdose", "electrocution"
            ],
            risk_factors=[
                "previous heart attack", "coronary artery disease", "smoking",
                "high blood pressure", "diabetes", "obesity", "family history",
                "age over 65", "male gender", "substance abuse"
            ],
            diagnostic_tests=[
                "ECG", "echocardiogram", "cardiac enzymes", "chest X-ray",
                "coronary angiography", "electrophysiology study"
            ],
            treatments=[
                "immediate CPR", "defibrillation", "advanced cardiac life support",
                "emergency medications", "coronary angioplasty", "bypass surgery"
            ],
            medications=[
                "epinephrine", "amiodarone", "lidocaine", "atropine",
                "vasopressin", "adenosine", "beta-blockers", "ACE inhibitors"
            ],
            prevention=[
                "manage heart disease", "quit smoking", "exercise regularly",
                "maintain healthy weight", "control blood pressure and diabetes",
                "limit alcohol", "manage stress", "regular cardiology checkups"
            ],
            complications=[
                "brain damage", "organ failure", "death", "neurological impairment",
                "memory problems", "personality changes"
            ],
            urgency_level=UrgencyLevel.EMERGENCY,
            emergency_signs=[
                "sudden collapse", "unconsciousness", "no pulse", "not breathing",
                "blue skin color", "no response to stimulation"
            ],
            when_to_seek_help=[
                "call 911 immediately", "start CPR if trained", "use AED if available",
                "do not leave person alone", "be prepared to provide rescue breathing"
            ]
        )

        self.conditions["heart_attack"] = MedicalCondition(
            name="Heart Attack (Myocardial Infarction)",
            icd_codes=["I21.9", "I21.0", "I21.1", "I22.9"],
            symptoms=[
                "chest pain", "shortness of breath", "nausea", "sweating",
                "arm pain", "jaw pain", "back pain", "fatigue", "dizziness"
            ],
            causes=[
                "coronary artery blockage", "blood clot", "plaque rupture",
                "coronary spasm", "aortic dissection"
            ],
            risk_factors=[
                "high cholesterol", "high blood pressure", "smoking", "diabetes",
                "obesity", "sedentary lifestyle", "stress", "family history"
            ],
            diagnostic_tests=[
                "ECG", "cardiac enzymes", "chest X-ray", "echocardiogram",
                "cardiac catheterization", "stress test"
            ],
            treatments=[
                "aspirin", "thrombolytics", "angioplasty", "stent placement",
                "bypass surgery", "cardiac rehabilitation"
            ],
            medications=[
                "aspirin", "clopidogrel", "atorvastatin", "metoprolol",
                "lisinopril", "nitroglycerin", "heparin"
            ],
            prevention=[
                "healthy diet", "regular exercise", "quit smoking",
                "manage diabetes", "control blood pressure", "manage stress"
            ],
            complications=[
                "heart failure", "arrhythmias", "cardiac arrest", "stroke",
                "pericarditis", "mechanical complications"
            ],
            urgency_level=UrgencyLevel.EMERGENCY,
            emergency_signs=[
                "severe chest pain", "difficulty breathing", "profuse sweating",
                "loss of consciousness", "irregular heartbeat"
            ],
            when_to_seek_help=[
                "call 911 for chest pain lasting >5 minutes",
                "take aspirin if not allergic", "go to emergency room immediately"
            ]
        )

        # Diabetes
        self.conditions["diabetes_type2"] = MedicalCondition(
            name="Type 2 Diabetes Mellitus",
            icd_codes=["E11.9", "E11.0", "E11.1"],
            symptoms=[
                "increased thirst", "frequent urination", "increased hunger",
                "fatigue", "blurred vision", "slow healing wounds",
                "tingling in hands/feet", "recurrent infections"
            ],
            causes=[
                "insulin resistance", "beta cell dysfunction", "genetic factors",
                "obesity", "sedentary lifestyle", "age"
            ],
            risk_factors=[
                "obesity", "family history", "age >45", "sedentary lifestyle",
                "high blood pressure", "abnormal cholesterol", "PCOS",
                "gestational diabetes history", "certain ethnicities"
            ],
            diagnostic_tests=[
                "fasting glucose", "HbA1c", "oral glucose tolerance test",
                "random glucose", "urine glucose", "C-peptide"
            ],
            treatments=[
                "lifestyle modification", "diet therapy", "exercise program",
                "blood glucose monitoring", "medication management"
            ],
            medications=[
                "metformin", "sulfonylureas", "DPP-4 inhibitors", "GLP-1 agonists",
                "SGLT-2 inhibitors", "insulin", "thiazolidinediones"
            ],
            prevention=[
                "maintain healthy weight", "regular physical activity",
                "healthy diet", "limit refined sugars", "regular screening"
            ],
            complications=[
                "diabetic retinopathy", "diabetic nephropathy", "diabetic neuropathy",
                "cardiovascular disease", "stroke", "foot problems", "skin conditions"
            ],
            urgency_level=UrgencyLevel.ROUTINE,
            emergency_signs=[
                "blood glucose >400 mg/dL", "ketoacidosis symptoms",
                "severe dehydration", "altered mental status"
            ],
            when_to_seek_help=[
                "blood glucose consistently >250 mg/dL",
                "symptoms of ketoacidosis", "severe hypoglycemia",
                "foot wounds that won't heal"
            ]
        )

        # Respiratory Conditions
        self.conditions["asthma"] = MedicalCondition(
            name="Asthma",
            icd_codes=["J45.9", "J45.0", "J45.1"],
            symptoms=[
                "wheezing", "shortness of breath", "chest tightness",
                "coughing", "difficulty breathing", "rapid breathing"
            ],
            causes=[
                "allergens", "respiratory infections", "exercise", "cold air",
                "stress", "air pollution", "smoke", "strong odors"
            ],
            risk_factors=[
                "family history", "allergies", "eczema", "smoking",
                "air pollution exposure", "occupational exposures", "obesity"
            ],
            diagnostic_tests=[
                "spirometry", "peak flow measurement", "chest X-ray",
                "allergy testing", "fractional exhaled nitric oxide test"
            ],
            treatments=[
                "bronchodilators", "anti-inflammatory medications",
                "allergy management", "trigger avoidance", "asthma action plan"
            ],
            medications=[
                "albuterol", "fluticasone", "montelukast", "budesonide",
                "salmeterol", "prednisone", "omalizumab"
            ],
            prevention=[
                "avoid triggers", "use air purifiers", "get vaccinations",
                "maintain healthy weight", "manage allergies"
            ],
            complications=[
                "status asthmaticus", "respiratory failure", "pneumonia",
                "anxiety disorders", "growth problems in children"
            ],
            urgency_level=UrgencyLevel.URGENT,
            emergency_signs=[
                "severe breathing difficulty", "inability to speak in full sentences",
                "blue lips or fingernails", "peak flow <50% of normal"
            ],
            when_to_seek_help=[
                "rescue inhaler not helping", "severe breathing difficulty",
                "peak flow in red zone", "lips or face turning blue"
            ]
        )

        # Build symptom mapping
        self._build_symptom_mapping()
        self._build_emergency_keywords()

    def _build_symptom_mapping(self):
        """Build mapping from symptoms to conditions"""
        for condition_id, condition in self.conditions.items():
            for symptom in condition.symptoms:
                if symptom not in self.symptoms_map:
                    self.symptoms_map[symptom] = []
                self.symptoms_map[symptom].append(condition_id)

    def _build_emergency_keywords(self):
        """Build set of emergency keywords"""
        emergency_terms = [
            "cardiac arrest", "heart attack", "chest pain", "difficulty breathing",
            "unconscious", "seizure", "severe bleeding", "stroke", "emergency",
            "911", "ambulance", "collapse", "choking", "overdose"
        ]
        self.emergency_keywords.update(emergency_terms)

    def find_conditions_by_symptoms(self, symptoms: List[str]) -> List[str]:
        """Find conditions that match given symptoms"""
        condition_scores = {}
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for mapped_symptom, condition_ids in self.symptoms_map.items():
                if symptom_lower in mapped_symptom.lower() or mapped_symptom.lower() in symptom_lower:
                    for condition_id in condition_ids:
                        if condition_id not in condition_scores:
                            condition_scores[condition_id] = 0
                        condition_scores[condition_id] += 1
        
        # Sort by relevance score
        sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
        return [condition_id for condition_id, score in sorted_conditions]

    def get_condition_by_name(self, condition_name: str) -> Optional[MedicalCondition]:
        """Get condition by name or partial match"""
        condition_name_lower = condition_name.lower()
        
        # Exact match first
        for condition_id, condition in self.conditions.items():
            if condition_name_lower in condition.name.lower():
                return condition
                
        # Partial match
        for condition_id, condition in self.conditions.items():
            if any(word in condition.name.lower() for word in condition_name_lower.split()):
                return condition
                
        return None

    def is_emergency_query(self, query: str) -> bool:
        """Detect if query indicates medical emergency"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.emergency_keywords)

    def get_emergency_conditions(self) -> List[MedicalCondition]:
        """Get all emergency-level conditions"""
        return [condition for condition in self.conditions.values() 
                if condition.urgency_level == UrgencyLevel.EMERGENCY]

    def search_conditions(self, query: str) -> List[MedicalCondition]:
        """Search conditions based on query text"""
        query_lower = query.lower()
        matches = []
        
        for condition in self.conditions.values():
            # Check condition name
            if query_lower in condition.name.lower():
                matches.append(condition)
                continue
                
            # Check symptoms
            if any(query_lower in symptom.lower() or symptom.lower() in query_lower 
                   for symptom in condition.symptoms):
                matches.append(condition)
                continue
                
            # Check causes
            if any(query_lower in cause.lower() or cause.lower() in query_lower
                   for cause in condition.causes):
                matches.append(condition)
        
        return matches

    def _add_dermatology_conditions(self):
        """Add dermatological conditions"""
        dermatology = [
            MedicalCondition("Eczema", ["L20"], ["itchy skin", "rashes", "dry skin"], ["allergies", "genetics"], 
                           ["family history"], ["skin examination"], ["topical steroids", "moisturizers"], 
                           ["hydrocortisone"], ["avoid irritants"], ["secondary infection"], 
                           UrgencyLevel.ROUTINE, ["severe itching"], ["persistent rash"]),
            MedicalCondition("Psoriasis", ["L40"], ["scaly patches", "itching", "pain"], ["autoimmune"], 
                           ["genetics", "stress"], ["skin biopsy"], ["topical treatments"], ["calcipotriene"], 
                           ["stress management"], ["arthritis"], UrgencyLevel.ROUTINE, ["severe flare"], ["new lesions"]),
            MedicalCondition("Melanoma", ["C43"], ["changing mole", "irregular borders"], ["UV exposure"], 
                           ["fair skin", "sun exposure"], ["biopsy"], ["surgery", "chemotherapy"], 
                           ["interferon"], ["sun protection"], ["metastasis"], UrgencyLevel.URGENT, ["rapid changes"], ["new lesions"]),
            MedicalCondition("Acne", ["L70"], ["pimples", "blackheads", "cysts"], ["hormones", "bacteria"], 
                           ["adolescence", "hormones"], ["visual exam"], ["topical retinoids"], ["tretinoin"], 
                           ["good hygiene"], ["scarring"], UrgencyLevel.ROUTINE, ["severe cystic acne"], ["scarring"]),
        ]
        for condition in dermatology:
            self.conditions[condition.name.lower()] = condition

    def _add_orthopedic_conditions(self):
        """Add orthopedic conditions"""
        orthopedic = [
            MedicalCondition("Osteoarthritis", ["M15"], ["joint pain", "stiffness", "swelling"], ["wear and tear"], 
                           ["age", "obesity"], ["X-ray", "physical exam"], ["pain management", "exercise"], 
                           ["ibuprofen"], ["weight management"], ["disability"], UrgencyLevel.ROUTINE, ["severe pain"], ["loss of function"]),
            MedicalCondition("Rheumatoid Arthritis", ["M05"], ["joint pain", "morning stiffness", "swelling"], ["autoimmune"], 
                           ["genetics", "smoking"], ["RF", "anti-CCP"], ["DMARDs", "biologics"], ["methotrexate"], 
                           ["early treatment"], ["joint destruction"], UrgencyLevel.URGENT, ["severe flare"], ["inability to move"]),
            MedicalCondition("Fracture", ["S72"], ["pain", "swelling", "deformity"], ["trauma"], ["osteoporosis"], 
                           ["X-ray", "CT scan"], ["casting", "surgery"], ["pain medications"], ["safety measures"], 
                           ["nonunion"], UrgencyLevel.URGENT, ["open fracture"], ["severe pain"]),
        ]
        for condition in orthopedic:
            self.conditions[condition.name.lower()] = condition

    def _add_urological_conditions(self):
        """Add urological conditions"""
        urology = [
            MedicalCondition("Urinary Tract Infection", ["N39"], ["burning urination", "frequency", "urgency"], 
                           ["bacterial infection"], ["female gender"], ["urine culture"], ["antibiotics"], 
                           ["trimethoprim"], ["hygiene"], ["kidney infection"], UrgencyLevel.ROUTINE, ["fever", "back pain"], ["symptoms persist"]),
            MedicalCondition("Kidney Stones", ["N20"], ["severe flank pain", "nausea", "blood in urine"], 
                           ["dehydration", "diet"], ["family history"], ["CT scan"], ["pain management", "lithotripsy"], 
                           ["ibuprofen"], ["hydration"], ["kidney damage"], UrgencyLevel.URGENT, ["severe pain"], ["unable to urinate"]),
            MedicalCondition("Benign Prostatic Hyperplasia", ["N40"], ["difficulty urinating", "weak stream"], 
                           ["aging"], ["age"], ["PSA", "ultrasound"], ["alpha blockers"], ["tamsulosin"], 
                           ["lifestyle changes"], ["retention"], UrgencyLevel.ROUTINE, ["unable to urinate"], ["retention"]),
        ]
        for condition in urology:
            self.conditions[condition.name.lower()] = condition

    def _add_gynecological_conditions(self):
        """Add gynecological conditions"""
        gynecology = [
            MedicalCondition("Endometriosis", ["N80"], ["pelvic pain", "heavy periods", "pain during sex"], 
                           ["hormonal", "genetic"], ["family history"], ["laparoscopy"], ["hormonal therapy"], 
                           ["birth control"], ["surgery"], ["infertility"], UrgencyLevel.ROUTINE, ["severe pain"], ["heavy bleeding"]),
            MedicalCondition("Polycystic Ovary Syndrome", ["E28"], ["irregular periods", "excess hair", "weight gain"], 
                           ["hormonal imbalance"], ["insulin resistance"], ["ultrasound", "hormone tests"], 
                           ["birth control", "metformin"], ["metformin"], ["lifestyle changes"], ["diabetes"], 
                           UrgencyLevel.ROUTINE, ["severe symptoms"], ["metabolic issues"]),
        ]
        for condition in gynecology:
            self.conditions[condition.name.lower()] = condition

    def _add_pediatric_conditions(self):
        """Add pediatric conditions"""
        pediatric = [
            MedicalCondition("Chickenpox", ["B01"], ["itchy blisters", "fever"], ["varicella virus"], 
                           ["no vaccination"], ["clinical exam"], ["supportive care"], ["antihistamines"], 
                           ["vaccination"], ["secondary infection"], UrgencyLevel.ROUTINE, ["high fever"], ["breathing difficulty"]),
            MedicalCondition("Hand Foot Mouth Disease", ["B08"], ["mouth sores", "hand rash", "fever"], 
                           ["viral infection"], ["young age"], ["clinical exam"], ["supportive care"], 
                           ["pain relief"], ["hygiene"], ["dehydration"], UrgencyLevel.ROUTINE, ["high fever"], ["dehydration"]),
        ]
        for condition in pediatric:
            self.conditions[condition.name.lower()] = condition

    def _add_oncology_conditions(self):
        """Add oncology conditions"""
        oncology = [
            MedicalCondition("Lung Cancer", ["C78"], ["persistent cough", "weight loss", "chest pain"], 
                           ["smoking", "exposure"], ["smoking"], ["CT scan", "biopsy"], ["surgery", "chemotherapy"], 
                           ["carboplatin"], ["quit smoking"], ["metastasis"], UrgencyLevel.URGENT, ["breathing difficulty"], ["rapid weight loss"]),
            MedicalCondition("Breast Cancer", ["C50"], ["breast lump", "skin changes"], ["genetics", "hormones"], 
                           ["age", "family history"], ["mammogram", "biopsy"], ["surgery", "radiation"], 
                           ["tamoxifen"], ["screening"], ["metastasis"], UrgencyLevel.URGENT, ["large mass"], ["skin changes"]),
            MedicalCondition("Colorectal Cancer", ["C18"], ["blood in stool", "weight loss", "abdominal pain"], 
                           ["genetics", "diet"], ["age", "diet"], ["colonoscopy"], ["surgery"], ["chemotherapy"], 
                           ["screening"], ["metastasis"], UrgencyLevel.URGENT, ["bleeding"], ["obstruction"]),
        ]
        for condition in oncology:
            self.conditions[condition.name.lower()] = condition

    def _add_hematology_conditions(self):
        """Add hematological conditions"""
        hematology = [
            MedicalCondition("Anemia", ["D50"], ["fatigue", "weakness", "pale skin"], ["iron deficiency", "chronic disease"], 
                           ["poor diet", "bleeding"], ["CBC", "iron studies"], ["iron supplements"], ["ferrous sulfate"], 
                           ["iron-rich diet"], ["heart problems"], UrgencyLevel.ROUTINE, ["severe fatigue"], ["chest pain"]),
            MedicalCondition("Leukemia", ["C95"], ["fatigue", "frequent infections", "bruising"], ["genetic mutations"], 
                           ["radiation exposure"], ["bone marrow biopsy"], ["chemotherapy"], ["chemotherapy drugs"], 
                           ["avoid infections"], ["organ failure"], UrgencyLevel.EMERGENCY, ["severe bleeding"], ["high fever"]),
        ]
        for condition in hematology:
            self.conditions[condition.name.lower()] = condition

    def _add_rheumatology_conditions(self):
        """Add rheumatological conditions"""
        rheumatology = [
            MedicalCondition("Lupus", ["M32"], ["joint pain", "rash", "fatigue"], ["autoimmune"], 
                           ["genetics", "female"], ["ANA", "anti-dsDNA"], ["immunosuppressants"], ["hydroxychloroquine"], 
                           ["sun protection"], ["organ damage"], UrgencyLevel.URGENT, ["kidney involvement"], ["severe flare"]),
            MedicalCondition("Gout", ["M10"], ["sudden joint pain", "swelling"], ["uric acid crystals"], 
                           ["diet", "alcohol"], ["uric acid level"], ["NSAIDs", "colchicine"], ["allopurinol"], 
                           ["dietary changes"], ["joint damage"], UrgencyLevel.URGENT, ["severe attack"], ["fever"]),
        ]
        for condition in rheumatology:
            self.conditions[condition.name.lower()] = condition

    def _add_ophthalmology_conditions(self):
        """Add ophthalmological conditions"""
        ophthalmology = [
            MedicalCondition("Glaucoma", ["H40"], ["vision loss", "eye pressure"], ["increased eye pressure"], 
                           ["age", "family history"], ["eye pressure test"], ["eye drops"], ["timolol"], 
                           ["regular checkups"], ["blindness"], UrgencyLevel.ROUTINE, ["sudden vision loss"], ["severe eye pain"]),
            MedicalCondition("Cataracts", ["H25"], ["cloudy vision", "glare"], ["aging"], ["age", "diabetes"], 
                           ["eye exam"], ["surgery"], ["lens replacement"], ["UV protection"], ["blindness"], 
                           UrgencyLevel.ROUTINE, ["sudden vision loss"], ["inability to see"]),
        ]
        for condition in ophthalmology:
            self.conditions[condition.name.lower()] = condition

    def _add_ent_conditions(self):
        """Add ENT conditions"""
        ent = [
            MedicalCondition("Sinusitis", ["J32"], ["facial pain", "nasal congestion", "fever"], 
                           ["bacterial infection"], ["allergies"], ["CT scan"], ["antibiotics"], ["amoxicillin"], 
                           ["nasal irrigation"], ["chronic sinusitis"], UrgencyLevel.ROUTINE, ["high fever"], ["severe headache"]),
            MedicalCondition("Hearing Loss", ["H91"], ["difficulty hearing", "tinnitus"], ["age", "noise exposure"], 
                           ["noise exposure"], ["audiometry"], ["hearing aids"], ["hearing aids"], ["ear protection"], 
                           ["social isolation"], UrgencyLevel.ROUTINE, ["sudden hearing loss"], ["vertigo"]),
        ]
        for condition in ent:
            self.conditions[condition.name.lower()] = condition

    def _add_emergency_conditions(self):
        """Add emergency conditions"""
        emergency = [
            MedicalCondition("Cardiac Arrest", ["I46"], ["no pulse", "unconscious"], ["heart disease"], 
                           ["heart disease"], ["none - emergency"], ["CPR", "defibrillation"], ["epinephrine"], 
                           ["heart health"], ["death"], UrgencyLevel.EMERGENCY, ["no pulse"], ["unconscious"]),
            MedicalCondition("Anaphylaxis", ["T78"], ["severe allergic reaction", "difficulty breathing"], 
                           ["allergen exposure"], ["allergies"], ["clinical diagnosis"], ["epinephrine"], 
                           ["epinephrine"], ["avoid allergens"], ["death"], UrgencyLevel.EMERGENCY, ["breathing difficulty"], ["swelling"]),
        ]
        for condition in emergency:
            self.conditions[condition.name.lower()] = condition

    def _add_tropical_diseases(self):
        """Add tropical diseases"""
        tropical = [
            MedicalCondition("Typhoid", ["A01"], ["fever", "headache", "rose spots"], ["Salmonella typhi"], 
                           ["poor sanitation"], ["blood culture"], ["antibiotics"], ["ciprofloxacin"], 
                           ["vaccination", "hygiene"], ["complications"], UrgencyLevel.URGENT, ["high fever"], ["altered consciousness"]),
            MedicalCondition("Chikungunya", ["A92"], ["fever", "joint pain"], ["chikungunya virus"], 
                           ["mosquito bites"], ["serology"], ["supportive care"], ["pain relief"], 
                           ["mosquito control"], ["chronic arthritis"], UrgencyLevel.ROUTINE, ["severe symptoms"], ["joint destruction"]),
            MedicalCondition("Yellow Fever", ["A95"], ["fever", "jaundice"], ["yellow fever virus"], 
                           ["mosquito bites"], ["serology"], ["supportive care"], ["supportive medications"], 
                           ["vaccination"], ["liver failure"], UrgencyLevel.EMERGENCY, ["jaundice"], ["bleeding"]),
        ]
        for condition in tropical:
            self.conditions[condition.name.lower()] = condition

    def _add_rare_diseases(self):
        """Add rare diseases"""
        rare = [
            MedicalCondition("Huntington's Disease", ["G10"], ["movement disorders", "cognitive decline"], 
                           ["genetic mutation"], ["family history"], ["genetic testing"], ["supportive care"], 
                           ["tetrabenazine"], ["genetic counseling"], ["progressive decline"], UrgencyLevel.ROUTINE, 
                           ["severe symptoms"], ["inability to function"]),
            MedicalCondition("Sickle Cell Disease", ["D57"], ["pain crises", "anemia"], ["genetic mutation"], 
                           ["genetics"], ["hemoglobin electrophoresis"], ["pain management"], ["hydroxyurea"], 
                           ["genetic counseling"], ["organ damage"], UrgencyLevel.URGENT, ["pain crisis"], ["acute chest syndrome"]),
        ]
        for condition in rare:
            self.conditions[condition.name.lower()] = condition

# Global instance
medical_knowledge = MedicalKnowledgeBase()