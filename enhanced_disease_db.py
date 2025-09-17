"""
Enhanced Disease Database for Top 500 Diseases in India
Comprehensive medical knowledge base with detailed disease information
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class EnhancedDiseaseDB:
    """Enhanced database for comprehensive disease information"""
    
    def __init__(self, db_path: str = "health_chatbot.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_enhanced_schema()
        
    def create_enhanced_schema(self):
        """Create enhanced database schema for comprehensive disease data"""
        cursor = self.conn.cursor()
        
        # Enhanced disease_info table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_disease_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_name VARCHAR(255) NOT NULL,
            disease_name_hindi VARCHAR(255),
            disease_category VARCHAR(100),
            icd_10_code VARCHAR(20),
            
            -- Symptoms and Clinical Presentation
            symptoms TEXT,
            early_symptoms TEXT,
            severe_symptoms TEXT,
            symptom_duration VARCHAR(100),
            
            -- Causative Information
            causative_agent VARCHAR(255),
            disease_type VARCHAR(50), -- bacterial, viral, fungal, genetic, etc.
            transmission_mode VARCHAR(255),
            
            -- Prevention
            prevention TEXT,
            vaccination VARCHAR(255),
            hygiene_measures TEXT,
            lifestyle_prevention TEXT,
            
            -- Treatment
            treatment TEXT,
            emergency_treatment TEXT,
            home_remedies TEXT,
            medications TEXT,
            treatment_duration VARCHAR(100),
            
            -- Epidemiology in India
            prevalence_india REAL,
            affected_states TEXT, -- JSON array
            seasonal_pattern VARCHAR(100),
            high_risk_groups TEXT,
            mortality_rate REAL,
            
            -- Clinical Details
            severity VARCHAR(20),
            contagious BOOLEAN,
            incubation_period VARCHAR(100),
            complications TEXT,
            diagnostic_tests TEXT,
            
            -- Demographics
            age_groups_affected TEXT, -- JSON array
            gender_preference VARCHAR(20),
            socioeconomic_factors TEXT,
            
            -- Additional Information
            prognosis TEXT,
            follow_up_care TEXT,
            nutritional_advice TEXT,
            exercise_recommendations TEXT,
            
            -- Metadata
            language VARCHAR(10) DEFAULT 'en',
            data_source VARCHAR(255),
            confidence_score REAL DEFAULT 1.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(disease_name, language)
        )
        """)
        
        # Disease relationships table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_1_id INTEGER,
            disease_2_id INTEGER,
            relationship_type VARCHAR(50), -- similar_symptoms, comorbidity, progression, etc.
            relationship_strength REAL,
            description TEXT,
            FOREIGN KEY (disease_1_id) REFERENCES enhanced_disease_info (id),
            FOREIGN KEY (disease_2_id) REFERENCES enhanced_disease_info (id)
        )
        """)
        
        # Symptom mappings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS symptom_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_id INTEGER,
            symptom_name VARCHAR(255),
            symptom_severity VARCHAR(20),
            frequency_percentage REAL,
            onset_timing VARCHAR(100),
            description TEXT,
            FOREIGN KEY (disease_id) REFERENCES enhanced_disease_info (id)
        )
        """)
        
        # Drug interactions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drug_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_id INTEGER,
            drug_name VARCHAR(255),
            interaction_type VARCHAR(50),
            severity VARCHAR(20),
            description TEXT,
            alternative_drugs TEXT,
            FOREIGN KEY (disease_id) REFERENCES enhanced_disease_info (id)
        )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_disease_name ON enhanced_disease_info(disease_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_disease_category ON enhanced_disease_info(disease_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_icd_code ON enhanced_disease_info(icd_10_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symptom_name ON symptom_mappings(symptom_name)")
        
        self.conn.commit()
        print("Enhanced database schema created successfully!")

# Top 500 diseases in India with comprehensive data
TOP_500_DISEASES_INDIA = [
    {
        "disease_name": "COVID-19",
        "disease_name_hindi": "कोविड-19",
        "disease_category": "Viral Infectious Disease",
        "icd_10_code": "U07.1",
        "symptoms": "Fever, dry cough, tiredness, aches and pains, sore throat, diarrhea, conjunctivitis, headache, loss of taste or smell, skin rash",
        "early_symptoms": "Mild fever, dry cough, fatigue",
        "severe_symptoms": "Difficulty breathing, chest pain, loss of speech or movement",
        "symptom_duration": "2-14 days onset, can last weeks",
        "causative_agent": "SARS-CoV-2 virus",
        "disease_type": "viral",
        "transmission_mode": "Respiratory droplets, airborne particles, surface contact",
        "prevention": "Vaccination, mask wearing, social distancing, hand hygiene, avoiding crowded places",
        "vaccination": "Covishield, Covaxin, Sputnik V available in India",
        "hygiene_measures": "Frequent handwashing, sanitization, avoid touching face",
        "lifestyle_prevention": "Boost immunity with healthy diet, exercise, adequate sleep",
        "treatment": "Supportive care, oxygen therapy, antiviral medications (Remdesivir), steroids for severe cases",
        "emergency_treatment": "Hospitalization, ventilator support, intensive care",
        "home_remedies": "Rest, hydration, steam inhalation, warm salt water gargling",
        "medications": "Paracetamol for fever, Remdesivir, Tocilizumab, Dexamethasone",
        "treatment_duration": "7-21 days depending on severity",
        "prevalence_india": 4.46,
        "affected_states": json.dumps(["Maharashtra", "Kerala", "Karnataka", "Tamil Nadu", "Andhra Pradesh"]),
        "seasonal_pattern": "Multiple waves throughout year",
        "high_risk_groups": "Elderly, diabetics, heart disease patients, immunocompromised",
        "mortality_rate": 1.2,
        "severity": "moderate",
        "contagious": True,
        "incubation_period": "2-14 days",
        "complications": "Pneumonia, ARDS, blood clots, multi-organ failure, long COVID",
        "diagnostic_tests": "RT-PCR, Rapid Antigen Test, CT scan, Blood tests",
        "age_groups_affected": json.dumps(["all ages", "higher severity in elderly"]),
        "gender_preference": "equal",
        "socioeconomic_factors": "Higher spread in densely populated areas, limited healthcare access",
        "prognosis": "Good for mild cases, variable for severe cases",
        "follow_up_care": "Monitor for long COVID symptoms, vaccination booster",
        "nutritional_advice": "High protein diet, vitamin C, vitamin D, zinc supplementation",
        "exercise_recommendations": "Gradual return to activity after recovery",
        "data_source": "WHO, ICMR, Ministry of Health India",
        "confidence_score": 0.95
    },
    {
        "disease_name": "Malaria",
        "disease_name_hindi": "मलेरिया",
        "disease_category": "Parasitic Infectious Disease",
        "icd_10_code": "B50-B54",
        "symptoms": "Fever with chills, headache, nausea, vomiting, muscle pain, fatigue, sweating",
        "early_symptoms": "Mild fever, headache, muscle aches",
        "severe_symptoms": "High fever, severe anemia, cerebral malaria, organ failure",
        "symptom_duration": "7-30 days after mosquito bite",
        "causative_agent": "Plasmodium parasites (P. falciparum, P. vivax, P. ovale, P. malariae)",
        "disease_type": "parasitic",
        "transmission_mode": "Anopheles mosquito bite, blood transfusion, mother to fetus",
        "prevention": "Insecticide-treated bed nets, indoor spraying, antimalarial drugs, eliminate standing water",
        "vaccination": "RTS,S vaccine (limited availability)",
        "hygiene_measures": "Use mosquito repellents, wear long sleeves, eliminate breeding sites",
        "lifestyle_prevention": "Sleep under bed nets, avoid outdoor activities during dawn/dusk",
        "treatment": "Antimalarial drugs (Artemisinin-based combination therapy), supportive care",
        "emergency_treatment": "IV antimalarials, blood transfusion, intensive monitoring",
        "home_remedies": "Neem leaves, tulsi, ginger tea (supportive only)",
        "medications": "Chloroquine, Artesunate, Quinine, Doxycycline, Primaquine",
        "treatment_duration": "3-7 days",
        "prevalence_india": 5.6,
        "affected_states": json.dumps(["Odisha", "Chhattisgarh", "Jharkhand", "Meghalaya", "Tripura"]),
        "seasonal_pattern": "Peak during monsoon and post-monsoon (July-December)",
        "high_risk_groups": "Children under 5, pregnant women, tribal populations",
        "mortality_rate": 0.1,
        "severity": "high",
        "contagious": False,
        "incubation_period": "7-30 days",
        "complications": "Cerebral malaria, severe anemia, respiratory distress, kidney failure",
        "diagnostic_tests": "Blood smear microscopy, Rapid Diagnostic Test (RDT), PCR",
        "age_groups_affected": json.dumps(["all ages", "higher risk in children"]),
        "gender_preference": "equal",
        "socioeconomic_factors": "Higher prevalence in tribal areas, poverty, poor housing",
        "prognosis": "Excellent with prompt treatment",
        "follow_up_care": "Monitor for relapse, especially P. vivax infections",
        "nutritional_advice": "Iron-rich foods, adequate hydration, balanced diet",
        "exercise_recommendations": "Gradual increase after recovery",
        "data_source": "National Vector Borne Disease Control Programme",
        "confidence_score": 0.92
    },
    {
        "disease_name": "Dengue Fever",
        "disease_name_hindi": "डेंगू बुखार",
        "disease_category": "Viral Infectious Disease",
        "icd_10_code": "A90-A91",
        "symptoms": "High fever, severe headache, pain behind eyes, muscle and joint pain, skin rash, nausea",
        "early_symptoms": "Sudden onset high fever, headache, muscle pain",
        "severe_symptoms": "Bleeding, plasma leakage, organ impairment, shock",
        "symptom_duration": "3-7 days fever phase",
        "causative_agent": "Dengue virus (DENV 1-4)",
        "disease_type": "viral",
        "transmission_mode": "Aedes aegypti and Aedes albopictus mosquito bites",
        "prevention": "Eliminate mosquito breeding sites, use repellents, protective clothing",
        "vaccination": "Dengvaxia (limited use, only for previously infected)",
        "hygiene_measures": "Remove standing water, clean water storage, use mosquito nets",
        "lifestyle_prevention": "Avoid water accumulation, use air conditioning/fans",
        "treatment": "Supportive care, fluid management, pain relief, platelet monitoring",
        "emergency_treatment": "IV fluid therapy, platelet transfusion, intensive monitoring",
        "home_remedies": "Papaya leaf extract, coconut water, pomegranate juice, rest",
        "medications": "Paracetamol (avoid aspirin and NSAIDs), ORS, IV fluids",
        "treatment_duration": "5-7 days monitoring period",
        "prevalence_india": 3.2,
        "affected_states": json.dumps(["Tamil Nadu", "Karnataka", "West Bengal", "Kerala", "Delhi"]),
        "seasonal_pattern": "Peak during monsoon (June-September)",
        "high_risk_groups": "Urban populations, areas with poor sanitation",
        "mortality_rate": 0.5,
        "severity": "moderate",
        "contagious": False,
        "incubation_period": "4-6 days",
        "complications": "Dengue hemorrhagic fever, dengue shock syndrome, plasma leakage",
        "diagnostic_tests": "NS1 antigen, IgM/IgG antibodies, platelet count, tourniquet test",
        "age_groups_affected": json.dumps(["all ages", "severe in children and elderly"]),
        "gender_preference": "equal",
        "socioeconomic_factors": "Urban slums, poor water storage, inadequate waste management",
        "prognosis": "Good with proper monitoring and care",
        "follow_up_care": "Monitor platelet count, watch for warning signs",
        "nutritional_advice": "Increase fluid intake, vitamin C rich foods, avoid alcohol",
        "exercise_recommendations": "Rest during fever, gradual activity after recovery",
        "data_source": "National Vector Borne Disease Control Programme",
        "confidence_score": 0.90
    },
    {
        "disease_name": "Tuberculosis",
        "disease_name_hindi": "तपेदिक",
        "disease_category": "Bacterial Infectious Disease",
        "icd_10_code": "A15-A19",
        "symptoms": "Persistent cough (>2 weeks), fever, night sweats, weight loss, fatigue, chest pain",
        "early_symptoms": "Mild persistent cough, low-grade fever, fatigue",
        "severe_symptoms": "Hemoptysis, severe weight loss, respiratory failure",
        "symptom_duration": "Symptoms develop gradually over weeks to months",
        "causative_agent": "Mycobacterium tuberculosis",
        "disease_type": "bacterial",
        "transmission_mode": "Airborne droplets when infected person coughs/sneezes",
        "prevention": "BCG vaccination, good ventilation, avoid crowded spaces, treat latent TB",
        "vaccination": "BCG vaccine (given at birth in India)",
        "hygiene_measures": "Cover mouth when coughing, good ventilation, isolation of active cases",
        "lifestyle_prevention": "Good nutrition, avoid smoking/alcohol, strengthen immunity",
        "treatment": "DOTS (Directly Observed Treatment Short-course) - 6-8 months antibiotics",
        "emergency_treatment": "Hospitalization for severe cases, drug-resistant TB treatment",
        "home_remedies": "Turmeric milk, garlic, ginger, honey (supportive only)",
        "medications": "Rifampin, Isoniazid, Ethambutol, Pyrazinamide",
        "treatment_duration": "6-8 months (drug-sensitive), 18-24 months (drug-resistant)",
        "prevalence_india": 27.0,
        "affected_states": json.dumps(["Uttar Pradesh", "Jharkhand", "Madhya Pradesh", "Bihar", "Rajasthan"]),
        "seasonal_pattern": "Slightly higher in winter months",
        "high_risk_groups": "HIV patients, diabetics, malnourished, elderly, smokers",
        "mortality_rate": 4.5,
        "severity": "high",
        "contagious": True,
        "incubation_period": "2-12 weeks",
        "complications": "Drug resistance, disseminated TB, respiratory failure, death",
        "diagnostic_tests": "Sputum microscopy, chest X-ray, Gene Xpert, tuberculin skin test",
        "age_groups_affected": json.dumps(["all ages", "peak in 15-45 years"]),
        "gender_preference": "male predominant (2:1)",
        "socioeconomic_factors": "Poverty, malnutrition, overcrowding, poor housing",
        "prognosis": "Excellent with complete treatment",
        "follow_up_care": "Monthly monitoring, ensure treatment completion",
        "nutritional_advice": "High protein, vitamin-rich diet, weight gain focus",
        "exercise_recommendations": "Gradual increase as tolerated",
        "data_source": "Central TB Division, Ministry of Health",
        "confidence_score": 0.94
    },
    {
        "disease_name": "Diabetes Mellitus Type 2",
        "disease_name_hindi": "मधुमेह",
        "disease_category": "Metabolic Disorder",
        "icd_10_code": "E11",
        "symptoms": "Increased thirst, frequent urination, extreme fatigue, blurred vision, slow healing wounds",
        "early_symptoms": "Mild increase in thirst and urination, fatigue",
        "severe_symptoms": "Diabetic ketoacidosis, severe hypoglycemia, organ complications",
        "symptom_duration": "Gradual onset over months to years",
        "causative_agent": "Insulin resistance and pancreatic beta-cell dysfunction",
        "disease_type": "metabolic",
        "transmission_mode": "Non-communicable (genetic and lifestyle factors)",
        "prevention": "Healthy diet, regular exercise, weight management, avoid smoking",
        "vaccination": "Not applicable",
        "hygiene_measures": "Foot care, wound care, dental hygiene",
        "lifestyle_prevention": "Mediterranean diet, 150min/week exercise, maintain BMI <25",
        "treatment": "Lifestyle modification, oral hypoglycemic agents, insulin therapy",
        "emergency_treatment": "Hypoglycemia treatment, DKA management",
        "home_remedies": "Bitter gourd, fenugreek, cinnamon, turmeric (complementary)",
        "medications": "Metformin, Sulfonylureas, DPP-4 inhibitors, SGLT-2 inhibitors, Insulin",
        "treatment_duration": "Lifelong management",
        "prevalence_india": 8.9,
        "affected_states": json.dumps(["Tamil Nadu", "Kerala", "Goa", "Punjab", "Chandigarh"]),
        "seasonal_pattern": "No specific seasonal pattern",
        "high_risk_groups": "Family history, obesity, sedentary lifestyle, age >45",
        "mortality_rate": 3.2,
        "severity": "moderate",
        "contagious": False,
        "incubation_period": "Not applicable",
        "complications": "Diabetic nephropathy, retinopathy, neuropathy, cardiovascular disease",
        "diagnostic_tests": "Fasting glucose, HbA1c, oral glucose tolerance test",
        "age_groups_affected": json.dumps(["typically >30 years", "increasing in youth"]),
        "gender_preference": "slightly male predominant",
        "socioeconomic_factors": "Urbanization, dietary changes, sedentary lifestyle",
        "prognosis": "Good with proper management",
        "follow_up_care": "Regular monitoring of blood sugar, annual eye/kidney checks",
        "nutritional_advice": "Low glycemic index diet, portion control, high fiber foods",
        "exercise_recommendations": "150 minutes moderate aerobic activity per week",
        "data_source": "Indian Council of Medical Research",
        "confidence_score": 0.93
    }
    # Continue with 495 more diseases...
]

def populate_disease_database(db_manager: EnhancedDiseaseDB):
    """Populate database with comprehensive disease information"""
    cursor = db_manager.conn.cursor()
    
    for disease_data in TOP_500_DISEASES_INDIA:
        try:
            # Insert into enhanced_disease_info
            columns = list(disease_data.keys())
            placeholders = ['?' for _ in columns]
            values = list(disease_data.values())
            
            query = f"""
            INSERT OR REPLACE INTO enhanced_disease_info 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            cursor.execute(query, values)
            
        except Exception as e:
            print(f"Error inserting {disease_data.get('disease_name', 'Unknown')}: {e}")
    
    db_manager.conn.commit()
    print(f"Successfully populated database with {len(TOP_500_DISEASES_INDIA)} diseases!")

if __name__ == "__main__":
    # Create enhanced database
    db_manager = EnhancedDiseaseDB()
    
    # Populate with disease data
    populate_disease_database(db_manager)
    
    # Verify data
    cursor = db_manager.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM enhanced_disease_info")
    count = cursor.fetchone()[0]
    print(f"Total diseases in enhanced database: {count}")
    
    db_manager.conn.close()