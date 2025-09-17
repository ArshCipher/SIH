"""
Immediate Access Medical Databases - No License Required
================================================

These APIs work RIGHT NOW without any approval delays!
"""

import requests
import json
from typing import Dict, List, Optional
import sqlite3
from datetime import datetime

class ImmediateMedicalAPIs:
    """Access multiple free medical databases instantly"""
    
    def __init__(self):
        self.apis = {
            "openfda": "https://api.fda.gov",
            "rxnorm": "https://rxnav.nlm.nih.gov/REST",
            "mesh": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "mondo": "https://api.monarchinitiative.org/api",
            "drugbank_open": "https://go.drugbank.com/releases/latest#open-data",
            "icd_api": "https://icd11restapi-developer-test.azurewebsites.net"
        }
        
    def get_fda_drug_info(self, drug_name: str) -> Dict:
        """
        FDA OpenFDA API - Immediate access to drug information
        🟢 NO LICENSE REQUIRED
        """
        try:
            url = f"{self.apis['openfda']}/drug/label.json"
            params = {
                "search": f"openfda.brand_name:{drug_name}",
                "limit": 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "FDA OpenFDA",
                    "drug": drug_name,
                    "results": data.get("results", []),
                    "status": "success"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_rxnorm_drug_info(self, drug_name: str) -> Dict:
        """
        RxNorm API - Free drug terminology
        🟢 NO LICENSE REQUIRED
        """
        try:
            # Search for drug concept
            search_url = f"{self.apis['rxnorm']}/drugs.json"
            params = {"name": drug_name}
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "RxNorm",
                    "drug": drug_name,
                    "concepts": data.get("drugGroup", {}).get("conceptGroup", []),
                    "status": "success"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_mesh_disease_info(self, disease_term: str) -> Dict:
        """
        MeSH (Medical Subject Headings) via NCBI E-utilities
        🟢 NO LICENSE REQUIRED
        """
        try:
            # Search MeSH database
            search_url = f"{self.apis['mesh']}/esearch.fcgi"
            params = {
                "db": "mesh",
                "term": disease_term,
                "retmode": "json",
                "retmax": 10
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "MeSH/NCBI",
                    "disease": disease_term,
                    "ids": data.get("esearchresult", {}).get("idlist", []),
                    "status": "success"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_disease_ontology_info(self, disease_name: str) -> Dict:
        """
        Disease Ontology via Monarch Initiative
        🟢 NO LICENSE REQUIRED
        """
        try:
            search_url = f"{self.apis['mondo']}/search/entity"
            params = {
                "q": disease_name,
                "category": "disease",
                "rows": 10
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "source": "Monarch/Disease Ontology",
                    "disease": disease_name,
                    "results": data.get("docs", []),
                    "status": "success"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

class OpenMedicalDataCollector:
    """Collect and store data from multiple free sources"""
    
    def __init__(self, db_path: str = "open_medical_data.db"):
        self.db_path = db_path
        self.apis = ImmediateMedicalAPIs()
        self.setup_database()
    
    def setup_database(self):
        """Create enhanced database schema for open data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced diseases table with multiple sources
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            icd_code TEXT,
            mesh_id TEXT,
            mondo_id TEXT,
            symptoms TEXT,
            treatments TEXT,
            prevalence_india TEXT,
            severity TEXT,
            emergency_signs TEXT,
            prevention TEXT,
            source_fda TEXT,
            source_mesh TEXT,
            source_mondo TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_score REAL DEFAULT 0.0
        )
        """)
        
        # Drugs table with multiple sources
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_drugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rxnorm_id TEXT,
            fda_data TEXT,
            indications TEXT,
            contraindications TEXT,
            side_effects TEXT,
            dosage_forms TEXT,
            interactions TEXT,
            source_fda TEXT,
            source_rxnorm TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
    
    def collect_disease_data(self, disease_list: List[str]) -> Dict:
        """Collect comprehensive disease data from multiple free sources"""
        results = {
            "collected": 0,
            "errors": 0,
            "sources_used": [],
            "details": []
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for disease in disease_list:
            print(f"📊 Collecting data for: {disease}")
            
            # Get data from multiple sources
            mesh_data = self.apis.get_mesh_disease_info(disease)
            mondo_data = self.apis.get_disease_ontology_info(disease)
            
            # Combine and store data
            try:
                cursor.execute("""
                INSERT OR REPLACE INTO enhanced_diseases 
                (name, source_mesh, source_mondo, last_updated, confidence_score)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    disease,
                    json.dumps(mesh_data) if mesh_data.get("status") == "success" else None,
                    json.dumps(mondo_data) if mondo_data.get("status") == "success" else None,
                    datetime.now().isoformat(),
                    0.8 if mesh_data.get("status") == "success" else 0.5
                ))
                
                results["collected"] += 1
                results["details"].append({
                    "disease": disease,
                    "mesh_success": mesh_data.get("status") == "success",
                    "mondo_success": mondo_data.get("status") == "success"
                })
                
            except Exception as e:
                results["errors"] += 1
                print(f"❌ Error storing {disease}: {e}")
        
        conn.commit()
        conn.close()
        
        results["sources_used"] = ["MeSH/NCBI", "Monarch/Disease Ontology"]
        return results
    
    def collect_drug_data(self, drug_list: List[str]) -> Dict:
        """Collect comprehensive drug data from free sources"""
        results = {
            "collected": 0,
            "errors": 0,
            "sources_used": [],
            "details": []
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for drug in drug_list:
            print(f"💊 Collecting data for: {drug}")
            
            # Get data from multiple sources
            fda_data = self.apis.get_fda_drug_info(drug)
            rxnorm_data = self.apis.get_rxnorm_drug_info(drug)
            
            # Store combined data
            try:
                cursor.execute("""
                INSERT OR REPLACE INTO enhanced_drugs 
                (name, source_fda, source_rxnorm, last_updated)
                VALUES (?, ?, ?, ?)
                """, (
                    drug,
                    json.dumps(fda_data) if fda_data.get("status") == "success" else None,
                    json.dumps(rxnorm_data) if rxnorm_data.get("status") == "success" else None,
                    datetime.now().isoformat()
                ))
                
                results["collected"] += 1
                results["details"].append({
                    "drug": drug,
                    "fda_success": fda_data.get("status") == "success",
                    "rxnorm_success": rxnorm_data.get("status") == "success"
                })
                
            except Exception as e:
                results["errors"] += 1
                print(f"❌ Error storing {drug}: {e}")
        
        conn.commit()
        conn.close()
        
        results["sources_used"] = ["FDA OpenFDA", "RxNorm"]
        return results

def demo_immediate_access():
    """Demonstrate immediate access to medical databases"""
    print("🚀 IMMEDIATE ACCESS MEDICAL DATABASES DEMO")
    print("=" * 50)
    
    # Test APIs that work RIGHT NOW
    apis = ImmediateMedicalAPIs()
    
    # Test 1: FDA Drug Information
    print("\n🔍 Testing FDA OpenFDA API...")
    fda_result = apis.get_fda_drug_info("aspirin")
    print(f"FDA Status: {fda_result.get('status')}")
    if fda_result.get("status") == "success":
        print(f"Found {len(fda_result.get('results', []))} FDA drug records")
    
    # Test 2: RxNorm Drug Information
    print("\n🔍 Testing RxNorm API...")
    rx_result = apis.get_rxnorm_drug_info("metformin")
    print(f"RxNorm Status: {rx_result.get('status')}")
    if rx_result.get("status") == "success":
        concepts = rx_result.get('concepts', [])
        print(f"Found {len(concepts)} drug concepts")
    
    # Test 3: MeSH Disease Information
    print("\n🔍 Testing MeSH/NCBI API...")
    mesh_result = apis.get_mesh_disease_info("diabetes")
    print(f"MeSH Status: {mesh_result.get('status')}")
    if mesh_result.get("status") == "success":
        ids = mesh_result.get('ids', [])
        print(f"Found {len(ids)} MeSH disease IDs")
    
    # Test 4: Disease Ontology
    print("\n🔍 Testing Monarch/Disease Ontology API...")
    mondo_result = apis.get_disease_ontology_info("hypertension")
    print(f"Mondo Status: {mondo_result.get('status')}")
    if mondo_result.get("status") == "success":
        results = mondo_result.get('results', [])
        print(f"Found {len(results)} disease ontology matches")
    
    return {
        "fda": fda_result,
        "rxnorm": rx_result,
        "mesh": mesh_result,
        "mondo": mondo_result
    }

# Top Indian diseases for immediate collection
IMMEDIATE_DISEASES = [
    "diabetes", "hypertension", "tuberculosis", "malaria", "dengue",
    "chikungunya", "hepatitis", "pneumonia", "asthma", "arthritis",
    "depression", "anxiety", "migraine", "gastritis", "obesity",
    "anemia", "thyroid disorders", "kidney stones", "heart disease",
    "stroke", "cancer", "epilepsy", "osteoporosis", "cataracts"
]

# Common drugs in India for immediate collection
IMMEDIATE_DRUGS = [
    "paracetamol", "aspirin", "metformin", "insulin", "amoxicillin",
    "ciprofloxacin", "omeprazole", "atorvastatin", "amlodipine",
    "losartan", "clopidogrel", "prednisolone", "diclofenac"
]

if __name__ == "__main__":
    # Run immediate demo
    demo_results = demo_immediate_access()
    
    print("\n" + "="*50)
    print("✅ READY FOR IMMEDIATE DATA COLLECTION!")
    print("No waiting for licenses - these APIs work NOW!")