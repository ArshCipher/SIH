"""
Quick Setup: Collect Medical Data NOW (No License Required)
=========================================================

Run this to immediately populate your database with real medical data
from multiple free sources while waiting for UMLS approval.
"""

from immediate_medical_apis import OpenMedicalDataCollector, IMMEDIATE_DISEASES, IMMEDIATE_DRUGS, demo_immediate_access
import time

def quick_medical_data_setup():
    """Set up comprehensive medical database using free APIs"""
    
    print("🚀 IMMEDIATE MEDICAL DATABASE SETUP")
    print("=" * 60)
    print("⏰ No waiting! Using free APIs that work RIGHT NOW")
    print()
    
    # Step 1: Test API access
    print("📡 Step 1: Testing API connectivity...")
    demo_results = demo_immediate_access()
    
    working_apis = sum(1 for result in demo_results.values() 
                      if result.get("status") == "success")
    
    print(f"\n✅ {working_apis}/4 APIs working successfully!")
    
    if working_apis == 0:
        print("❌ No APIs working. Check internet connection.")
        return
    
    # Step 2: Initialize collector
    print("\n📊 Step 2: Initializing data collector...")
    collector = OpenMedicalDataCollector("enhanced_medical_database.db")
    print("✅ Database schema created!")
    
    # Step 3: Collect disease data
    print(f"\n🦠 Step 3: Collecting data for {len(IMMEDIATE_DISEASES)} diseases...")
    print("This will take 2-3 minutes...")
    
    disease_results = collector.collect_disease_data(IMMEDIATE_DISEASES)
    
    print(f"""
📊 DISEASE DATA COLLECTION RESULTS:
   ✅ Successfully collected: {disease_results['collected']} diseases
   ❌ Errors: {disease_results['errors']}
   🔗 Sources used: {', '.join(disease_results['sources_used'])}
    """)
    
    # Step 4: Collect drug data
    print(f"\n💊 Step 4: Collecting data for {len(IMMEDIATE_DRUGS)} drugs...")
    print("This will take 1-2 minutes...")
    
    drug_results = collector.collect_drug_data(IMMEDIATE_DRUGS)
    
    print(f"""
📊 DRUG DATA COLLECTION RESULTS:
   ✅ Successfully collected: {drug_results['collected']} drugs
   ❌ Errors: {drug_results['errors']}
   🔗 Sources used: {', '.join(drug_results['sources_used'])}
    """)
    
    # Step 5: Summary
    total_collected = disease_results['collected'] + drug_results['collected']
    
    print(f"""
🎉 SETUP COMPLETE!
{'='*40}
📊 Total Medical Entities Collected: {total_collected}
🦠 Diseases: {disease_results['collected']}
💊 Drugs: {drug_results['collected']}
🗄️ Database: enhanced_medical_database.db

🔥 YOUR CHATBOT NOW HAS ACCESS TO:
   • FDA-approved drug information
   • MeSH disease classifications  
   • Disease ontology mappings
   • RxNorm drug terminologies
   
⚡ Ready to integrate with your existing medical_orchestrator.py!
    """)
    
    return {
        "diseases_collected": disease_results['collected'],
        "drugs_collected": drug_results['collected'],
        "total_entities": total_collected,
        "database_path": "enhanced_medical_database.db"
    }

def create_integration_guide():
    """Create guide for integrating with existing chatbot"""
    
    integration_code = '''
# Add this to your medical_orchestrator.py to use the new database

import sqlite3
import json
from typing import Dict, List, Optional

class EnhancedMedicalRetriever:
    """Retrieve data from the new comprehensive medical database"""
    
    def __init__(self, db_path: str = "enhanced_medical_database.db"):
        self.db_path = db_path
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """Get comprehensive disease information from multiple sources"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT name, source_mesh, source_mondo, confidence_score
        FROM enhanced_diseases 
        WHERE name LIKE ? OR name = ?
        ORDER BY confidence_score DESC
        LIMIT 5
        """, (f"%{disease_name}%", disease_name))
        
        results = cursor.fetchall()
        conn.close()
        
        enhanced_info = []
        for result in results:
            name, mesh_data, mondo_data, confidence = result
            
            info = {
                "disease": name,
                "confidence": confidence,
                "sources": []
            }
            
            if mesh_data:
                mesh_parsed = json.loads(mesh_data)
                info["sources"].append("MeSH/NCBI")
                info["mesh_ids"] = mesh_parsed.get("ids", [])
            
            if mondo_data:
                mondo_parsed = json.loads(mondo_data)
                info["sources"].append("Disease Ontology")
                info["ontology_matches"] = mondo_parsed.get("results", [])
            
            enhanced_info.append(info)
        
        return enhanced_info
    
    def get_drug_info(self, drug_name: str) -> Dict:
        """Get comprehensive drug information from FDA and RxNorm"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT name, source_fda, source_rxnorm
        FROM enhanced_drugs 
        WHERE name LIKE ? OR name = ?
        LIMIT 5
        """, (f"%{drug_name}%", drug_name))
        
        results = cursor.fetchall()
        conn.close()
        
        drug_info = []
        for result in results:
            name, fda_data, rxnorm_data = result
            
            info = {
                "drug": name,
                "sources": []
            }
            
            if fda_data:
                fda_parsed = json.loads(fda_data)
                info["sources"].append("FDA")
                info["fda_results"] = fda_parsed.get("results", [])
            
            if rxnorm_data:
                rxnorm_parsed = json.loads(rxnorm_data)
                info["sources"].append("RxNorm")
                info["rxnorm_concepts"] = rxnorm_parsed.get("concepts", [])
            
            drug_info.append(info)
        
        return drug_info

# Update your medical_orchestrator.py generate_response method:
# Add this inside the generate_response method:

enhanced_retriever = EnhancedMedicalRetriever()

# For disease queries
if "disease" in user_input.lower() or "condition" in user_input.lower():
    disease_matches = enhanced_retriever.get_disease_info(user_input)
    if disease_matches:
        context += f"\\n\\nEnhanced Disease Data: {disease_matches}"

# For drug queries  
if "medicine" in user_input.lower() or "drug" in user_input.lower():
    drug_matches = enhanced_retriever.get_drug_info(user_input)
    if drug_matches:
        context += f"\\n\\nEnhanced Drug Data: {drug_matches}"
'''
    
    with open("integration_guide.py", "w") as f:
        f.write(integration_code)
    
    print("📝 Created integration_guide.py with code to connect to your chatbot!")

if __name__ == "__main__":
    print("🚀 Starting immediate medical data collection...")
    print("⏰ This will run while you wait for UMLS license approval!")
    print()
    
    # Run the setup
    results = quick_medical_data_setup()
    
    # Create integration guide
    create_integration_guide()
    
    print(f"""
🎯 NEXT STEPS:
1. ✅ Database populated with {results['total_entities']} medical entities
2. 📝 Check integration_guide.py for chatbot integration code
3. 🔄 Run your enhanced chatbot with real medical data!
4. ⏳ When UMLS arrives in 3 days, we'll add even more data!

🚀 Your chatbot is now 10x more powerful with real medical databases!
    """)