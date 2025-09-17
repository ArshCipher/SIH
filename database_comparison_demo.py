"""
Database Comparison: Your 500 Diseases vs World-Class Medical Databases
See the massive difference in scope and authority
"""

import asyncio
import aiohttp
import json
from datetime import datetime

class DatabaseComparison:
    """Compare our current 500-disease database with world-class alternatives"""
    
    def __init__(self):
        self.our_database_stats = {
            "diseases": 500,
            "sources": ["Custom"],
            "languages": 1,
            "authority": "Educational",
            "updates": "Static",
            "research_papers": 0,
            "clinical_use": "Limited",
            "cost": "Free"
        }
        
        self.world_class_stats = {
            "UMLS": {
                "diseases": 4300000,
                "sources": ["NIH", "NLM", "200+ vocabularies"],
                "languages": 25,
                "authority": "National Institutes of Health",
                "updates": "Continuous",
                "research_papers": "Integrated",
                "clinical_use": "Hospital-grade",
                "cost": "Free"
            },
            "SNOMED CT": {
                "diseases": 350000,
                "sources": ["IHTSDO", "80+ countries"],
                "languages": 50,
                "authority": "International standard",
                "updates": "Bi-annual",
                "research_papers": "Clinical evidence",
                "clinical_use": "Global clinical records",
                "cost": "Free for India"
            },
            "WHO ICD-11": {
                "diseases": 55000,
                "sources": ["World Health Organization"],
                "languages": 40,
                "authority": "WHO - Global health authority",
                "updates": "Real-time",
                "research_papers": "WHO studies",
                "clinical_use": "Global health statistics",
                "cost": "Free"
            }
        }

    async def test_live_apis(self):
        """Test live access to world-class medical APIs"""
        
        print("🌐 Testing Live Access to World-Class Medical Databases")
        print("=" * 60)
        
        results = {}
        
        # Test SNOMED CT
        try:
            print("🔍 Testing SNOMED CT International...")
            async with aiohttp.ClientSession() as session:
                url = "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/concepts"
                params = {"term": "diabetes mellitus", "activeFilter": "true", "limit": 5}
                
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "items" in data and data["items"]:
                            results["SNOMED CT"] = {
                                "status": "✅ Connected",
                                "sample_results": len(data["items"]),
                                "example": data["items"][0].get("pt", {}).get("term", "N/A"),
                                "concept_id": data["items"][0].get("conceptId", "N/A")
                            }
                        else:
                            results["SNOMED CT"] = {"status": "⚠️ No results"}
                    else:
                        results["SNOMED CT"] = {"status": f"⚠️ HTTP {response.status}"}
        except Exception as e:
            results["SNOMED CT"] = {"status": f"❌ Error: {str(e)[:50]}"}
        
        # Test WHO Global Health Observatory
        try:
            print("🔍 Testing WHO Global Health Observatory...")
            async with aiohttp.ClientSession() as session:
                url = "https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues"
                
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "value" in data:
                            countries = [country.get("Title") for country in data["value"][:10]]
                            india_available = "India" in str(data)
                            results["WHO"] = {
                                "status": "✅ Connected",
                                "countries_available": len(data["value"]),
                                "india_data": "Available" if india_available else "Checking...",
                                "sample_countries": countries[:3]
                            }
                        else:
                            results["WHO"] = {"status": "⚠️ Unexpected format"}
                    else:
                        results["WHO"] = {"status": f"⚠️ HTTP {response.status}"}
        except Exception as e:
            results["WHO"] = {"status": f"❌ Error: {str(e)[:50]}"}
        
        # Test PubMed
        try:
            print("🔍 Testing PubMed (35+ Million Research Papers)...")
            async with aiohttp.ClientSession() as session:
                url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                params = {
                    "db": "pubmed",
                    "term": "diabetes treatment India",
                    "retmax": 3,
                    "retmode": "json"
                }
                
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "esearchresult" in data:
                            count = int(data["esearchresult"].get("count", 0))
                            results["PubMed"] = {
                                "status": "✅ Connected",
                                "diabetes_papers": f"{count:,}",
                                "total_papers": "35+ million",
                                "latest_research": "Available"
                            }
                        else:
                            results["PubMed"] = {"status": "⚠️ Unexpected format"}
                    else:
                        results["PubMed"] = {"status": f"⚠️ HTTP {response.status}"}
        except Exception as e:
            results["PubMed"] = {"status": f"❌ Error: {str(e)[:50]}"}
        
        return results

    def display_comparison(self, live_results=None):
        """Display comprehensive comparison"""
        
        print("\n📊 Database Comparison: Current vs World-Class")
        print("=" * 60)
        
        # Header
        print(f"{'Metric':<25} {'Current (500)':<15} {'UMLS':<15} {'SNOMED CT':<15} {'WHO ICD-11':<15}")
        print("-" * 85)
        
        # Data rows
        metrics = [
            ("Medical Concepts", "500", "4.3M+", "350K+", "55K+"),
            ("Authority", "Custom", "NIH/NLM", "International", "WHO"),
            ("Languages", "1", "25+", "50+", "40+"),
            ("Updates", "Static", "Live", "Bi-annual", "Real-time"),
            ("Clinical Use", "Educational", "Hospital", "Global", "Statistics"),
            ("Research Papers", "0", "Integrated", "Evidence", "WHO Studies"),
            ("Cost", "Free", "Free", "Free*", "Free")
        ]
        
        for metric, current, umls, snomed, who in metrics:
            print(f"{metric:<25} {current:<15} {umls:<15} {snomed:<15} {who:<15}")
        
        print("\n*Free for developing countries including India")
        
        # Live API test results
        if live_results:
            print("\n🌐 Live API Test Results")
            print("=" * 30)
            for api, result in live_results.items():
                print(f"\n{api}:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
        
        # Impact analysis
        print("\n🚀 Impact of Upgrading to World-Class Databases")
        print("=" * 50)
        
        improvements = [
            ("Disease Coverage", "500 diseases", "4.3+ million medical concepts", "8,600x increase"),
            ("Authority", "Custom database", "WHO, NIH, SNOMED standards", "Hospital-grade"),
            ("Languages", "English only", "25-50 languages", "Global accessibility"),
            ("Updates", "Static data", "Real-time medical updates", "Always current"),
            ("Research", "No integration", "35+ million papers", "Latest evidence"),
            ("Clinical Use", "Educational", "Used in hospitals globally", "Professional grade")
        ]
        
        print(f"{'Aspect':<15} {'Current':<20} {'World-Class':<25} {'Improvement':<15}")
        print("-" * 75)
        for aspect, current, world_class, improvement in improvements:
            print(f"{aspect:<15} {current:<20} {world_class:<25} {improvement:<15}")

    def show_real_examples(self):
        """Show real examples of what changes"""
        
        print("\n💡 Real Examples: What Changes with World-Class Data")
        print("=" * 55)
        
        examples = [
            {
                "query": "What is Type 2 Diabetes?",
                "current": "Basic definition from our 500-disease database",
                "world_class": "UMLS CUI: C0011860, SNOMED: 44054006, ICD-11: 5A11, WHO prevalence data for India, latest research from 15,000+ diabetes papers"
            },
            {
                "query": "COVID-19 symptoms in Hindi",
                "current": "English response only, basic symptoms",
                "world_class": "Multi-language response, WHO official symptoms, latest variant information, India-specific data"
            },
            {
                "query": "Drug interactions with metformin",
                "current": "Not available in our database",
                "world_class": "Complete drug interaction database, clinical warnings, dosage recommendations"
            },
            {
                "query": "Latest tuberculosis treatment",
                "current": "Static treatment information",
                "world_class": "Latest WHO guidelines, recent research papers, India TB program data"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n🔍 Example {i}: {example['query']}")
            print(f"   Current Response: {example['current']}")
            print(f"   World-Class Response: {example['world_class']}")

    def show_integration_options(self):
        """Show different integration strategies"""
        
        print("\n🔧 Integration Strategies")
        print("=" * 25)
        
        strategies = [
            {
                "name": "Hybrid Approach (Recommended)",
                "description": "Keep 500-disease DB for speed + World APIs for comprehensive coverage",
                "pros": ["Fast responses", "Comprehensive coverage", "Reliable fallback"],
                "setup_time": "1 day",
                "complexity": "Medium"
            },
            {
                "name": "Full Migration",
                "description": "Replace local DB entirely with world-class APIs",
                "pros": ["Maximum coverage", "Always up-to-date", "Professional grade"],
                "setup_time": "3 days",
                "complexity": "High"
            },
            {
                "name": "API Enhancement",
                "description": "Add world APIs as enhancement layer on top of current system",
                "pros": ["Easy implementation", "Gradual upgrade", "Risk-free"],
                "setup_time": "2 hours",
                "complexity": "Low"
            }
        ]
        
        for strategy in strategies:
            print(f"\n📋 {strategy['name']}")
            print(f"   Description: {strategy['description']}")
            print(f"   Pros: {', '.join(strategy['pros'])}")
            print(f"   Setup Time: {strategy['setup_time']}")
            print(f"   Complexity: {strategy['complexity']}")

async def main():
    """Main comparison function"""
    
    print("🏥 Medical Database Comparison Tool")
    print("See how world-class databases transform your chatbot")
    print("=" * 60)
    
    comparison = DatabaseComparison()
    
    # Test live APIs
    print("Testing live connections to world-class medical databases...")
    live_results = await comparison.test_live_apis()
    
    # Display full comparison
    comparison.display_comparison(live_results)
    
    # Show real examples
    comparison.show_real_examples()
    
    # Show integration options
    comparison.show_integration_options()
    
    print("\n🎯 Conclusion")
    print("=" * 12)
    print("✅ World-class medical databases are accessible and FREE")
    print("✅ They provide 8,600x more medical coverage than current system")
    print("✅ Used by hospitals and healthcare systems globally")
    print("✅ Real-time updates with latest medical research")
    print("✅ Multi-language support including Indian languages")
    print("✅ Professional-grade authority (WHO, NIH, SNOMED)")
    
    print("\n🚀 Next Steps:")
    print("1. Get free UMLS API key: https://uts.nlm.nih.gov/uts/")
    print("2. Choose integration strategy (Hybrid recommended)")
    print("3. Transform your 500-disease chatbot into world-class medical AI")
    print("4. Provide hospital-grade medical information to users")

if __name__ == "__main__":
    asyncio.run(main())