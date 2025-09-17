"""
Comprehensive Medical AI Testing and Improvement System

This script continuously tests the medical AI with hundreds of diverse queries
across all medical specialties and provides detailed analysis and improvement suggestions.
"""

import asyncio
import json
import time
import requests
import random
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    query: str
    response: str
    response_time: float
    query_category: str
    expected_keywords: List[str]
    found_keywords: List[str]
    completeness_score: float
    relevance_score: float
    emergency_detected: bool
    language_detected: str
    timestamp: datetime

class MedicalAITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.improvement_suggestions = []
        
        # Comprehensive test queries covering all medical specialties
        self.test_queries = {
            "cardiovascular": [
                "chest pain", "heart attack symptoms", "high blood pressure", "irregular heartbeat",
                "heart palpitations", "shortness of breath", "chest tightness", "cardiac arrest",
                "angina", "heart failure", "arrhythmia", "hypertension treatment", "blood pressure medication",
                "cholesterol management", "heart disease prevention", "bypass surgery", "angioplasty",
                "stroke symptoms", "heart valve disease", "peripheral artery disease"
            ],
            "respiratory": [
                "difficulty breathing", "asthma attack", "pneumonia symptoms", "chronic cough",
                "lung infection", "bronchitis", "COPD", "tuberculosis", "lung cancer",
                "pulmonary embolism", "sleep apnea", "wheezing", "chest congestion",
                "respiratory failure", "oxygen therapy", "inhaler use", "lung disease",
                "breathing exercises", "pulmonary rehabilitation", "ventilator support"
            ],
            "gastrointestinal": [
                "stomach pain", "diarrhea", "constipation", "nausea", "vomiting",
                "acid reflux", "ulcers", "inflammatory bowel disease", "liver disease",
                "gallstones", "appendicitis", "food poisoning", "gastritis",
                "colon cancer", "irritable bowel syndrome", "hemorrhoids", "hepatitis",
                "cirrhosis", "pancreatitis", "celiac disease"
            ],
            "neurological": [
                "headache", "migraine", "seizure", "stroke", "epilepsy", "Alzheimer's",
                "Parkinson's disease", "multiple sclerosis", "brain tumor", "concussion",
                "nerve pain", "numbness", "tingling", "memory loss", "confusion",
                "dizziness", "fainting", "tremor", "paralysis", "spinal cord injury"
            ],
            "endocrine": [
                "diabetes", "thyroid problems", "insulin resistance", "blood sugar",
                "hormone imbalance", "adrenal insufficiency", "growth hormone",
                "osteoporosis", "metabolic syndrome", "polycystic ovary syndrome",
                "menopause", "testosterone deficiency", "cushing's syndrome",
                "hyperthyroidism", "hypothyroidism", "diabetic ketoacidosis"
            ],
            "infectious": [
                "fever", "flu symptoms", "COVID-19", "malaria", "dengue", "tuberculosis",
                "urinary tract infection", "pneumonia", "sepsis", "meningitis",
                "hepatitis", "HIV", "sexually transmitted diseases", "fungal infection",
                "bacterial infection", "viral infection", "antibiotic resistance",
                "vaccination", "tropical diseases", "food-borne illness"
            ],
            "musculoskeletal": [
                "joint pain", "arthritis", "back pain", "muscle strain", "fracture",
                "osteoporosis", "fibromyalgia", "lupus", "gout", "tendonitis",
                "sports injury", "carpal tunnel syndrome", "herniated disc",
                "bone cancer", "muscle weakness", "joint swelling", "stiffness",
                "rheumatoid arthritis", "osteoarthritis", "bone density"
            ],
            "dermatological": [
                "skin rash", "eczema", "psoriasis", "acne", "melanoma", "skin cancer",
                "allergic reaction", "hives", "dermatitis", "fungal infection",
                "warts", "moles", "sunburn", "wound healing", "scar treatment",
                "hair loss", "nail problems", "skin discoloration", "itching", "dry skin"
            ],
            "mental_health": [
                "depression", "anxiety", "panic attacks", "bipolar disorder", "schizophrenia",
                "PTSD", "eating disorders", "substance abuse", "suicide prevention",
                "stress management", "insomnia", "mood disorders", "attention deficit",
                "autism spectrum", "personality disorders", "grief counseling",
                "therapy options", "psychiatric medications", "mental health crisis"
            ],
            "pediatric": [
                "child fever", "infant feeding", "growth development", "vaccination schedule",
                "childhood diseases", "ear infection", "strep throat", "chickenpox",
                "hand foot mouth disease", "colic", "teething", "developmental delays",
                "ADHD in children", "childhood obesity", "pediatric emergencies",
                "newborn care", "breastfeeding", "formula feeding", "child safety"
            ],
            "women_health": [
                "pregnancy symptoms", "menstrual problems", "breast cancer", "cervical cancer",
                "menopause", "fertility issues", "contraception", "prenatal care",
                "postpartum depression", "ovarian cysts", "endometriosis", "PMS",
                "vaginal infection", "pelvic pain", "mammogram", "pap smear",
                "hormone replacement therapy", "pregnancy complications", "miscarriage"
            ],
            "emergency": [
                "cardiac arrest", "stroke", "severe allergic reaction", "poisoning",
                "severe trauma", "unconsciousness", "severe bleeding", "burns",
                "choking", "drug overdose", "electrical shock", "drowning",
                "heat stroke", "hypothermia", "anaphylaxis", "respiratory distress",
                "acute abdomen", "medical emergency", "911 situations", "life threatening"
            ],
            "hindi_queries": [
                "मुझे बुखार है", "सिर दर्द", "पेट में दर्द", "सांस लेने में तकलीफ",
                "मधुमेह के लक्षण", "हृदय रोग", "उच्च रक्तचाप", "कैंसर के संकेत",
                "संक्रमण", "दवा की जानकारी", "डॉक्टर से कब मिलें", "आपातकालीन स्थिति"
            ]
        }
        
        # Expected keywords for each category to evaluate response quality
        self.expected_keywords = {
            "cardiovascular": ["heart", "cardiac", "blood pressure", "chest", "artery", "circulation"],
            "respiratory": ["lung", "breathing", "respiratory", "oxygen", "airway", "pulmonary"],
            "gastrointestinal": ["stomach", "intestine", "digestive", "bowel", "liver", "gastric"],
            "neurological": ["brain", "nerve", "neurological", "cognitive", "motor", "sensory"],
            "endocrine": ["hormone", "gland", "metabolism", "insulin", "thyroid", "diabetes"],
            "infectious": ["infection", "virus", "bacteria", "fever", "immune", "pathogen"],
            "musculoskeletal": ["bone", "joint", "muscle", "skeletal", "arthritis", "fracture"],
            "dermatological": ["skin", "dermatological", "rash", "lesion", "dermatitis"],
            "mental_health": ["mental", "psychological", "mood", "anxiety", "depression", "psychiatric"],
            "pediatric": ["child", "infant", "pediatric", "development", "growth", "vaccination"],
            "women_health": ["pregnancy", "menstrual", "reproductive", "gynecological", "breast"],
            "emergency": ["emergency", "urgent", "immediate", "911", "life threatening", "critical"],
            "hindi_queries": ["medical", "health", "condition", "symptoms", "treatment"]
        }

    async def test_medical_ai(self, query: str, category: str) -> TestResult:
        """Test a single query against the medical AI"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"message": query},
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                ai_response = response.json().get("response", "")
                
                # Analyze response quality
                expected_keywords = self.expected_keywords.get(category, [])
                found_keywords = [kw for kw in expected_keywords if kw.lower() in ai_response.lower()]
                
                completeness_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
                relevance_score = self._calculate_relevance_score(query, ai_response)
                emergency_detected = self._detect_emergency_response(ai_response)
                language_detected = self._detect_response_language(ai_response)
                
                return TestResult(
                    query=query,
                    response=ai_response,
                    response_time=response_time,
                    query_category=category,
                    expected_keywords=expected_keywords,
                    found_keywords=found_keywords,
                    completeness_score=completeness_score,
                    relevance_score=relevance_score,
                    emergency_detected=emergency_detected,
                    language_detected=language_detected,
                    timestamp=datetime.now()
                )
            else:
                logger.error(f"HTTP {response.status_code} for query: {query}")
                return None
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
            return None

    def _calculate_relevance_score(self, query: str, response: str) -> float:
        """Calculate how relevant the response is to the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate overlap
        overlap = len(query_words.intersection(response_words))
        total_query_words = len(query_words)
        
        # Basic relevance score
        base_score = overlap / total_query_words if total_query_words > 0 else 0.0
        
        # Bonus for medical context
        medical_terms = ["medical", "health", "condition", "symptoms", "treatment", "doctor", "diagnosis"]
        medical_score = sum(1 for term in medical_terms if term in response.lower()) / len(medical_terms)
        
        return min(1.0, (base_score * 0.7) + (medical_score * 0.3))

    def _detect_emergency_response(self, response: str) -> bool:
        """Detect if emergency response was triggered"""
        emergency_indicators = ["emergency", "911", "call doctor", "seek immediate", "urgent", "life threatening"]
        return any(indicator in response.lower() for indicator in emergency_indicators)

    def _detect_response_language(self, response: str) -> str:
        """Detect the primary language of the response"""
        hindi_chars = sum(1 for char in response if '\u0900' <= char <= '\u097F')
        return "hindi" if hindi_chars > 10 else "english"

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive testing across all categories"""
        logger.info("Starting comprehensive medical AI testing...")
        
        all_results = []
        category_scores = {}
        
        for category, queries in self.test_queries.items():
            logger.info(f"Testing {category} category with {len(queries)} queries...")
            category_results = []
            
            # Test a sample of queries from each category
            test_sample = random.sample(queries, min(10, len(queries)))
            
            for query in test_sample:
                result = await self.test_medical_ai(query, category)
                if result:
                    category_results.append(result)
                    all_results.append(result)
                
                # Small delay to avoid overwhelming the server
                await asyncio.sleep(0.5)
            
            # Calculate category performance
            if category_results:
                avg_completeness = sum(r.completeness_score for r in category_results) / len(category_results)
                avg_relevance = sum(r.relevance_score for r in category_results) / len(category_results)
                avg_response_time = sum(r.response_time for r in category_results) / len(category_results)
                
                category_scores[category] = {
                    "avg_completeness": avg_completeness,
                    "avg_relevance": avg_relevance,
                    "avg_response_time": avg_response_time,
                    "total_queries": len(category_results),
                    "emergency_detection_rate": sum(1 for r in category_results if r.emergency_detected) / len(category_results)
                }
        
        self.test_results = all_results
        return self._generate_test_report(category_scores)

    def _generate_test_report(self, category_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        overall_completeness = sum(r.completeness_score for r in self.test_results) / total_tests if total_tests > 0 else 0
        overall_relevance = sum(r.relevance_score for r in self.test_results) / total_tests if total_tests > 0 else 0
        overall_response_time = sum(r.response_time for r in self.test_results) / total_tests if total_tests > 0 else 0
        
        # Identify areas for improvement
        weak_categories = [cat for cat, scores in category_scores.items() 
                          if scores['avg_completeness'] < 0.6 or scores['avg_relevance'] < 0.7]
        
        slow_categories = [cat for cat, scores in category_scores.items() 
                          if scores['avg_response_time'] > 5.0]
        
        report = {
            "test_summary": {
                "total_queries_tested": total_tests,
                "overall_completeness_score": overall_completeness,
                "overall_relevance_score": overall_relevance,
                "average_response_time": overall_response_time,
                "categories_tested": len(category_scores)
            },
            "category_performance": category_scores,
            "improvement_areas": {
                "weak_content_categories": weak_categories,
                "slow_response_categories": slow_categories,
                "recommendations": self._generate_recommendations(category_scores)
            },
            "test_timestamp": datetime.now().isoformat()
        }
        
        return report

    def _generate_recommendations(self, category_scores: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []
        
        for category, scores in category_scores.items():
            if scores['avg_completeness'] < 0.6:
                recommendations.append(f"Improve knowledge base for {category} - low completeness score: {scores['avg_completeness']:.2f}")
            
            if scores['avg_relevance'] < 0.7:
                recommendations.append(f"Enhance response relevance for {category} - score: {scores['avg_relevance']:.2f}")
            
            if scores['avg_response_time'] > 5.0:
                recommendations.append(f"Optimize response time for {category} - current: {scores['avg_response_time']:.2f}s")
            
            if category == "emergency" and scores['emergency_detection_rate'] < 0.8:
                recommendations.append(f"Critical: Improve emergency detection - only {scores['emergency_detection_rate']:.1%} detection rate")
        
        return recommendations

    def save_results(self, filename: str = None):
        """Save test results to file"""
        if not filename:
            filename = f"medical_ai_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert test results to serializable format
        results_data = []
        for result in self.test_results:
            results_data.append({
                "query": result.query,
                "response": result.response,
                "response_time": result.response_time,
                "query_category": result.query_category,
                "completeness_score": result.completeness_score,
                "relevance_score": result.relevance_score,
                "emergency_detected": result.emergency_detected,
                "language_detected": result.language_detected,
                "timestamp": result.timestamp.isoformat()
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to {filename}")

    async def continuous_testing(self, interval_minutes: int = 30):
        """Run continuous testing and improvement"""
        logger.info(f"Starting continuous testing every {interval_minutes} minutes...")
        
        while True:
            try:
                # Run comprehensive test
                report = await self.run_comprehensive_test()
                
                # Log key metrics
                logger.info(f"Test completed: {report['test_summary']['total_queries_tested']} queries")
                logger.info(f"Overall scores - Completeness: {report['test_summary']['overall_completeness_score']:.2f}, "
                           f"Relevance: {report['test_summary']['overall_relevance_score']:.2f}")
                
                # Save results
                self.save_results()
                
                # Print recommendations
                if report['improvement_areas']['recommendations']:
                    logger.info("Improvement recommendations:")
                    for rec in report['improvement_areas']['recommendations']:
                        logger.info(f"  - {rec}")
                
                # Wait for next test cycle
                logger.info(f"Waiting {interval_minutes} minutes until next test cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous testing: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

async def main():
    """Main testing function"""
    tester = MedicalAITester()
    
    print("🏥 Medical AI Comprehensive Testing System")
    print("=" * 50)
    
    choice = input("""
Choose testing mode:
1. Single comprehensive test
2. Continuous testing (every 30 minutes)
3. Quick test (sample queries)
4. Emergency-focused testing
5. Hindi language testing

Enter choice (1-5): """)
    
    if choice == "1":
        print("\n🔬 Running comprehensive test...")
        report = await tester.run_comprehensive_test()
        tester.save_results()
        
        print("\n📊 Test Results Summary:")
        print(f"Total queries tested: {report['test_summary']['total_queries_tested']}")
        print(f"Overall completeness: {report['test_summary']['overall_completeness_score']:.2%}")
        print(f"Overall relevance: {report['test_summary']['overall_relevance_score']:.2%}")
        print(f"Average response time: {report['test_summary']['average_response_time']:.2f}s")
        
        if report['improvement_areas']['recommendations']:
            print("\n🎯 Recommendations:")
            for rec in report['improvement_areas']['recommendations']:
                print(f"  • {rec}")
    
    elif choice == "2":
        await tester.continuous_testing()
    
    elif choice == "3":
        print("\n⚡ Running quick test...")
        # Test a few queries from each category
        quick_queries = [
            ("chest pain", "cardiovascular"),
            ("difficulty breathing", "respiratory"), 
            ("stomach pain", "gastrointestinal"),
            ("headache", "neurological"),
            ("diabetes symptoms", "endocrine"),
            ("मुझे बुखार है", "hindi_queries")
        ]
        
        for query, category in quick_queries:
            result = await tester.test_medical_ai(query, category)
            if result:
                print(f"\n🔍 Query: {query}")
                print(f"📝 Response time: {result.response_time:.2f}s")
                print(f"📊 Scores - Completeness: {result.completeness_score:.2%}, Relevance: {result.relevance_score:.2%}")
                print(f"🚨 Emergency detected: {result.emergency_detected}")
    
    elif choice == "4":
        print("\n🚨 Testing emergency detection...")
        emergency_queries = tester.test_queries["emergency"]
        for query in emergency_queries[:5]:
            result = await tester.test_medical_ai(query, "emergency")
            if result:
                print(f"\n🔍 Query: {query}")
                print(f"🚨 Emergency detected: {result.emergency_detected}")
                print(f"⏱️ Response time: {result.response_time:.2f}s")
    
    elif choice == "5":
        print("\n🌐 Testing Hindi language support...")
        hindi_queries = tester.test_queries["hindi_queries"]
        for query in hindi_queries:
            result = await tester.test_medical_ai(query, "hindi_queries")
            if result:
                print(f"\n🔍 Query: {query}")
                print(f"🗣️ Language detected: {result.language_detected}")
                print(f"📊 Relevance score: {result.relevance_score:.2%}")

if __name__ == "__main__":
    asyncio.run(main())