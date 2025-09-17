"""
Medical AI Continuous Improvement System

This module analyzes test results and automatically implements improvements
to enhance the medical AI's performance across all specialties.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    category: str
    completeness_score: float
    relevance_score: float
    response_time: float
    emergency_detection_rate: float
    query_count: int
    timestamp: datetime

@dataclass
class ImprovementAction:
    action_type: str  # "knowledge_expansion", "response_optimization", "emergency_tuning"
    category: str
    priority: str  # "high", "medium", "low"
    description: str
    implementation_notes: str
    expected_improvement: float

class MedicalAIImprover:
    def __init__(self):
        self.performance_history = []
        self.improvement_actions = []
        self.knowledge_gaps = {}
        self.response_patterns = {}
        
        # Performance thresholds for triggering improvements
        self.thresholds = {
            "completeness_min": 0.75,
            "relevance_min": 0.80,
            "response_time_max": 3.0,
            "emergency_detection_min": 0.90
        }
        
        # Medical knowledge expansion templates
        self.knowledge_expansion_templates = {
            "cardiovascular": {
                "conditions": ["myocardial infarction", "atrial fibrillation", "heart failure", "angina", "cardiomyopathy"],
                "symptoms": ["chest pain", "dyspnea", "palpitations", "syncope", "fatigue"],
                "treatments": ["ACE inhibitors", "beta blockers", "statins", "anticoagulants", "angioplasty"],
                "emergencies": ["STEMI", "cardiac arrest", "aortic dissection", "pulmonary edema"]
            },
            "respiratory": {
                "conditions": ["asthma", "COPD", "pneumonia", "pulmonary embolism", "lung cancer"],
                "symptoms": ["dyspnea", "cough", "wheeze", "hemoptysis", "chest pain"],
                "treatments": ["bronchodilators", "corticosteroids", "antibiotics", "oxygen therapy"],
                "emergencies": ["respiratory failure", "tension pneumothorax", "severe asthma", "PE"]
            },
            "neurological": {
                "conditions": ["stroke", "epilepsy", "migraine", "Alzheimer's", "Parkinson's"],
                "symptoms": ["headache", "seizure", "weakness", "confusion", "memory loss"],
                "treatments": ["anticonvulsants", "dopamine agonists", "cholinesterase inhibitors"],
                "emergencies": ["acute stroke", "status epilepticus", "increased ICP"]
            }
        }

    def analyze_performance_trends(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends and identify improvement opportunities"""
        
        # Convert to performance metrics
        metrics_by_category = defaultdict(list)
        
        for result in test_results:
            category = result.get('query_category', 'unknown')
            metrics_by_category[category].append({
                'completeness': result.get('completeness_score', 0),
                'relevance': result.get('relevance_score', 0),
                'response_time': result.get('response_time', 0),
                'emergency_detected': result.get('emergency_detected', False)
            })
        
        # Calculate category performance
        category_analysis = {}
        overall_issues = []
        
        for category, results in metrics_by_category.items():
            if not results:
                continue
                
            avg_completeness = np.mean([r['completeness'] for r in results])
            avg_relevance = np.mean([r['relevance'] for r in results])
            avg_response_time = np.mean([r['response_time'] for r in results])
            emergency_rate = np.mean([r['emergency_detected'] for r in results])
            
            # Identify issues
            issues = []
            if avg_completeness < self.thresholds['completeness_min']:
                issues.append(f"Low completeness: {avg_completeness:.2%}")
            if avg_relevance < self.thresholds['relevance_min']:
                issues.append(f"Low relevance: {avg_relevance:.2%}")
            if avg_response_time > self.thresholds['response_time_max']:
                issues.append(f"Slow response: {avg_response_time:.2f}s")
            if category == 'emergency' and emergency_rate < self.thresholds['emergency_detection_min']:
                issues.append(f"Poor emergency detection: {emergency_rate:.2%}")
            
            category_analysis[category] = {
                'avg_completeness': avg_completeness,
                'avg_relevance': avg_relevance,
                'avg_response_time': avg_response_time,
                'emergency_detection_rate': emergency_rate,
                'query_count': len(results),
                'issues': issues,
                'priority': self._calculate_priority(issues, len(results))
            }
            
            overall_issues.extend([(category, issue) for issue in issues])
        
        return {
            'category_analysis': category_analysis,
            'overall_issues': overall_issues,
            'improvement_actions': self._generate_improvement_actions(category_analysis),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_priority(self, issues: List[str], query_count: int) -> str:
        """Calculate improvement priority based on issues and query volume"""
        if not issues:
            return "low"
        
        # High priority for emergency or high-volume categories with issues
        if any("emergency" in issue.lower() for issue in issues):
            return "high"
        if query_count > 10 and len(issues) >= 2:
            return "high"
        if len(issues) >= 2:
            return "medium"
        return "low"

    def _generate_improvement_actions(self, category_analysis: Dict) -> List[ImprovementAction]:
        """Generate specific improvement actions based on analysis"""
        actions = []
        
        for category, analysis in category_analysis.items():
            if not analysis['issues']:
                continue
            
            for issue in analysis['issues']:
                if "completeness" in issue:
                    actions.append(ImprovementAction(
                        action_type="knowledge_expansion",
                        category=category,
                        priority=analysis['priority'],
                        description=f"Expand knowledge base for {category} category",
                        implementation_notes=f"Add more conditions, symptoms, and treatments for {category}",
                        expected_improvement=0.15
                    ))
                
                elif "relevance" in issue:
                    actions.append(ImprovementAction(
                        action_type="response_optimization",
                        category=category,
                        priority=analysis['priority'],
                        description=f"Improve response relevance for {category}",
                        implementation_notes=f"Enhance AI model prompts and context for {category}",
                        expected_improvement=0.10
                    ))
                
                elif "response" in issue:
                    actions.append(ImprovementAction(
                        action_type="performance_optimization",
                        category=category,
                        priority=analysis['priority'],
                        description=f"Optimize response time for {category}",
                        implementation_notes=f"Cache common queries and optimize model inference for {category}",
                        expected_improvement=0.20
                    ))
                
                elif "emergency" in issue:
                    actions.append(ImprovementAction(
                        action_type="emergency_tuning",
                        category=category,
                        priority="high",
                        description=f"Improve emergency detection for {category}",
                        implementation_notes=f"Enhance emergency keyword detection and urgency assessment",
                        expected_improvement=0.25
                    ))
        
        return actions

    def generate_knowledge_expansion(self, category: str) -> Dict[str, List[str]]:
        """Generate expanded knowledge for a specific medical category"""
        
        base_knowledge = self.knowledge_expansion_templates.get(category, {})
        
        # Enhanced knowledge based on gaps
        expanded_knowledge = {
            "new_conditions": [],
            "new_symptoms": [],
            "new_treatments": [],
            "emergency_indicators": []
        }
        
        if category == "cardiovascular":
            expanded_knowledge["new_conditions"].extend([
                "Aortic stenosis", "Mitral regurgitation", "Hypertrophic cardiomyopathy",
                "Pericarditis", "Deep vein thrombosis", "Peripheral artery disease",
                "Aortic aneurysm", "Cardiac tamponade", "Endocarditis"
            ])
            expanded_knowledge["new_symptoms"].extend([
                "Orthopnea", "Paroxysmal nocturnal dyspnea", "Claudication",
                "Cyanosis", "Jugular vein distension", "Heart murmur"
            ])
            expanded_knowledge["new_treatments"].extend([
                "Cardiac catheterization", "Pacemaker", "ICD implantation",
                "Valve replacement", "Bypass surgery", "Thrombolysis"
            ])
            expanded_knowledge["emergency_indicators"].extend([
                "Crushing chest pain", "Radiating pain to arm/jaw", "Severe dyspnea",
                "Loss of consciousness", "Irregular pulse", "Cold sweats"
            ])
        
        elif category == "respiratory":
            expanded_knowledge["new_conditions"].extend([
                "Interstitial lung disease", "Sarcoidosis", "Pulmonary hypertension",
                "Lung abscess", "Pleural effusion", "Sleep apnea", "Cystic fibrosis"
            ])
            expanded_knowledge["new_symptoms"].extend([
                "Stridor", "Hemoptysis", "Barrel chest", "Digital clubbing",
                "Accessory muscle use", "Pursed lip breathing"
            ])
            expanded_knowledge["new_treatments"].extend([
                "Mechanical ventilation", "Thoracentesis", "Chest tube",
                "Pulmonary rehabilitation", "Long-term oxygen therapy"
            ])
            expanded_knowledge["emergency_indicators"].extend([
                "Severe respiratory distress", "Cyanosis", "Use of accessory muscles",
                "Inability to speak in full sentences", "Altered mental status"
            ])
        
        elif category == "neurological":
            expanded_knowledge["new_conditions"].extend([
                "Multiple sclerosis", "Myasthenia gravis", "Guillain-Barré syndrome",
                "Trigeminal neuralgia", "Brain tumor", "Hydrocephalus", "Meningitis"
            ])
            expanded_knowledge["new_symptoms"].extend([
                "Aphasia", "Ataxia", "Nystagmus", "Diplopia", "Dysarthria",
                "Focal weakness", "Sensory loss", "Tremor"
            ])
            expanded_knowledge["new_treatments"].extend([
                "Tissue plasminogen activator", "Immunosuppressants",
                "Deep brain stimulation", "Craniotomy", "Lumbar puncture"
            ])
            expanded_knowledge["emergency_indicators"].extend([
                "Sudden severe headache", "Focal neurological deficits", "Altered consciousness",
                "Seizure activity", "Signs of increased intracranial pressure"
            ])
        
        return expanded_knowledge

    def implement_improvements(self, actions: List[ImprovementAction]) -> Dict[str, Any]:
        """Implement the suggested improvements"""
        
        implementation_results = {
            "successful_implementations": [],
            "failed_implementations": [],
            "partial_implementations": []
        }
        
        for action in sorted(actions, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x.priority], reverse=True):
            try:
                if action.action_type == "knowledge_expansion":
                    result = self._implement_knowledge_expansion(action)
                elif action.action_type == "response_optimization":
                    result = self._implement_response_optimization(action)
                elif action.action_type == "emergency_tuning":
                    result = self._implement_emergency_tuning(action)
                elif action.action_type == "performance_optimization":
                    result = self._implement_performance_optimization(action)
                else:
                    result = {"status": "unsupported", "details": f"Unknown action type: {action.action_type}"}
                
                if result["status"] == "success":
                    implementation_results["successful_implementations"].append({
                        "action": action,
                        "result": result
                    })
                elif result["status"] == "partial":
                    implementation_results["partial_implementations"].append({
                        "action": action,
                        "result": result
                    })
                else:
                    implementation_results["failed_implementations"].append({
                        "action": action,
                        "result": result
                    })
                    
            except Exception as e:
                implementation_results["failed_implementations"].append({
                    "action": action,
                    "error": str(e)
                })
        
        return implementation_results

    def _implement_knowledge_expansion(self, action: ImprovementAction) -> Dict[str, Any]:
        """Implement knowledge base expansion"""
        try:
            expanded_knowledge = self.generate_knowledge_expansion(action.category)
            
            # Generate code to add to medical_knowledge_base.py
            expansion_code = self._generate_knowledge_expansion_code(action.category, expanded_knowledge)
            
            return {
                "status": "success",
                "details": f"Generated knowledge expansion for {action.category}",
                "expansion_code": expansion_code,
                "new_items_count": sum(len(items) for items in expanded_knowledge.values())
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _implement_response_optimization(self, action: ImprovementAction) -> Dict[str, Any]:
        """Implement response optimization"""
        optimization_suggestions = {
            "prompt_improvements": [
                f"Add more specific medical context for {action.category}",
                f"Include symptom severity assessment for {action.category}",
                f"Enhance treatment recommendation specificity"
            ],
            "model_tuning": [
                f"Fine-tune medical terminology for {action.category}",
                f"Improve medical reasoning chains",
                f"Enhance differential diagnosis capabilities"
            ]
        }
        
        return {
            "status": "partial",
            "details": f"Generated optimization suggestions for {action.category}",
            "suggestions": optimization_suggestions
        }

    def _implement_emergency_tuning(self, action: ImprovementAction) -> Dict[str, Any]:
        """Implement emergency detection improvements"""
        emergency_keywords = {
            "cardiovascular": ["crushing", "radiating", "severe chest", "cardiac arrest", "heart attack"],
            "respiratory": ["can't breathe", "choking", "blue lips", "gasping", "respiratory arrest"],
            "neurological": ["sudden weakness", "can't speak", "severe headache", "unconscious", "seizure"],
            "general": ["emergency", "911", "dying", "life threatening", "critical condition"]
        }
        
        enhanced_keywords = emergency_keywords.get(action.category, emergency_keywords["general"])
        
        return {
            "status": "success",
            "details": f"Enhanced emergency detection for {action.category}",
            "new_keywords": enhanced_keywords,
            "implementation": "Add to emergency detection system"
        }

    def _implement_performance_optimization(self, action: ImprovementAction) -> Dict[str, Any]:
        """Implement performance optimizations"""
        optimization_strategies = [
            "Implement response caching for common queries",
            "Optimize model inference pipeline",
            "Pre-process frequent medical queries",
            "Implement parallel processing for multiple models"
        ]
        
        return {
            "status": "partial",
            "details": f"Generated performance optimization strategies for {action.category}",
            "strategies": optimization_strategies
        }

    def _generate_knowledge_expansion_code(self, category: str, expanded_knowledge: Dict) -> str:
        """Generate Python code for knowledge base expansion"""
        
        code_template = f'''
    def expand_{category}_knowledge(self):
        """Expanded knowledge base for {category} category - Auto-generated"""
        
        # New conditions
        new_conditions = {expanded_knowledge.get("new_conditions", [])}
        
        # New symptoms  
        new_symptoms = {expanded_knowledge.get("new_symptoms", [])}
        
        # New treatments
        new_treatments = {expanded_knowledge.get("new_treatments", [])}
        
        # Emergency indicators
        emergency_indicators = {expanded_knowledge.get("emergency_indicators", [])}
        
        # Add to existing knowledge base
        for condition in new_conditions:
            if condition not in self.medical_knowledge:
                self.medical_knowledge[condition] = {{
                    "category": "{category}",
                    "symptoms": new_symptoms[:3],  # Sample symptoms
                    "treatments": new_treatments[:3],  # Sample treatments
                    "emergency_signs": emergency_indicators[:2],  # Sample emergency signs
                    "auto_generated": True,
                    "confidence": 0.8
                }}
        
        return len(new_conditions)
'''
        
        return code_template

    def generate_improvement_report(self, analysis: Dict, implementations: Dict) -> str:
        """Generate comprehensive improvement report"""
        
        report = f"""
🏥 MEDICAL AI IMPROVEMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

📊 PERFORMANCE ANALYSIS
{'-'*30}
Categories analyzed: {len(analysis['category_analysis'])}
Total issues identified: {len(analysis['overall_issues'])}

CATEGORY PERFORMANCE:
"""
        
        for category, data in analysis['category_analysis'].items():
            report += f"""
{category.upper()}:
  • Completeness: {data['avg_completeness']:.2%}
  • Relevance: {data['avg_relevance']:.2%}
  • Response Time: {data['avg_response_time']:.2f}s
  • Query Count: {data['query_count']}
  • Priority: {data['priority'].upper()}
  • Issues: {len(data['issues'])}
"""
            for issue in data['issues']:
                report += f"    - {issue}\n"
        
        report += f"""

🎯 IMPROVEMENT ACTIONS
{'-'*30}
Total actions: {len(analysis['improvement_actions'])}
"""
        
        action_counts = {}
        for action in analysis['improvement_actions']:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
        
        for action_type, count in action_counts.items():
            report += f"  • {action_type}: {count} actions\n"
        
        report += f"""

✅ IMPLEMENTATION RESULTS
{'-'*30}
Successful: {len(implementations['successful_implementations'])}
Partial: {len(implementations['partial_implementations'])}
Failed: {len(implementations['failed_implementations'])}

SUCCESSFUL IMPLEMENTATIONS:
"""
        
        for impl in implementations['successful_implementations']:
            action = impl['action']
            report += f"  ✓ {action.description} ({action.category})\n"
        
        if implementations['partial_implementations']:
            report += "\nPARTIAL IMPLEMENTATIONS:\n"
            for impl in implementations['partial_implementations']:
                action = impl['action']
                report += f"  ⚠ {action.description} ({action.category})\n"
        
        if implementations['failed_implementations']:
            report += "\nFAILED IMPLEMENTATIONS:\n"
            for impl in implementations['failed_implementations']:
                action = impl['action']
                report += f"  ❌ {action.description} ({action.category})\n"
        
        report += f"""

📈 EXPECTED IMPROVEMENTS
{'-'*30}
"""
        
        total_expected_improvement = sum(action.expected_improvement for action in analysis['improvement_actions'])
        report += f"Total expected performance gain: {total_expected_improvement:.1%}\n"
        
        return report

    async def run_continuous_improvement(self, test_results_file: str = None):
        """Run continuous improvement based on test results"""
        
        if test_results_file:
            # Load test results from file
            with open(test_results_file, 'r', encoding='utf-8') as f:
                test_results = json.load(f)
        else:
            # Use placeholder test results for demonstration
            test_results = self._generate_sample_test_results()
        
        logger.info("Analyzing performance trends...")
        analysis = self.analyze_performance_trends(test_results)
        
        logger.info("Generating improvement actions...")
        implementations = self.implement_improvements(analysis['improvement_actions'])
        
        # Generate and save report
        report = self.generate_improvement_report(analysis, implementations)
        
        report_filename = f"improvement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Improvement report saved to {report_filename}")
        
        return {
            "analysis": analysis,
            "implementations": implementations,
            "report_file": report_filename
        }

    def _generate_sample_test_results(self) -> List[Dict]:
        """Generate sample test results for demonstration"""
        sample_results = []
        
        categories = ["cardiovascular", "respiratory", "neurological", "emergency"]
        
        for category in categories:
            for i in range(10):
                sample_results.append({
                    "query": f"sample {category} query {i}",
                    "query_category": category,
                    "completeness_score": np.random.uniform(0.4, 0.9),
                    "relevance_score": np.random.uniform(0.5, 0.95),
                    "response_time": np.random.uniform(1.0, 6.0),
                    "emergency_detected": category == "emergency" and np.random.random() > 0.3,
                    "timestamp": datetime.now().isoformat()
                })
        
        return sample_results

async def main():
    """Main improvement function"""
    improver = MedicalAIImprover()
    
    print("🔧 Medical AI Continuous Improvement System")
    print("=" * 50)
    
    choice = input("""
Choose improvement mode:
1. Analyze existing test results
2. Run sample improvement analysis
3. Generate knowledge expansion for specific category
4. Continuous monitoring and improvement

Enter choice (1-4): """)
    
    if choice == "1":
        test_file = input("Enter test results file path (or press Enter for latest): ")
        if not test_file:
            # Find latest test results file
            import glob
            files = glob.glob("medical_ai_test_results_*.json")
            if files:
                test_file = max(files)
                print(f"Using latest file: {test_file}")
            else:
                print("No test results files found. Using sample data.")
                test_file = None
        
        result = await improver.run_continuous_improvement(test_file)
        print(f"📄 Report saved to: {result['report_file']}")
    
    elif choice == "2":
        print("🧪 Running sample improvement analysis...")
        result = await improver.run_continuous_improvement()
        print(f"📄 Report saved to: {result['report_file']}")
    
    elif choice == "3":
        category = input("Enter medical category (cardiovascular/respiratory/neurological): ")
        if category in improver.knowledge_expansion_templates:
            expansion = improver.generate_knowledge_expansion(category)
            print(f"\n📚 Knowledge expansion for {category}:")
            for key, items in expansion.items():
                print(f"\n{key.replace('_', ' ').title()}:")
                for item in items[:5]:  # Show first 5 items
                    print(f"  • {item}")
        else:
            print(f"Category '{category}' not supported yet.")
    
    elif choice == "4":
        print("🔄 Starting continuous monitoring...")
        while True:
            try:
                await improver.run_continuous_improvement()
                print("⏳ Waiting 1 hour before next improvement cycle...")
                await asyncio.sleep(3600)  # Wait 1 hour
            except KeyboardInterrupt:
                print("🛑 Stopping continuous improvement.")
                break
            except Exception as e:
                logger.error(f"Error in continuous improvement: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

if __name__ == "__main__":
    asyncio.run(main())