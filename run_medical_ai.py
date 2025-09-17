"""
Medical AI - Easy Test & Improve Runner

This script makes it super easy to test and improve the medical AI system.
Just run this and choose what you want to do!
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'requests', 'pandas', 'matplotlib', 'seaborn', 
        'rich', 'numpy', 'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("🔧 Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("✅ All packages installed!")

def run_medical_ai_server():
    """Start the medical AI server"""
    print("🚀 Starting Medical AI Server...")
    print("This will start the FastAPI server with the 500+ disease database")
    print("Server will run at: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start server. Make sure main.py exists.")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")

def run_quick_test():
    """Run a quick test of the medical AI"""
    print("⚡ Running Quick Test...")
    print("This will test a few medical queries to check if everything works")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_tester.py'], input='3\n', text=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Quick test failed. Make sure the server is running first.")

def run_comprehensive_test():
    """Run comprehensive testing"""
    print("🔬 Running Comprehensive Test...")
    print("This will test hundreds of medical queries across all specialties")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_tester.py'], input='1\n', text=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Comprehensive test failed. Make sure the server is running first.")

def run_continuous_testing():
    """Run continuous testing (every 30 minutes)"""
    print("🔄 Starting Continuous Testing...")
    print("This will test the AI every 30 minutes and keep improving it")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_tester.py'], input='2\n', text=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Continuous testing failed.")
    except KeyboardInterrupt:
        print("\n🛑 Continuous testing stopped by user.")

def run_performance_dashboard():
    """Run the performance dashboard"""
    print("📊 Starting Performance Dashboard...")
    print("This shows real-time performance metrics and testing results")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_dashboard.py'], check=True)
    except subprocess.CalledProcessError:
        print("❌ Dashboard failed to start.")
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user.")

def run_improvement_analysis():
    """Run improvement analysis"""
    print("🎯 Running Improvement Analysis...")
    print("This analyzes test results and suggests improvements")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_improver.py'], input='2\n', text=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Improvement analysis failed.")

def test_emergency_detection():
    """Test emergency detection specifically"""
    print("🚨 Testing Emergency Detection...")
    print("This tests if the AI properly detects medical emergencies")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_tester.py'], input='4\n', text=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Emergency detection test failed.")

def test_hindi_support():
    """Test Hindi language support"""
    print("🌐 Testing Hindi Language Support...")
    print("This tests if the AI can handle Hindi medical queries")
    
    try:
        subprocess.run([sys.executable, 'medical_ai_tester.py'], input='5\n', text=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Hindi language test failed.")

def show_system_info():
    """Show information about the medical AI system"""
    print("\n" + "="*60)
    print("🏥 MEDICAL AI SYSTEM INFORMATION")
    print("="*60)
    print("📚 Database: 500+ diseases across all medical specialties")
    print("🤖 AI Models: BioBERT, ClinicalBERT, PubMedBERT, Medical NER")
    print("🌐 Languages: English, Hindi (with auto-detection)")
    print("⚡ Features:")
    print("  • AI-driven responses (no hardcoded answers)")
    print("  • Emergency detection and urgent care routing")
    print("  • Multilingual support with translation")
    print("  • Continuous testing and improvement")
    print("  • Real-time performance monitoring")
    print("  • 24/7 automated quality assurance")
    print("\n📂 Files:")
    print("  • main.py - FastAPI server with medical AI")
    print("  • medical_ai_tester.py - Comprehensive testing system")
    print("  • medical_ai_improver.py - Continuous improvement system") 
    print("  • medical_ai_dashboard.py - Performance monitoring dashboard")
    print("  • chatbot/ - Medical AI modules and knowledge base")
    print("\n🎯 Covered Medical Specialties:")
    specialties = [
        "Cardiovascular", "Respiratory", "Neurological", "Gastrointestinal",
        "Endocrine", "Infectious Diseases", "Orthopedic", "Dermatology",
        "Psychiatry", "Pediatrics", "Gynecology", "Urology", "Oncology",
        "Hematology", "Rheumatology", "Ophthalmology", "ENT", "Emergency",
        "Tropical Medicine", "Rare Diseases"
    ]
    
    for i, specialty in enumerate(specialties, 1):
        print(f"  {i:2d}. {specialty}")
    
    print("="*60)

def main():
    """Main menu for easy access to all functionality"""
    
    print("\n" + "🏥" * 20)
    print("🏥 MEDICAL AI - EASY TEST & IMPROVE SYSTEM 🏥")
    print("🏥" + " " * 36 + "🏥")
    print("🏥  500+ Diseases • AI-Driven • Continuous Learning  🏥")
    print("🏥" * 20)
    
    while True:
        print("\n" + "="*50)
        print("🎯 WHAT WOULD YOU LIKE TO DO?")
        print("="*50)
        
        options = [
            ("1", "🚀 Start Medical AI Server", "Start the main AI server"),
            ("2", "⚡ Quick Test", "Test a few medical queries"),
            ("3", "🔬 Comprehensive Test", "Test hundreds of medical queries"),
            ("4", "🔄 Continuous Testing", "Keep testing every 30 minutes"),
            ("5", "📊 Performance Dashboard", "Real-time monitoring dashboard"),
            ("6", "🎯 Improvement Analysis", "Analyze and improve performance"),
            ("7", "🚨 Emergency Detection Test", "Test emergency response"),
            ("8", "🌐 Hindi Language Test", "Test multilingual support"),
            ("9", "ℹ️  System Information", "Learn about the system"),
            ("0", "🚪 Exit", "Quit this program")
        ]
        
        for code, title, desc in options:
            print(f"  {code}. {title:<25} - {desc}")
        
        print("="*50)
        choice = input("👉 Enter your choice (0-9): ").strip()
        
        if choice == "0":
            print("\n👋 Thanks for using Medical AI Test & Improve System!")
            print("🏥 Keep making healthcare AI better!")
            break
            
        elif choice == "1":
            check_dependencies()
            run_medical_ai_server()
            
        elif choice == "2":
            run_quick_test()
            
        elif choice == "3":
            run_comprehensive_test()
            
        elif choice == "4":
            run_continuous_testing()
            
        elif choice == "5":
            check_dependencies()
            run_performance_dashboard()
            
        elif choice == "6":
            run_improvement_analysis()
            
        elif choice == "7":
            test_emergency_detection()
            
        elif choice == "8":
            test_hindi_support()
            
        elif choice == "9":
            show_system_info()
            
        else:
            print("❌ Invalid choice. Please enter a number from 0-9.")
        
        if choice != "0" and choice != "9":
            input("\n📋 Press Enter to return to main menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try running individual components separately if this continues.")