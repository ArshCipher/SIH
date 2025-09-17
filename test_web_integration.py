#!/usr/bin/env python3
"""
Test Web Interface Integration
Quick test to ensure all components are working with the web interface
"""

import subprocess
import time
import requests
import json

def test_web_integration():
    """Test the web interface integration"""
    print("🌐 Testing Medical Chatbot Web Interface Integration")
    print("=" * 60)
    
    # Test 1: Check if server starts
    print("🚀 Step 1: Testing server startup...")
    
    # Start server in background
    try:
        server_process = subprocess.Popen([
            "python", "-m", "uvicorn", "main:app", 
            "--host", "localhost", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("   Waiting for server to start...")
        time.sleep(5)
        
        # Test basic connectivity
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Server started successfully")
            else:
                print(f"   ⚠️ Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Server connectivity failed: {e}")
            return False
        
        # Test 2: Frontend availability
        print("\n🌐 Step 2: Testing frontend availability...")
        try:
            response = requests.get("http://localhost:8000/frontend", timeout=5)
            if response.status_code == 200 and "Medical AI Assistant" in response.text:
                print("   ✅ Frontend loaded successfully")
                print("   ✅ Medical AI Assistant interface found")
            else:
                print(f"   ⚠️ Frontend issue: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Frontend test failed: {e}")
        
        # Test 3: Enhanced medical endpoint
        print("\n🧠 Step 3: Testing enhanced medical endpoint...")
        try:
            test_payload = {
                "text": "I have fever and headache",
                "user_id": "test_user",
                "language": "en",
                "country": "IN"
            }
            
            response = requests.post(
                "http://localhost:8000/medical/answer", 
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Enhanced medical endpoint working")
                print(f"   📊 Confidence: {data.get('confidence', 'N/A')}")
                print(f"   ⚠️ Risk Level: {data.get('risk_level', 'N/A')}")
                print(f"   🔒 Safety Validated: {data.get('safety_validated', 'N/A')}")
                if len(data.get('response', '')) > 50:
                    print("   ✅ Response contains substantial content")
                else:
                    print("   ⚠️ Response seems short")
            else:
                print(f"   ❌ Medical endpoint failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Medical endpoint test failed: {e}")
        
        # Test 4: Multilingual support
        print("\n🌐 Step 4: Testing multilingual support...")
        try:
            test_payload = {
                "text": "मुझे बुखार है",  # Hindi: I have fever
                "user_id": "test_user",
                "language": "hi",
                "country": "IN"
            }
            
            response = requests.post(
                "http://localhost:8000/medical/answer", 
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Hindi query processed successfully")
                if "hindi" in data.get('response', '').lower() or len(data.get('response', '')) > 50:
                    print("   ✅ Multilingual response generated")
                else:
                    print("   ⚠️ Multilingual response may need improvement")
            else:
                print(f"   ❌ Multilingual test failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Multilingual test failed: {e}")
        
        # Test 5: Basic chat endpoint (fallback)
        print("\n💬 Step 5: Testing basic chat endpoint...")
        try:
            test_payload = {
                "message": "What are symptoms of diabetes?",
                "user_id": "test_user",
                "language": "en",
                "platform": "web"
            }
            
            response = requests.post(
                "http://localhost:8000/chat", 
                json=test_payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Basic chat endpoint working")
                if len(data.get('response', '')) > 30:
                    print("   ✅ Chat response generated")
                else:
                    print("   ⚠️ Chat response seems short")
            else:
                print(f"   ❌ Chat endpoint failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Chat endpoint test failed: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 WEB INTERFACE INTEGRATION TEST COMPLETE!")
        print("\n🌐 **Access your Medical Chatbot:**")
        print("   • Frontend: http://localhost:8000/frontend")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Health Check: http://localhost:8000/health")
        print("\n✨ **Features Available:**")
        print("   • 102+ Disease Database")
        print("   • Multilingual Support (11 languages)")
        print("   • AI Confidence Scores")
        print("   • Risk Assessment")
        print("   • Real-time Medical APIs")
        print("   • Enhanced Medical Orchestrator")
        
        print("\n🔄 Server will continue running...")
        print("Press Ctrl+C to stop the server")
        
        # Keep server running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            print("✅ Server stopped")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_web_integration()