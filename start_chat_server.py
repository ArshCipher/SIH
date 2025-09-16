#!/usr/bin/env python3
"""
Medical AI Chatbot Server Launcher
Simple script to start the FastAPI server with the frontend interface
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def main():
    print("🏥 Starting Medical AI Chatbot Server...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if templates directory exists
    if not os.path.exists("templates"):
        print("📁 Creating templates directory...")
        os.makedirs("templates", exist_ok=True)
    
    # Check if chat.html exists
    if not os.path.exists("templates/chat.html"):
        print("❌ Error: templates/chat.html not found. Please ensure the frontend file exists.")
        sys.exit(1)
    
    print("✅ All files found!")
    print("🚀 Starting server on http://localhost:8000")
    print("🌐 Frontend available at: http://localhost:8000/frontend")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    process = None
    try:
        # Start the server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ]
        
        # Start server in background
        process = subprocess.Popen(cmd)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser to frontend
        try:
            webbrowser.open("http://localhost:8000/frontend")
            print("🌐 Browser opened to chat interface")
        except:
            print("🌐 Please open http://localhost:8000/frontend in your browser")
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        if process:
            process.terminate()
        print("✅ Server stopped successfully")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        if process:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()