@echo off
echo Starting Medical AI Chatbot Server...
echo =====================================
echo.
echo Opening medical AI chatbot with ChatGPT-like interface
echo Server will start at: http://localhost:8000
echo Chat interface at: http://localhost:8000/chat
echo.

cd /d "c:\Users\arshd\OneDrive\Documents\GitHub\SIH-Git"

REM Activate virtual environment
call medical_ai_env\Scripts\activate.bat

REM Start the FastAPI server
echo Starting server...
python main.py

pause