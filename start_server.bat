@echo off
echo.
echo ===============================================
echo   Medical AI Chatbot - Starting Server
echo ===============================================
echo.

REM Check if virtual environment exists
if exist "medical_ai_env\Scripts\python.exe" (
    echo Using virtual environment: medical_ai_env
    set PYTHON_CMD=medical_ai_env\Scripts\python.exe
) else (
    echo Using system Python
    set PYTHON_CMD=python
)

echo.
echo Starting FastAPI server...
echo Frontend will be available at: http://localhost:8000/frontend
echo API docs will be available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
%PYTHON_CMD% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo Server stopped.
pause