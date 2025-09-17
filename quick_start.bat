@echo off
title Medical AI - Quick Start
color 0A

echo.
echo ===============================================
echo    MEDICAL AI - QUICK START LAUNCHER
echo ===============================================
echo.
echo 🏥 Starting Medical AI Test ^& Improve System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo 🔧 Installing required packages...

REM Install required packages
pip install fastapi uvicorn requests pandas matplotlib seaborn rich numpy transformers torch sentence-transformers chromadb sqlalchemy

echo.
echo 🚀 Launching Medical AI Runner...
echo.

python run_medical_ai.py

pause