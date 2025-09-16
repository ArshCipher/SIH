#!/bin/bash

# Public Health Chatbot Startup Script

echo "🏥 Starting Public Health Chatbot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model if spaCy is installed
if python - <<'PY'
try:
    import importlib
    found = __import__('importlib').import_module('importlib')
    import importlib.machinery  # ensure importlib is available
    import pkgutil
    has_spacy = pkgutil.find_loader('spacy') is not None
    raise SystemExit(0 if has_spacy else 1)
except Exception:
    raise SystemExit(1)
PY
then
  echo "🧠 Downloading spaCy English model..."
  python -m spacy download en_core_web_sm || echo "Skipping spaCy model download"
else
  echo "ℹ️  spaCy not installed; skipping model download (optional)."
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp config.env.example .env
    echo "📝 Please edit .env file with your API credentials before running the application."
fi

# Initialize database
echo "🗄️  Initializing database..."
python -c "from chatbot.database import DatabaseManager; DatabaseManager()"

# Run tests
echo "🧪 Running tests..."
python -m pytest test_chatbot.py -v

# Start the application
echo "🚀 Starting the application..."
echo "📱 WhatsApp webhook: http://localhost:8000/whatsapp/webhook"
echo "📱 SMS webhook: http://localhost:8000/sms/webhook"
echo "📊 Analytics dashboard: http://localhost:8000/analytics/dashboard/html"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""
echo "✅ Application is starting..."

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
