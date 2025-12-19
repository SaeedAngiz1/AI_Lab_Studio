#!/bin/bash

# AI Lab Studio - Quick Start Script

echo "🧪 AI Lab Studio - Starting Application"
echo "======================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"

# Check if requirements are installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "🚀 Starting Streamlit application..."
echo ""
echo "📝 Note: The application will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

# Start Streamlit
streamlit run app.py
