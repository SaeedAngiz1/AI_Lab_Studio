@echo off
REM AI Lab Studio - Quick Start Script (Windows)

echo 🧪 AI Lab Studio - Starting Application
echo =======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✓ Python found

REM Check if requirements are installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    echo ✓ Dependencies installed
) else (
    echo ✓ Dependencies already installed
)

echo.
echo 🚀 Starting Streamlit application...
echo.
echo 📝 Note: The application will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

REM Start Streamlit
streamlit run app.py
