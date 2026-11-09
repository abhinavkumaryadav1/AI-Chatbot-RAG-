@echo off
echo ========================================================
echo   Starting SmartRAG - Document-Based AI Chatbot
echo ========================================================
echo.

REM 1. Activate venv
if exist ".\venv\Scripts\activate.bat" (
    call .\venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found! Please run setup.bat first.
    pause
    exit /b
)

REM 2. Check / Start Ollama
echo [*] Checking Ollama service...
powershell -Command "if (!(Get-Process -Name ollama -ErrorAction SilentlyContinue)) { Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden }" >nul 2>&1

REM 3. Launch Streamlit
echo [*] Launching Streamlit web interface...
echo App will open in your browser at http://localhost:8501
streamlit run app.py

