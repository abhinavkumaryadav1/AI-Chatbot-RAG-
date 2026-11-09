@echo off
echo ========================================================
echo   SmartRAG - One-Click Setup Script
echo ========================================================
echo.

REM 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b
)

echo [*] Creating virtual environment (venv)...
python -m venv venv

echo [*] Activating virtual environment...
call .\venv\Scripts\activate.bat

echo [*] Upgrading pip and installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================================
echo   Setup Complete! You can now run "run.bat"
echo ========================================================
pause

