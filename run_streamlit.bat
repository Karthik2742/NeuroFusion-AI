@echo off
TITLE NeuroFusion AI - Streamlit Web Application
COLOR 0B
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo   NeuroFusion AI & BrainSight - Streamlit Launcher
echo ===================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PYTHON_CMD=python"
if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    echo [INFO] Using virtual environment in .venv
    set "PYTHON_CMD=%ROOT_DIR%.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%venv\Scripts\python.exe" (
    echo [INFO] Using virtual environment in venv
    set "PYTHON_CMD=%ROOT_DIR%venv\Scripts\python.exe"
)

echo [INFO] Launching Streamlit on http://localhost:8501 ...
"%PYTHON_CMD%" -m streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause
