@echo off
TITLE NeuroFusion AI Launcher
COLOR 0A
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo          NeuroFusion AI - Launching All Services
echo ===================================================
echo.

:: Get script directory
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: Check for Python virtual environments
set "PYTHON_CMD=python"
if exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    echo [INFO] Found virtual environment in .venv
    set "PYTHON_CMD=%ROOT_DIR%.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%venv\Scripts\python.exe" (
    echo [INFO] Found virtual environment in venv
    set "PYTHON_CMD=%ROOT_DIR%venv\Scripts\python.exe"
)

echo [1/2] Launching Backend FastAPI Server (http://localhost:8000)...
start "NeuroFusion AI - Backend" /D "%ROOT_DIR%" cmd /k "%PYTHON_CMD%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

echo [2/2] Launching Frontend Vite Dev Server (http://localhost:5173)...
start "NeuroFusion AI - Frontend" /D "%ROOT_DIR%frontend" cmd /k npm run dev

echo.
echo ===================================================
echo   Services are running in separate command windows:
echo   - Backend:  http://localhost:8000 (Docs: http://localhost:8000/docs)
echo   - Frontend: http://localhost:5173
echo ===================================================
echo.
