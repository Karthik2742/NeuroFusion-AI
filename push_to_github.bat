@echo off
TITLE NeuroFusion AI - GitHub Push Helper
COLOR 0A
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo     Push NeuroFusion AI to GitHub Repository
echo ===================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

echo Current Git Status:
git status
echo.

set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/username/neurofusion-ai.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No repository URL entered. Exiting.
    pause
    exit /b
)

echo.
echo [1/3] Setting remote origin to: %REPO_URL%
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo [2/3] Setting main branch...
git branch -M main

echo [3/3] Pushing to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo [SUCCESS] Code successfully pushed to GitHub!
    echo ===================================================
) else (
    echo.
    echo [NOTE] If prompted by GitHub, sign in via your browser window.
)

pause
