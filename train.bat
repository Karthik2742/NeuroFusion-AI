@echo off
TITLE NeuroFusion AI - Brain MRI Model Trainer
COLOR 0B
SETLOCAL EnableDelayedExpansion

echo ============================================================
echo   NeuroFusion AI - Training All Brain MRI Models
echo   (UNet, Attention UNet, UNet++, Restormer, SwinIR, MIRNet)
echo ============================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if exist "%ROOT_DIR%.venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment .venv...
    call "%ROOT_DIR%.venv\Scripts\activate.bat"
) else if exist "%ROOT_DIR%venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment venv...
    call "%ROOT_DIR%venv\Scripts\activate.bat"
)

echo.
echo Launching full Brain MRI training pipeline...
echo.
python -m backend.training.train_all_brain_models %*

echo.
echo ============================================================
echo   Training complete! All weights saved to ./weights
echo ============================================================
pause
