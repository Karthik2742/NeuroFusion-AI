@echo off
TITLE Open NeuroFusion AI in Google Chrome
echo Opening NeuroFusion AI on http://localhost:8501 in Google Chrome...

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://localhost:8501"
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "http://localhost:8501"
) else (
    start "" "http://localhost:8501"
)
