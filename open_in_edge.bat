@echo off
TITLE Open NeuroFusion AI in Microsoft Edge
echo Opening NeuroFusion AI on http://localhost:8501 in Microsoft Edge...

if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "http://localhost:8501"
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" "http://localhost:8501"
) else (
    start "" "http://localhost:8501"
)
