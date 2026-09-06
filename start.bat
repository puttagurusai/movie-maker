@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  Avatar start
echo  - web server  : http://localhost:8000
echo  - ws bridge   : ws://localhost:8765
echo  - agents      : orchestrator_agents.py
echo ============================================
echo.

REM 1) Static files for face_viewer.html
start "avatar-http" cmd /k "cd /d "%~dp0web_viewer" && python -m http.server 8000"

REM 2) UDP (agents) -> WebSocket (browser)
start "avatar-ws" cmd /k "cd /d "%~dp0web_viewer" && python ws_bridge.py"

REM Give servers a moment to bind ports
timeout /t 2 /nobreak >nul

REM 3) Open viewer (hard-refresh with Ctrl+Shift+R if cache is stale)
start "" "http://localhost:8000/face_viewer.html"

echo.
echo Browser opened. This window runs the agents.
echo When prompted, paste JSON or type:
echo   @temp/director_actions_test.json
echo.
echo Close the other two windows to stop HTTP / WebSocket.
echo ============================================
echo.

REM 4) Agents pipeline (interactive — stays in this window)
python orchestrator_agents.py

echo.
echo Agents exited.
pause
