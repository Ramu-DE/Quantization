@echo off
REM Start Quantization Visualizer - Windows
REM Usage: double-click start.bat or run from command prompt

echo =========================================
echo   Quantization Visualizer - Starting
echo =========================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check Node
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)

REM Check uv
uv --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing uv...
    pip install uv
)

SET SCRIPT_DIR=%~dp0

REM Start Backend
echo [1/2] Starting Backend (port 8000)...
cd /d "%SCRIPT_DIR%backend"
start "Quantization Backend" /B cmd /c "uv sync --quiet && uv run uvicorn main:app --host 0.0.0.0 --port 8000"
echo       Backend starting...

REM Wait for backend
timeout /t 8 /nobreak >nul
echo       Checking backend...
curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo       Backend ready!
) else (
    echo       Backend still loading (first run downloads model, may take a minute)...
)

REM Start Frontend
echo.
echo [2/2] Starting Frontend (port 3000)...
cd /d "%SCRIPT_DIR%"
if not exist "node_modules" (
    echo       Installing dependencies...
    call npm install
)
start "Quantization Frontend" /B cmd /c "npm run dev"

timeout /t 8 /nobreak >nul

echo.
echo =========================================
echo   Ready!
echo   UI:       http://localhost:3000
echo   API:      http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo =========================================
echo.
echo Press any key to stop services...
pause >nul

REM Cleanup
taskkill /FI "WINDOWTITLE eq Quantization Backend" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Quantization Frontend" /F >nul 2>&1
echo Stopped.
