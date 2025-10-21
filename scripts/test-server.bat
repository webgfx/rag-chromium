@echo off
REM Test Chromium RAG Server

echo ========================================
echo Chromium RAG - Server Test
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo Testing MCP server startup...
echo This will take 45-60 seconds for first-time warmup
echo Press Ctrl+C to stop the test
echo.
echo Starting server...
echo.

python scripts\rag_mcp_server.py

pause
