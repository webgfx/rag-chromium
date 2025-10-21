@echo off
REM Start Chromium RAG MCP Server
REM This keeps the server running in the background

echo ========================================
echo Chromium RAG - Start MCP Server
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or higher
    exit /b 1
)

REM Check if server is already running
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [WARNING] Python processes are already running
    echo If the server is already running, you don't need to start it again
    echo.
)

REM Check if server script exists
if not exist "scripts\rag_mcp_server.py" (
    echo [ERROR] Server script not found at: scripts\rag_mcp_server.py
    echo Please ensure you're running this from the package root directory
    exit /b 1
)

echo Starting MCP server...
echo This will run in the foreground. Keep this window open.
echo.
echo Server starting in 3 seconds...
timeout /t 3 >nul
echo.
echo ========================================
echo Server is now running
echo ========================================
echo.
echo Keep this window open while using @rag-chromium in VS Code
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python scripts\rag_mcp_server.py
