@echo off
REM Start Chromium RAG HTTP Server for remote connections

echo ========================================
echo Chromium RAG - HTTP/WebSocket Server
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

REM Check if server script exists
if not exist "scripts\rag_http_server.py" (
    echo [ERROR] Server script not found at: scripts\rag_http_server.py
    echo Please ensure you're running this from the package root directory
    pause
    exit /b 1
)

echo Starting HTTP/WebSocket server...
echo.
echo Server endpoints:
echo   - HTTP API: http://localhost:8080/api/search
echo   - WebSocket: ws://localhost:8080/ws
echo   - MCP WebSocket: ws://localhost:8080/mcp
echo.
echo For remote access, replace 'localhost' with your server's IP address
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python scripts\rag_http_server.py --host 0.0.0.0 --port 8080

pause
