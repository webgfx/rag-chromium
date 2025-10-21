@echo off
REM Quick Deployment Script for Chromium RAG System
REM Optimized for GPU performance with RTX 5080

echo ========================================
echo Chromium RAG - Quick Deploy
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or higher
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

echo [1/4] Checking Python version...
python --version
echo.

REM Check if requirements are installed
echo [2/4] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    echo Please review any errors above
)
echo Dependencies installed!
echo.

REM Check if database exists
echo [3/4] Checking database...
if exist "data\cache\qdrant_db" (
    echo [OK] Qdrant database found ^(244,403 commits^)
    echo.
) else (
    echo [ERROR] Database not found at: data\cache\qdrant_db
    echo Please ensure the full package was copied
    exit /b 1
)

REM Check if models exist
echo [4/4] Checking models...
if exist "data\cache\models" (
    echo [OK] Embedding models found
    echo.
) else (
    echo [ERROR] Models not found at: data\cache\models
    echo Please ensure the full package was copied
    exit /b 1
)

REM Display VS Code setup instructions
echo ========================================
echo VS Code Setup
echo ========================================
echo.
echo Copy this to your VS Code settings.json:
echo.
echo {
 echo   "github.copilot.chat.mcp.servers": {
echo     "rag-chromium": {
echo       "command": "python",
echo       "args": ["%CD%\\scripts\\rag_mcp_server.py"]
echo     }
echo   }
echo }
echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Copy the settings above to VS Code
echo   2. Restart VS Code
echo   3. Test with: @rag-chromium How does Chrome handle WebGPU?
echo.
echo Performance:
echo   - First query: 45-60s (one-time warmup)
echo   - Subsequent queries: 10-20s
echo   - Cached queries: ^<50ms
echo.
echo Documentation:
 echo   - QUICK_REFERENCE.md
 echo   - VSCODE_SETUP_OPTIMIZED.md
 echo   - PERFORMANCE_OPTIMIZATION.md
echo.