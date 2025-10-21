@echo off
REM Generate Deployment Package for Chromium RAG System
REM Creates a complete deployment package with all files

setlocal EnableDelayedExpansion

echo ========================================
echo Chromium RAG - Generate Deployment Package
echo ========================================
echo.

REM Get current date for package name
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
set packageName=rag-chromium-%mydate%
set packagePath=deployment\%packageName%

echo Creating package: %packageName%
echo Location: %packagePath%
echo.

REM Check if deployment folder exists
if not exist "deployment" mkdir deployment

REM Remove old package if exists
if exist "%packagePath%" (
    echo [WARNING] Package already exists. Removing old package...
    rmdir /s /q "%packagePath%"
)

REM Create directory structure
echo [1/8] Creating directory structure...
mkdir "%packagePath%"
mkdir "%packagePath%\scripts"
mkdir "%packagePath%\rag_system"
mkdir "%packagePath%\data\cache"
echo    [OK] Structure created
echo.

REM Copy core scripts
echo [2/8] Copying scripts...
copy "scripts\copilot_rag_interface.py" "%packagePath%\scripts\" >nul
copy "scripts\rag_mcp_server.py" "%packagePath%\scripts\" >nul
copy "scripts\rag_http_server.py" "%packagePath%\scripts\" >nul
copy "scripts\rag_remote_client.py" "%packagePath%\scripts\" >nul
copy "scripts\benchmark_performance.py" "%packagePath%\scripts\" >nul
copy "scripts\quick-deploy.bat" "%packagePath%\" >nul
copy "scripts\test-server.bat" "%packagePath%\" >nul
copy "scripts\start-server.bat" "%packagePath%\" >nul
copy "scripts\start-http-server.bat" "%packagePath%\" >nul
echo    [OK] Scripts copied (9 files)
echo.

REM Copy RAG system
echo [3/8] Copying RAG system...
xcopy /E /I /Q "rag_system" "%packagePath%\rag_system" >nul
echo    [OK] RAG system copied
echo.

REM Copy documentation
echo [4/8] Copying documentation...
copy "README.md" "%packagePath%\" >nul
copy "QUICK_REFERENCE.md" "%packagePath%\" >nul
copy "VSCODE_SETUP_OPTIMIZED.md" "%packagePath%\" >nul
copy "PERFORMANCE_OPTIMIZATION.md" "%packagePath%\" >nul
copy "requirements.txt" "%packagePath%\" >nul
copy "config.yaml" "%packagePath%\" >nul
REM Create DEPLOYMENT_README.md
if not exist "%packagePath%\DEPLOYMENT_README.md" (
    echo # Chromium RAG Deployment Package > "%packagePath%\DEPLOYMENT_README.md"
    echo. >> "%packagePath%\DEPLOYMENT_README.md"
    echo Run quick-deploy.bat to get started. >> "%packagePath%\DEPLOYMENT_README.md"
)
echo    [OK] Documentation copied (7 files)
echo.

REM Copy database
echo [5/8] Copying Qdrant database ^(this may take a minute^)...
if exist "data\cache\qdrant_db" (
    xcopy /E /I /Q "data\cache\qdrant_db" "%packagePath%\data\cache\qdrant_db" >nul
    echo    [OK] Database copied ^(244,403 commits^)
    echo.
) else (
    echo    [WARNING] Database not found at data\cache\qdrant_db
    echo.
)

REM Copy models
echo [6/8] Copying embedding models ^(this may take a minute^)...
if exist "data\cache\models" (
    xcopy /E /I /Q "data\cache\models" "%packagePath%\data\cache\models" >nul
    echo    [OK] Models copied ^(~11 GB^)
    echo.
) else (
    echo    [WARNING] Models not found at data\cache\models
    echo.
)

REM Calculate package size
echo [7/8] Calculating package size...
powershell -Command "$size = (Get-ChildItem '%packagePath%' -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; $sizeGB = [math]::Round($size / 1GB, 2); Write-Host \"   [OK] Package size: $sizeGB GB\""
echo.

REM Display summary
echo [8/8] Package created successfully!
echo.
echo ========================================
echo Package Details
echo ========================================
echo Name: %packageName%
echo Location: %CD%\%packagePath%
powershell -Command "$size = (Get-ChildItem '%packagePath%' -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; $sizeGB = [math]::Round($size / 1GB, 2); Write-Host \"Size: $sizeGB GB\""
echo.
echo Contents:
echo   - scripts/ ^(Python scripts + deployment tools^)
echo   - rag_system/ ^(Core modules^)
echo   - data/cache/qdrant_db/ ^(244,403 commits^)
echo   - data/cache/models/ ^(BAAI/bge-large-en-v1.5^)
echo   - Documentation ^(guides + README^)
echo   - quick-deploy.bat ^(deployment script^)
echo   - test-server.bat ^(testing script^)
echo.
echo ========================================
echo Deployment Options
echo ========================================
echo.
echo Option 1: Local Development
echo   1. Update VS Code settings to point to this package
echo   2. Restart VS Code
echo   3. Test with @rag-chromium
echo.
echo Option 2: Remote Server
echo   1. Transfer package: robocopy "%CD%\%packagePath%" "\\SERVER\path\" /E /Z /MT:8
echo   2. On server: Run quick-deploy.bat
echo   3. Update colleagues' VS Code settings
echo.
echo ========================================
echo.
