@echo off
REM Update Chromium Data - Ingest New Commits
REM Simple script to update the database with latest Chromium commits

setlocal EnableDelayedExpansion

echo ========================================
echo Chromium RAG - Update Data
echo ========================================
echo.

REM Check if repository path is configured
if not exist "config.yaml" (
    echo [ERROR] config.yaml not found!
    echo Please ensure you're in the project root directory
    pause
    exit /b 1
)

echo This script will:
echo   1. Check your local Chromium repository
echo   2. Ingest new commits since last update
echo   3. Generate embeddings
echo   4. Update vector database
echo.
echo Choose update method:
echo   [1] Recent commits (last 7 days)
echo   [2] Recent commits (last 30 days)
echo   [3] Custom date range
echo   [4] By commit index range
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    set days=7
    goto :recent
)
if "%choice%"=="2" (
    set days=30
    goto :recent
)
if "%choice%"=="3" (
    goto :daterange
)
if "%choice%"=="4" (
    goto :indexrange
)

echo Invalid choice. Exiting.
pause
exit /b 1

:recent
echo.
echo [1/3] Extracting recent commits (%days% days)...
python -m rag_system.data.ingest --mode recent --days %days% --batch-size 1000
if errorlevel 1 (
    echo [ERROR] Data extraction failed
    pause
    exit /b 1
)
echo [OK] Recent commits extracted
echo.

echo [2/3] Generating embeddings...
python scripts\generate_embeddings.py
if errorlevel 1 (
    echo [ERROR] Embedding generation failed
    pause
    exit /b 1
)
echo [OK] Embeddings generated
echo.

echo [3/3] Ingesting into vector database...
python scripts\ingest_vectors.py
if errorlevel 1 (
    echo [ERROR] Vector ingestion failed
    pause
    exit /b 1
)
echo [OK] Vector database updated
echo.
goto :complete

:daterange
echo.
set /p start_date="Enter start date (YYYY-MM-DD): "
set /p end_date="Enter end date (YYYY-MM-DD): "
echo.

echo [1/3] Extracting commits from %start_date% to %end_date%...
python -m rag_system.data.ingest --mode range --since %start_date% --until %end_date% --batch-size 1000
if errorlevel 1 (
    echo [ERROR] Data extraction failed
    pause
    exit /b 1
)
echo [OK] Commits extracted
echo.

echo [2/3] Generating embeddings...
python scripts\generate_embeddings.py
if errorlevel 1 (
    echo [ERROR] Embedding generation failed
    pause
    exit /b 1
)
echo [OK] Embeddings generated
echo.

echo [3/3] Ingesting into vector database...
python scripts\ingest_vectors.py
if errorlevel 1 (
    echo [ERROR] Vector ingestion failed
    pause
    exit /b 1
)
echo [OK] Vector database updated
echo.
goto :complete

:indexrange
echo.
echo For large-scale ingestion by index, use the advanced script:
echo   python scripts\ingestion\ingest_by_index.py --repo-path PATH --start-index N --end-index M
echo.
echo Current database status:
python -m rag_system.data.ingest --mode stats
echo.
pause
exit /b 0

:complete
echo ========================================
echo Update Complete!
echo ========================================
echo.
echo Database has been updated with new commits.
echo.
echo Next steps:
echo   - Test queries to verify new data
echo   - Run backup script to save changes
echo   - Monitor performance in VS Code
echo.
echo Usage stats:
python -m rag_system.data.ingest --mode stats
echo.
pause
