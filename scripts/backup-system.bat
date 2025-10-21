@echo off
REM Backup System - Save Database, Embeddings, and Critical Files
REM Creates timestamped backup of all important data

setlocal EnableDelayedExpansion

echo ========================================
echo Chromium RAG - System Backup
echo ========================================
echo.

REM Generate timestamp for backup folder name
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a%%b)
set timestamp=%mydate%_%mytime: =0%
set backupName=backup_%timestamp%
set backupPath=backups\%backupName%

echo Creating backup: %backupName%
echo Location: %backupPath%
echo.

REM Create backup directory
if not exist "backups" mkdir backups
mkdir "%backupPath%"
mkdir "%backupPath%\database"
mkdir "%backupPath%\embeddings"
mkdir "%backupPath%\config"
mkdir "%backupPath%\logs"

REM Calculate sizes
echo [1/6] Calculating backup size...
set totalSize=0

if exist "data\cache\qdrant_db" (
    for /r "data\cache\qdrant_db" %%F in (*) do set /a totalSize+=%%~zF
)
if exist "data\embeddings" (
    for /r "data\embeddings" %%F in (*) do set /a totalSize+=%%~zF
)

set /a sizeMB=!totalSize! / 1048576
set /a sizeGB=!sizeMB! / 1024

if !sizeGB! GTR 0 (
    echo    Backup size: ~!sizeGB! GB
) else (
    echo    Backup size: ~!sizeMB! MB
)
echo.

REM Ask for confirmation if backup is large
if !sizeGB! GTR 5 (
    echo [WARNING] Large backup detected (^>5 GB^)
    echo This may take several minutes...
    set /p continue="Continue? (Y/N): "
    if /i not "!continue!"=="Y" (
        echo Backup cancelled.
        pause
        exit /b 0
    )
    echo.
)

REM Backup vector database
echo [2/6] Backing up vector database...
if exist "data\cache\qdrant_db" (
    xcopy /E /I /Q /Y "data\cache\qdrant_db" "%backupPath%\database\qdrant_db" >nul
    echo    [OK] Database backed up
) else (
    echo    [WARNING] No database found to backup
)
echo.

REM Backup embeddings
echo [3/6] Backing up embeddings...
if exist "data\embeddings" (
    xcopy /E /I /Q /Y "data\embeddings" "%backupPath%\embeddings" >nul
    echo    [OK] Embeddings backed up
) else (
    echo    [WARNING] No embeddings found to backup
)
echo.

REM Backup embedding cache (models metadata)
if exist "data\cache\models" (
    xcopy /E /I /Q /Y "data\cache\models" "%backupPath%\models" >nul
    echo    [OK] Model cache backed up
)
echo.

REM Backup configuration files
echo [4/6] Backing up configuration...
if exist "config.yaml" copy "config.yaml" "%backupPath%\config\" >nul
if exist "requirements.txt" copy "requirements.txt" "%backupPath%\config\" >nul
if exist ".env" copy ".env" "%backupPath%\config\" >nul
if exist "data\status.json" copy "data\status.json" "%backupPath%\config\" >nul
echo    [OK] Configuration backed up
echo.

REM Backup recent logs
echo [5/6] Backing up recent logs...
if exist "logs" (
    REM Copy only logs from last 7 days
    forfiles /P "logs" /D -7 /C "cmd /c copy @path \"%backupPath%\logs\" >nul 2>&1" 2>nul
    echo    [OK] Recent logs backed up
) else (
    echo    [WARNING] No logs found to backup
)
echo.

REM Create backup manifest
echo [6/6] Creating backup manifest...
(
    echo Chromium RAG System Backup
    echo =========================
    echo.
    echo Backup Date: %date% %time%
    echo Backup Name: %backupName%
    echo.
    echo Contents:
    echo   - Vector Database: data\cache\qdrant_db
    echo   - Embeddings: data\embeddings
    echo   - Configuration: config.yaml, requirements.txt
    echo   - Status: data\status.json
    echo   - Recent Logs: last 7 days
    echo.
    echo Database Stats:
) > "%backupPath%\BACKUP_INFO.txt"

REM Add database stats if available
if exist "data\status.json" (
    python -c "import json; data=json.load(open('data/status.json')); print(f\"   Total Commits: {data.get('stats', {}).get('total_commits_ingested', 'N/A')}\")" >> "%backupPath%\BACKUP_INFO.txt" 2>nul
)

echo    [OK] Manifest created
echo.

REM Calculate final backup size
set backupSize=0
for /r "%backupPath%" %%F in (*) do set /a backupSize+=%%~zF
set /a backupMB=!backupSize! / 1048576
set /a backupGB=!backupMB! / 1024

REM Display summary
echo ========================================
echo Backup Complete!
echo ========================================
echo.
echo Backup Details:
echo   Name: %backupName%
echo   Location: %CD%\%backupPath%
if !backupGB! GTR 0 (
    echo   Size: !backupGB! GB
) else (
    echo   Size: !backupMB! MB
)
echo.
echo Backed up:
echo   [√] Vector database
echo   [√] Embeddings
echo   [√] Configuration files
echo   [√] Recent logs
echo.
echo To restore from this backup:
echo   1. Stop any running servers
echo   2. Copy contents to project directory
echo   3. Restart services
echo.
echo Backup location: %backupPath%
echo.

REM Ask about compression
set /p compress="Compress backup to .zip? (Y/N): "
if /i "%compress%"=="Y" (
    echo.
    echo Creating compressed archive...
    powershell -Command "Compress-Archive -Path '%backupPath%' -DestinationPath 'backups\%backupName%.zip' -CompressionLevel Optimal"
    if errorlevel 1 (
        echo [WARNING] Compression failed
    ) else (
        echo [OK] Backup compressed: backups\%backupName%.zip
        echo.
        set /p delete="Delete uncompressed backup folder? (Y/N): "
        if /i "!delete!"=="Y" (
            rmdir /s /q "%backupPath%"
            echo [OK] Uncompressed backup removed
        )
    )
)

echo.
pause
