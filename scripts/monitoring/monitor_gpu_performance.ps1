# GPU Performance Monitor for Chromium Ingestion
# Displays real-time GPU and processing metrics

Write-Host "`n🔍 GPU Performance Monitor" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

$iteration = 0
while ($true) {
    Clear-Host
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " GPU PERFORMANCE MONITOR - RTX 5080" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # Get GPU stats
    $gpuStats = nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits
    $stats = $gpuStats -split ','
    
    $memUsed = [int]$stats[0].Trim()
    $memFree = [int]$stats[1].Trim()
    $memTotal = [int]$stats[2].Trim()
    $gpuUtil = [int]$stats[3].Trim()
    $memUtil = [int]$stats[4].Trim()
    $power = [float]$stats[5].Trim()
    $temp = [int]$stats[6].Trim()
    
    $memUsedGB = [math]::Round($memUsed / 1024, 2)
    $memFreeGB = [math]::Round($memFree / 1024, 2)
    $memTotalGB = [math]::Round($memTotal / 1024, 2)
    $memPercent = [math]::Round(($memUsed / $memTotal) * 100, 1)
    
    # GPU Memory
    Write-Host " 💾 GPU Memory:" -ForegroundColor Yellow
    Write-Host "    Used: $memUsedGB GB / $memTotalGB GB ($memPercent%)"
    Write-Host "    Free: $memFreeGB GB"
    
    # Utilization bar
    $bar = ""
    for ($i = 0; $i -lt 50; $i++) {
        if ($i -lt ($memPercent / 2)) { $bar += "█" }
        else { $bar += "░" }
    }
    
    $color = if ($memPercent -lt 50) { "Green" } elseif ($memPercent -lt 80) { "Yellow" } else { "Red" }
    Write-Host "    [$bar] " -NoNewline
    Write-Host "$memPercent%" -ForegroundColor $color
    Write-Host ""
    
    # GPU Compute
    Write-Host " ⚡ GPU Compute:" -ForegroundColor Yellow
    Write-Host "    Utilization: $gpuUtil%"
    
    $bar = ""
    for ($i = 0; $i -lt 50; $i++) {
        if ($i -lt ($gpuUtil / 2)) { $bar += "█" }
        else { $bar += "░" }
    }
    
    $color = if ($gpuUtil -lt 30) { "Red" } elseif ($gpuUtil -lt 70) { "Yellow" } else { "Green" }
    Write-Host "    [$bar] " -NoNewline
    Write-Host "$gpuUtil%" -ForegroundColor $color
    Write-Host ""
    
    # Power & Temperature
    Write-Host " 🔋 Power & Temp:" -ForegroundColor Yellow
    Write-Host "    Power Draw: $power W"
    Write-Host "    Temperature: $temp°C"
    Write-Host ""
    
    # Python processes
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | 
        Select-Object Id, @{N='CPU';E={[math]::Round($_.CPU,1)}}, @{N='MemGB';E={[math]::Round($_.WorkingSet64/1GB,2)}}
    
    if ($pythonProcesses) {
        Write-Host " 🐍 Python Processes:" -ForegroundColor Yellow
        foreach ($proc in $pythonProcesses) {
            Write-Host "    PID $($proc.Id): CPU=$($proc.CPU)s, RAM=$($proc.MemGB)GB"
        }
    } else {
        Write-Host " 🐍 Python Processes:" -ForegroundColor Yellow
        Write-Host "    No Python processes running" -ForegroundColor Red
    }
    Write-Host ""
    
    # Progress file check
    if (Test-Path "data\massive_cache\progress.json") {
        $progress = Get-Content "data\massive_cache\progress.json" | ConvertFrom-Json
        Write-Host " 📊 Processing Stats:" -ForegroundColor Yellow
        Write-Host "    Commits: $($progress.commits_processed)"
        Write-Host "    Batches: $($progress.batches_completed)"
        if ($progress.last_batch_time) {
            $rate = [math]::Round($progress.last_batch_time, 2)
            Write-Host "    Last Batch: $rate seconds"
        }
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " Refreshing in 5 seconds... (Iteration: $iteration)" -ForegroundColor Gray
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    
    $iteration++
    Start-Sleep -Seconds 5
}
