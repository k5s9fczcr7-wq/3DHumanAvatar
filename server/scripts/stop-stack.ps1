# Graceful stop sequence for the kiosk stack.
# Avoids Stop-Process -Force on the orchestrator (which wedges the camera
# in MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED). Falls back to -Force only
# when the graceful path times out.
#
# Order: orchestrator graceful WS shutdown -> llama-server -> (delay) -> FastAPI -> orphan sweep.
# FastAPI is killed LAST so the admin dashboard's status poll (every 2s)
# can show llama + orchestrator going red before FastAPI itself dies.
# FastAPIDelaySeconds gives the dashboard one-or-more poll cycles in the
# all-other-red state before this very server stops responding.

param(
    [int]$GracefulTimeoutSeconds = 8,
    [int]$FastAPIDelaySeconds = 3,    # delay between llama-kill and FastAPI-kill (dashboard observability)
    [switch]$KeepLlama = $false,    # leave llama-server running (warm cache)
    [switch]$KeepFastAPI = $false   # leave FastAPI running
)

$root = Split-Path -Parent $PSScriptRoot
Write-Output ("=== stop-stack @ {0:HH:mm:ss.fff} ===" -f (Get-Date))

# 1. Graceful orchestrator shutdown via WS
Write-Output "[1/4] Orchestrator graceful shutdown via WS..."
$orchPid = (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($orchPid) {
    & "$root\.venv\Scripts\python.exe" "$PSScriptRoot\shutdown-orchestrator.py"
    $deadline = (Get-Date).AddSeconds($GracefulTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
        $alive = Get-Process -Id $orchPid -ErrorAction SilentlyContinue
        if (-not $alive) { Write-Output "  graceful exit OK"; break }
    }
    $alive = Get-Process -Id $orchPid -ErrorAction SilentlyContinue
    if ($alive) {
        Write-Output "  graceful timeout - falling back to Stop-Process -Force"
        Stop-Process -Id $orchPid -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Output "  not running"
}

# 2. llama-server (before FastAPI, so FastAPI stops last)
if (-not $KeepLlama) {
    Write-Output "[2/4] llama-server Stop-Process..."
    $llamaPid = (Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue).OwningProcess
    if ($llamaPid) { Stop-Process -Id $llamaPid -Force -ErrorAction SilentlyContinue; Write-Output "  killed PID $llamaPid" }
    else { Write-Output "  not running" }
} else { Write-Output "[2/4] llama-server: kept (--KeepLlama)" }

# 3. FastAPI delay + kill (last; see the header)
if (-not $KeepFastAPI) {
    if ($FastAPIDelaySeconds -gt 0) {
        Write-Output ("[3/4] FastAPI delay {0}s (lets dashboard observe other components down)..." -f $FastAPIDelaySeconds)
        Start-Sleep -Seconds $FastAPIDelaySeconds
    }
    Write-Output "[3/4] FastAPI Stop-Process..."
    $apiPid = (Get-NetTCPConnection -LocalPort 8100 -State Listen -ErrorAction SilentlyContinue).OwningProcess
    if ($apiPid) { Stop-Process -Id $apiPid -Force -ErrorAction SilentlyContinue; Write-Output "  killed PID $apiPid" }
    else { Write-Output "  not running" }
} else { Write-Output "[3/4] FastAPI: kept (--KeepFastAPI)" }

# 4. Orphan sweep — any python.exe running orchestrator/uvicorn/cmd wrappers.
# The Jabra keeper is deliberately NOT swept: the Jabra drops its own mute when its last capture stream closes,
# so stopping the keeper with the orchestrator down would unmute a room mic that was muted on purpose. The keeper
# only holds that stream open and throws the audio away.
Write-Output "[4/4] Orphan sweep..."
$kills = 0
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*orchestrator.orchestrator*" -or
                   $_.CommandLine -like "*uvicorn server:app*" -or
                   $_.CommandLine -like "*start-llama*" -or
                   $_.CommandLine -like "*start-fastapi*" -or
                   $_.CommandLine -like "*start-orchestrator*" } |
    ForEach-Object {
        Write-Output ("  killing PID {0} ({1})" -f $_.ProcessId, $_.Name)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $kills++
    }
Write-Output "  swept $kills orphan(s)"

Start-Sleep -Milliseconds 500

# Verify
Write-Output ""
Write-Output "=== Ports listening after stop ==="
$listen = Get-NetTCPConnection -LocalPort 8080,8100,8765 -State Listen -ErrorAction SilentlyContinue
if ($listen) { $listen | Select LocalPort,OwningProcess | Format-Table -AutoSize }
else { Write-Output "  all ports clear" }
