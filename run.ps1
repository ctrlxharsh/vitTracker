#Requires -Version 5.1
<#
.SYNOPSIS
    Reliable startup and environment setup script for AI Vision Tracker on Windows.

.DESCRIPTION
    Automates environment detection, virtual environment creation (.venv),
    dependency installation, OpenCV-contrib / CSRT compatibility verification,
    and launches the AI Vision Tracker.

.PARAMETER Cuda
    Install or use PyTorch with CUDA acceleration (NVIDIA GPU).

.PARAMETER Pi
    Use lightweight profile (requirements-pi.txt, no PyTorch / YOLO).

.PARAMETER Install
    Force reinstall/update of dependencies before launching.

.PARAMETER Help
    Show this help message.

.EXAMPLE
    .\run.ps1
    Launches GUI in default YOLO auto-detect mode.

.EXAMPLE
    .\run.ps1 --mode csrt
    Launches directly into OpenCV CSRT tracking mode.

.EXAMPLE
    .\run.ps1 --cv-only
    Runs lightweight OpenCV standalone window without Tkinter GUI.

.EXAMPLE
    .\run.ps1 -Cuda
    Installs/uses PyTorch with CUDA acceleration for NVIDIA GPUs.
#>

[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Cuda,
    [switch]$Pi,
    [switch]$Install,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# Set console title, UTF-8 encoding, and disable buggy MSMF backend for OpenCV
$host.UI.RawUI.WindowTitle = "AI Vision Tracker"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:OPENCV_VIDEOIO_PRIORITY_MSMF = "0"
$env:OPENCV_LOG_LEVEL = "ERROR"
$env:PYTHONUNBUFFERED = "1"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       AI Vision Tracker & Pan-Tilt Servoing              " -ForegroundColor Cyan
Write-Host "                  Windows Launcher                        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Locate Python 3 Interpreter Executable
$pythonExe = $null

# Check Windows Python Launcher (py.exe) and resolve actual python.exe path
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    try {
        $resolved = & py -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $candidate = $resolved.Trim()
            if (Test-Path $candidate) {
                $pythonExe = $candidate
            }
        }
    } catch {}
}

# Check standard python in PATH
if (-not $pythonExe -and (Get-Command "python" -ErrorAction SilentlyContinue)) {
    try {
        $resolved = & python -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $candidate = $resolved.Trim()
            # Guard against Windows Store 0-byte execution alias stub
            if ((Test-Path $candidate) -and (Get-Item $candidate).Length -gt 0) {
                $verMajor = & $candidate -c "import sys; print(sys.version_info.major)" 2>$null
                if ($verMajor -match "3") {
                    $pythonExe = $candidate
                }
            }
        }
    } catch {}
}

# Check python3 in PATH
if (-not $pythonExe -and (Get-Command "python3" -ErrorAction SilentlyContinue)) {
    try {
        $resolved = & python3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $candidate = $resolved.Trim()
            if (Test-Path $candidate) {
                $pythonExe = $candidate
            }
        }
    } catch {}
}

# Search standard Windows installation folders if not in PATH
if (-not $pythonExe) {
    $searchPatterns = @(
        "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe",
        "$env:ProgramFiles\Python3*\python.exe",
        "${env:ProgramFiles(x86)}\Python3*\python.exe",
        "C:\Python3*\python.exe"
    )
    foreach ($pattern in $searchPatterns) {
        $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
        foreach ($item in $found) {
            if (Test-Path $item.FullName) {
                $verMajor = & $item.FullName -c "import sys; print(sys.version_info.major)" 2>$null
                if ($verMajor -match "3") {
                    $pythonExe = $item.FullName
                    break
                }
            }
        }
        if ($pythonExe) { break }
    }
}

if (-not $pythonExe) {
    Write-Host "[ERROR] Python 3 was not found on your system." -ForegroundColor Red
    Write-Host "Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/windows/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add python.exe to PATH' during installation." -ForegroundColor Yellow
    Write-Host "Or install via winget: winget install Python.Python.3.11" -ForegroundColor Yellow
    exit 1
}

$pyVersion = & $pythonExe -c "import sys; print(sys.version.split()[0])"
Write-Host "[OK] Detected Python $pyVersion ($pythonExe)" -ForegroundColor Green

# 2. Select profile / requirements
$reqFile = "requirements.txt"
if ($Pi -or ($RemainingArgs -contains "--pi") -or ($RemainingArgs -contains "--csrt-only")) {
    $reqFile = "requirements-pi.txt"
    Write-Host "==> Using lightweight profile ($reqFile)..." -ForegroundColor Yellow
}

# 3. Virtual Environment Setup
$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = (Get-Location).Path }
$venvDir = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "==> Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & $pythonExe -m venv $venvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }

    Write-Host "==> Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip setuptools wheel --quiet

    if ($Cuda) {
        Write-Host "==> Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Cyan
        & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    }

    Write-Host "==> Installing project dependencies from $reqFile..." -ForegroundColor Cyan
    # Install with lapx fallback for Windows (prevents MSVC compilation failure on 'lap')
    & $venvPip install lapx --quiet
    & $venvPip install -r (Join-Path $scriptDir $reqFile)
    
    # Remove conflicting headless or base opencv if ultralytics pulled it
    & $venvPip uninstall -y opencv-python opencv-python-headless 2>$null | Out-Null
    & $venvPip install --force-reinstall --no-deps "opencv-contrib-python>=4.10.0" --quiet

    New-Item -ItemType File -Path (Join-Path $venvDir ".installed") -Force | Out-Null
    Write-Host "[OK] Environment setup complete." -ForegroundColor Green
}

# 4. Handle explicit --install or dependency re-verification
if ($Install -or ($RemainingArgs -contains "--install")) {
    Write-Host "==> Updating dependencies..." -ForegroundColor Cyan
    if ($Cuda) {
        Write-Host "==> Ensuring PyTorch with CUDA..." -ForegroundColor Cyan
        & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    }
    & $venvPip install lapx --quiet
    & $venvPip install -r (Join-Path $scriptDir $reqFile)
    & $venvPip uninstall -y opencv-python opencv-python-headless 2>$null | Out-Null
    & $venvPip install --force-reinstall --no-deps "opencv-contrib-python>=4.10.0" --quiet
}

# 5. Verify Ultralytics & OpenCV Contrib (CSRT) health
if ($reqFile -eq "requirements.txt") {
    $yoloOk = & $venvPython -c "import ultralytics" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> Ultralytics missing in .venv. Installing dependencies..." -ForegroundColor Yellow
        & $venvPip install lapx --quiet
        & $venvPip install -r (Join-Path $scriptDir $reqFile)
    }
}

$csrtOk = & $venvPython -c "import cv2; assert hasattr(cv2, 'TrackerCSRT_create')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Fixing OpenCV Contrib bindings for CSRT tracker..." -ForegroundColor Yellow
    & $venvPip uninstall -y opencv-python opencv-python-headless 2>$null | Out-Null
    & $venvPip install --force-reinstall --no-deps "opencv-contrib-python>=4.10.0" --quiet
}

# Filter out internal script flags from arguments forwarded to main.py
$cleanArgs = @()
foreach ($arg in $RemainingArgs) {
    if ($arg -notin @("--install", "-Install", "--cuda", "-Cuda", "--pi", "-Pi")) {
        $cleanArgs += $arg
    }
}

Write-Host "==> Starting AI Vision Tracker..." -ForegroundColor Green
Write-Host ""

# Launch main.py within virtual environment
& $venvPython (Join-Path $scriptDir "main.py") @cleanArgs
exit $LASTEXITCODE
