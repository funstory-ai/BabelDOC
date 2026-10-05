<#
.SYNOPSIS
    Build standalone, fully-offline BabelDOC packages for Windows x64 and
    Linux x86_64 (portable Python + BabelDOC + all dependencies + offline
    assets). No Python is required on the target machine.

.DESCRIPTION
    Windows layout (zip):
        1-init.bat, babeldoc.bat, python\ (embeddable layout), assets\
    Linux layout (zip):
        install.sh, python-<ver>-linux-x86_64-babeldoc.tar.gz, assets\
        (install.sh extracts the tarball next to itself on first run)

    Hard-won knowledge encoded here:
      * uv-managed standalone Python refuses modification; we COPY it first
        and install with pip --break-system-packages (never `uv pip install`
        into the managed interpreter).
      * Linux Python and dependencies are built INSIDE a Linux container and
        tarred to the mounted output dir. Do NOT `docker cp` it out on a
        Windows host: symlinks cannot be re-created on NTFS.
      * docker -v "X":/y tokenization is broken in PowerShell 5.1; use
        --mount with a single quoted string.
      * pip-generated console scripts carry a build-time absolute shebang,
        so the Linux wrapper invokes `python -m babeldoc.main`.
      * Zip entries must use forward slashes (CreateFromDirectory on .NET
        Framework writes backslashes). All packaged file names are ASCII.
      * Linux wheels require glibc >= 2.34 (onnx 1.21+ baseline).

.PARAMETER Platform
    Target platforms: win64, linux-x64. Default: both.

.PARAMETER PythonVersion
    CPython version to embed. Default 3.13.3 (must be >=3.10,<3.14).

.PARAMETER OutputDir
    Staging/output directory. Default _offline_babeldoc under the repo root.

.PARAMETER PipIndex
    Optional pip index URL used while BUILDING (e.g. a Tsinghua mirror).
    Empty means PyPI default. Not used at all on offline target machines.

.PARAMETER DockerImage
    Linux build image with uv + glibc baseline compatible with the wheels.
    Default ghcr.io/astral-sh/uv:python3.13-bookworm-slim.

.PARAMETER AssetsZip
    Prebuilt offline_assets_<tag>.zip to embed. If omitted the script looks
    in ./dist and ~/.cache/babeldoc/assets, then generates one (network
    warmup) as a last resort.

.PARAMETER Force
    Rebuild portable Python + dependencies even when staged artifacts exist.

.PARAMETER SkipVerify
    Skip post-build smoke tests.

.PARAMETER SkipZip
    Only stage directories, do not produce the zip files.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_offline_packages.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build_offline_packages.ps1 -Platform win64 -Force
#>
[CmdletBinding()]
param(
    [ValidateSet("win64", "linux-x64")]
    [string[]]$Platform = @("win64", "linux-x64"),

    [string]$PythonVersion = "3.13.3",

    [string]$OutputDir = "",

    [string]$PipIndex = "https://pypi.tuna.tsinghua.edu.cn/simple",

    [string]$DockerImage = "ghcr.io/astral-sh/uv:python3.13-bookworm-slim",

    [string]$AssetsZip = "",

    [switch]$Force,
    [switch]$SkipVerify,
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RootDir

if (-not $OutputDir) { $OutputDir = Join-Path $RootDir "_offline_babeldoc" }
$PyFull = $PythonVersion
$LinuxPyDir = "cpython-$PyFull-linux-x86_64-gnu"
$LinuxTarName = "python-$PyFull-linux-x86_64-babeldoc.tar.gz"

function Invoke-Step($msg, [scriptblock]$body) {
    Write-Host "`n==== $msg ====" -ForegroundColor Cyan
    & $body
}

function Assert-Native($msg) {
    if ($LASTEXITCODE -ne 0) { throw "$msg (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------------------
Invoke-Step "Build BabelDOC wheel from local sources" {
    uv build --wheel
    Assert-Native "uv build failed"
    $script:Wheel = Get-ChildItem (Join-Path $RootDir "dist") -Filter "BabelDOC-*.whl" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $script:Wheel) { throw "wheel not found under dist/" }
    Write-Host "wheel: $($script:Wheel.Name)"
    if ($script:Wheel.BaseName -match "^BabelDOC-(.+)-py3-none-any$") {
        $script:BabelVersion = $Matches[1]
    } else {
        throw "cannot parse version from wheel name: $($script:Wheel.Name)"
    }
}

$DistDir = Join-Path $RootDir "dist"
$WinStage = Join-Path $OutputDir "win64"
$LinuxStage = Join-Path $OutputDir "linux-x64"

# ---------------------------------------------------------------------------
if ($Platform -contains "win64") {
    Invoke-Step "win64: prepare portable Python and install BabelDOC" {
        uv python install $PyFull
        Assert-Native "uv python install failed"
        # `uv python find` may return a .local shim; use the managed dir directly
        $uvPythonRoot = (uv python dir).Trim()
        $managedRoot = Join-Path $uvPythonRoot "cpython-$PyFull-windows-x86_64-none"
        if (-not (Test-Path (Join-Path $managedRoot "python.exe"))) {
            throw "managed standalone python not found: $managedRoot"
        }
        $stagedPy = Join-Path $WinStage "python\python.exe"
        if ($Force -or -not (Test-Path $stagedPy)) {
            New-Item -ItemType Directory -Force $WinStage | Out-Null
            # /MIR also wipes packages left over from a previous build
            robocopy $managedRoot (Join-Path $WinStage "python") /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
            if ($LASTEXITCODE -gt 7) { throw "robocopy portable python failed: $LASTEXITCODE" }
            $global:LASTEXITCODE = 0
            $pipArgs = @("-m", "pip", "install", "--break-system-packages")
            if ($PipIndex) { $pipArgs += @("-i", $PipIndex) }
            $pipArgs += $script:Wheel.FullName
            & $stagedPy @pipArgs
            Assert-Native "pip install into portable python failed"
        } else {
            Write-Host "staged python exists, reuse it (use -Force to rebuild)"
        }
    }
}

# ---------------------------------------------------------------------------
if ($Platform -contains "linux-x64") {
    Invoke-Step "linux-x64: build portable Python and dependencies inside container" {
        New-Item -ItemType Directory -Force $LinuxStage | Out-Null
        $tarPath = Join-Path $LinuxStage $LinuxTarName
        if ($Force -or -not (Test-Path $tarPath)) {
            docker version --format "{{.Server.Version}}" | Out-Null
            Assert-Native "docker is required for the linux-x64 build"
            $indexArgs = ""
            if ($PipIndex) { $indexArgs = "-i $PipIndex" }
            # Single-quoted PowerShell string; only __TOKENS__ are substituted.
            $sh = @'
set -e
uv python install __PYFULL__
PY=/root/.local/share/uv/python/__LINDIR__/bin/python3
$PY -m pip install --break-system-packages __INDEX__ /whl/__WHEEL__
cd /root/.local/share/uv/python
tar czf /out/__TARNAME__ __LINDIR__
echo LINUX_BUILD_OK
'@
            $sh = $sh.Replace("__PYFULL__", $PyFull).
                     Replace("__LINDIR__", $LinuxPyDir).
                     Replace("__INDEX__", $indexArgs).
                     Replace("__WHEEL__", $script:Wheel.Name).
                     Replace("__TARNAME__", $LinuxTarName)
            docker run --rm `
                --mount "type=bind,source=$DistDir,target=/whl,readonly" `
                --mount "type=bind,source=$LinuxStage,target=/out" `
                $DockerImage sh -c $sh
            Assert-Native "linux container build failed"
        } else {
            Write-Host "$LinuxTarName exists, reuse it (use -Force to rebuild)"
        }
    }
}

# ---------------------------------------------------------------------------
Invoke-Step "Resolve offline assets package" {
    # Compute the expected tag with the freshly built Windows interpreter.
    $tagPy = Join-Path $WinStage "python\python.exe"
    if (-not (Test-Path $tagPy)) {
        throw "win64 stage is required to compute the assets tag; build win64 too"
    }
    Push-Location $env:TEMP
    $assetsTag = (& $tagPy -c "from babeldoc.assets.assets import get_offline_assets_tag; print(get_offline_assets_tag())").Trim()
    Pop-Location
    $expectedName = "offline_assets_$assetsTag.zip"
    Write-Host "expected assets: $expectedName"

    $candidates = @()
    if ($AssetsZip) { $candidates += $AssetsZip }
    $candidates += (Join-Path $DistDir $expectedName)
    $candidates += (Join-Path $HOME ".cache\babeldoc\assets\$expectedName")
    $candidates += (Join-Path $OutputDir "win64\assets\$expectedName")
    $candidates += (Join-Path $OutputDir "linux-x64\assets\$expectedName")

    $resolved = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $resolved) {
        Write-Host "no prebuilt assets found, generating (network warmup) ..."
        $genDir = Join-Path $OutputDir "_assets_gen"
        New-Item -ItemType Directory -Force $genDir | Out-Null
        Push-Location $env:TEMP
        & $tagPy -m babeldoc.main --generate-offline-assets $genDir
        Assert-Native "generate-offline-assets failed"
        Pop-Location
        $resolved = Join-Path $genDir $expectedName
    }
    if (-not (Test-Path $resolved)) { throw "assets zip missing after resolution: $resolved" }
    $script:AssetsResolved = $resolved
    Write-Host "using assets: $resolved"
}

# ---------------------------------------------------------------------------
function Write-WinFiles {
    $initBat = Join-Path $WinStage "1-init.bat"
    $bat = @'
@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo  BabelDOC offline package - initialization (run once per user)
echo ============================================================
echo.
echo [1/2] Restoring offline assets to %USERPROFILE%\.cache\babeldoc
"python\Scripts\babeldoc.exe" --restore-offline-assets assets
if errorlevel 1 (
    echo [ERROR] Failed to restore offline assets.
    pause
    exit /b 1
)
echo.
echo [2/2] Verifying installation ...
"python\Scripts\babeldoc.exe" --version
if errorlevel 1 (
    echo [ERROR] babeldoc is not runnable.
    pause
    exit /b 1
)
echo.
echo Done. Translate with:
echo   babeldoc.bat --files sample.pdf --lang-in en --lang-out zh --openai ^
--openai-base-url http://YOUR_LLM_HOST/v1 --openai-api-key YOUR_KEY ^
--openai-model YOUR_MODEL -o output
echo.
pause
'@
    $wrapper = @'
@echo off
"%~dp0python\Scripts\babeldoc.exe" %*
'@
    Set-Ascii -Path $initBat -Content $bat -Crlf $true
    Set-Ascii -Path (Join-Path $WinStage "babeldoc.bat") -Content $wrapper -Crlf $true
}

function Write-LinuxFiles {
    $install = @'
#!/bin/sh
# BabelDOC offline package - one-time initialization (Linux x86_64, glibc >= 2.34)
# Usage: sh install.sh
set -e
cd "$(dirname "$0")"

if [ ! -d python ]; then
    echo "[1/3] Extracting portable Python ..."
    tar xzf __TARNAME__
    mv __LINDIR__ python
fi
chmod -R +x python/bin

# The pip-generated python/bin/babeldoc entry has a build-time absolute
# shebang; invoke the module through the relocated interpreter instead.
RUN="./python/bin/python3 -m babeldoc.main"

echo "[2/3] Restoring offline assets to $HOME/.cache/babeldoc ..."
$RUN --restore-offline-assets assets

echo "[3/3] Verifying ..."
$RUN --version

cat > babeldoc.sh <<'EOF'
#!/bin/sh
exec "$(dirname "$0")/python/bin/python3" -m babeldoc.main "$@"
EOF
chmod +x babeldoc.sh

echo ""
echo "Done. Translate with:"
echo "  ./babeldoc.sh --files sample.pdf --lang-in en --lang-out zh \\"
echo "    --openai --openai-base-url http://YOUR_LLM_HOST/v1 \\"
echo "    --openai-api-key YOUR_KEY --openai-model YOUR_MODEL -o output"
'@
    $install = $install.Replace("__TARNAME__", $LinuxTarName).Replace("__LINDIR__", $LinuxPyDir)
    Set-Ascii -Path (Join-Path $LinuxStage "install.sh") -Content $install -Crlf $false
}

function Set-Ascii($Path, $Content, [bool]$Crlf) {
    $text = $Content -replace "`r`n", "`n"
    if ($Crlf) { $text = $text -replace "`n", "`r`n" } else { $text = $text.TrimEnd([char]10) + "`n" }
    # No BOM: all content is ASCII by design.
    [System.IO.File]::WriteAllText($Path, $text, (New-Object System.Text.ASCIIEncoding))
}

Invoke-Step "Write launcher scripts and copy assets into stages" {
    foreach ($st in @($WinStage, $LinuxStage)) {
        if (Test-Path $st) {
            $ad = Join-Path $st "assets"
            New-Item -ItemType Directory -Force $ad | Out-Null
            $dst = Join-Path $ad $expectedName
            $same = (Test-Path $dst) -and `
                ((Resolve-Path $dst).Path -eq (Resolve-Path $script:AssetsResolved).Path)
            if (-not $same -and ($Force -or -not (Test-Path $dst))) {
                Copy-Item $script:AssetsResolved $dst -Force
            }
        }
    }
    if ($Platform -contains "win64") { Write-WinFiles }
    if ($Platform -contains "linux-x64") { Write-LinuxFiles }
}

# ---------------------------------------------------------------------------
if (-not $SkipVerify) {
    if ($Platform -contains "win64") {
        Invoke-Step "win64: smoke test staged interpreter" {
            Push-Location $env:TEMP
            & (Join-Path $WinStage "python\python.exe") -c "import babeldoc; from babeldoc.format.pdf.document_il.midend.paragraph_finder import TOC_SPACED_DOT_MIN_COUNT as T, ParagraphFinder as P; print('verify', babeldoc.__version__, T, hasattr(P, '_split_line_visual_bands'))"
            Assert-Native "win64 import/feature check failed"
            & (Join-Path $WinStage "python\Scripts\babeldoc.exe") --version
            Assert-Native "win64 babeldoc --version failed"
            Pop-Location
        }
    }
    if ($Platform -contains "linux-x64") {
        Invoke-Step "linux-x64: smoke test in a clean container" {
            $sh = @'
set -e
rm -rf /t && mkdir /t
tar xzf /out/__TARNAME__ -C /t
D=/t/__LINDIR__
$D/bin/python3 -m babeldoc.main --version
grep -c _split_line_visual_bands $D/lib/python3.13/site-packages/babeldoc/format/pdf/document_il/midend/paragraph_finder.py
$D/bin/python3 -c 'import onnxruntime,cv2,sklearn,scipy,skimage,freetype,hyperscan,tiktoken,Levenshtein,uharfbuzz,pymupdf; print(42)'
echo LINUX_VERIFY_OK
'@
            $sh = $sh.Replace("__TARNAME__", $LinuxTarName).Replace("__LINDIR__", $LinuxPyDir)
            docker run --rm --mount "type=bind,source=$LinuxStage,target=/out" $DockerImage sh -c $sh
            Assert-Native "linux verification failed"
        }
    }
}

# ---------------------------------------------------------------------------
if (-not $SkipZip) {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    function New-PortableZip($Src, $Dst, $Level) {
        if (Test-Path $Dst) { Remove-Item $Dst -Force }
        $fs = [System.IO.File]::Open($Dst, [System.IO.FileMode]::CreateNew)
        $zip = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            $full = (Resolve-Path $Src).Path.TrimEnd("\")
            foreach ($f in (Get-ChildItem -Recurse -File $full)) {
                $rel = $f.FullName.Substring($full.Length + 1).Replace("\", "/")
                [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $rel, $Level)
            }
        } finally {
            $zip.Dispose()
            $fs.Dispose()
        }
    }

    Invoke-Step "Create distribution zip files (forward-slash entries)" {
        if ($Platform -contains "win64") {
            $z = Join-Path $OutputDir "babeldoc-$($script:BabelVersion)-win64-offline.zip"
            New-PortableZip $WinStage $z ([System.IO.Compression.CompressionLevel]::Optimal)
        }
        if ($Platform -contains "linux-x64") {
            $z = Join-Path $OutputDir "babeldoc-$($script:BabelVersion)-linux-x64-offline.zip"
            # payload tar.gz is already compressed; store-fast to save time
            New-PortableZip $LinuxStage $z ([System.IO.Compression.CompressionLevel]::Fastest)
        }
        Get-ChildItem (Join-Path $OutputDir "babeldoc-*-offline.zip") |
            Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }, LastWriteTime |
            Format-Table -AutoSize | Out-String | Write-Host
    }
}

Write-Host "`nAll done. Artifacts in: $OutputDir" -ForegroundColor Green
