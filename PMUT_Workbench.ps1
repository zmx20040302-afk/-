param([ValidateSet('Open', 'Start', 'Install', 'Uninstall')][string]$Action = 'Open')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$appDir = Join-Path $root 'short_circuit\inverse_design\S237'
$appFile = Join-Path $appDir 'workbench.py'
$url = 'http://127.0.0.1:8797'
$linkPath = Join-Path ([Environment]::GetFolderPath('Startup')) 'PMUT Full Field Workbench.lnk'
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $root 'requirements.txt'

function Test-Workbench {
    try {
        $health = Invoke-RestMethod "$url/health" -TimeoutSec 2 -UseBasicParsing
        return ($health.app -eq 'pmut-program-v3' -and $health.protected_batch -eq $true)
    } catch { return $false }
}

function Find-BasePython {
    # Resolve the real executable, including Python installations behind py.exe.
    $candidates = @()
    $candidates += Get-Command python.exe -All -CommandType Application -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike '*\Microsoft\WindowsApps\*' }
    $candidates += Get-Command py.exe -All -CommandType Application -ErrorAction SilentlyContinue
    $candidates += Get-Command python.exe -All -CommandType Application -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -like '*\Microsoft\WindowsApps\*' }
    foreach ($candidate in $candidates) {
        if ($null -eq $candidate) { continue }
        $flags = @()
        if ($candidate.Name -eq 'py.exe') { $flags = @('-3') }
        try {
            $exe = & $candidate.Source @flags -I -c 'import sys; assert sys.version_info >= (3,11); print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path -LiteralPath $exe)) { return $exe }
        } catch { continue }
    }
    throw 'Install Python 3.11 or newer (3.12 recommended), then rerun this launcher.'
}

function Get-WorkbenchPython {
    if (!(Test-Path -LiteralPath $venvPython)) {
        $base = Find-BasePython
        & $base -m venv $venv | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Could not create a Python environment in $venv." }
    }
    $required = (Get-Content -LiteralPath $requirements | Where-Object { $_ -match '^numpy==' }) -replace '^numpy==', ''
    if (!$required) { throw "Missing NumPy version in $requirements." }
    $probe = "import importlib.metadata as m, sys; sys.exit(m.version('numpy') != sys.argv[1])"
    try { & $venvPython -I -c $probe $required 2>$null; $installed = ($LASTEXITCODE -eq 0) }
    catch { $installed = $false }
    if (!$installed) {
        & $venvPython -m pip --disable-pip-version-check install -r $requirements | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Restore network access or install the required wheel in .venv, then retry.' }
    }
    & $venvPython -I -c 'import numpy, sys; assert hasattr(numpy, sys.argv[1])' 'bitwise_count'
    if ($LASTEXITCODE -ne 0) { throw 'Could not load NumPy in .venv.' }
    return (Join-Path $venv 'Scripts\pythonw.exe')
}

if ($Action -eq 'Uninstall') {
    if (Test-Path -LiteralPath $linkPath) { Remove-Item -LiteralPath $linkPath }
    Write-Output 'Automatic startup has been disabled for this user.'
    exit 0
}

$mutex = New-Object System.Threading.Mutex($false, 'Local\PMUTWorkbenchLaunch8797')
$locked = $false
try {
    try { $locked = $mutex.WaitOne(300000) }
    catch [System.Threading.AbandonedMutexException] { $locked = $true }
    if (!$locked) { throw 'Another launcher is still setting up the workbench.' }
    if (!(Test-Path -LiteralPath $appFile)) { throw "Missing workbench: $appFile" }
    $running = Test-Workbench
    if (!$running -or $Action -eq 'Install') { $pythonw = Get-WorkbenchPython }
    if (!$running) {
        $proc = Start-Process -FilePath $pythonw -ArgumentList '-B','workbench.py' -WorkingDirectory $appDir -WindowStyle Hidden -PassThru
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            if (Test-Workbench) { $running = $true; break }
            if ($proc.HasExited) { break }
            Start-Sleep -Milliseconds 250
        }
        if (!$running) { throw "Workbench did not start on $url. Check port 8797 or run .venv\Scripts\python.exe short_circuit\inverse_design\S237\workbench.py to see the error." }
    }
    if ($Action -eq 'Install') {
        # Logon runs the already prepared local interpreter, without PowerShell,
        # a console, a browser, or another dependency download.
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($linkPath)
        $link.TargetPath = $pythonw
        $link.Arguments = '-B workbench.py'
        $link.WorkingDirectory = $appDir
        $link.Description = 'Start the local PMUT full-field workbench on port 8797'
        $link.Save()
        Write-Output "Enabled automatic startup at sign-in: $linkPath"
    }
    if ($Action -eq 'Open') { Start-Process "$url/#inverse" }
    Write-Output "PMUT workbench ready: $url/#inverse"
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
