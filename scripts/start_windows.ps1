param([ValidateSet('Capture', 'Analysis')][string]$Workbench)

$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$venv = Join-Path $project '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$entry = @(if ($Workbench -eq 'Capture') { '-m', 'straintrace_capture.gui' } else { 'dic_virtual_extensometer_gui_v7_multi_roi_range.py' })
$entryFile = if ($Workbench -eq 'Capture') { 'straintrace_capture\gui.py' } else { $entry[0] }
$workbenchArgs = @($entry) + @($args)

if ($env:EZDIC_LAUNCHER_SMOKE_TEST) {
    if (!(Test-Path -LiteralPath (Join-Path $project $entryFile) -PathType Leaf)) { exit 1 }
    $label = if ($Workbench -eq 'Analysis') { 'EZDIC' } else { 'Capture' }
    Write-Host "$label launcher smoke test"
    Write-Host "Entry $($entry -join ' ')"
    exit 0
}

function Quote-WindowsArgument([AllowEmptyString()][string]$Value) {
    # The Windows CRT doubles backslashes before quotes and the closing quote.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    return '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
}

function Invoke-Interpreter([string[]]$Command, [string[]]$Arguments, [int]$TimeoutMs = 0) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Command[0]
    $tokens = @($Command | Select-Object -Skip 1) + $Arguments
    $info.Arguments = ($tokens | ForEach-Object { Quote-WindowsArgument $_ }) -join ' '
    $info.WorkingDirectory = $project
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $info.StandardOutputEncoding = [Text.Encoding]::UTF8
    $info.StandardErrorEncoding = [Text.Encoding]::UTF8
    $processor = New-Object Diagnostics.Process
    $processor.StartInfo = $info
    try {
        [void]$processor.Start()
        $stdout = $processor.StandardOutput.ReadToEndAsync()
        $stderr = $processor.StandardError.ReadToEndAsync()
        if ($TimeoutMs -gt 0 -and !$processor.WaitForExit($TimeoutMs)) {
            $processor.Kill()
            $processor.WaitForExit()
            return 1
        }
        $processor.WaitForExit()
        $outputText = ($stdout.Result + $stderr.Result).Trim()
        if ($outputText) { Add-Content -LiteralPath $startupLog -Value $outputText -Encoding UTF8 }
        if ($outputText -and $TimeoutMs -eq 0) { Write-Host $outputText }
        return $processor.ExitCode
    } finally {
        $processor.Dispose()
    }
}

function Test-Python([string[]]$Command) {
    try {
        return (Invoke-Interpreter $Command @('-c', 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))') 7000) -eq 0
    } catch {
        Add-Content -LiteralPath $startupLog -Value $_.Exception.Message -Encoding UTF8
        return $false
    }
}

function Find-Python {
    foreach ($launcher in (Get-Command -Name py.exe -CommandType Application -All -ErrorAction SilentlyContinue)) {
        foreach ($flag in @('-3.13', '-3.12', '-3.11')) {
            $candidate = @($launcher.Source, $flag)
            if (Test-Python $candidate) { return $candidate }
        }
    }
    foreach ($app in (Get-Command -Name python.exe -CommandType Application -All -ErrorAction SilentlyContinue)) {
        if ($app.Source -like '*\Microsoft\WindowsApps\python.exe') { continue }
        if (Test-Python @($app.Source)) { return @($app.Source) }
    }
    throw 'Python 3.11 to 3.13 was not found. Install Python or open the Windows portable version.'
}

function Prepare-Environment {
    if (!(Test-Path -LiteralPath (Join-Path $venv 'pyvenv.cfg') -PathType Leaf) -or !(Test-Python @($python))) {
        $basePython = @(Find-Python)
        if (Test-Path -LiteralPath $venv) {
            $item = Get-Item -Force -LiteralPath $venv
            if ($item.FullName -ne $venv -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw 'The unusable .venv is a directory link. Move the link out of the project before using the source launcher.'
            }
            $savedName = '.venv.backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + "-$PID"
            Rename-Item -LiteralPath $venv -NewName $savedName
            Write-Host "Previous Python environment saved as $savedName"
            Add-Content -LiteralPath $startupLog -Value "Environment moved to $savedName" -Encoding UTF8
        }
        Write-Host 'Preparing the project Python environment...'
        if ((Invoke-Interpreter $basePython @('-m', 'venv', $venv)) -ne 0 -or !(Test-Python @($python))) {
            throw 'The project Python environment could not be created.'
        }
    }

    $requirements = Join-Path $project 'requirements.txt'
    $marker = Join-Path $venv '.ezDIC_requirements.txt'
    if (Test-Path -LiteralPath $requirements -PathType Leaf) {
        $requested = [Convert]::ToBase64String([IO.File]::ReadAllBytes($requirements))
        if (!(Test-Path -LiteralPath $marker -PathType Leaf) -or [Convert]::ToBase64String([IO.File]::ReadAllBytes($marker)) -cne $requested) {
            Write-Host 'Installing the project dependencies...'
            if ((Invoke-Interpreter @($python) @('-m', 'pip', 'install', '-r', $requirements)) -ne 0) {
                throw 'Dependency installation failed. Check the startup log and the connection to the package index.'
            }
            Copy-Item -LiteralPath $requirements -Destination $marker
        }
    }
}

$startupLog = $null
try {
    $state = if ($env:EZDIC_STATE_DIR) { $env:EZDIC_STATE_DIR } else { Join-Path $env:LOCALAPPDATA 'ezDIC\state' }
    $state = [IO.Path]::GetFullPath($state)
    [void][IO.Directory]::CreateDirectory($state)
    $startupLog = Join-Path $state ("startup-$Workbench-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + "-$PID.log")
    Set-Content -LiteralPath $startupLog -Value "$Workbench source startup in $project" -Encoding UTF8
    if (!(Test-Path -LiteralPath (Join-Path $project $entryFile) -PathType Leaf)) { throw "Source entry unavailable $entryFile" }

    $sha256 = [Security.Cryptography.SHA256]::Create()
    try { $key = [BitConverter]::ToString($sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($project.ToUpperInvariant()))) }
    finally { $sha256.Dispose() }
    $mutex = [Threading.Mutex]::new($false, "Local\StrainTrace-startup-$key")
    try {
        try { $acquired = $mutex.WaitOne(120000) }
        catch [Threading.AbandonedMutexException] { $acquired = $true }
        if (!$acquired) { throw 'The project environment is still being prepared in another launcher.' }
        try { Prepare-Environment }
        finally { $mutex.ReleaseMutex() }
    } finally { $mutex.Dispose() }

    $exitCode = Invoke-Interpreter @($python) $workbenchArgs
    Add-Content -LiteralPath $startupLog -Value "Exit code $exitCode" -Encoding UTF8
    if ($exitCode -ne 0) { Write-Host "Startup log $startupLog" }
    exit $exitCode
} catch {
    $message = $_.Exception.Message
    Write-Host $message
    if ($startupLog) {
        Add-Content -LiteralPath $startupLog -Value $message -Encoding UTF8 -ErrorAction SilentlyContinue
        Write-Host "Startup log $startupLog"
    }
    exit 1
}
