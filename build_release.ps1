param(
    [string]$Version = "0.1.4",
    [switch]$SmokeTest,
    [switch]$GuardTest,
    [string]$GuardPath,
    [switch]$BundleTest,
    [string]$BundlePath
)

$ErrorActionPreference = "Stop"

$Root = (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Root = [IO.Path]::GetFullPath($Root)
$VenvDir = Join-Path $Root ".venv-build"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$BuildRoot = Join-Path $Root "build"
$DistRoot = Join-Path $Root "dist"
$DistDir = Join-Path $DistRoot "ezDIC_Windows_x64"
$ReleaseRoot = Join-Path $Root "release"
$ReleaseDir = Join-Path $ReleaseRoot "ezDIC_Windows_x64"
$ZipPath = Join-Path $ReleaseRoot "ezDIC_Windows_x64_v$Version.zip"
$MainScriptName = "ezdic_frozen_entrypoint.py"
$CliScriptName = "ezdic_cli_entrypoint.py"

# Build this filename from code points so Windows PowerShell 5.1 can read this
# UTF-8 script even when the file has no BOM.
$ReadmeBundleFile = "README_" + [char]0x4F7F + [char]0x7528 + [char]0x8BF4 + [char]0x660E + ".txt"
$LegacyBundleFiles = @(
    $ReadmeBundleFile,
    "VERSION.txt",
    "NOTICE_Attribution_and_Usage.txt",
    "LICENSE.txt",
    "CITATION.cff"
)
$BundleFiles = @(
    "README.md",
    $ReadmeBundleFile,
    "RELEASE_NOTES_v0.2.0-dev.md",
    "VERSION.txt",
    "NOTICE_Attribution_and_Usage.txt",
    "LICENSE.txt",
    "CITATION.cff"
)
$ModernSourceFiles = @(
    "dic_virtual_extensometer_gui_v7_multi_roi_range.py",
    "ezdic_core.py",
    "ezdic_cli.py",
    "ezdic_benchmark.py",
    $MainScriptName,
    $CliScriptName,
    "schemas\run_config_v1.json",
    "benchmarks\cases_v1.json",
    "benchmarks\run_benchmark.py",
    "benchmarks\synthetic_cases.py",
    "ezDIC.spec"
)
# A copied legacy fixture used by the historical launcher tests does not carry
# the new core/schema files.  Only an actual modern source root enters the
# strict branch; a partial modern root fails closed.
$ModernMarkerFiles = @(
    "ezdic_core.py",
    "ezdic_cli.py",
    "ezdic_benchmark.py",
    $MainScriptName,
    $CliScriptName,
    "schemas\run_config_v1.json",
    "benchmarks\cases_v1.json",
    "benchmarks\run_benchmark.py",
    "benchmarks\synthetic_cases.py"
)
$SmokeRequested = $SmokeTest -or @("1", "true", "TRUE", "yes", "YES") -contains ([string]$env:EZDIC_BUILD_SMOKE_TEST)

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [IO.Path]::GetFullPath($Path)
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = $null
    $hasher = $null
    try {
        $stream = [IO.File]::OpenRead((Get-FullPath $Path))
        $hasher = [Security.Cryptography.SHA256]::Create()
        return ([BitConverter]::ToString($hasher.ComputeHash($stream)) -replace "-", "").ToLowerInvariant()
    }
    finally {
        if ($null -ne $hasher) { $hasher.Dispose() }
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

function Assert-NoReparsePath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $fullPath = Get-FullPath $Path
    $rootPart = [IO.Path]::GetPathRoot($fullPath)
    if ([string]::IsNullOrWhiteSpace($rootPart)) {
        throw "$Label has no filesystem root: $fullPath"
    }

    # Walk the lexical path one existing component at a time.  GetFullPath
    # alone is insufficient here: a junction/symlink can redirect an otherwise
    # contained path after the lexical containment check has passed.
    $current = $rootPart
    $lastExisting = $rootPart
    $tail = $fullPath.Substring($rootPart.Length)
    $components = $tail.Split(
        [char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar),
        [StringSplitOptions]::RemoveEmptyEntries
    )
    foreach ($component in $components) {
        $current = Join-Path $current $component
        $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
        if ($null -eq $item) {
            # The remaining suffix is not yet present.  Its parent chain has
            # been checked; callers must re-run this guard after creating it.
            break
        }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label contains a reparse point (junction/symlink): $current"
        }
        $lastExisting = $current
    }

    # Resolve the deepest existing component and reject a physical escape as a
    # second defence.  This also catches providers that expose a reparse target
    # without preserving the attribute on the final Get-Item result.
    try {
        $resolved = (Resolve-Path -LiteralPath $lastExisting -ErrorAction Stop).Path
        $resolvedFull = Get-FullPath $resolved
        if (-not $resolvedFull.Equals((Get-FullPath $lastExisting), [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Label resolves through an unexpected path: $fullPath -> $resolvedFull"
        }
    }
    catch {
        if ($_.Exception.Message -like "$Label resolves through an unexpected path:*") {
            throw
        }
        throw "$Label could not be resolved safely: $fullPath ($($_.Exception.Message))"
    }
    return $fullPath
}

function Assert-NoReparseTree {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $fullPath = Assert-NoReparsePath -Path $Path -Label $Label
    $rootItem = Get-Item -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue
    if ($null -eq $rootItem -or -not ($rootItem.PSIsContainer)) {
        return $fullPath
    }

    # Recursive copy/removal/compression can encounter a reparse point below a
    # normal top-level directory.  Walk only normal directories and fail as
    # soon as a junction/symlink is observed; never recurse through it.
    $pending = New-Object System.Collections.Generic.Queue[object]
    $pending.Enqueue($rootItem)
    while ($pending.Count -gt 0) {
        $directory = $pending.Dequeue()
        foreach ($child in @(Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction Stop)) {
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "$Label contains a reparse point below the target: $($child.FullName)"
            }
            if ($child.PSIsContainer) {
                $pending.Enqueue($child)
            }
        }
    }
    return $fullPath
}

function Assert-SafeBuildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $fullPath = Get-FullPath $Path
    $safeRoots = @($BuildRoot, $DistRoot, $ReleaseRoot, $VenvDir) | ForEach-Object { Get-FullPath $_ }
    $isSafe = $false
    foreach ($safeRoot in $safeRoots) {
        $rootWithSeparator = $safeRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if ($fullPath -eq $safeRoot -or $fullPath.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
            $isSafe = $true
            break
        }
    }
    if (-not $isSafe) {
        throw "$Label is outside the explicit build/dist/release/.venv-build cleanup roots: $fullPath"
    }
    [void](Assert-NoReparsePath -Path $fullPath -Label $Label)
    return $fullPath
}

function Remove-SafeBuildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $fullPath = Assert-SafeBuildPath -Path $Path -Label $Label
    if (Test-Path -LiteralPath $fullPath) {
        [void](Assert-NoReparseTree -Path $fullPath -Label $Label)
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$CommandParts
    )

    if ($CommandParts.Count -lt 1) {
        throw "$Label has no executable."
    }
    $exe = $CommandParts[0]
    $args = @()
    if ($CommandParts.Count -gt 1) {
        $args = $CommandParts[1..($CommandParts.Count - 1)]
    }
    try {
        & $exe @args
    }
    catch {
        throw "$Label could not start: $($_.Exception.Message)"
    }
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code ${exitCode}: $($CommandParts -join ' ')"
    }
}

function Test-PythonCommand {
    param([string[]]$CommandParts)
    if ($CommandParts.Count -lt 1) { return $false }
    try {
        Invoke-NativeChecked -Label "Python probe" -CommandParts ($CommandParts + @("--version"))
        return $true
    }
    catch { return $false }
}

function Invoke-PythonCommand {
    param(
        [string[]]$CommandParts,
        [string[]]$Arguments,
        [string]$Label = "Python command"
    )
    Invoke-NativeChecked -Label $Label -CommandParts (@($CommandParts) + @($Arguments))
}

function Invoke-WindowedChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )

    try {
        # PowerShell does not reliably populate $LASTEXITCODE for a GUI
        # subsystem executable.  Wait explicitly and inspect Process.ExitCode
        # so a frozen smoke cannot be reported before it has finished.
        $process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WindowStyle Hidden -Wait -PassThru
    }
    catch {
        throw "$Label could not start: $($_.Exception.Message)"
    }
    if ($process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($process.ExitCode): $Executable $($Arguments -join ' ')"
    }
}

function Test-VenvPython {
    param([string]$PythonPath)
    $venvRoot = Split-Path -Parent (Split-Path -Parent $PythonPath)
    [void](Assert-NoReparseTree -Path $venvRoot -Label ".venv-build before Python probe")
    if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) { return $false }
    if (-not (Test-PythonCommand -CommandParts @($PythonPath))) { return $false }
    try {
        Invoke-PythonCommand -CommandParts @($PythonPath) -Arguments @("-m", "pip", "--version") -Label "Build-environment pip probe"
        return $true
    }
    catch { return $false }
}

function Get-BasePythonCommand {
    $candidates = @(
        ,@("py", "-3.11")
        ,@("py", "-3")
        ,@("python")
    )
    foreach ($candidate in $candidates) {
        if (Test-PythonCommand -CommandParts $candidate) { return $candidate }
    }
    throw "Python was not found. Install Python 3.11+ or ensure python is on PATH."
}

function Ensure-BuildVenv {
    [void](Assert-NoReparseTree -Path $VenvDir -Label ".venv-build before validation")
    if (Test-Path -LiteralPath $VenvPython) {
        [void](Assert-NoReparseTree -Path $VenvDir -Label ".venv-build before existing Python probe")
        if (-not (Test-VenvPython -PythonPath $VenvPython)) {
            Write-Warning "Existing build virtual environment is not usable. Recreating .venv-build..."
            Remove-SafeBuildPath -Path $VenvDir -Label ".venv-build"
        }
    }
    if (!(Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        $BasePython = Get-BasePythonCommand
        [void](Assert-NoReparsePath -Path $VenvDir -Label ".venv-build creation target")
        try {
            Invoke-PythonCommand -CommandParts $BasePython -Arguments @("-m", "venv", $VenvDir) -Label "Build virtual-environment creation"
        }
        catch {
            [void](Assert-NoReparseTree -Path $VenvDir -Label ".venv-build after failed creation")
            if (Test-VenvPython -PythonPath $VenvPython) {
                Write-Warning "Virtual environment creation returned an error, but .venv-build Python and pip are usable. Continuing..."
            }
            else { throw }
        }
        [void](Assert-NoReparsePath -Path $VenvDir -Label ".venv-build created directory")
    }
    [void](Assert-NoReparseTree -Path $VenvDir -Label ".venv-build before final Python probe")
    if (-not (Test-VenvPython -PythonPath $VenvPython)) {
        throw "Created .venv-build, but its Python or pip is not usable: $VenvPython"
    }
}

function Read-RequiredText {
    param([Parameter(Mandatory = $true)][string]$Path)
    [void](Assert-NoReparsePath -Path $Path -Label "required path")
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required release file is missing: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8
}

function Test-ModernSourceInventory {
    $presentMarkers = @($ModernMarkerFiles | Where-Object { Test-Path -LiteralPath (Join-Path $Root $_) -PathType Leaf })
    if ($presentMarkers.Count -eq 0) { return $false }
    if ($presentMarkers.Count -ne $ModernMarkerFiles.Count) {
        $missing = @($ModernMarkerFiles | Where-Object { !(Test-Path -LiteralPath (Join-Path $Root $_) -PathType Leaf) })
        throw "Partial modern source inventory; missing: $($missing -join ', ')"
    }
    foreach ($relativePath in $ModernSourceFiles) {
        [void](Read-RequiredText (Join-Path $Root $relativePath))
    }
    return $true
}

function Test-SpecDataInventory {
    param(
        [Parameter(Mandatory = $true)][string]$SpecText,
        [Parameter(Mandatory = $true)][bool]$Modern
    )
    $requiredData = if ($Modern) { $BundleFiles + @("schemas/run_config_v1.json", "benchmarks/cases_v1.json") } else { $LegacyBundleFiles }
    foreach ($relativePath in $requiredData) {
        $token = [regex]::Escape(($relativePath -replace "\\", "/"))
        $normalizedSpec = $SpecText -replace "\\", "/"
        if ($normalizedSpec -notmatch $token) {
            throw "ezDIC.spec is missing required data entry: $relativePath"
        }
        if ($relativePath -notmatch "schemas/") {
            [void](Read-RequiredText (Join-Path $Root $relativePath))
        }
    }
    if ($Modern) {
        [void](Read-RequiredText (Join-Path $Root "schemas\run_config_v1.json"))
    }
}

function Test-ReleaseContracts {
    param([switch]$Modern)

    if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') {
        throw "-Version must be a semantic version such as 0.1.4; received '$Version'."
    }

    $versionText = Read-RequiredText (Join-Path $Root "VERSION.txt")
    if ($versionText -notmatch '(?m)^\s*ezDIC\s+v([0-9]+\.[0-9]+\.[0-9]+)\s*$') {
        throw "VERSION.txt must start with 'ezDIC v<version>'."
    }
    $versionFileValue = $Matches[1]
    $guiSourceText = Read-RequiredText (Join-Path $Root "dic_virtual_extensometer_gui_v7_multi_roi_range.py")
    if ($guiSourceText -notmatch '(?m)^\s*APP_VERSION\s*=\s*["'']([^"'']+)["'']') {
        throw "dic_virtual_extensometer_gui_v7_multi_roi_range.py does not define a readable APP_VERSION."
    }
    $appVersionValue = $Matches[1]
    if ($Version -ne $versionFileValue -or $Version -ne $appVersionValue) {
        throw "Version mismatch: -Version=$Version, VERSION.txt=$versionFileValue, APP_VERSION=$appVersionValue."
    }

    $citationText = Read-RequiredText (Join-Path $Root "CITATION.cff")
    if ($citationText -notmatch '(?m)^\s*version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$') {
        throw "CITATION.cff does not define a readable version."
    }
    if ($Matches[1] -ne $Version) { throw "CITATION.cff version $($Matches[1]) does not match $Version." }
    # An unreleased preparation candidate must not invent a release date.
    $citationDate = $null
    if ($citationText -match '(?m)^date-released:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$') {
        $citationDate = $Matches[1]
    }

    $zenodoText = Read-RequiredText (Join-Path $Root ".zenodo.json")
    try { $zenodo = $zenodoText | ConvertFrom-Json }
    catch { throw ".zenodo.json is not valid JSON: $($_.Exception.Message)" }
    if ([string]$zenodo.version -ne $Version) { throw ".zenodo.json version $($zenodo.version) does not match $Version." }
    $zenodoDate = $zenodo.PSObject.Properties["publication_date"]
    $publicationDate = if ($null -eq $zenodoDate) { $null } else { [string]$zenodoDate.Value }
    if ([string]$publicationDate -ne [string]$citationDate) {
        throw "CITATION.cff and .zenodo.json release dates must agree, or both be absent for an unreleased candidate."
    }

    $specText = Read-RequiredText (Join-Path $Root "ezDIC.spec")
    if ($specText -notmatch 'name\s*=\s*[''\"]ezDIC[''\"]') { throw "ezDIC.spec does not define the ezDIC executable." }
    if ($specText -notmatch 'console=False') { throw "ezDIC.spec must build a windowed executable (console=False)." }
    if ($Modern -and $specText -notmatch 'name\s*=\s*[''\"]ezDIC-cli[''\"]') { throw "ezDIC.spec does not define the console ezDIC-cli executable." }
    Test-SpecDataInventory -SpecText $specText -Modern ([bool]$Modern)

    $requirementsText = Read-RequiredText (Join-Path $Root "requirements.txt")
    if ($requirementsText -match '(?im)^\s*originpro\s*(?:[<>=!~].*)?$') {
        throw "requirements.txt must not include optional originpro; use requirements-origin.txt."
    }
    $originRequirementsText = Read-RequiredText (Join-Path $Root "requirements-origin.txt")
    if ($originRequirementsText -notmatch '(?im)^\s*originpro\s*(?:[<>=!~].*)?$') {
        throw "requirements-origin.txt must declare originpro."
    }
    $buildRequirementsText = Read-RequiredText (Join-Path $Root "requirements-build.txt")
    if ($buildRequirementsText -notmatch '(?im)^\s*-r\s+requirements\.txt\s*$') {
        throw "requirements-build.txt must include requirements.txt."
    }
    if ($buildRequirementsText -notmatch '(?im)pyinstaller\s*==') { throw "requirements-build.txt must pin PyInstaller." }
    if ($buildRequirementsText -notmatch '(?im)pytest\s*==') { throw "requirements-build.txt must pin pytest." }

    if ($Modern) { Test-ModernSourceInventory | Out-Null }
    Write-Host "Release contract: version $Version; source/portable manifest verified (modern=$Modern)."
}

function Invoke-SourceSmoke {
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    $markerRoot = Join-Path ([IO.Path]::GetTempPath()) ("ezdic-source-smoke-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $markerRoot -Force | Out-Null
    $markerPath = Join-Path $markerRoot "smoke.json"
    $oldMarker = $env:EZDIC_FROZEN_SMOKE_MARKER
    $env:EZDIC_FROZEN_SMOKE_MARKER = $markerPath
    try {
        Invoke-PythonCommand -CommandParts @($PythonPath) -Arguments @("-B", $MainScriptName, "--smoke-test") -Label "source/frozen contract smoke"
    }
    finally {
        if ($null -eq $oldMarker) { Remove-Item Env:EZDIC_FROZEN_SMOKE_MARKER -ErrorAction SilentlyContinue }
        else { $env:EZDIC_FROZEN_SMOKE_MARKER = $oldMarker }
    }
    $markerText = Read-RequiredText $markerPath
    try { $marker = $markerText | ConvertFrom-Json }
    catch { throw "Source smoke marker is not valid JSON: $markerPath" }
    if ($marker.smoke -ne "passed") { throw "Source smoke did not pass: $markerText" }
    Write-Host "Source/frozen contract smoke passed: $markerPath"
}

function Test-BundleInventory {
    param([Parameter(Mandatory = $true)][string]$BundleRoot)
    [void](Assert-NoReparseTree -Path $BundleRoot -Label "portable bundle")
    $required = @(
        "ezDIC.exe",
        "ezDIC-cli.exe",
        "_internal\schemas\run_config_v1.json",
        "_internal\sources\dic_virtual_extensometer_gui_v7_multi_roi_range.py",
        "_internal\sources\ezdic_core.py",
        "_internal\sources\ezdic_cli.py",
        "_internal\sources\ezdic_benchmark.py",
        "_internal\benchmarks\cases_v1.json",
        "_internal\sources\benchmarks\run_benchmark.py",
        "_internal\sources\benchmarks\synthetic_cases.py"
    ) + $BundleFiles
    foreach ($relativePath in $required) {
        $path = Join-Path $BundleRoot $relativePath
        [void](Assert-NoReparsePath -Path $path -Label "portable bundle entry $relativePath")
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Portable package is missing required file: $relativePath"
        }
    }
    $internal = Join-Path $BundleRoot "_internal"
    [void](Assert-NoReparsePath -Path $internal -Label "portable _internal directory")
    if (!(Test-Path -LiteralPath $internal -PathType Container)) {
        throw "Portable package is missing PyInstaller _internal directory: $internal"
    }
    Write-Host "Portable bundle inventory verified: $BundleRoot"
}

function Assert-FrozenGuiProvenance {
    param([Parameter(Mandatory = $true)][string]$BundleRoot)

    $relative = "_internal\sources\dic_virtual_extensometer_gui_v7_multi_roi_range.py"
    $bundlePath = Join-Path $BundleRoot $relative
    $sourcePath = Join-Path $Root "dic_virtual_extensometer_gui_v7_multi_roi_range.py"
    [void](Assert-NoReparsePath -Path $bundlePath -Label "frozen GUI provenance source")
    [void](Assert-NoReparsePath -Path $sourcePath -Label "source GUI provenance file")
    if (!(Test-Path -LiteralPath $bundlePath -PathType Leaf)) {
        throw "Frozen GUI provenance source is missing: $bundlePath"
    }
    if (!(Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Source GUI provenance file is missing: $sourcePath"
    }
    $bundleHash = Get-Sha256 $bundlePath
    $sourceHash = Get-Sha256 $sourcePath
    if ($bundleHash -notmatch '^[0-9a-fA-F]{64}$' -or $bundleHash -ne $sourceHash) {
        throw "Frozen GUI provenance hash mismatch: $relative"
    }
    Write-Host "Frozen GUI provenance verified: $relative ($bundleHash)"
    return $bundleHash
}

function Assert-JsonProperties {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ($null -eq $Object -or $null -eq $Object.PSObject) {
        throw "$Label is missing or is not a JSON object."
    }
    foreach ($name in $Names) {
        if ($null -eq $Object.PSObject.Properties[$name]) {
            throw "$Label is missing required field: $name"
        }
    }
    return $Object
}

function Assert-BenchmarkV6Report {
    param(
        [Parameter(Mandatory = $true)][string]$ReportPath,
        [Parameter(Mandatory = $true)][string]$CasesPath,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $reportText = Read-RequiredText $ReportPath
    try { $report = $reportText | ConvertFrom-Json }
    catch { throw "$Label benchmark report is not valid JSON: $ReportPath" }
    [void](Assert-JsonProperties -Object $report -Names @("report_version", "cases_version", "locked_cases_hash", "overall_pass", "exit_code", "quality_auc", "gate_summary", "gates", "cases", "quality_error", "quality_score", "quality_contract", "artifacts", "code") -Label "$Label benchmark report")
    if ($reportText -match '"(?:status|threshold_status)"\s*:\s*"CALIBRATED"') { throw "$Label must reject a calibrated quality-threshold claim." }
    if ([string]$report.report_version -ne "ezdic-benchmark-report-v6") { throw "$Label benchmark report_version is not v6." }
    if ([string]$report.cases_version -ne "ezdic-benchmark-cases-v4") { throw "$Label cases_version is not the locked v6 document." }
    if ([string]$report.locked_cases_hash -ne "0fb244586def285fe0ca6b816eac588ddae8d4a0de0ce1a313b183bf369db138") { throw "$Label locked case hash is not the v6 contract." }
    if ($report.overall_pass -ne $true -or [int]$report.exit_code -ne 0) { throw "$Label locked benchmark did not pass." }
    $gateNames = @("code_provenance", "csv_exists", "csv_rows", "near_1d_preflight_pass", "numeric_baseline_pass", "quality_ranking_pass", "quality_threshold_evaluated", "quality_threshold_pass")
    [void](Assert-JsonProperties -Object $report.gate_summary -Names $gateNames -Label "$Label gate_summary")
    [void](Assert-JsonProperties -Object $report.gates -Names $gateNames -Label "$Label gates")
    foreach ($gate in @("code_provenance", "csv_exists", "csv_rows", "near_1d_preflight_pass", "numeric_baseline_pass", "quality_ranking_pass")) {
        if ($report.gate_summary.PSObject.Properties[$gate].Value -ne $true -or $report.gates.PSObject.Properties[$gate].Value -ne $true) { throw "$Label benchmark gate failed: $gate" }
    }
    foreach ($gateObject in @($report.gate_summary, $report.gates)) {
        if ($gateObject.PSObject.Properties["quality_threshold_evaluated"].Value -ne $false -or $null -ne $gateObject.PSObject.Properties["quality_threshold_pass"].Value) { throw "$Label must not evaluate or pass an illustrative quality threshold." }
    }

    try { $cases = (Read-RequiredText $CasesPath) | ConvertFrom-Json }
    catch { throw "$Label locked case definition is not valid JSON: $CasesPath" }
    [void](Assert-JsonProperties -Object $cases -Names @("version", "cases", "contract", "quality_contract", "thresholds") -Label "$Label locked case definition")
    if ([string]$cases.version -ne "ezdic-benchmark-cases-v4") { throw "$Label locked case definition version is wrong." }
    $expectedIds = @("small_translation", "large_translation", "small_affine_strain", "near_1d_periodic")
    $caseList = @($cases.cases)
    $caseIds = @($caseList | ForEach-Object { [string]$_.case_id })
    if (($caseIds -join "|") -ne ($expectedIds -join "|")) { throw "$Label required v6 cases are missing or reordered." }
    $caseById = @{}
    foreach ($case in $caseList) { $caseById[[string]$case.case_id] = $case }
    $smallCase = $caseById["small_translation"]
    $largeCase = $caseById["large_translation"]
    $affineCase = $caseById["small_affine_strain"]
    $nearCase = $caseById["near_1d_periodic"]
    if ([double]$smallCase.translation[0] -ne 2.3 -or [double]$smallCase.translation[1] -ne -1.2 -or [int]$smallCase.search_radius -ne 8 -or [int]$smallCase.pyramid_levels -ne 1) { throw "$Label small translation definition is not [2.3,-1.2]/pyramid1/radius8." }
    if ([double]$largeCase.translation[0] -ne 28.0 -or [double]$largeCase.translation[1] -ne -18.0 -or [int]$largeCase.pyramid_levels -ne 3 -or [int]$largeCase.search_radius -ne 8) { throw "$Label large translation definition is not [28,-18]/pyramid3/radius8." }
    if ([string]$affineCase.kind -ne "affine" -or [int]$affineCase.search_radius -ne 8) { throw "$Label affine case definition is incomplete." }
    if ([string]$nearCase.kind -ne "near_1d_periodic") { throw "$Label near-1D case is not typed near_1d_periodic." }
    [void](Assert-JsonProperties -Object $cases.quality_contract -Names @("version", "error_tolerance_px", "roc_auc_min", "minimum_bad_label_count", "minimum_corruption_row_count", "quality_validity_required", "illustrative_quality_threshold", "corruption_panel") -Label "$Label quality_contract")
    if ([string]$cases.quality_contract.version -ne "quality_score_v1" -or [double]$cases.quality_contract.error_tolerance_px -ne 0.25 -or [double]$cases.quality_contract.roc_auc_min -lt 0.90) { throw "$Label quality-score v1 contract is incomplete." }
    [void](Assert-JsonProperties -Object $cases.quality_contract.illustrative_quality_threshold -Names @("status", "quality_accept_score_min") -Label "$Label illustrative threshold")
    if ([string]$cases.quality_contract.illustrative_quality_threshold.status -ne "NOT_CALIBRATED") { throw "$Label illustrative threshold must remain NOT_CALIBRATED." }

    $caseByReportId = @{}
    foreach ($case in @($report.cases)) { $caseByReportId[[string]$case.case_id] = $case }
    foreach ($caseId in @("small_translation", "large_translation", "small_affine_strain")) {
        [void](Assert-JsonProperties -Object $caseByReportId[$caseId] -Names @("status", "benchmark_pass", "metrics") -Label "$Label case $caseId")
        if ($caseByReportId[$caseId].status -ne "PASS" -or $caseByReportId[$caseId].benchmark_pass -ne $true) { throw "$Label required case failed: $caseId" }
    }
    $smallMetrics = $caseByReportId["small_translation"].metrics
    $largeMetrics = $caseByReportId["large_translation"].metrics
    $affineMetrics = $caseByReportId["small_affine_strain"].metrics
    [void](Assert-JsonProperties -Object $smallMetrics -Names @("rmse_px", "p95_error_px", "max_error_px", "false_accept_count", "quality_false_accept_count") -Label "$Label small metrics")
    [void](Assert-JsonProperties -Object $largeMetrics -Names @("rmse_px", "p95_error_px", "max_error_px", "false_accept_count", "quality_false_accept_count") -Label "$Label large metrics")
    [void](Assert-JsonProperties -Object $affineMetrics -Names @("rmse_px", "p95_error_px", "max_error_px", "strain_component_abs_error_max", "strain_consistency_abs_error_max") -Label "$Label affine metrics")
    if ([math]::Abs([double]$smallMetrics.rmse_px - 0.000242870585615446) -gt 1e-6 -or [math]::Abs([double]$smallMetrics.p95_error_px - 0.000471525609103834) -gt 1e-6 -or [math]::Abs([double]$smallMetrics.max_error_px - 0.000541210285865718) -gt 1e-6) { throw "$Label small clean metrics differ from v6." }
    if ([math]::Abs([double]$largeMetrics.rmse_px - 0.000142756487216106) -gt 1e-6 -or [math]::Abs([double]$largeMetrics.p95_error_px - 0.000199604933633678) -gt 1e-6 -or [math]::Abs([double]$largeMetrics.max_error_px - 0.000649429296399418) -gt 1e-6) { throw "$Label large clean metrics differ from v6." }
    if ([math]::Abs([double]$affineMetrics.rmse_px - 0.000144908819464799) -gt 1e-6 -or [math]::Abs([double]$affineMetrics.p95_error_px - 0.000356810906530228) -gt 1e-6 -or [math]::Abs([double]$affineMetrics.max_error_px - 0.000639258340868429) -gt 1e-6 -or [math]::Abs([double]$affineMetrics.strain_component_abs_error_max - 0.000012809688012261) -gt 1e-6 -or [math]::Abs([double]$affineMetrics.strain_consistency_abs_error_max - 0.000012733433353507) -gt 1e-6) { throw "$Label affine clean metrics differ from v6." }
    if ([int]$smallMetrics.false_accept_count -ne 0 -or [int]$smallMetrics.quality_false_accept_count -ne 0 -or [int]$largeMetrics.false_accept_count -ne 0 -or [int]$largeMetrics.quality_false_accept_count -ne 0) { throw "$Label clean baseline contains unexpected false accepts." }

    $nearReport = $caseByReportId["near_1d_periodic"]
    [void](Assert-JsonProperties -Object $nearReport -Names @("status", "failure_code", "outcome", "solver_calls", "successful_export_artifacts", "texture_preflight") -Label "$Label near-1D case")
    if ([string]$nearReport.status -ne "REJECTED" -or [string]$nearReport.failure_code -ne "AMBIGUOUS_TEXTURE" -or [string]$nearReport.outcome -ne "AMBIGUOUS_TEXTURE" -or [int]$nearReport.solver_calls -ne 0 -or [int]$nearReport.successful_export_artifacts -ne 0) { throw "$Label near-1D typed rejection/zero-artifact contract failed." }
    if ([string]$nearReport.texture_preflight.code -ne "AMBIGUOUS_TEXTURE") { throw "$Label near-1D preflight code is not typed." }

    $qualityError = $report.quality_error
    [void](Assert-JsonProperties -Object $qualityError -Names @("version", "error_tolerance_px", "point_count", "good_label_count", "bad_label_count", "finite_error_label_count", "ranking_point_count", "ranking_good_label_count", "ranking_bad_label_count", "ranking_rejected_bad_count", "corruption_row_count", "roc_auc", "roc_auc_min", "false_accept_count", "false_accept_rate", "ranking_false_accept_count", "ranking_false_accept_rate", "quality_threshold_evaluated", "quality_threshold_pass", "threshold_status") -Label "$Label quality_error")
    if ([string]$qualityError.version -ne "quality_score_v1" -or [double]$qualityError.error_tolerance_px -ne 0.25 -or [int]$qualityError.point_count -ne 551 -or [int]$qualityError.good_label_count -ne 551 -or [int]$qualityError.bad_label_count -ne 0 -or [int]$qualityError.finite_error_label_count -ne 551 -or [int]$qualityError.ranking_point_count -ne 567 -or [int]$qualityError.ranking_good_label_count -ne 551 -or [int]$qualityError.ranking_bad_label_count -ne 16 -or [int]$qualityError.ranking_rejected_bad_count -ne 16 -or [int]$qualityError.corruption_row_count -ne 4) { throw "$Label quality-score label populations are not the v6 contract." }
    if ([math]::Abs([double]$qualityError.roc_auc - 1.0) -gt 1e-12 -or [math]::Abs([double]$report.quality_auc - [double]$qualityError.roc_auc) -gt 1e-12 -or [double]$qualityError.roc_auc_min -lt 0.90 -or [int]$qualityError.false_accept_count -ne 0 -or $null -ne $qualityError.false_accept_rate -or [int]$qualityError.ranking_false_accept_count -ne 0 -or [double]$qualityError.ranking_false_accept_rate -ne 0.0) { throw "$Label quality ranking/finite-error rates differ from v6." }
    if ($qualityError.quality_threshold_evaluated -ne $false -or $null -ne $qualityError.quality_threshold_pass -or [string]$qualityError.threshold_status -ne "NOT_CALIBRATED") { throw "$Label quality threshold must remain unevaluated/NOT_CALIBRATED." }
    [void](Assert-JsonProperties -Object $report.quality_score -Names @("version", "roc_auc", "roc_auc_min", "error_tolerance_px", "ratio_direction", "threshold_status", "quality_threshold_evaluated", "quality_threshold_pass") -Label "$Label quality_score")
    if ([string]$report.quality_score.version -ne "quality_score_v1" -or [string]$report.quality_score.ratio_direction -ne "best_over_second" -or [string]$report.quality_score.threshold_status -ne "NOT_CALIBRATED" -or $report.quality_score.quality_threshold_evaluated -ne $false -or $null -ne $report.quality_score.quality_threshold_pass) { throw "$Label quality_score calibration status is invalid." }
    [void](Assert-JsonProperties -Object $report.quality_contract -Names @("version", "error_tolerance_px", "roc_auc_min", "illustrative_quality_threshold", "quality_validity_required") -Label "$Label report quality_contract")
    [void](Assert-JsonProperties -Object $report.quality_contract.illustrative_quality_threshold -Names @("status", "quality_accept_score_min") -Label "$Label report illustrative threshold")
    if ([string]$report.quality_contract.version -ne "quality_score_v1" -or [string]$report.quality_contract.illustrative_quality_threshold.status -ne "NOT_CALIBRATED") { throw "$Label report quality_contract is calibrated or incomplete." }

    [void](Assert-JsonProperties -Object $report.artifacts -Names @("benchmark_report_csv", "benchmark_report_csv_sha256") -Label "$Label artifacts")
    $csvName = [string]$report.artifacts.benchmark_report_csv
    if ($csvName -ne "benchmark_report.csv") { throw "$Label report does not point to benchmark_report.csv." }
    $csvPath = Join-Path (Split-Path -Parent (Get-FullPath $ReportPath)) $csvName
    [void](Assert-NoReparsePath -Path $csvPath -Label "$Label benchmark CSV")
    if (!(Test-Path -LiteralPath $csvPath -PathType Leaf)) { throw "$Label benchmark CSV is missing: $csvPath" }
    $csvInfo = Get-Item -LiteralPath $csvPath -Force
    if ([int64]$csvInfo.Length -le 0) { throw "$Label benchmark CSV is empty: $csvPath" }
    $csvHash = Get-Sha256 $csvPath
    if ($csvHash -ne ([string]$report.artifacts.benchmark_report_csv_sha256).ToLowerInvariant()) { throw "$Label benchmark CSV hash does not match the JSON report." }
    # CSV bytes are content-checked against this run; numeric baselines above use explicit tolerances.

    $hashPattern = '^[0-9a-fA-F]{64}$'
    $codeFiles = @{
        "benchmark_source_sha256" = "ezdic_benchmark.py"
        "benchmark_runner_source_sha256" = "benchmarks\run_benchmark.py"
        "synthetic_cases_source_sha256" = "benchmarks\synthetic_cases.py"
        "cases_json_sha256" = "benchmarks\cases_v1.json"
        "core_source_sha256" = "ezdic_core.py"
        "cli_source_sha256" = "ezdic_cli.py"
    }
    [void](Assert-JsonProperties -Object $report.code -Names @($codeFiles.Keys) -Label "$Label code")
    foreach ($field in $codeFiles.Keys) {
        $reported = [string]$report.code.PSObject.Properties[$field].Value
        if ($reported -notmatch $hashPattern) { throw "$Label report is missing a real code hash: $field" }
        $sourcePath = Join-Path $Root $codeFiles[$field]
        [void](Assert-NoReparsePath -Path $sourcePath -Label "$Label source $field")
        $observed = Get-Sha256 $sourcePath
        if ($reported.ToLowerInvariant() -ne $observed) { throw "$Label code/data hash mismatch: $field" }
    }
    [void](Assert-JsonProperties -Object $report.code.benchmark_facade -Names @("source_sha256") -Label "$Label benchmark facade code")
    if ([string]$report.code.benchmark_facade.source_sha256 -notmatch $hashPattern -or [string]$report.code.benchmark_facade.source_sha256 -ne $report.code.benchmark_source_sha256) { throw "$Label benchmark facade hash is missing or inconsistent." }
    [void](Assert-NoReparsePath -Path (Join-Path $Root "dic_virtual_extensometer_gui_v7_multi_roi_range.py") -Label "$Label GUI provenance source")
    if ((Get-Sha256 (Join-Path $Root "dic_virtual_extensometer_gui_v7_multi_roi_range.py")) -notmatch $hashPattern) { throw "$Label GUI provenance hash is unavailable." }
    return $report
}

# Check the repository root before Set-Location or any build/output operation.
# A root junction would otherwise make every subsequent lexical containment
# check meaningless.  -GuardTest is a non-building harness used by the static
# release tests to exercise this guard against a real junction/symlink.
[void](Assert-NoReparsePath -Path $Root -Label "repository root")
Set-Location $Root
if ($GuardTest) {
    if ([string]::IsNullOrWhiteSpace($GuardPath)) {
        throw "-GuardTest requires -GuardPath."
    }
    [void](Assert-SafeBuildPath -Path $GuardPath -Label "guard test target")
    [void](Assert-NoReparseTree -Path $GuardPath -Label "guard test tree")
    Write-Host "Reparse guard accepted: $GuardPath"
    exit 0
}
if ($BundleTest) {
    if ([string]::IsNullOrWhiteSpace($BundlePath)) {
        throw "-BundleTest requires -BundlePath."
    }
    Test-BundleInventory -BundleRoot $BundlePath
    [void](Assert-FrozenGuiProvenance -BundleRoot $BundlePath)
    Write-Host "Bundle inventory/provenance test accepted: $BundlePath"
    exit 0
}

if ($SmokeRequested) {
    Ensure-BuildVenv
    Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("--version") -Label "Smoke-test Python"
    Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-m", "pip", "--version") -Label "Smoke-test pip"

    $modern = Test-ModernSourceInventory
    if ($modern) {
        Test-ReleaseContracts -Modern
        $compileFiles = @($ModernSourceFiles | Where-Object { $_ -like "*.py" } | ForEach-Object { Join-Path $Root $_ })
        Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments (@("-B", "-m", "py_compile") + $compileFiles) -Label "source entrypoint/module compilation"
        Invoke-SourceSmoke -PythonPath $VenvPython
        $smokeBenchmarkRoot = Join-Path ([IO.Path]::GetTempPath()) ("ezdic-smoke-benchmark-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $smokeBenchmarkRoot -Force | Out-Null
        Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-B", "ezdic_cli.py", "benchmark", "--cases", (Join-Path $Root "benchmarks\cases_v1.json"), "--output", $smokeBenchmarkRoot) -Label "source v6 benchmark smoke"
        [void](Assert-BenchmarkV6Report -ReportPath (Join-Path $smokeBenchmarkRoot "benchmark_report.json") -CasesPath (Join-Path $Root "benchmarks\cases_v1.json") -Label "source smoke")
        Write-Host "ezDIC build smoke test: modern source contract passed."
    }
    else {
        # Compatibility mode is solely for the historical copied-fixture tests
        # that intentionally contain only the v0.1.4 launcher metadata.
        Test-ReleaseContracts
        Write-Host "ezDIC build smoke test: legacy metadata fixture passed."
    }
    Write-Host "Project: $Root"
    exit 0
}

Test-ModernSourceInventory | Out-Null
Test-ReleaseContracts -Modern
Ensure-BuildVenv

Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-m", "pip", "install", "--upgrade", "pip") -Label "pip upgrade"
Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-m", "pip", "install", "-r", (Join-Path $Root "requirements-build.txt")) -Label "build requirements installation"

$compileFiles = @($ModernSourceFiles | Where-Object { $_ -like "*.py" } | ForEach-Object { Join-Path $Root $_ })
Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments (@("-B", "-m", "py_compile") + $compileFiles) -Label "all entrypoint/module compilation"
Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-B", "-m", "pytest", "-q", "-p", "no:cacheprovider") -Label "pytest"

$benchmarkRoot = Join-Path ([IO.Path]::GetTempPath()) ("ezdic-build-benchmark-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $benchmarkRoot -Force | Out-Null
Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-B", "ezdic_cli.py", "benchmark", "--cases", (Join-Path $Root "benchmarks\cases_v1.json"), "--output", $benchmarkRoot) -Label "locked v6 synthetic benchmark"
$benchmarkReportPath = Join-Path $benchmarkRoot "benchmark_report.json"
$benchmarkReport = Assert-BenchmarkV6Report -ReportPath $benchmarkReportPath -CasesPath (Join-Path $Root "benchmarks\cases_v1.json") -Label "source"

Remove-SafeBuildPath -Path $BuildRoot -Label "build output"
Remove-SafeBuildPath -Path $DistRoot -Label "dist output"
 [void](Assert-SafeBuildPath -Path $BuildRoot -Label "build output creation target")
 [void](Assert-SafeBuildPath -Path $DistRoot -Label "dist output creation target")
Invoke-PythonCommand -CommandParts @($VenvPython) -Arguments @("-B", "-m", "PyInstaller", (Join-Path $Root "ezDIC.spec"), "--clean", "--noconfirm") -Label "PyInstaller"

 [void](Assert-NoReparseTree -Path $DistDir -Label "dist bundle")
if (!(Test-Path -LiteralPath $DistDir -PathType Container)) { throw "PyInstaller output was not found: $DistDir" }

if (Test-Path -LiteralPath $ReleaseDir) { Remove-SafeBuildPath -Path $ReleaseDir -Label "release package" }
 [void](Assert-SafeBuildPath -Path $ReleaseRoot -Label "release root creation target")
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
 [void](Assert-SafeBuildPath -Path $ReleaseRoot -Label "release root")
 [void](Assert-SafeBuildPath -Path $ReleaseDir -Label "release package creation target")
Copy-Item -LiteralPath $DistDir -Destination $ReleaseDir -Recurse -Force
 [void](Assert-NoReparseTree -Path $ReleaseDir -Label "copied release package")
foreach ($bundleFile in $BundleFiles) {
    $source = Join-Path $Root $bundleFile
    $destination = Join-Path $ReleaseDir $bundleFile
    [void](Assert-NoReparsePath -Path $source -Label "release source $bundleFile")
    [void](Assert-SafeBuildPath -Path $destination -Label "release destination $bundleFile")
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
Test-BundleInventory -BundleRoot $ReleaseDir
[void](Assert-FrozenGuiProvenance -BundleRoot $DistDir)

$frozenMarkerRoot = Join-Path ([IO.Path]::GetTempPath()) ("ezdic-frozen-smoke-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $frozenMarkerRoot -Force | Out-Null
$frozenMarker = Join-Path $frozenMarkerRoot "smoke.json"
$frozenExe = Join-Path $DistDir "ezDIC.exe"
$oldFrozenMarker = $env:EZDIC_FROZEN_SMOKE_MARKER
$env:EZDIC_FROZEN_SMOKE_MARKER = $frozenMarker
try {
    Invoke-WindowedChecked -Label "frozen GUI contract smoke" -Executable $frozenExe -Arguments @("--smoke-test")
}
finally {
    if ($null -eq $oldFrozenMarker) { Remove-Item Env:EZDIC_FROZEN_SMOKE_MARKER -ErrorAction SilentlyContinue }
    else { $env:EZDIC_FROZEN_SMOKE_MARKER = $oldFrozenMarker }
}
$frozenMarkerText = Read-RequiredText $frozenMarker
try { $frozenResult = $frozenMarkerText | ConvertFrom-Json }
catch { throw "Frozen smoke marker is not valid JSON: $frozenMarker" }
if ($frozenResult.smoke -ne "passed") { throw "Frozen smoke did not pass: $frozenMarkerText" }
$frozenHashSources = @{
    "core_sha256" = "ezdic_core.py"
    "cli_sha256" = "ezdic_cli.py"
    "benchmark_sha256" = "ezdic_benchmark.py"
    "schema_sha256" = "schemas\run_config_v1.json"
}
foreach ($field in $frozenHashSources.Keys) {
    $reported = [string]$frozenResult.PSObject.Properties[$field].Value
    if ($reported -notmatch '^[0-9a-fA-F]{64}$') { throw "Frozen smoke marker is missing a real code/data hash: $field" }
    $sourcePath = Join-Path $Root $frozenHashSources[$field]
    [void](Assert-NoReparsePath -Path $sourcePath -Label "frozen smoke source $field")
    $observed = Get-Sha256 $sourcePath
    if ($reported.ToLowerInvariant() -ne $observed) { throw "Frozen smoke hash mismatch: $field" }
}

$cliExe = Join-Path $DistDir "ezDIC-cli.exe"
Invoke-NativeChecked -Label "frozen CLI help smoke" -CommandParts @($cliExe, "--help")
$frozenBenchmarkRoot = Join-Path ([IO.Path]::GetTempPath()) ("ezdic-frozen-benchmark-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $frozenBenchmarkRoot -Force | Out-Null
Invoke-NativeChecked -Label "frozen CLI locked v6 benchmark" -CommandParts @($cliExe, "benchmark", "--output", $frozenBenchmarkRoot)
$frozenBenchmarkReportPath = Join-Path $frozenBenchmarkRoot "benchmark_report.json"
$frozenBenchmarkReport = Assert-BenchmarkV6Report -ReportPath $frozenBenchmarkReportPath -CasesPath (Join-Path $Root "benchmarks\cases_v1.json") -Label "frozen CLI"

if (Test-Path -LiteralPath $ZipPath) { Remove-SafeBuildPath -Path $ZipPath -Label "release archive" }
 [void](Assert-NoReparseTree -Path $ReleaseDir -Label "release package before compression")
 [void](Assert-SafeBuildPath -Path $ZipPath -Label "release archive creation target")
Compress-Archive -Path $ReleaseDir -DestinationPath $ZipPath -Force
 [void](Assert-SafeBuildPath -Path $ZipPath -Label "release archive")
if (!(Test-Path -LiteralPath $ZipPath -PathType Leaf)) { throw "Release zip was not created: $ZipPath" }

Write-Host "Release folder: $ReleaseDir"
Write-Host "Release zip: $ZipPath"
Write-Host "Locked benchmark report: $benchmarkReportPath"
Write-Host "Frozen CLI benchmark report: $frozenBenchmarkReportPath"
Write-Host "Frozen smoke marker: $frozenMarker"
