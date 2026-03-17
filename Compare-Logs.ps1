# =============================================================================
# Compare-Logs.ps1
# Structural and statistical comparison between real M365 audit logs
# and the synthetic data produced by generate_logs.py.
#
# USAGE:
#   .\Compare-Logs.ps1
#   .\Compare-Logs.ps1 -RealFile real_logs.json -SyntheticFile data\train.jsonl
#
# OUTPUT:
#   Console report + comparison_report.json
# =============================================================================

param(
    [string]$RealFile      = "real_logs.json",
    [string]$SyntheticFile = "data\train.jsonl",
    [string]$ReportFile    = "comparison_report.json"
)

Write-Host "`n=== Compare-Logs.ps1 ===" -ForegroundColor Cyan

# -- Load data -----------------------------------------------------------------

if (-not (Test-Path $RealFile)) {
    Write-Error "Real logs not found: $RealFile - run Fetch-AuditLogs.ps1 first."
    exit 1
}
if (-not (Test-Path $SyntheticFile)) {
    Write-Error "Synthetic logs not found: $SyntheticFile - run generate_logs.py first."
    exit 1
}

Write-Host "Loading real logs from $RealFile..."
$real = Get-Content $RealFile -Raw | ConvertFrom-Json
Write-Host "  $($real.Count) real events loaded"

Write-Host "Loading synthetic logs from $SyntheticFile..."
# JSONL: one JSON object per line
$synthetic = Get-Content $SyntheticFile |
    Where-Object { $_.Trim() -ne "" } |
    ForEach-Object { $_ | ConvertFrom-Json }
Write-Host "  $($synthetic.Count) synthetic events loaded"


# -- Helper --------------------------------------------------------------------

function Distribution($items, $property) {
    # Returns hashtable: value -> percentage
    $total  = $items.Count
    $groups = $items | Group-Object -Property $property | Sort-Object Count -Descending
    $result = [ordered]@{}
    foreach ($g in $groups) {
        $result[$g.Name] = [math]::Round(($g.Count / $total) * 100, 1)
    }
    return $result
}

function PrintComparison($label, $real, $synth) {
    Write-Host "`n  $label" -ForegroundColor Yellow
    $allKeys = ($real.Keys + $synth.Keys) | Sort-Object -Unique
    $rows    = @()
    foreach ($k in $allKeys) {
        $r = if ($real.Contains($k))  { "$($real[$k])%"  } else { "N/A" }
        $s = if ($synth.Contains($k)) { "$($synth[$k])%" } else { "N/A" }
        $rows += [PSCustomObject]@{ Value = $k; Real = $r; Synthetic = $s }
    }
    $rows | Format-Table -AutoSize
}

function FieldCoverage($events, $field) {
    # What % of events have this field populated
    $present = ($events | Where-Object { $null -ne $_.$field -and $_.$field -ne "" }).Count
    return [math]::Round(($present / [math]::Max($events.Count, 1)) * 100, 1)
}


# -- 1. Schema field coverage --------------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  1. SCHEMA FIELD COVERAGE" -ForegroundColor Cyan
Write-Host "============================================"

$coreFields = @(
    "Id", "CreationTime", "RecordType", "Operation",
    "UserId", "ClientIP", "Workload", "ResultStatus",
    "AppDisplayName", "DeviceProperties", "Location", "ExtendedProperties"
)

$coverageReport = @()
foreach ($field in $coreFields) {
    $r = FieldCoverage $real $field
    $s = FieldCoverage $synthetic $field
    $delta = [math]::Abs($r - $s)
    $flag  = if ($delta -gt 20) { "  <-- MISMATCH" } else { "" }
    $coverageReport += [PSCustomObject]@{
        Field     = $field
        Real      = "$r%"
        Synthetic = "$s%"
        Delta     = "$delta%"
        Status    = if ($delta -le 5) { "OK" } elseif ($delta -le 20) { "CLOSE" } else { "MISMATCH" }
    }
}
$coverageReport | Format-Table -AutoSize

$mismatches = ($coverageReport | Where-Object Status -eq "MISMATCH").Count
if ($mismatches -eq 0) {
    Write-Host "  All core fields present in both datasets." -ForegroundColor Green
} else {
    Write-Host "  $mismatches field(s) have significant coverage gaps - review synthetic generator." -ForegroundColor Yellow
}


# -- 2. Workload distribution --------------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  2. WORKLOAD DISTRIBUTION" -ForegroundColor Cyan
Write-Host "============================================"

$realWorkloads  = Distribution $real "Workload"
$synthWorkloads = Distribution $synthetic "Workload"
PrintComparison "Workload %" $realWorkloads $synthWorkloads


# -- 3. Top operations ---------------------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  3. TOP OPERATIONS" -ForegroundColor Cyan
Write-Host "============================================"

$realOps  = Distribution $real "Operation"
$synthOps = Distribution $synthetic "Operation"

# Limit to top 10 per dataset for readability
$top10Real  = [ordered]@{}; $realOps.Keys  | Select-Object -First 10 | ForEach-Object { $top10Real[$_]  = $realOps[$_]  }
$top10Synth = [ordered]@{}; $synthOps.Keys | Select-Object -First 10 | ForEach-Object { $top10Synth[$_] = $synthOps[$_] }
PrintComparison "Operation % (top 10)" $top10Real $top10Synth

# Check for operations in real that are missing from synthetic entirely
$realOnlyOps = $realOps.Keys | Where-Object { -not $synthOps.Contains($_) }
if ($realOnlyOps) {
    Write-Host "  Operations in REAL but NOT in synthetic:" -ForegroundColor Yellow
    $realOnlyOps | ForEach-Object { Write-Host "    - $_ ($($realOps[$_])%)" }
    Write-Host "  --> Consider adding these to OPERATIONS dict in generate_logs.py" -ForegroundColor Yellow
}


# -- 4. UserType distribution --------------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  4. USER TYPE DISTRIBUTION" -ForegroundColor Cyan
Write-Host "============================================"
Write-Host "  (0=Regular, 2=Admin, 3=DcAdmin, 4=System)"

$realUT  = Distribution $real "UserType"
$synthUT = Distribution $synthetic "UserType"
PrintComparison "UserType %" $realUT $synthUT


# -- 5. Result status distribution --------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  5. RESULT STATUS" -ForegroundColor Cyan
Write-Host "============================================"

$realRS  = Distribution $real "ResultStatus"
$synthRS = Distribution $synthetic "ResultStatus"
PrintComparison "ResultStatus %" $realRS $synthRS


# -- 6. Location / country presence -------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  6. LOCATION FIELD PRESENCE" -ForegroundColor Cyan
Write-Host "============================================"

$realHasLoc  = FieldCoverage $real "Location"
$synthHasLoc = FieldCoverage $synthetic "Location"
Write-Host "  Location field coverage:  Real=$realHasLoc%  Synthetic=$synthHasLoc%"

# Country distribution for events that have location
$realWithLoc  = $real      | Where-Object { $null -ne $_.Location }
$synthWithLoc = $synthetic | Where-Object { $null -ne $_.Location }

if ($realWithLoc.Count -gt 0) {
    $realCountries  = $realWithLoc  | ForEach-Object { $_.Location.CountryCode } |
                      Group-Object | Sort-Object Count -Descending |
                      Select-Object -First 8 |
                      ForEach-Object { "$($_.Name)=$($_.Count)" }
    $synthCountries = $synthWithLoc | ForEach-Object { $_.Location.CountryCode } |
                      Group-Object | Sort-Object Count -Descending |
                      Select-Object -First 8 |
                      ForEach-Object { "$($_.Name)=$($_.Count)" }

    Write-Host "  Real countries (top 8):      $($realCountries -join ', ')"
    Write-Host "  Synthetic countries (top 8): $($synthCountries -join ', ')"
}


# -- 7. DeviceProperties structure check --------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  7. DEVICEPROPERTIES STRUCTURE" -ForegroundColor Cyan
Write-Host "============================================"

$realWithDev  = $real      | Where-Object { $null -ne $_.DeviceProperties -and $_.DeviceProperties.Count -gt 0 }
$synthWithDev = $synthetic | Where-Object { $null -ne $_.DeviceProperties -and $_.DeviceProperties.Count -gt 0 }

Write-Host "  Events with DeviceProperties:  Real=$($realWithDev.Count)  Synthetic=$($synthWithDev.Count)"

if ($realWithDev.Count -gt 0) {
    # Extract OS values
    $realOS = $realWithDev | ForEach-Object {
        ($_.DeviceProperties | Where-Object Name -eq "OS").Value
    } | Where-Object { $_ } | Group-Object | Sort-Object Count -Descending | Select-Object -First 6

    $synthOS = $synthWithDev | ForEach-Object {
        ($_.DeviceProperties | Where-Object Name -eq "OS").Value
    } | Where-Object { $_ } | Group-Object | Sort-Object Count -Descending | Select-Object -First 6

    Write-Host "`n  OS distribution:"
    Write-Host "  Real:      $(($realOS  | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ', ')"
    Write-Host "  Synthetic: $(($synthOS | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ', ')"

    # Check that key property names match
    $realPropNames  = ($realWithDev  | Select-Object -First 20 | ForEach-Object { $_.DeviceProperties.Name }) | Sort-Object -Unique
    $synthPropNames = ($synthWithDev | Select-Object -First 20 | ForEach-Object { $_.DeviceProperties.Name }) | Sort-Object -Unique
    $missingProps   = $realPropNames | Where-Object { $synthPropNames -notcontains $_ }

    if ($missingProps) {
        Write-Host "  Property names in real but missing from synthetic: $($missingProps -join ', ')" -ForegroundColor Yellow
    } else {
        Write-Host "  DeviceProperties key names match." -ForegroundColor Green
    }
}


# -- 8. Time-of-day distribution -----------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  8. TIME-OF-DAY DISTRIBUTION" -ForegroundColor Cyan
Write-Host "============================================"

function HourBuckets($events) {
    $buckets = @{ "Night(0-6)"=0; "Morning(7-9)"=0; "Work(10-17)"=0; "Evening(18-23)"=0 }
    foreach ($e in $events) {
        try {
            $hour = ([datetime]$e.CreationTime).Hour
            if ($hour -lt 7)                      { $buckets["Night(0-6)"]++   }
            elseif ($hour -lt 10)                 { $buckets["Morning(7-9)"]++ }
            elseif ($hour -lt 18)                 { $buckets["Work(10-17)"]++  }
            else                                  { $buckets["Evening(18-23)"]++ }
        } catch {}
    }
    $total = [math]::Max(($buckets.Values | Measure-Object -Sum).Sum, 1)
    $pct   = [ordered]@{}
    foreach ($k in @("Night(0-6)", "Morning(7-9)", "Work(10-17)", "Evening(18-23)")) {
        $pct[$k] = "$([math]::Round(($buckets[$k] / $total) * 100, 1))%"
    }
    return $pct
}

$realHours  = HourBuckets $real
$synthHours = HourBuckets $synthetic
PrintComparison "Hour bucket %" $realHours $synthHours


# -- 9. Overall verdict --------------------------------------------------------

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  9. SUMMARY" -ForegroundColor Cyan
Write-Host "============================================"

$issues = @()

if ($mismatches -gt 0) {
    $issues += "$mismatches core field(s) have >20% coverage gap"
}
if ($realOnlyOps -and $realOnlyOps.Count -gt 0) {
    $issues += "$($realOnlyOps.Count) operation type(s) in real logs not present in synthetic"
}

$workloadDelta = ($realWorkloads.Keys | ForEach-Object {
    $r = if ($realWorkloads.Contains($_))  { $realWorkloads[$_]  } else { 0 }
    $s = if ($synthWorkloads.Contains($_)) { $synthWorkloads[$_] } else { 0 }
    [math]::Abs($r - $s)
} | Measure-Object -Maximum).Maximum

if ($workloadDelta -gt 25) {
    $issues += "Workload distribution gap >25% on at least one workload"
}

if ($issues.Count -eq 0) {
    Write-Host "`n  SYNTHETIC DATA LOOKS GOOD - schema and distributions match real logs." -ForegroundColor Green
    Write-Host "  Safe to proceed to tokenisation." -ForegroundColor Green
} else {
    Write-Host "`n  ISSUES FOUND - review and tune generate_logs.py:" -ForegroundColor Yellow
    $issues | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}


# -- Write report JSON ---------------------------------------------------------

$report = @{
    generated_at         = (Get-Date -Format "o")
    real_event_count     = $real.Count
    synthetic_event_count = $synthetic.Count
    field_coverage       = $coverageReport
    workload_real        = $realWorkloads
    workload_synthetic   = $synthWorkloads
    operations_real_only = $realOnlyOps
    issues               = $issues
}

$report | ConvertTo-Json -Depth 5 | Out-File $ReportFile -Encoding UTF8
Write-Host "`nFull report written -> $ReportFile" -ForegroundColor Cyan