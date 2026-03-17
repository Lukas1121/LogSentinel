# =============================================================================
# Fetch-AuditLogs.ps1
# Fetches real M365 Unified Audit Log events from your tenant and exports
# them to real_logs.json for comparison against synthetic data.
#
# REQUIREMENTS:
#   Install-Module ExchangeOnlineManagement -Force   (run once as admin)
#
# USAGE:
#   .\Fetch-AuditLogs.ps1 -UPN you@yourtenant.com
#   .\Fetch-AuditLogs.ps1 -UPN you@yourtenant.com -Days 14 -MaxEvents 500
#
# OUTPUT:
#   real_logs.json   — array of parsed audit events, same shape as synthetic
# =============================================================================

param(
    [Parameter(Mandatory)]
    [string]$UPN,                    # Your admin UPN

    [int]$Days      = 7,             # How many days back to fetch
    [int]$MaxEvents = 200,           # Cap — keep small for comparison purposes
    [string]$OutFile = "real_logs.json"
)

# ── Connect ───────────────────────────────────────────────────────────────────

Write-Host "`n=== Fetch-AuditLogs.ps1 ===" -ForegroundColor Cyan
Write-Host "Connecting to Exchange Online as $UPN..."

try {
    Import-Module ExchangeOnlineManagement -ErrorAction Stop
} catch {
    Write-Error "ExchangeOnlineManagement module not found. Run: Install-Module ExchangeOnlineManagement -Force"
    exit 1
}

Connect-ExchangeOnline -UserPrincipalName $UPN -ShowBanner:$false

# Verify auditing is enabled
$auditConfig = Get-AdminAuditLogConfig | Select-Object -ExpandProperty UnifiedAuditLogIngestionEnabled
if (-not $auditConfig) {
    Write-Warning "Unified Audit Log ingestion is DISABLED on this tenant."
    Write-Warning "Enable it with: Set-AdminAuditLogConfig -UnifiedAuditLogIngestionEnabled `$true"
    Write-Warning "Note: it can take up to 60 minutes to start collecting after enabling."
    Disconnect-ExchangeOnline -Confirm:$false
    exit 1
}
Write-Host "  Audit logging: ENABLED" -ForegroundColor Green

# ── Fetch ─────────────────────────────────────────────────────────────────────

$EndDate   = Get-Date
$StartDate = $EndDate.AddDays(-$Days)

Write-Host "`nFetching audit events..."
Write-Host "  Period:    $($StartDate.ToString('yyyy-MM-dd')) → $($EndDate.ToString('yyyy-MM-dd'))"
Write-Host "  Max:       $MaxEvents events"
Write-Host "  Workloads: AzureActiveDirectory, SharePoint, Exchange, MicrosoftTeams`n"

# Fetch across the four workloads we care about.
# We use a session loop to handle pagination — default cap is 100 per call.
$allResults  = @()
$sessionId   = [System.Guid]::NewGuid().ToString()

$recordTypes = @(
    "AzureActiveDirectory",
    "SharePointFileOperation",
    "ExchangeItem",
    "MicrosoftTeams"
)

foreach ($rt in $recordTypes) {
    if ($allResults.Count -ge $MaxEvents) { break }

    Write-Host "  Fetching $rt..." -NoNewline
    $rtResults = @()
    $page      = 0

    do {
        $page++
        $batch = Search-UnifiedAuditLog `
            -StartDate      $StartDate `
            -EndDate        $EndDate `
            -RecordType     $rt `
            -SessionId      "$sessionId-$rt" `
            -SessionCommand ReturnLargeSet `
            -ResultSize     100

        if ($batch) {
            $rtResults += $batch
        }
    } while ($batch -and $rtResults.Count -lt ($MaxEvents / $recordTypes.Count) -and $page -lt 10)

    Write-Host " $($rtResults.Count) events"
    $allResults += $rtResults
}

Write-Host "`n  Total raw events fetched: $($allResults.Count)"

if ($allResults.Count -eq 0) {
    Write-Warning "No events returned. Check that auditing is enabled and you have log data in the past $Days days."
    Disconnect-ExchangeOnline -Confirm:$false
    exit 1
}

# ── Parse AuditData JSON ──────────────────────────────────────────────────────
# Each result's AuditData property is a raw JSON string — parse it.

Write-Host "Parsing AuditData payloads..."

$parsed = @()
$errors = 0

foreach ($record in $allResults) {
    try {
        $data = $record.AuditData | ConvertFrom-Json

        # Normalise to match our synthetic schema as closely as possible
        $event = [ordered]@{
            Id                = $data.Id
            CreationTime      = $data.CreationTime
            RecordType        = $data.RecordType
            Operation         = $data.Operation
            OrganizationId    = $data.OrganizationId
            UserType          = $data.UserType
            UserKey           = $data.UserKey
            UserId            = $data.UserId
            ClientIP          = $data.ClientIP
            Workload          = $data.Workload
            ResultStatus      = $data.ResultStatus
            AppDisplayName    = $data.AppDisplayName
            UserDisplayName   = $data.UserDisplayName
            DeviceProperties  = $data.DeviceProperties
            Location          = $data.Location
            ExtendedProperties = $data.ExtendedProperties
            # SharePoint-specific
            ObjectId          = $data.ObjectId
            SiteUrl           = $data.SiteUrl
            SourceFileName    = $data.SourceFileName
            # Exchange-specific
            MailboxOwnerUPN   = $data.MailboxOwnerUPN
            ClientInfoString  = $data.ClientInfoString
        }

        # Strip null fields to keep output clean
        $clean = [ordered]@{}
        foreach ($key in $event.Keys) {
            if ($null -ne $event[$key]) {
                $clean[$key] = $event[$key]
            }
        }

        $parsed += $clean
    } catch {
        $errors++
    }
}

Write-Host "  Parsed: $($parsed.Count)  Errors: $errors"

# ── Write output ──────────────────────────────────────────────────────────────

$parsed | ConvertTo-Json -Depth 10 | Out-File $OutFile -Encoding UTF8
Write-Host "`nWrote $($parsed.Count) events → $OutFile" -ForegroundColor Green

# Quick summary to console
Write-Host "`n--- Quick summary ---"
$parsed | Group-Object -Property Workload | Sort-Object Count -Descending |
    Format-Table Name, Count -AutoSize

$parsed | Group-Object -Property Operation | Sort-Object Count -Descending |
    Select-Object -First 10 |
    Format-Table Name, Count -AutoSize

Disconnect-ExchangeOnline -Confirm:$false
Write-Host "Done. Run Compare-Logs.ps1 next." -ForegroundColor Cyan