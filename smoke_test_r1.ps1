param(
  [string]$BaseUrl = "https://spad-python-ml.onrender.com",
  [string]$ZipPath = ".\SPAD_V4_WEB_INPUT_V2.zip",
  [string]$LotId = "R1_SMOKE",
  [string]$ApiKey = $env:SPAD_API_KEY
)

$ErrorActionPreference = "Stop"

$healthHeaders = @{}
if ($ApiKey) { $healthHeaders["X-SPAD-API-Key"] = $ApiKey }

Write-Host "Checking $BaseUrl/health ..."
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -Headers $healthHeaders
$health | ConvertTo-Json -Depth 8

if ($health.serviceRevision -ne "R1") { throw "Unexpected serviceRevision: $($health.serviceRevision)" }
if ($health.serviceBuild -ne "R1.1-HARDENED") { throw "Unexpected serviceBuild: $($health.serviceBuild)" }
if ($health.randomForest -ne "loaded" -or $health.isolationForest -ne "loaded") { throw "Required model is not loaded." }
if ($health.apiKeyRequired -and -not $ApiKey) { throw "Deployment requires SPAD_API_KEY; set the environment variable before running the smoke test." }

$form = @{
  file = Get-Item $ZipPath
  lotId = $LotId
  engineeringLimits = '{"rdson":{"limitValue":0.7,"direction":"UPPER"}}'
}

$headers = @{}
if ($ApiKey) { $headers["X-SPAD-API-Key"] = $ApiKey }

Write-Host "Running screening smoke test ..."
$result = Invoke-RestMethod -Uri "$BaseUrl/run-screening" -Method Post -Headers $headers -Form $form
$result | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 ".\R1_smoke_response.json"

if (-not $result.success) { throw "Screening response reported success=false." }
if ($result.serviceRevision -ne "R1") { throw "Unexpected response serviceRevision." }
if ($result.serviceBuild -ne "R1.1-HARDENED") { throw "Unexpected response serviceBuild." }
if ($result.records.Count -ne 15) { throw "Expected 15 records, got $($result.records.Count)." }
if ($result.dataFormatVersion -ne "SPAD_V4_WEB_CSV_V2") { throw "Unexpected input format version." }
if ($result.modelPackageSha256 -ne $health.modelPackageSha256) { throw "Model hash changed between health and screening response." }

$totalSamples = 0
$totalTransientGroups = 0
$totalViolatingSamples = 0
$totalViolatingRuns = 0

foreach ($r in $result.records) {
  if ($r.prediction.futureRiskScore -ne $null -or $r.prediction.futureRiskPercent -ne $null) {
    throw "Unexpected future risk field detected for $($r.componentId)."
  }
  if ($r.lotAnomaly.Module_A_Authority -ne "EVIDENCE_ONLY") {
    throw "Module A authority mismatch for $($r.componentId)."
  }
  $e = $r.engineering.rdson
  if ($null -eq $e.limitExceedanceCount -or $null -eq $e.limitExceedanceTransientCount -or $null -eq $e.limitExceedanceRunCount) {
    throw "Engineering count fields missing for $($r.componentId)."
  }
  $totalSamples += [int]$e.validRdsSampleCount
  $totalTransientGroups += [int]$e.targetConditionTransientCount
  $totalViolatingSamples += [int]$e.limitExceedanceCount
  $totalViolatingRuns += [int]$e.limitExceedanceRunCount
}

if ($totalSamples -ne 3656923) { throw "Unexpected valid sample count: $totalSamples" }
if ($totalTransientGroups -ne 6900) { throw "Unexpected transient group count: $totalTransientGroups" }
if ($totalViolatingSamples -ne 327975) { throw "Unexpected violating sample count: $totalViolatingSamples" }
if ($totalViolatingRuns -ne 27) { throw "Unexpected violating run count: $totalViolatingRuns" }

Write-Host "R1 smoke test PASSED. Response saved to R1_smoke_response.json"
