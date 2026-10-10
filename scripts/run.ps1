param(
    [string]$Prompt,
    [string]$PromptFile,
    [switch]$Interactive,
    [ValidateRange(512,262144)][int]$Context,
    [int]$MaxTokens,
    [ValidateRange(1,256)][int]$Threads,
    [ValidateRange(0,256)][int]$ThreadsBatch,
    [ValidateRange(1,262144)][int]$Batch,
    [ValidateSet('on','off')][string]$PersistentThreadpool,
    [ValidateSet('on','off')][string]$PrefillLastOnly,
    [ValidateSet('on','off')][string]$CpuWeightRepacking,
    [Alias('Profile')][switch]$ProfileTimings,
    [switch]$ProfileOps,
    [string]$Config = 'configs/local.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$exe = Join-Path $projectRoot 'build/bin/Release/wllm.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Run scripts/build.ps1 first' }
$params = @('--config', $Config)
if ($Interactive) { $params += '--interactive' }
if ($ProfileTimings) { $params += '--profile' }
if ($ProfileOps) { $params += '--profile-ops' }
if ($PSBoundParameters.ContainsKey('Context')) { $params += @('--context', $Context) }
if ($PSBoundParameters.ContainsKey('MaxTokens')) { $params += @('--max-tokens', $MaxTokens) }
foreach ($entry in @(@('Threads','--threads'), @('ThreadsBatch','--threads-batch'), @('Batch','--batch'), @('PersistentThreadpool','--persistent-threadpool'), @('PrefillLastOnly','--prefill-last-only'), @('CpuWeightRepacking','--cpu-weight-repacking'))) {
    if ($PSBoundParameters.ContainsKey($entry[0])) { $params += @($entry[1], $PSBoundParameters[$entry[0]]) }
}
if ($Prompt -and $PromptFile) { throw 'Use either Prompt or PromptFile' }
if ($PromptFile) { $params += @('--prompt-file', $PromptFile) }
elseif ($Prompt) { $params += @('--prompt', $Prompt) }
& $exe @params
exit $LASTEXITCODE
