param(
    [string]$Model,
    [switch]$Tui,
    [switch]$Serve,
    [ValidateSet('all','openai','systemone')][string]$Service,
    [string]$BindHost,
    [ValidateRange(1,65535)][int]$Port,
    [string]$ModelAlias,
    [string]$ApiKeyEnv,
    [string]$Prompt,
    [string]$PromptFile,
    [switch]$Raw,
    [string]$ChatTemplateFile,
    [string]$BuildDirectory = 'build/public',
    [switch]$Interactive,
    [ValidateSet('generate','candidate_score','system_one')][string]$Mode,
    [string]$CandidatesFile,
    [string]$RequestFile,
    [ValidateRange(512,262144)][int]$Context,
    [int]$MaxTokens,
    [ValidateRange(1,256)][int]$Threads,
    [ValidateRange(0,256)][int]$ThreadsBatch,
    [ValidateRange(1,262144)][int]$Batch,
    [ValidateSet('on','off')][string]$PersistentThreadpool,
    [ValidateSet('on','off')][string]$PrefillLastOnly,
    [ValidateSet('on','off')][string]$CpuWeightRepacking,
    [ValidateRange(0,4096)][int]$PrefixCacheMiB,
    [Alias('Profile')][switch]$ProfileTimings,
    [switch]$ProfileOps,
    [string]$Config
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $projectRoot "$BuildDirectory/bin/Release/wllm.exe"
if (-not (Test-Path -LiteralPath $exe)) { $exe = Join-Path $projectRoot 'build/bin/Release/wllm.exe' }
if (-not (Test-Path -LiteralPath $exe)) { throw 'Run scripts/build.ps1 first' }
$params = @()
if ($Config) { $params += @('--config', $Config) }
if ($Model) { $params += @('--model', $Model) }
if ($Tui) { $params += '--tui' }
if ($Serve) { $params += '--serve' }
foreach ($entry in @(@('Service','--service'), @('BindHost','--host'), @('Port','--port'), @('ModelAlias','--model-alias'), @('ApiKeyEnv','--api-key-env'))) {
    if ($PSBoundParameters.ContainsKey($entry[0])) { $params += @($entry[1], $PSBoundParameters[$entry[0]]) }
}
if ($Interactive) { $params += '--interactive' }
if ($Raw) { $params += '--raw' }
if ($ChatTemplateFile) { $params += @('--chat-template-file', $ChatTemplateFile) }
if ($PSBoundParameters.ContainsKey('Mode')) { $params += @('--mode', $Mode) }
if ($CandidatesFile) { $params += @('--candidates', $CandidatesFile) }
if ($RequestFile) { $params += @('--request-file', $RequestFile) }
if ($ProfileTimings) { $params += '--profile' }
if ($ProfileOps) { $params += '--profile-ops' }
if ($PSBoundParameters.ContainsKey('Context')) { $params += @('--context', $Context) }
if ($PSBoundParameters.ContainsKey('MaxTokens')) { $params += @('--max-tokens', $MaxTokens) }
foreach ($entry in @(@('Threads','--threads'), @('ThreadsBatch','--threads-batch'), @('Batch','--batch'), @('PersistentThreadpool','--persistent-threadpool'), @('PrefillLastOnly','--prefill-last-only'), @('CpuWeightRepacking','--cpu-weight-repacking'), @('PrefixCacheMiB','--prefix-cache-mib'))) {
    if ($PSBoundParameters.ContainsKey($entry[0])) { $params += @($entry[1], $PSBoundParameters[$entry[0]]) }
}
if ($Prompt -and $PromptFile) { throw 'Use either Prompt or PromptFile' }
if ($PromptFile) { $params += @('--prompt-file', $PromptFile) }
elseif ($Prompt) { $params += @('--prompt', $Prompt) }
& $exe @params
exit $LASTEXITCODE
