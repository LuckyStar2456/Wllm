param(
    [string]$Prompt,
    [string]$PromptFile,
    [ValidateRange(512,262144)][int]$Context,
    [int]$MaxTokens,
    [string]$Config = 'configs/local.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$exe = Join-Path $projectRoot 'build/bin/Release/wllm.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Run scripts/build.ps1 first' }
$params = @('--config', $Config)
if ($PSBoundParameters.ContainsKey('Context')) { $params += @('--context', $Context) }
if ($PSBoundParameters.ContainsKey('MaxTokens')) { $params += @('--max-tokens', $MaxTokens) }
if ($Prompt -and $PromptFile) { throw 'Use either Prompt or PromptFile' }
if ($PromptFile) { $params += @('--prompt-file', $PromptFile) }
elseif ($Prompt) { $params += @('--prompt', $Prompt) }
& $exe @params
exit $LASTEXITCODE
