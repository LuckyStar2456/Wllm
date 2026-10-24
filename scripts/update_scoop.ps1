param([string]$Version)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$repository = 'https://github.com/LuckyStar2456/Wllm'
$api = 'https://api.github.com/repos/LuckyStar2456/Wllm/releases'
$endpoint = if ($Version) { "$api/tags/v$Version" } else { "$api/latest" }
$release = Invoke-RestMethod -Uri $endpoint -Headers @{ 'User-Agent' = 'Wllm-Scoop' }
if ($release.draft -or $release.prerelease -or $release.tag_name -notmatch '^v(\d+\.\d+\.\d+)$') { throw 'Select a published stable release' }
$packageVersion = $Matches[1]
$filename = "wllm-$packageVersion-windows-x64.zip"
$asset = @($release.assets | Where-Object name -EQ $filename)
$checksum = @($release.assets | Where-Object name -EQ "$filename.sha256")
if ($asset.Count -ne 1 -or $checksum.Count -ne 1) { throw 'Release ZIP and SHA256 assets are required' }
$hashFile = (Invoke-WebRequest -Uri $checksum[0].browser_download_url -UseBasicParsing).Content
if ($hashFile -is [byte[]]) { $hashFile = [Text.Encoding]::UTF8.GetString($hashFile) }
if ($hashFile -notmatch '^([a-fA-F0-9]{64})\s+') { throw 'Invalid release checksum file' }
$digest = $Matches[1].ToLowerInvariant()
if ($asset[0].digest -and $asset[0].digest -ne "sha256:$digest") { throw 'Release asset digest differs from checksum file' }
$manifest = [ordered]@{
    version = $packageVersion
    description = 'Local CPU text inference and Reflex decisions in a terminal'
    homepage = $repository
    license = 'MIT'
    architecture = [ordered]@{
        '64bit' = [ordered]@{ url = $asset[0].browser_download_url; hash = $digest }
    }
    extract_dir = "wllm-$packageVersion-windows-x64"
    bin = 'wllm.exe'
    notes = @('Start with: wllm --tui --model "D:\Models\model.gguf"', 'Example files are under: scoop prefix wllm', 'Requires an AVX2 CPU. Model weights are supplied separately.')
    checkver = [ordered]@{ github = $repository }
    autoupdate = [ordered]@{
        architecture = [ordered]@{
            '64bit' = [ordered]@{
                url = 'https://github.com/LuckyStar2456/Wllm/releases/download/v$version/wllm-$version-windows-x64.zip'
                hash = [ordered]@{ url = '$url.sha256'; regex = '^([a-fA-F0-9]{64})' }
            }
        }
        extract_dir = 'wllm-$version-windows-x64'
    }
}
$bucket = Join-Path $projectRoot 'bucket'
New-Item -ItemType Directory -Path $bucket -Force | Out-Null
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $bucket 'wllm.json')
Write-Output "Scoop manifest: $packageVersion / $digest"
