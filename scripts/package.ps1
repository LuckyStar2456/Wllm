param([string]$BuildDirectory = 'build/public', [string]$OutputDirectory = 'dist')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $projectRoot "$BuildDirectory/bin/Release/wllm.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build Wllm first' }
$version = (& $exe --version).Split(' ')[1]
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^\d+\.\d+\.\d+$') { throw 'Cannot read executable version' }
$destination = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$stage = Join-Path $projectRoot "build/package-$([guid]::NewGuid().ToString('N'))/wllm-$version-windows-x64"
New-Item -ItemType Directory -Path $stage -Force | Out-Null
Copy-Item -LiteralPath $exe -Destination (Join-Path $stage 'wllm.exe')
foreach ($name in @('README.md','LICENSE','THIRD_PARTY_NOTICES.md','CHANGELOG.md','wllm.cmd','dependencies.json')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $stage
}
foreach ($name in @('configs','guides','assets')) { New-Item -ItemType Directory -Path (Join-Path $stage $name) | Out-Null }
foreach ($name in @('default.json','light-switch.candidates.json','system-one.example.json')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "configs/$name") -Destination (Join-Path $stage 'configs')
}
Copy-Item -Path (Join-Path $projectRoot 'guides/*.md') -Destination (Join-Path $stage 'guides')
Copy-Item -Path (Join-Path $projectRoot 'assets/*') -Destination (Join-Path $stage 'assets')
$licenses = Join-Path $stage 'licenses'
New-Item -ItemType Directory -Path $licenses | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'vendor/llama.cpp/LICENSE') -Destination (Join-Path $licenses 'llama.cpp-MIT.txt')
Copy-Item -LiteralPath (Join-Path $projectRoot 'vendor/llama.cpp/licenses') -Destination (Join-Path $licenses 'upstream') -Recurse
foreach ($notice in (Get-ChildItem (Join-Path $projectRoot 'vendor/llama.cpp/vendor') -File -Recurse | Where-Object { $_.Name -match '^(LICENSE|COPYING|NOTICE)(\..*)?$' })) {
    $relative = $notice.FullName.Substring((Join-Path $projectRoot 'vendor/llama.cpp/vendor').Length + 1)
    $target = Join-Path $licenses "vendor/$relative"
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $notice.FullName -Destination $target
}
$manifest = Get-ChildItem -LiteralPath $stage -File -Recurse | ForEach-Object {
    [ordered]@{ path = $_.FullName.Substring($stage.Length + 1).Replace('\','/'); sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
@{ version = $version; platform = 'windows-x64-avx2'; files = @($manifest) } | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $stage 'manifest.json')
$zip = Join-Path $destination "wllm-$version-windows-x64.zip"
if (Test-Path -LiteralPath $zip) { throw "Package already exists: $zip; choose another OutputDirectory" }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
$digest = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$digest  $([IO.Path]::GetFileName($zip))" | Set-Content -Encoding ascii "$zip.sha256"
Write-Output $zip
