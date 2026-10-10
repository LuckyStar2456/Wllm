param([int]$Jobs = 4)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$lock = Get-Content dependencies.json -Raw | ConvertFrom-Json
$pin = $lock.'llama.cpp'.commit
if (-not (Test-Path vendor/llama.cpp/CMakeLists.txt)) {
    git clone --depth 1 --no-checkout $lock.'llama.cpp'.url vendor/llama.cpp
    if ($LASTEXITCODE -ne 0) { throw 'Dependency clone failed' }
    git -C vendor/llama.cpp fetch --depth 1 origin $pin
    if ($LASTEXITCODE -ne 0) { throw 'Pinned fetch failed' }
    git -C vendor/llama.cpp checkout --detach $pin
    if ($LASTEXITCODE -ne 0) { throw 'Pinned checkout failed' }
}
$actual = git -C vendor/llama.cpp rev-parse HEAD
if ($actual -ne $pin) { throw "Dependency mismatch: expected $pin, got $actual" }
if (git -C vendor/llama.cpp status --porcelain --untracked-files=no) { throw 'Dependency has local changes; record and review them before building' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 toolchain required' }
$cmake = Join-Path $installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
if (-not (Test-Path -LiteralPath $cmake)) { $cmake = (Get-Command cmake -ErrorAction Stop).Source }
& $cmake -S . -B build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
& $cmake --build build --config Release --target wllm wllm-tests --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$ctest = Join-Path (Split-Path $cmake -Parent) 'ctest.exe'
& $ctest --test-dir build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
