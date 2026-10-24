param([int]$Jobs = 4, [ValidateSet('avx2','avx512')][string]$CpuTarget = 'avx2', [string]$BuildDirectory = 'build', [switch]$CpuKernelExperiments)
$ErrorActionPreference = 'Stop'
if ($CpuKernelExperiments -and $CpuTarget -ne 'avx2') { throw 'CPU kernel experiments require the AVX2 baseline' }
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
& $cmake -S . -B $BuildDirectory -G 'Visual Studio 17 2022' -A x64 "-DWLLM_CPU_TARGET=$CpuTarget"
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
$buildTargets = @('wllm', 'wllm-tests')
if ($CpuKernelExperiments) { $buildTargets += 'wllm-cpu-kernel-bench' }
& $cmake --build $BuildDirectory --config Release --target @buildTargets --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$ctest = Join-Path (Split-Path $cmake -Parent) 'ctest.exe'
& $ctest --test-dir $BuildDirectory -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
if ($CpuKernelExperiments) {
    New-Item -ItemType Directory -Path reports -Force | Out-Null
    $kernelSelfTestReport = Join-Path $projectRoot "reports/cpu-kernel-selftest-$([guid]::NewGuid().ToString('N')).json"
    $kernelExperimentExe = Join-Path $BuildDirectory 'bin/Release/wllm-cpu-kernel-bench.exe'
    & $kernelExperimentExe --self-test --output $kernelSelfTestReport
    if ($LASTEXITCODE -ne 0) { throw "CPU kernel numerical self-test failed; see $kernelSelfTestReport" }
}
