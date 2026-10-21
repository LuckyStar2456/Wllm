param(
    [Parameter(Mandatory=$true)][string]$VulkanSdk,
    [string]$BuildDirectory = 'build/vulkan-probe',
    [ValidateRange(1,8)][int]$Jobs = 2,
    [switch]$WllmValidation
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$sdkPath = (Resolve-Path -LiteralPath $VulkanSdk).Path
foreach ($required in @('Include/vulkan/vulkan.h','Lib/vulkan-1.lib','Bin/glslc.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $sdkPath $required))) { throw "Missing Vulkan SDK input: $required" }
}
$lock = Get-Content dependencies.json -Raw | ConvertFrom-Json
$actual = git -C vendor/llama.cpp rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actual -ne $lock.'llama.cpp'.commit) { throw 'Pinned llama.cpp checkout required' }
if (git -C vendor/llama.cpp status --porcelain --untracked-files=no) { throw 'Dependency has local changes' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 toolchain required' }
$cmake = Join-Path $installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
# Only this process receives SDK search paths; no persistent environment or registry changes.
$previousSdk = $env:VULKAN_SDK
try {
    $env:VULKAN_SDK = $sdkPath
    $validationHook = if ($WllmValidation) { Join-Path $PSScriptRoot 'cmake/vulkan-validation.cmake' } else { '' }
    & $cmake -S vendor/llama.cpp -B $BuildDirectory -G 'Visual Studio 17 2022' -A x64 `
        "-DCMAKE_PREFIX_PATH=$sdkPath" -DBUILD_SHARED_LIBS=OFF `
        "-DCMAKE_PROJECT_llama.cpp_INCLUDE=$validationHook" `
        -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF `
        -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_OPENMP=OFF `
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON `
        -DLLAMA_OPENSSL=OFF
    if ($LASTEXITCODE -ne 0) { throw 'Vulkan probe configure failed' }
    $targets = if ($WllmValidation) { @('llama-bench','llama-simple','wllm-vulkan') } else { @('llama-bench','llama-simple') }
    & $cmake --build $BuildDirectory --config Release --target $targets --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'Vulkan probe build failed' }
} finally {
    $env:VULKAN_SDK = $previousSdk
}
