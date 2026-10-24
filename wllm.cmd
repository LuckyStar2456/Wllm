@echo off
setlocal
if exist "%~dp0wllm.exe" (
  "%~dp0wllm.exe" %*
  exit /b
)
if exist "%~dp0build\public\bin\Release\wllm.exe" (
  "%~dp0build\public\bin\Release\wllm.exe" %*
  exit /b
)
if exist "%~dp0build\bin\Release\wllm.exe" (
  "%~dp0build\bin\Release\wllm.exe" %*
  exit /b
)
echo Build Wllm with scripts\build.ps1 first. 1>&2
exit /b 1
