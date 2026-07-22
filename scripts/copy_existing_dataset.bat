@echo off
setlocal

if "%~1"=="" (
    echo Usage:
    echo copy_existing_dataset.bat F:\Python\Q1_COVID19_Research_Package\data\prepared\xray_3class
    pause
    exit /b 1
)

set SOURCE=%~1
set DEST=data\prepared\xray_3class

if not exist "%SOURCE%" (
    echo Source folder does not exist:
    echo %SOURCE%
    pause
    exit /b 1
)

if exist "%DEST%" rmdir /s /q "%DEST%"
mkdir "%DEST%"

robocopy "%SOURCE%" "%DEST%" /E
if errorlevel 8 (
    echo Copy failed.
    pause
    exit /b 1
)

echo Dataset copied to:
echo %CD%\%DEST%
pause
