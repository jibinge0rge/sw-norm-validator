@echo off
setlocal enabledelayedexpansion

set VENV_DIR=venv
set PYTHON=%VENV_DIR%\Scripts\python.exe
set PIP=%VENV_DIR%\Scripts\pip.exe
set BRANCH=main

:: Function: Colored title
set PS_OUT=powershell -Command "Write-Host"

:: Colors
:: Green  = Success
:: Yellow = Step Info
:: Red    = Error

:: Step 0: Git pull if updates are available
%PS_OUT% '[*] Checking for Git updates...' -ForegroundColor Yellow
if exist ".git" (
    git remote update >nul 2>&1

    for /f "tokens=*" %%i in ('git rev-parse HEAD') do set LOCAL=%%i
    for /f "tokens=*" %%i in ('git rev-parse origin/%BRANCH%') do set REMOTE=%%i

    if not "!LOCAL!"=="!REMOTE!" (
        git pull origin %BRANCH%
        if errorlevel 1 (
            %PS_OUT% '[!] Git pull failed.' -ForegroundColor Red
            pause
            exit /b 1
        )
        %PS_OUT% '[+] Git updated successfully.' -ForegroundColor Green
    ) else (
        %PS_OUT% '[+] Already up to date.' -ForegroundColor Green
    )
) else (
    %PS_OUT% '[!] Git repository not found. Skipping update check.' -ForegroundColor Red
)

:: Step 1: Create virtual environment if missing
if not exist %PYTHON% (
    %PS_OUT% '[*] Creating virtual environment...' -ForegroundColor Yellow
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        %PS_OUT% '[!] Failed to create venv. Is Python installed and in PATH?' -ForegroundColor Red
        pause
        exit /b 1
    )
    %PS_OUT% '[+] Virtual environment created.' -ForegroundColor Green
) else (
    %PS_OUT% '[+] Virtual environment exists.' -ForegroundColor Green
)

:: Step 2: Upgrade pip
%PS_OUT% '[*] Upgrading pip...' -ForegroundColor Yellow
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 (
    %PS_OUT% '[!] pip upgrade failed.' -ForegroundColor Red
    pause
    exit /b 1
)
%PS_OUT% '[+] pip upgrade complete.' -ForegroundColor Green

:: Step 3: Install requirements
%PS_OUT% '[*] Installing requirements...' -ForegroundColor Yellow
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    %PS_OUT% '[!] Failed to install dependencies.' -ForegroundColor Red
    pause
    exit /b 1
)
%PS_OUT% '[+] All dependencies installed.' -ForegroundColor Green

:: Step 4: Launch Streamlit app
%PS_OUT% '[*] Launching Streamlit app...' -ForegroundColor Yellow
%PYTHON% -m streamlit run app.py

pause
endlocal
