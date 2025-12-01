@echo off
echo ========================================
echo Stopping processes on port 5000
echo ========================================
echo.

REM Find processes using port 5000
netstat -ano | findstr :5000 >nul 2>&1
if %errorlevel% neq 0 (
    echo No process found using port 5000
    echo Port is free!
    pause
    exit /b
)

echo Processes using port 5000:
netstat -ano | findstr :5000
echo.

REM Kill processes
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    echo Killing process %%a...
    taskkill /F /PID %%a >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Process %%a stopped
    ) else (
        echo [WARNING] Could not stop process %%a (may need admin rights)
    )
)

echo.
echo ========================================
echo Done! Port 5000 should now be free.
echo ========================================
pause

