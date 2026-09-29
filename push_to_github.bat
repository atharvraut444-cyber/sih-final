@echo off
cd /d "%~dp0"
echo ========================================================
echo Pushing fix to GitHub (sih-final)...
echo ========================================================
echo.

git config user.name "Atharv"
git config user.email "atharvraut444-cyber@users.noreply.github.com"

git add .
git commit -m "Fix Render deploy: add gunicorn and correct start command"
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin https://github.com/atharvraut444-cyber/sih-final.git

echo.
echo Pushing fix to GitHub...
git push -u origin main

echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo SUCCESS! Fix pushed. Render will auto-deploy now.
    echo ========================================================
) else (
    echo ========================================================
    echo Push failed. Please check network or GitHub login.
    echo ========================================================
)
pause
