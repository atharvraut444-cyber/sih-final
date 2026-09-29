@echo off
cd /d "%~dp0"
echo ========================================================
echo Pushing Vercel Configuration to GitHub (sih-final)...
echo ========================================================
echo.

git config user.name "Atharv"
git config user.email "atharvraut444-cyber@users.noreply.github.com"

git add -A
git commit -m "Configure project for Vercel deployment (api/index.py, vercel.json, serverless requirements)"
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin https://github.com/atharvraut444-cyber/sih-final.git

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo SUCCESS! Code pushed to GitHub.
    echo Now import or deploy repository directly in Vercel!
    echo ========================================================
) else (
    echo ========================================================
    echo Push failed. Please check your network or GitHub login.
    echo ========================================================
)
pause
