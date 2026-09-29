@echo off
cd /d "%~dp0"
echo ========================================================
echo Pushing Vercel serverless fixes to GitHub...
echo ========================================================
echo.

git config user.name "Atharv"
git config user.email "atharvraut444-cyber@users.noreply.github.com"

git add -A
git commit -m "Fix: serverless safe filesystem, multiprocessing disabled on lambda, diagnostic handler"
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
    echo Vercel is auto-deploying the fix right now!
    echo ========================================================
) else (
    echo ========================================================
    echo Push failed. Please check network or GitHub login.
    echo ========================================================
)
pause
