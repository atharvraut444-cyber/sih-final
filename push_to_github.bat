@echo off
cd /d "%~dp0"
echo ========================================================
echo Pushing SENORITA project to GitHub (sih-final)...
echo ========================================================
echo.

git config user.name "Atharv"
git config user.email "atharvraut444-cyber@users.noreply.github.com"

git add .
git commit -m "Setup project with Docker and Render deployment"
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin https://github.com/atharvraut444-cyber/sih-final.git

echo.
echo Syncing and pushing to GitHub...
git push -u origin main --force

echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo SUCCESS! Your code has been pushed to GitHub.
    echo ========================================================
) else (
    echo ========================================================
    echo PUSH FAILED:
    echo Please make sure you sign in or paste your GitHub Token.
    echo ========================================================
)
pause
