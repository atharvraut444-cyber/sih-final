@echo off
cd /d "%~dp0"
echo ========================================================
echo Triggering fresh deployment on Render via GitHub...
echo ========================================================
echo.

git config user.name "Atharv"
git config user.email "atharvraut444-cyber@users.noreply.github.com"

git add .
git commit -m "Trigger Render auto-deployment" --allow-empty
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin https://github.com/atharvraut444-cyber/sih-final.git

echo.
echo Pushing new trigger commit to GitHub...
git push -u origin main

echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo SUCCESS! New commit pushed. 
    echo Render will now automatically detect this and start deploying!
    echo ========================================================
) else (
    echo ========================================================
    echo Push failed. Please check network or GitHub login.
    echo ========================================================
)
pause
