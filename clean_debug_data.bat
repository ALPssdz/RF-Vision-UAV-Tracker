@echo off
echo ========================================================
echo       RF-Vision Test Data Cleanup Utility
echo ========================================================
echo [!] Preparing to free up memory and storage space...
echo.

echo [-] Deleting historical alert database (rf_alert_history.db)...
del /Q /F "database\rf_alert_history.db" 2>nul

if exist "database\rf_alert_history.db" (
    echo [!] WARNING: Access Denied! Please ensure system_hub.py is completely closed.
) else (
    echo  -^> SQLite database shredded. It will be recreated upon next system startup.
)

echo.
echo [-] Erasing physical optical evidence records (database/evidences)...
del /Q /F "database\alert_images\*.jpg" 2>nul
del /Q /F "database\alert_images\*.png" 2>nul
echo  -^> K230 NPU snapshot cache cleared.

echo.
echo ========================================================
echo [OK] System environment successfully restored to clean state!
echo ========================================================
echo.
pause
