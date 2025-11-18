@echo off
setlocal
chcp 65001 >nul
title ai_recorder_live (NIE ZAMYKA SIE SAM)

cd /d "E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT"

set "PYTHON=python"
set "AI_REC=dom_renderer\ai_recorder_live.py"
set "REC_OUTPUT_DIR=E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT\dom_live"

echo ============================================================
echo [ai_recorder] CWD: %CD%
echo [ai_recorder] Uzywany python: %PYTHON%
echo [ai_recorder] Skrypt:        %AI_REC%
echo [ai_recorder] Output dir:    %REC_OUTPUT_DIR%
echo ============================================================
echo.
echo [ai_recorder] CMD:
echo   %PYTHON% %AI_REC% --output-dir "%REC_OUTPUT_DIR%" --url "https://chat.openai.com" --fps 2 --dom-only --log-file "%REC_OUTPUT_DIR%\rec.log" --verbose
echo.

%PYTHON% %AI_REC% --output-dir "%REC_OUTPUT_DIR%" --url "https://chat.openai.com" --fps 2 --dom-only --log-file "%REC_OUTPUT_DIR%\rec.log" --verbose
set "ERR=%ERRORLEVEL%"

echo.
echo [ai_recorder] Proces Pythona zakonczony, ERRORLEVEL=%ERR%
echo [ai_recorder] Okno NIE zamknie sie samo.
echo          Wcisnij dowolny klawisz, aby zamknac to okno.
echo.
pause
endlocal
exit /b 0
