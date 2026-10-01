@echo off

echo ========================================
echo          PostureGuard
echo ========================================
echo.

call conda activate postureguard

if errorlevel 1 (
    echo.
    echo ERRORE: ambiente Conda "postureguard" non trovato.
    echo.
    echo Crea prima l'ambiente con:
    echo conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

echo Avvio PostureGuard...
echo.

python src\main.py

echo.
echo PostureGuard terminato.
pause