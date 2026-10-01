@echo off

echo ========================================
echo       PostureGuard - Installation
echo ========================================
echo.

echo Creazione dell'ambiente Conda...
echo.

call conda env create -f environment.yml

if errorlevel 1 (
    echo.
    echo ERRORE durante la creazione dell'ambiente.
    echo Controlla che Conda sia installato.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installazione completata!
echo ========================================
echo.
echo Ora puoi avviare PostureGuard eseguendo:
echo run.bat
echo.

pause