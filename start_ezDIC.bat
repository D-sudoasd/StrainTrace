@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ezDIC launcher

pushd "%~dp0" || (
    echo [ERROR] Failed to enter the ezDIC project directory:
    echo         %~dp0
    pause
    exit /b 1
)

set "ENTRY=dic_virtual_extensometer_gui_v7_multi_roi_range.py"
set "REQ=requirements.txt"
set "VENV_PY=.venv\Scripts\python.exe"
set "REQ_MARKER=.venv\.ezDIC_requirements.txt"

if not exist "%ENTRY%" (
    echo [ERROR] Cannot find the ezDIC entry script:
    echo         %CD%\%ENTRY%
    echo Make sure this .bat file is still in the project root folder.
    popd
    pause
    exit /b 1
)

if defined EZDIC_LAUNCHER_SMOKE_TEST (
    echo EZDIC launcher smoke test
    echo Project: %CD%
    echo Python: %VENV_PY%
    echo Entry: %ENTRY%
    popd
    exit /b 0
)

if exist "%VENV_PY%" (
    "%VENV_PY%" --version >nul 2>nul
    if errorlevel 1 (
        echo Existing local Python environment is not usable. Recreating .venv...
        rmdir /s /q ".venv" >nul 2>nul
        if exist "%VENV_PY%" (
            echo [ERROR] Failed to remove the unusable .venv folder.
            echo Close programs using %CD%\.venv and try again.
            echo.
            popd
            pause
            exit /b 1
        )
    )
)

if not exist "%VENV_PY%" (
    set "BASE_PYTHON="
    where py >nul 2>nul
    if not errorlevel 1 (
        for %%V in (3.11 3.12 3.13) do (
            if not defined BASE_PYTHON (
                py -%%V --version >nul 2>nul
                if not errorlevel 1 set "BASE_PYTHON=py -%%V"
            )
        )
        if not defined BASE_PYTHON (
            py -3 --version >nul 2>nul
            if not errorlevel 1 set "BASE_PYTHON=py -3"
        )
    )

    if not defined BASE_PYTHON (
        where python >nul 2>nul
        if not errorlevel 1 (
            python --version >nul 2>nul
            if not errorlevel 1 set "BASE_PYTHON=python"
        )
    )

    if not defined BASE_PYTHON (
        echo [ERROR] Python was not found.
        echo Install Python 3, then double-click this file again.
        echo.
        popd
        pause
        exit /b 1
    )

    echo Creating local Python environment: %CD%\.venv
    !BASE_PYTHON! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        echo.
        popd
        pause
        exit /b 1
    )

    "%VENV_PY%" --version >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Created .venv, but its Python executable is not usable.
        echo.
        popd
        pause
        exit /b 1
    )
)

set "PYTHON_CMD=%VENV_PY%"

if exist "%REQ%" (
    set "NEED_INSTALL="
    if not exist "%REQ_MARKER%" (
        set "NEED_INSTALL=1"
    ) else (
        fc /b "%REQ%" "%REQ_MARKER%" >nul 2>nul
        if errorlevel 1 set "NEED_INSTALL=1"
    )

    if defined NEED_INSTALL (
        echo Installing or updating ezDIC dependencies...
        "%PYTHON_CMD%" -m pip install -r "%REQ%"
        if errorlevel 1 (
            echo [ERROR] Dependency installation failed.
            echo.
            popd
            pause
            exit /b 1
        )
        copy /y "%REQ%" "%REQ_MARKER%" >nul
    )
)

echo Starting ezDIC...
echo Project: %CD%
echo Python: %PYTHON_CMD%
echo Entry: %ENTRY%
echo.

"%PYTHON_CMD%" "%ENTRY%"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] ezDIC exited with code %EXITCODE%.
    echo.
    pause
)

popd
exit /b %EXITCODE%
