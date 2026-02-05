@echo off
REM Create a Python virtual environment in .venv
python -m venv .venv

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required packages
REM Install all required and optional dependencies for the project, including paper/quarto support
pip install -e .[all]

REM Install Quarto CLI if not already installed (user must have it on PATH for paper rendering)
echo If you need Quarto CLI, download from https://quarto.org/docs/get-started/

echo Virtual environment setup complete.
echo To activate later, run:
echo    call .venv\Scripts\activate.bat
