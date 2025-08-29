@echo off

IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m backend.main %*