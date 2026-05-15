@echo off
cd /d "C:\Users\Bruce\source\repos\RegimeTrading"
call .venv\Scripts\activate.bat
python scripts\intraday_checks.py >> logs\intraday_checks.log 2>&1
