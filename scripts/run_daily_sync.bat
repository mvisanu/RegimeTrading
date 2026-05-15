@echo off
cd /d "C:\Users\Bruce\source\repos\RegimeTrading"
call .venv\Scripts\activate.bat
python scripts\daily_sync.py >> logs\daily_sync.log 2>&1
