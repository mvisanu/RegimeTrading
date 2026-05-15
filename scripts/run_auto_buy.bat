@echo off
cd /d "C:\Users\Bruce\source\repos\RegimeTrading"
call .venv\Scripts\activate.bat
python scripts\auto_buy.py >> logs\auto_buy.log 2>&1
