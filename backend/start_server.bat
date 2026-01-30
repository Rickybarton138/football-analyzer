@echo off
echo Starting Football Analyzer Backend Server...
cd /d "C:\Users\info\football-analyzer\backend"
call venv\Scripts\activate.bat
python main.py
pause
