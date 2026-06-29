@echo off
REM Nightly BTC model run + git publish. Point Task Scheduler at this file.
cd /d "C:\Users\white\Claude\Projects\BTC Model"
call .venv\Scripts\activate.bat
python -m btc_factor_model.daily_run --source real
if errorlevel 1 (echo daily_run failed & exit /b 1)
git add artifacts\daily_dashboard.html artifacts\valuation_cone.html artifacts\skill_metrics.csv artifacts\parent_scores.csv
git commit -m "daily run %date% %time%"
git push
