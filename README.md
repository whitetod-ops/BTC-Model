# BTC Factor Model

An attribution + **valuation** engine for Bitcoin. It does **not** try to forecast
price (that was tested honestly and has no out-of-sample edge); its trustworthy
output is the **power-law valuation cone** — where BTC sits versus its structural
trend and a plausible 3/6/12-month range.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r btc_factor_model/requirements.txt lxml html5lib
# put your free FRED key in a .env file:  FRED_API_KEY=xxxx
python -m btc_factor_model.daily_run --source real
```
Outputs land in `artifacts/`: open `daily_dashboard.html` and `valuation_cone.html`.

## Honest evaluation tools
```bash
python -m btc_factor_model.backtest --cone --source real          # valuation calibration
python -m btc_factor_model.backtest --cone-holdout --source real  # frozen out-of-sample test
python -m btc_factor_model.backtest --anchors --source real       # power-law vs Metcalfe vs blend
python -m btc_factor_model.backtest --regime --source real        # factor IC by volatility regime
python -m btc_factor_model.backtest --factors --source real       # expected sign vs realized IC
python -m btc_factor_model.backtest --composite --source real     # fixed-weight composite vs ML
```

## Daily auto-update
`.github/workflows/daily.yml` runs every night (00:00 Central) on GitHub Actions,
rebuilds the dashboard + cone, and commits them back. Set repo secret
`FRED_API_KEY`. See the workflow file to change the schedule.

*Research tooling. Not investment advice.*
