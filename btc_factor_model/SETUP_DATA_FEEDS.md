# SETUP_DATA_FEEDS.md — getting the model onto real data

The model runs on synthetic data with zero setup. This guide is for wiring **real**
feeds so you can refine it. Do the steps in order — **FRED + Yahoo alone give you
~60% of the model** (all macro factors + the power-law valuation backbone), so start
there, confirm it runs, then add the rest.

The contract for every source: return a **daily DataFrame indexed by date** whose
**columns are dictionary IDs** (see `data_dictionary.py`). Adapters return data
stamped at its observation date — do **not** apply publication lags in the adapter
(the pipeline does that once).

---

## 0. One-time setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Keep keys out of code. Create a `.env` file in the project root:

```
FRED_API_KEY=your_key_here
```

Load it at the top of any script: `from dotenv import load_dotenv; load_dotenv()`.

---

## 1. FRED — the only key in the free stack (do this first)

Unlocks every macro factor: net liquidity, yields, DXY, VIX, HY spreads, M2.

1. Create a free account at **fredaccount.stlouisfed.org** (email + password, instant).
2. Request a key at **fredaccount.stlouisfed.org/apikeys** (free, same day, 32-char).
3. Put it in `.env` as `FRED_API_KEY`.
4. Test:

```python
import os; from fredapi import Fred
fred = Fred(api_key=os.environ["FRED_API_KEY"])
print(fred.get_series("WALCL").tail())      # Fed balance sheet, weekly
```

Series IDs needed (already referenced in `data/macro_data.py`):
`WALCL, WTREGEN, RRPONTSYD, DGS2, DGS10, DFII10, DTWEXBGS, VIXCLS, BAMLH0A0HYM2, M2SL`.

**Unit gotcha:** WALCL & WTREGEN are in **millions**, RRPONTSYD is in **billions**.
Net liquidity = `(WALCL − WTREGEN − RRPONTSYD*1000) / 1e6` (trillions). `macro_data.py`
already handles this.

---

## 2. Keyless APIs — install and call

**Yahoo / yfinance** — BTC price (the target) + FX/equities:
```python
import yfinance as yf
btc = yf.download("BTC-USD", start="2018-01-01", interval="1d", auto_adjust=False)
# also "JPY=X", "^VIX", "^MOVE", "^GSPC", "GC=F", "HG=F"
```
If Yahoo throttles (empty result), retry or use `yf.Ticker("BTC-USD").history(period="max")`.
Keep Stooq as a backup price source.

**Coin Metrics Community** — on-chain, no key:
```python
from coinmetrics.api_client import CoinMetricsClient
client = CoinMetricsClient()                 # no key = community tier
df = client.get_asset_metrics(
    assets="btc",
    metrics=["AdrActCnt", "SplyCur", "CapMrktCurUSD", "CapRealUSD"],
    frequency="1d").to_dataframe()
```
Stay under ~1.6 req/s. Covers active addresses, supply, market cap, realized cap.

**Deribit** — options open interest (public, US-accessible):
```python
import requests
r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                 params={"currency": "BTC", "kind": "option"}).json()
options_oi = sum(x["open_interest"] for x in r["result"])
```

**alternative.me** — crypto Fear & Greed (`limit=0` = full history):
```python
import requests, pandas as pd
data = requests.get("https://api.alternative.me/fng/", params={"limit": 0}).json()["data"]
fng = pd.DataFrame(data)
```

**DefiLlama** — stablecoin supply:
```python
import requests
r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all").json()
```

---

## 3. Derivatives — US geo caveat

Options OI via Deribit works. But Binance/Bybit **futures** APIs (perp funding,
futures OI) are geo-blocked from US IPs. Options:
- pull what you can from Deribit,
- subscribe to Coinglass (~$29/mo) which aggregates funding/OI/ETF flows and sidesteps
  the geo block,
- or **de-weight the derivatives factor** at first — it's a fragility overlay, not the
  backbone. Recommended: de-weight, revisit later.

---

## 4. Scrapers

**Farside — spot BTC ETF flows** (the go-to free source since the Jan-2024 launch):
```python
import requests, pandas as pd, io
url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text   # UA avoids a block
raw = pd.read_html(io.StringIO(html))[0]
df = raw.copy()
df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns]
df = df.rename(columns={df.columns[0]: "date"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).set_index("date")
clean = (df.replace({r"[(]": "-", r"[)]": "", ",": "", "-": "0"}, regex=True)
           .apply(pd.to_numeric, errors="coerce"))
etf_net_flow = clean["Total"]      # -> the etf_net_flow column (US$m)
```
Scrape once a day, with the User-Agent set. Farside updates evenings US time.

**Treasury holdings** (low priority, episodic): scrape an aggregator like
**bitcointreasuries.net** the same way, or use **SEC EDGAR** full-text search at
`efts.sec.gov` (free, requires a `User-Agent: Your Name you@email.com` header, 10 req/s
cap; pulling clean 8-K purchase numbers is an NLP job). Scrape the aggregator for now.

---

## 5. Wiring it into `data/sources.py`

Each tested snippet becomes the body of a stub. Map sources → blocks:

| `data/sources.py` function | fill with |
|---|---|
| `macro_liquidity_block` / `funding_stress_block` | FRED (already scaffolded in `macro_data.py`) |
| `btc_price` | yfinance `BTC-USD` |
| `onchain_block` | Coin Metrics community |
| `etf_flows_block` | Farside scraper's `Total` column |
| `derivatives_block` | Deribit (+ Coinglass if you subscribe) |
| `treasury_block` | bitcointreasuries / EDGAR |

Each must return a daily DataFrame with **dictionary-ID column names**. Then:

```bash
python -m btc_factor_model.daily_run --source real
```

`daily_run.py` calls `pipeline_real_factors()`, which concatenates whichever blocks
you've implemented (skipping the rest), so you can go live incrementally.

**Recommended order:** FRED → Yahoo → Coin Metrics → Farside → derivatives. Validate
each against the synthetic baseline (`skill_metrics.csv`) before adding the next, so
you're never debugging ten sources at once.

---

## 6. Free-tier reference (verified)

| Source | Key? | Free? | Feeds |
|---|---|---|---|
| FRED | free key | ✅ | net liquidity, yields, DXY, VIX, HY OAS, M2 |
| Yahoo (yfinance) | none | ✅ | BTC OHLCV, USD/JPY, S&P, gold/copper, ^VIX, ^MOVE |
| Coin Metrics Community | none | ✅ | active addresses, supply, market cap, realized cap |
| Deribit | none | ✅ | options OI, put/call |
| alternative.me | none | ✅ | Fear & Greed |
| DefiLlama | none | ✅ | stablecoin supply |
| Farside | scrape | ✅ | spot BTC ETF daily net flows |
| Glassnode | paid | ❌ | illiquid/LTH supply cohorts (API is a Professional add-on) |
| Coinglass | paid | ❌ | aggregated OI/funding/ETF flows ($29/mo, no free tier) |
| Binance/Bybit futures | none | ⚠️ | funding/OI — **geo-blocked from US IPs** |
