# Python DCF Engine

An intrinsic valuation engine that rebuilds Aswath Damodaran's **FCFF "simple Ginzu"** spreadsheet in Python, pulls live financials for any ticker, stress-tests the answer with Monte Carlo simulation, and writes a four-page equity research report as a PDF.

```bash
pip install -r requirements.txt
python run.py MSFT                                   # live data, default assumptions
python run.py MSFT --config configs/example_override.json   # your story on top
python run.py --input examples/sample_company.json   # offline example
```

Output in `reports/`: `<TICKER>_DCF_report.pdf`, the projection table and the sensitivity grid as CSV.

## What the report contains

1. **The call.** Value per share, price, upside, Monte Carlo range, BUY / HOLD / SELL, a written investment view, every key driver with where it came from, and a value bridge chart.
2. **The model.** Ten-year projection (revenue, margin, taxes, reinvestment, FCFF, cost of capital, PV, ROIC), terminal value and charts.
3. **Risk.** 10,000-run Monte Carlo distribution, WACC × terminal-growth sensitivity grid, and a tornado chart ranking the inputs that matter most.
4. **Checks and method.** Damodaran-style sanity checks, the WACC build (CAPM + synthetic rating), methodology and data notes.

## How the model works

| Line | Mechanic |
|---|---|
| Revenue growth | Year-1 rate, flat years 2-5 rate, linear fade to terminal growth by year 10 |
| Operating margin | Year-1 margin converging linearly to a target by the year of convergence |
| Tax | Effective rate in years 1-5, moving to the marginal rate by year 10; NOLs shield profits |
| Reinvestment | Δ revenue ÷ sales-to-capital (lag 1 = invest a year ahead of the growth) |
| FCFF | EBIT(1 − t) − reinvestment |
| Discounting | Initial WACC for years 1-5, fading to a mature-firm WACC (risk-free + ERP) |
| Terminal value | FCFF₁₁ ÷ (WACC − g), where reinvestment = g ÷ ROC, so growth is never free |
| Equity bridge | Operating value − debt − minority interests + cash + non-operating assets − options |

The core (`dcf/model.py`) is fully vectorised in NumPy: pass an array for any input and you get one valuation per element. That is why 10,000 full ten-year valuations run in well under a second.

**Default assumptions from live data** (`dcf/data.py`): TTM revenue and operating income from the last four quarters; year-1 growth from last year; years 2-5 growth from the three-year CAGR; margin held at today's level; sales-to-capital from revenue ÷ invested capital; WACC from CAPM (US 10-year yield, Yahoo beta, Damodaran's January 2026 implied ERP of 4.23%) and a synthetic credit rating from interest coverage. Every default can be overridden in a JSON config, which is where your view of the company goes.

## Validation against Damodaran's spreadsheet

1. Download `fcffsimpleginzu.xlsx` from Damodaran's [spreadsheet page](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/spreadsh.htm) and fill its Input sheet for a company.
2. Copy the same inputs and the year-by-year outputs into a copy of `validation/template.json`.
3. Run `python -m dcf.validate validation/my_company.json`.

Every row is compared year by year (PASS / FAIL at 0.01% tolerance), so a mismatch points straight at the mechanic that differs. If only reinvestment and FCFF are off, flip `reinvestment_lag` between 0 and 1. The spreadsheet's sheet layout changes between versions, so check any default it applies (terminal WACC, terminal ROC) and copy it into the JSON explicitly.

## Tests

```bash
pytest
```

19 tests pin the engine to closed-form answers. Examples: zero growth must equal a perpetuity (FCFF ÷ WACC); constant growth with consistent reinvestment must equal the Gordon growth model exactly; NOLs, margin convergence, the equity bridge, Monte Carlo reproducibility and the data parser (with a mocked yfinance) are all checked.

## Project structure

```
dcf/model.py      core vectorised FCFF model + sanity checks
dcf/wacc.py       CAPM cost of equity, synthetic rating cost of debt
dcf/data.py       yfinance ingestion, defaults, config overrides
dcf/analysis.py   Monte Carlo, sensitivity grid, tornado, recommendation rule
dcf/report.py     charts and the PDF report
dcf/validate.py   line-by-line comparison against the Excel model
run.py            command-line entry point
```

## Limitations

- Yahoo data is free and occasionally messy; always sanity-check revenue, debt and share count against the annual report.
- Non-USD companies need `riskfree_rate` set in the config (e.g. the 10-year gilt yield for a UK firm), and ADRs can report statements and prices in different currencies (the report flags this).
- Banks and insurers need a dividend or excess-return model, not FCFF.
- Synthetic-rating spreads are a snapshot; refresh them from Damodaran's `ratings.xls` each January.

Educational project. Not investment advice.
