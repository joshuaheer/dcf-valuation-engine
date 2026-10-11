"""
Data layer: pull financials from Yahoo Finance (via yfinance) and turn them
into model inputs with sensible, Damodaran-style defaults.

Every default is recorded in a provenance dict so the report can show where
each number came from. Any default can be overridden by a JSON config, which
is where your *story* for the company lives.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .model import DCFInputs
from .wacc import WACCBreakdown, estimate_wacc

SCALE = 1e6  # report money in millions

DEFAULTS = dict(
    marginal_tax_rate=0.25,
    mature_erp=0.0423,          # Damodaran implied US ERP, January 2026
    riskfree_fallback=0.042,    # used only if the 10y yield can't be fetched
    growth_clamp=(-0.05, 0.25),
    s2c_clamp=(0.5, 5.0),
    s2c_fallback=1.5,
    beta_clamp=(0.6, 2.0),
)


@dataclass
class Fundamentals:
    """Raw numbers pulled from the data source (money in millions)."""
    ticker: str
    name: str
    currency: str
    financial_currency: str
    price: float
    shares: float
    beta: float | None
    revenue_ttm: float
    ebit_ttm: float
    pretax_ttm: float | None
    tax_ttm: float | None
    interest_ttm: float | None
    revenue_history: list[float]      # annual, most recent first
    ebit_history: list[float]
    debt: float
    cash: float
    equity_book: float | None
    minority: float
    riskfree_rate: float
    as_of: str
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers for messy statement frames
# --------------------------------------------------------------------------- #
def _row(df: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            if isinstance(s, pd.DataFrame):      # duplicated labels
                s = s.iloc[0]
            if s.notna().any():
                return s
    return None


def _ttm(q: pd.DataFrame | None, a: pd.DataFrame | None, names: list[str]):
    """Sum of last four quarters if complete, else latest annual figure."""
    s = _row(q, names)
    if s is not None and s.iloc[:4].notna().sum() == 4:
        return float(s.iloc[:4].sum()), "TTM (last 4 quarters)"
    s = _row(a, names)
    if s is not None:
        return float(s.dropna().iloc[0]), "latest annual"
    return None, "missing"


def _latest(q: pd.DataFrame | None, a: pd.DataFrame | None, names: list[str]):
    for df in (q, a):
        s = _row(df, names)
        if s is not None:
            return float(s.dropna().iloc[0])
    return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def fetch_fundamentals(ticker: str, yf=None) -> Fundamentals:
    """Pull everything we need for one ticker. `yf` is injectable for tests."""
    if yf is None:
        try:
            import yfinance as yf  # noqa: F811
        except ImportError as e:
            raise ImportError("pip install yfinance") from e

    t = yf.Ticker(ticker)
    info = t.info or {}
    notes: list[str] = []
    qi, ai = t.quarterly_income_stmt, t.income_stmt
    qb, ab = t.quarterly_balance_sheet, t.balance_sheet

    rev_names = ["Total Revenue", "Operating Revenue"]
    ebit_names = ["Operating Income", "EBIT"]
    revenue, src_rev = _ttm(qi, ai, rev_names)
    ebit, src_ebit = _ttm(qi, ai, ebit_names)
    if revenue is None or ebit is None:
        raise ValueError(f"{ticker}: could not find revenue / operating income.")
    notes.append(f"Revenue and operating income: {src_rev}.")
    pretax, _ = _ttm(qi, ai, ["Pretax Income"])
    tax, _ = _ttm(qi, ai, ["Tax Provision"])
    interest, _ = _ttm(qi, ai, ["Interest Expense", "Interest Expense Non Operating"])

    rev_hist = _row(ai, rev_names)
    ebit_hist = _row(ai, ebit_names)

    debt = _latest(qb, ab, ["Total Debt"])
    if debt is None:
        parts = [_latest(qb, ab, [n]) for n in (
            "Long Term Debt And Capital Lease Obligation",
            "Current Debt And Capital Lease Obligation")]
        debt = sum(p for p in parts if p is not None)
        notes.append("Total Debt missing; summed long- and short-term debt.")
    cash = _latest(qb, ab, ["Cash Cash Equivalents And Short Term Investments",
                            "Cash And Cash Equivalents"]) or 0.0
    equity_book = _latest(qb, ab, ["Stockholders Equity", "Common Stock Equity"])
    minority = _latest(qb, ab, ["Minority Interest"]) or 0.0

    price = (info.get("currentPrice") or info.get("regularMarketPrice")
             or _fast_price(t))
    shares = info.get("sharesOutstanding") or _latest(qb, ab, ["Ordinary Shares Number"])
    if not price or not shares:
        raise ValueError(f"{ticker}: missing price or share count.")

    currency = info.get("currency", "USD")
    fin_ccy = info.get("financialCurrency", currency)
    if fin_ccy != currency:
        notes.append(f"WARNING: statements in {fin_ccy} but price in {currency}; "
                     "convert before trusting the per-share value.")

    rf = _riskfree(yf, notes) if currency == "USD" else None
    if rf is None:
        rf = DEFAULTS["riskfree_fallback"]
        notes.append(f"Risk-free rate: fallback {rf:.2%} (set riskfree_rate in "
                     "config for non-USD firms, e.g. the 10y gilt for GBP).")

    def m(x):
        return None if x is None else x / SCALE

    return Fundamentals(
        ticker=ticker.upper(), name=info.get("longName") or info.get("shortName") or ticker,
        currency=currency, financial_currency=fin_ccy,
        price=float(price), shares=float(shares) / SCALE,
        beta=info.get("beta"),
        revenue_ttm=m(revenue), ebit_ttm=m(ebit), pretax_ttm=m(pretax), tax_ttm=m(tax),
        interest_ttm=m(interest),
        revenue_history=[x / SCALE for x in (rev_hist.dropna().tolist() if rev_hist is not None else [])],
        ebit_history=[x / SCALE for x in (ebit_hist.dropna().tolist() if ebit_hist is not None else [])],
        debt=m(debt), cash=m(cash), equity_book=m(equity_book), minority=m(minority),
        riskfree_rate=rf, as_of=pd.Timestamp.today().strftime("%d %b %Y"), notes=notes,
    )


def _fast_price(t):
    try:
        return t.fast_info["lastPrice"]
    except Exception:
        return None


def _riskfree(yf, notes):
    try:
        h = yf.Ticker("^TNX").history(period="5d")
        rf = float(h["Close"].dropna().iloc[-1]) / 100
        notes.append(f"Risk-free rate: US 10y Treasury yield {rf:.2%} (^TNX).")
        return rf
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Defaults -> DCFInputs
# --------------------------------------------------------------------------- #
def build_inputs(f: Fundamentals, overrides: dict | None = None
                 ) -> tuple[DCFInputs, dict, WACCBreakdown]:
    """
    Turn raw fundamentals into model inputs.
    Returns (inputs, provenance {field: explanation}, WACC breakdown).
    """
    o = dict(overrides or {})
    prov: dict[str, str] = {}
    lo, hi = DEFAULTS["growth_clamp"]
    marginal = o.get("marginal_tax_rate", DEFAULTS["marginal_tax_rate"])
    erp = o.get("mature_erp", DEFAULTS["mature_erp"])
    rf = o.get("riskfree_rate", f.riskfree_rate)

    # Growth: last annual YoY for year 1; up-to-3y CAGR for years 2-5
    h = f.revenue_history
    if len(h) >= 2 and h[1] > 0:
        g1 = _clamp(h[0] / h[1] - 1, lo, hi)
        prov["revenue_growth_y1"] = "Last annual revenue growth (clamped)"
    else:
        g1 = rf
        prov["revenue_growth_y1"] = "Insufficient history: set to risk-free rate"
    if len(h) >= 3 and h[-1] > 0:
        yrs = min(len(h) - 1, 3)
        g25 = _clamp((h[0] / h[yrs]) ** (1 / yrs) - 1, lo, hi)
        prov["revenue_growth_y2_5"] = f"{yrs}-year historical revenue CAGR (clamped)"
    else:
        g25 = g1
        prov["revenue_growth_y2_5"] = "Same as year 1"

    margin_now = f.ebit_ttm / f.revenue_ttm
    prov["op_margin_y1"] = "Current TTM operating margin"
    hist_m = [e / r for e, r in zip(f.ebit_history, f.revenue_history) if r]
    target = margin_now
    prov["target_margin"] = "Current margin (no improvement assumed)"
    if hist_m:
        avg = sum(hist_m) / len(hist_m)
        prov["target_margin"] += f"; historical average {avg:.1%}"

    # Sales-to-capital: revenue / (book equity + debt - cash)
    ic = None
    if f.equity_book is not None:
        ic = f.equity_book + f.debt - f.cash
    s_lo, s_hi = DEFAULTS["s2c_clamp"]
    if ic and ic > 0:
        s2c = _clamp(f.revenue_ttm / ic, s_lo, s_hi)
        prov["sales_to_capital"] = "Revenue / invested capital (clamped 0.5-5.0)"
    else:
        s2c = DEFAULTS["s2c_fallback"]
        prov["sales_to_capital"] = "Invested capital non-positive: fallback 1.5"
        ic = None

    # Effective tax
    if f.pretax_ttm and f.pretax_ttm > 0 and f.tax_ttm is not None:
        eff = _clamp(f.tax_ttm / f.pretax_ttm, 0.0, marginal)
        prov["effective_tax_rate"] = "Tax provision / pre-tax income (TTM)"
    else:
        eff = marginal
        prov["effective_tax_rate"] = "Loss-making or missing: marginal rate"

    # WACC
    b_lo, b_hi = DEFAULTS["beta_clamp"]
    beta = o.pop("beta", None)
    if beta is None:
        beta = _clamp(f.beta, b_lo, b_hi) if f.beta else 1.0
        beta_src = "Yahoo regression beta (clamped 0.6-2.0)" if f.beta else "Beta missing: 1.0"
    else:
        beta_src = "Config override"
    w = estimate_wacc(riskfree_rate=rf, beta=beta, erp=erp, ebit=f.ebit_ttm,
                      interest_expense=f.interest_ttm or 0.0,
                      market_cap=f.price * f.shares, debt=f.debt,
                      marginal_tax_rate=marginal)
    prov["wacc_initial"] = (f"CAPM + synthetic rating ({w.rating}); beta: {beta_src}")
    prov["riskfree_rate"] = (("Config override" if "riskfree_rate" in o else "Market 10y yield")
                             + "; terminal growth defaults to it")
    prov["terminal_roc"] = "Default: equals terminal WACC (growth is value-neutral)"

    inp = DCFInputs(
        company=f.name, ticker=f.ticker, currency=f.currency, units="millions",
        revenue=f.revenue_ttm, ebit=f.ebit_ttm, debt=f.debt, cash=f.cash,
        shares_outstanding=f.shares, price=f.price,
        minority_interests=f.minority,
        revenue_growth_y1=g1, revenue_growth_y2_5=g25,
        op_margin_y1=margin_now, target_margin=target,
        sales_to_capital=s2c,
        effective_tax_rate=eff, marginal_tax_rate=marginal,
        riskfree_rate=rf, mature_erp=erp, wacc_initial=w.wacc,
        invested_capital=ic,
    )
    inp = apply_overrides(inp, o, prov)
    return inp, prov, w


def apply_overrides(inp: DCFInputs, overrides: dict, prov: dict | None = None) -> DCFInputs:
    d = inp.to_dict()
    for k, v in overrides.items():
        if k.startswith("_"):          # allow "_comment" keys in JSON
            continue
        if k not in d:
            raise ValueError(f"Unknown override '{k}'")
        d[k] = v
        if prov is not None:
            prov[k] = "Config override (your story)"
    return DCFInputs.from_dict(d)


def load_json(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def inputs_from_file(path: str | Path) -> DCFInputs:
    """Load a fully specified input file (no internet needed)."""
    d = {k: v for k, v in load_json(path).items() if not k.startswith("_")}
    return DCFInputs.from_dict(d)
