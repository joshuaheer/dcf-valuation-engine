"""
Core FCFF valuation model.

A Python rebuild of the mechanics in Aswath Damodaran's "simple Ginzu"
spreadsheet (fcffsimpleginzu.xlsx):

  * Revenue growth: year-1 rate, a flat rate for years 2-5, then a linear fade
    to the terminal growth rate by year 10.
  * Operating margin: year-1 margin, converging linearly to a target margin
    by a chosen "year of convergence".
  * Tax rate: effective rate for years 1-5, moving linearly to the marginal
    rate by year 10. Net operating losses (NOLs) shield taxable income.
  * Reinvestment: change in revenue / sales-to-capital ratio (optionally lagged
    one year, i.e. you invest this year for next year's growth).
  * Cost of capital: initial WACC for years 1-5, fading to a mature-firm WACC.
  * Terminal value: FCFF in year 11 / (terminal WACC - terminal growth), with
    terminal reinvestment = g / terminal ROC, so growth is never "free".

Every computation is vectorised: pass arrays for any input and you get one
valuation per element. This is what makes 10,000-run Monte Carlo instant.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields, replace
from typing import Optional

import numpy as np
import pandas as pd

YEARS = 10


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class DCFInputs:
    """All inputs to the model. Money values in `units` of `currency`."""

    # -- Base year (trailing twelve months / latest balance sheet) --
    revenue: float
    ebit: float                         # operating income
    debt: float                         # book debt incl. lease liabilities
    cash: float                         # cash + marketable securities
    shares_outstanding: float           # in millions if units are millions
    price: float                        # current share price

    # -- The story --
    revenue_growth_y1: float
    revenue_growth_y2_5: float          # compounded annual growth, years 2-5
    op_margin_y1: float
    target_margin: float                # pre-tax operating margin at maturity
    sales_to_capital: float             # years 1-5

    # -- Tax & discount rates --
    effective_tax_rate: float
    riskfree_rate: float
    wacc_initial: float

    # -- Optional, with Damodaran-style defaults --
    convergence_year: int = 5
    sales_to_capital_y6_10: Optional[float] = None   # default: same as yrs 1-5
    marginal_tax_rate: float = 0.25
    mature_erp: float = 0.0423          # Damodaran implied ERP, Jan 2026
    wacc_terminal: Optional[float] = None   # default: riskfree + mature ERP
    terminal_growth: Optional[float] = None # default: riskfree rate
    terminal_roc: Optional[float] = None    # default: terminal WACC
    reinvestment_lag: int = 1           # 1 = invest a year ahead of growth
    nol_carryforward: float = 0.0
    minority_interests: float = 0.0
    non_operating_assets: float = 0.0
    options_value: float = 0.0          # value of employee options outstanding
    invested_capital: Optional[float] = None  # for ROIC path (optional)
    prob_failure: float = 0.0
    failure_proceeds_pct: float = 0.5   # recovery as % of operating value

    # -- Labels --
    company: str = "Company"
    ticker: str = ""
    currency: str = "USD"
    units: str = "millions"

    def resolved(self) -> "DCFInputs":
        """Fill in the Damodaran defaults for any None field."""
        wt = self.wacc_terminal
        if wt is None:
            wt = self.riskfree_rate + self.mature_erp
        g = self.terminal_growth if self.terminal_growth is not None else self.riskfree_rate
        return replace(
            self,
            wacc_terminal=wt,
            terminal_growth=g,
            terminal_roc=self.terminal_roc if self.terminal_roc is not None else wt,
            sales_to_capital_y6_10=(self.sales_to_capital_y6_10
                                    if self.sales_to_capital_y6_10 is not None
                                    else self.sales_to_capital),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DCFInputs":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown input field(s): {sorted(unknown)}")
        return cls(**d)


# Numeric fields that the vectorised core accepts as arrays
_NUMERIC = [
    "revenue", "ebit", "debt", "cash", "shares_outstanding",
    "revenue_growth_y1", "revenue_growth_y2_5", "op_margin_y1", "target_margin",
    "sales_to_capital", "sales_to_capital_y6_10", "convergence_year",
    "effective_tax_rate", "marginal_tax_rate", "wacc_initial", "wacc_terminal",
    "terminal_growth", "terminal_roc", "nol_carryforward", "minority_interests",
    "non_operating_assets", "options_value", "prob_failure", "failure_proceeds_pct",
]


# --------------------------------------------------------------------------- #
# Vectorised core
# --------------------------------------------------------------------------- #
def project(inp: DCFInputs, **arrays) -> dict:
    """
    Run the model. Any numeric input can be overridden with a 1-D array of
    length N (e.g. `wacc_initial=np.array([...])`); all arrays broadcast
    together. Returns a dict of arrays: per-year arrays are shape (N, 10),
    scalars are shape (N,).
    """
    r = inp.resolved()
    p = {k: getattr(r, k) for k in _NUMERIC}
    p.update(arrays)
    n = max(np.size(v) for v in p.values())
    P = {k: np.broadcast_to(np.asarray(v, dtype=float).reshape(-1), (n,)).copy()
         if np.size(v) in (1, n) else _bad(k, v, n) for k, v in p.items()}
    col = {k: v[:, None] for k, v in P.items()}  # (N,1) for year-axis maths

    years = np.arange(1, YEARS + 1)[None, :]     # (1,10)

    # --- Revenue growth path ---
    g = np.empty((n, YEARS))
    g[:, 0] = P["revenue_growth_y1"]
    g[:, 1:5] = col["revenue_growth_y2_5"]
    step_g = (P["revenue_growth_y2_5"] - P["terminal_growth"]) / 5
    for k in range(5, YEARS):
        g[:, k] = g[:, k - 1] - step_g

    revenue = col["revenue"] * np.cumprod(1 + g, axis=1)

    # --- Operating margin path (linear convergence to target) ---
    conv = col["convergence_year"]
    tgt, m1 = col["target_margin"], col["op_margin_y1"]
    margin = np.where(years > conv, tgt, tgt - (tgt - m1) / conv * (conv - years))
    margin[:, 0] = P["op_margin_y1"]
    ebit = revenue * margin

    # --- Tax rate path ---
    tax = np.empty((n, YEARS))
    tax[:, :5] = col["effective_tax_rate"]
    step_t = (P["marginal_tax_rate"] - P["effective_tax_rate"]) / 5
    for k in range(5, YEARS):
        tax[:, k] = tax[:, k - 1] + step_t

    # --- After-tax operating income with NOL shield ---
    ebit_at = np.empty((n, YEARS))
    nol = np.empty((n, YEARS))
    nol_prev = P["nol_carryforward"].copy()
    for k in range(YEARS):
        e = ebit[:, k]
        taxable = np.clip(e - nol_prev, 0, None)
        ebit_at[:, k] = e - taxable * tax[:, k]
        nol_prev = np.where(e < 0, nol_prev - e, np.clip(nol_prev - e, 0, None))
        nol[:, k] = nol_prev

    # --- Terminal year (year 11) revenue, needed for lagged reinvestment ---
    rev_term = revenue[:, -1] * (1 + P["terminal_growth"])

    # --- Reinvestment ---
    s2c = np.empty((n, YEARS))
    s2c[:, :5] = col["sales_to_capital"]
    s2c[:, 5:] = col["sales_to_capital_y6_10"]
    if inp.reinvestment_lag == 1:
        rev_next = np.column_stack([revenue[:, 1:], rev_term])
        reinvestment = (rev_next - revenue) / s2c
    elif inp.reinvestment_lag == 0:
        rev_prev = np.column_stack([P["revenue"], revenue[:, :-1]])
        reinvestment = (revenue - rev_prev) / s2c
    else:
        raise ValueError("reinvestment_lag must be 0 or 1")

    fcff = ebit_at - reinvestment

    # --- Cost of capital path & discounting ---
    wacc = np.empty((n, YEARS))
    wacc[:, :5] = col["wacc_initial"]
    step_w = (P["wacc_initial"] - P["wacc_terminal"]) / 5
    for k in range(5, YEARS):
        wacc[:, k] = wacc[:, k - 1] - step_w
    discount = np.cumprod(1 / (1 + wacc), axis=1)
    pv_fcff = fcff * discount

    # --- Terminal value ---
    gT, wT = P["terminal_growth"], P["wacc_terminal"]
    if np.any(wT <= gT):
        raise ValueError("Terminal WACC must exceed terminal growth.")
    ebit_term = rev_term * P["target_margin"]
    taxable_T = np.clip(ebit_term - nol_prev, 0, None)
    ebit_at_term = ebit_term - taxable_T * P["marginal_tax_rate"]
    reinv_term = np.where(gT > 0, gT / P["terminal_roc"] * ebit_at_term, 0.0)
    fcff_term = ebit_at_term - reinv_term
    terminal_value = fcff_term / (wT - gT)
    pv_terminal = terminal_value * discount[:, -1]

    # --- From operating value to value per share ---
    sum_pv = pv_fcff.sum(axis=1)
    pv_operating = sum_pv + pv_terminal
    p_f = P["prob_failure"]
    value_operating = pv_operating * (1 - p_f) + p_f * P["failure_proceeds_pct"] * pv_operating
    equity = (value_operating - P["debt"] - P["minority_interests"]
              + P["cash"] + P["non_operating_assets"])
    equity_common = equity - P["options_value"]
    value_per_share = equity_common / P["shares_outstanding"]

    out = dict(
        growth=g, revenue=revenue, margin=margin, ebit=ebit, tax_rate=tax,
        ebit_after_tax=ebit_at, nol=nol, reinvestment=reinvestment,
        sales_to_capital=s2c, fcff=fcff, wacc=wacc, discount_factor=discount,
        pv_fcff=pv_fcff,
        revenue_terminal=rev_term, ebit_terminal=ebit_term,
        ebit_after_tax_terminal=ebit_at_term, reinvestment_terminal=reinv_term,
        fcff_terminal=fcff_term, terminal_value=terminal_value,
        pv_terminal=pv_terminal, sum_pv_fcff=sum_pv, pv_operating=pv_operating,
        value_operating=value_operating, equity_value=equity,
        equity_common=equity_common, value_per_share=value_per_share,
    )

    # --- Optional: invested capital & ROIC path (Damodaran shows this row) ---
    if inp.invested_capital is not None:
        ic = inp.invested_capital + np.cumsum(reinvestment, axis=1)
        ic_prev = np.column_stack([np.full(n, inp.invested_capital), ic[:, :-1]])
        out["invested_capital"] = ic
        out["roic"] = ebit_at / ic_prev
    return out


def _bad(k, v, n):
    raise ValueError(f"Input '{k}' has size {np.size(v)}; expected 1 or {n}.")


# --------------------------------------------------------------------------- #
# Single-case convenience wrapper
# --------------------------------------------------------------------------- #
@dataclass
class DCFResult:
    inputs: DCFInputs
    table: pd.DataFrame                 # rows = line items, cols = Base,1..10,Terminal
    terminal_value: float
    pv_terminal: float
    sum_pv_fcff: float
    pv_operating: float
    value_operating: float
    equity_value: float
    equity_common: float
    value_per_share: float

    @property
    def upside(self) -> float:
        return self.value_per_share / self.inputs.price - 1

    @property
    def terminal_share(self) -> float:
        """Share of operating value coming from the terminal value."""
        return self.pv_terminal / self.pv_operating

    def bridge(self) -> list[tuple[str, float]]:
        r = self.inputs.resolved()
        items = [
            ("PV of FCFF (yrs 1-10)", self.sum_pv_fcff),
            ("PV of terminal value", self.pv_terminal),
        ]
        if r.prob_failure:
            items.append(("Failure adjustment", self.value_operating - self.pv_operating))
        items += [
            ("Less: debt", -r.debt),
            ("Less: minority interests", -r.minority_interests),
            ("Plus: cash", r.cash),
            ("Plus: non-operating assets", r.non_operating_assets),
            ("Less: options", -r.options_value),
        ]
        return [(k, v) for k, v in items if v != 0 or k.startswith("PV")]


def run_dcf(inp: DCFInputs) -> DCFResult:
    """Run one deterministic valuation and return a tidy result."""
    o = project(inp)
    r = inp.resolved()
    s = {k: (v[0] if np.ndim(v) == 1 else v[0]) for k, v in o.items()}

    cols = ["Base"] + [str(i) for i in range(1, YEARS + 1)] + ["Terminal"]
    base_margin = inp.ebit / inp.revenue
    nan = np.nan
    rows = {
        "Revenue growth": [nan, *s["growth"], r.terminal_growth],
        "Revenue": [inp.revenue, *s["revenue"], s["revenue_terminal"]],
        "Operating margin": [base_margin, *s["margin"], r.target_margin],
        "EBIT": [inp.ebit, *s["ebit"], s["ebit_terminal"]],
        "Tax rate": [r.effective_tax_rate, *s["tax_rate"], r.marginal_tax_rate],
        "EBIT(1-t)": [nan, *s["ebit_after_tax"], s["ebit_after_tax_terminal"]],
        "Reinvestment": [nan, *s["reinvestment"], s["reinvestment_terminal"]],
        "FCFF": [nan, *s["fcff"], s["fcff_terminal"]],
        "NOL": [inp.nol_carryforward, *s["nol"], nan],
        "Cost of capital": [nan, *s["wacc"], r.wacc_terminal],
        "Discount factor": [nan, *s["discount_factor"], nan],
        "PV(FCFF)": [nan, *s["pv_fcff"], nan],
    }
    if "roic" in s:
        rows["Invested capital"] = [inp.invested_capital, *s["invested_capital"], nan]
        rows["ROIC"] = [inp.ebit * (1 - r.effective_tax_rate) / inp.invested_capital,
                        *s["roic"], r.terminal_roc]
    table = pd.DataFrame(rows, index=cols).T

    return DCFResult(
        inputs=inp, table=table,
        terminal_value=float(s["terminal_value"]),
        pv_terminal=float(s["pv_terminal"]),
        sum_pv_fcff=float(s["sum_pv_fcff"]),
        pv_operating=float(s["pv_operating"]),
        value_operating=float(s["value_operating"]),
        equity_value=float(s["equity_value"]),
        equity_common=float(s["equity_common"]),
        value_per_share=float(s["value_per_share"]),
    )


# --------------------------------------------------------------------------- #
# Consistency checks (Damodaran's "sanity checks")
# --------------------------------------------------------------------------- #
def sanity_checks(res: DCFResult) -> list[tuple[str, str]]:
    """Return (status, message) pairs; status is 'ok' or 'warn'."""
    r = res.inputs.resolved()
    t = res.table
    out = []

    def add(ok, msg):
        out.append(("ok" if ok else "warn", msg))

    add(r.terminal_growth <= r.riskfree_rate + 1e-12,
        f"Terminal growth ({r.terminal_growth:.2%}) is at or below the risk-free "
        f"rate ({r.riskfree_rate:.2%}); no firm can outgrow the economy forever.")
    add(r.terminal_roc >= r.wacc_terminal - 1e-12,
        f"Terminal ROC ({r.terminal_roc:.2%}) vs terminal WACC "
        f"({r.wacc_terminal:.2%}): growth is value-neutral or better in steady state.")
    ts = res.terminal_share
    add(ts < 0.8, f"Terminal value is {ts:.0%} of operating value"
        + (" (high: the valuation leans heavily on the long run)." if ts >= 0.8 else "."))
    neg = int((t.loc["FCFF", "1":"10"] < 0).sum())
    add(neg == 0, f"{neg} of 10 projected years have negative FCFF.")
    if "ROIC" in t.index:
        roic10 = t.loc["ROIC", "10"]
        add(roic10 < 0.5, f"Implied year-10 ROIC is {roic10:.1%} "
            "(check the sales-to-capital ratio if this looks implausible).")
    rev_mult = t.loc["Revenue", "10"] / t.loc["Revenue", "Base"]
    add(rev_mult < 5, f"Revenue grows {rev_mult:.1f}x over ten years.")
    return out
