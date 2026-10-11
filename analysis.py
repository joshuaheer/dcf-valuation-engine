"""
Quant layer: Monte Carlo simulation, WACC x terminal-growth sensitivity grid,
and a tornado (one-at-a-time) sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import DCFInputs, project


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass
class MCSpec:
    """Standard deviations (absolute, in rate terms) for the uncertain inputs."""
    sims: int = 10_000
    sd_growth: float = 0.03          # years 2-5 revenue CAGR
    sd_margin: float = 0.02          # target operating margin
    sd_wacc: float = 0.0075          # initial cost of capital
    sd_s2c_pct: float = 0.15         # sales-to-capital, as % of base
    rho_growth_margin: float = 0.3   # good stories tend to come together
    seed: int = 42


@dataclass
class MCResult:
    values: np.ndarray
    samples: pd.DataFrame
    price: float

    @property
    def prob_undervalued(self) -> float:
        return float((self.values > self.price).mean())

    def pct(self, q) -> float:
        return float(np.percentile(self.values, q))

    def summary(self) -> dict:
        v = self.values
        return dict(mean=float(v.mean()), median=self.pct(50), std=float(v.std()),
                    p5=self.pct(5), p25=self.pct(25), p75=self.pct(75),
                    p95=self.pct(95), prob_undervalued=self.prob_undervalued)


def monte_carlo(inp: DCFInputs, spec: MCSpec = MCSpec()) -> MCResult:
    r = inp.resolved()
    rng = np.random.default_rng(spec.seed)
    n = spec.sims

    # Correlated normal shocks for growth and margin
    z = rng.standard_normal((n, 4))
    rho = spec.rho_growth_margin
    z_margin = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]

    growth = r.revenue_growth_y2_5 + spec.sd_growth * z[:, 0]
    margin = r.target_margin + spec.sd_margin * z_margin
    wacc = r.wacc_initial + spec.sd_wacc * z[:, 2]
    s2c_mult = np.exp(spec.sd_s2c_pct * z[:, 3] - 0.5 * spec.sd_s2c_pct ** 2)

    # Keep draws economically sensible
    growth = np.clip(growth, -0.10, 0.60)
    margin = np.clip(margin, -0.50, 0.80)
    wacc = np.clip(wacc, r.riskfree_rate + 0.005, None)

    out = project(
        inp,
        revenue_growth_y2_5=growth,
        target_margin=margin,
        wacc_initial=wacc,
        sales_to_capital=r.sales_to_capital * s2c_mult,
        sales_to_capital_y6_10=r.sales_to_capital_y6_10 * s2c_mult,
    )
    samples = pd.DataFrame(dict(growth_y2_5=growth, target_margin=margin,
                                wacc_initial=wacc,
                                sales_to_capital=r.sales_to_capital * s2c_mult,
                                value_per_share=out["value_per_share"]))
    return MCResult(values=out["value_per_share"], samples=samples, price=inp.price)


# --------------------------------------------------------------------------- #
# WACC x terminal growth grid
# --------------------------------------------------------------------------- #
def sensitivity_grid(inp: DCFInputs,
                     wacc_shifts=(-0.01, -0.005, 0.0, 0.005, 0.01),
                     growth_shifts=(-0.01, -0.005, 0.0, 0.005, 0.01)) -> pd.DataFrame:
    """
    Value per share for each (WACC shift, terminal growth) pair. The WACC
    shift moves the whole cost-of-capital path (initial and terminal), which
    is how bankers read a "discount rate" sensitivity.
    """
    r = inp.resolved()
    W, G = np.meshgrid(np.array(wacc_shifts), np.array(growth_shifts), indexing="ij")
    w0 = r.wacc_initial + W.ravel()
    wT = r.wacc_terminal + W.ravel()
    g = r.terminal_growth + G.ravel()
    valid = wT > g + 1e-6
    vals = np.full(W.size, np.nan)
    if valid.any():
        res = project(inp, wacc_initial=w0[valid], wacc_terminal=wT[valid],
                      terminal_growth=g[valid],
                      terminal_roc=(np.full(valid.sum(), r.terminal_roc)
                                    if inp.terminal_roc is not None else wT[valid]))
        vals[valid] = res["value_per_share"]
    df = pd.DataFrame(vals.reshape(W.shape),
                      index=[f"{r.wacc_initial + s:.2%}" for s in wacc_shifts],
                      columns=[f"{r.terminal_growth + s:.2%}" for s in growth_shifts])
    df.index.name = "Initial WACC"
    df.columns.name = "Terminal growth"
    return df


# --------------------------------------------------------------------------- #
# Tornado
# --------------------------------------------------------------------------- #
def tornado(inp: DCFInputs) -> pd.DataFrame:
    """Value per share when each key input is moved to a low / high case."""
    r = inp.resolved()
    cases = [
        ("Revenue growth yrs 2-5", "revenue_growth_y2_5", -0.03, 0.03, "pp"),
        ("Target operating margin", "target_margin", -0.02, 0.02, "pp"),
        ("Initial WACC", "wacc_initial", -0.01, 0.01, "pp"),
        ("Sales-to-capital", "sales_to_capital", -0.25, 0.25, "pct"),
        ("Terminal growth", "terminal_growth", -0.01, 0.0, "pp"),
        ("Revenue growth yr 1", "revenue_growth_y1", -0.03, 0.03, "pp"),
    ]
    base = project(inp)["value_per_share"][0]
    rows = []
    for label, key, dlo, dhi, kind in cases:
        b = getattr(r, key)
        lo, hi = (b + dlo, b + dhi) if kind == "pp" else (b * (1 + dlo), b * (1 + dhi))
        extra = {}
        if key == "sales_to_capital":
            f = np.array([lo, hi]) / b
            extra["sales_to_capital_y6_10"] = r.sales_to_capital_y6_10 * f
        if key == "terminal_growth" and inp.terminal_roc is None:
            extra["terminal_roc"] = np.full(2, r.terminal_roc)
        v = project(inp, **{key: np.array([lo, hi])}, **extra)["value_per_share"]
        fmt = (lambda x: f"{x:.1%}") if kind == "pp" else (lambda x: f"{x:.2f}")
        rows.append(dict(input=label, low_input=fmt(lo), high_input=fmt(hi),
                         value_low=v[0], value_high=v[1], base=base,
                         swing=abs(v[1] - v[0])))
    return pd.DataFrame(rows).sort_values("swing", ascending=True).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def recommendation(value: float, price: float, prob_under: float,
                   band: float = 0.15) -> tuple[str, str]:
    """
    BUY if base value is >15% above price AND most simulations agree;
    SELL if >15% below AND most simulations agree; otherwise HOLD.
    """
    up = value / price - 1
    if up > band and prob_under >= 0.6:
        return "BUY", (f"Intrinsic value is {up:.0%} above the market price and "
                       f"{prob_under:.0%} of simulations value the shares above it.")
    if up < -band and prob_under <= 0.4:
        return "SELL", (f"Intrinsic value is {-up:.0%} below the market price and "
                        f"only {prob_under:.0%} of simulations value the shares above it.")
    return "HOLD", (f"Intrinsic value is within the uncertainty band of the price "
                    f"({up:+.0%} upside; {prob_under:.0%} of simulations above price).")
