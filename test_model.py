"""
Tests that pin the engine to closed-form answers. If these pass, the
mechanics (growth fade, margin convergence, NOLs, discounting, terminal
value, equity bridge) are doing what they claim.
"""

import numpy as np
import pandas as pd
import pytest

from dcf import DCFInputs, project, run_dcf, monte_carlo, sensitivity_grid, tornado, MCSpec
from dcf.analysis import recommendation
from dcf.data import Fundamentals, build_inputs, fetch_fundamentals
from dcf.validate import compare
from dcf.wacc import synthetic_rating


def base(**kw):
    d = dict(revenue=1000.0, ebit=100.0, debt=0.0, cash=0.0, shares_outstanding=10.0,
             price=50.0, revenue_growth_y1=0.0, revenue_growth_y2_5=0.0,
             op_margin_y1=0.10, target_margin=0.10, sales_to_capital=2.0,
             effective_tax_rate=0.25, marginal_tax_rate=0.25, riskfree_rate=0.0,
             wacc_initial=0.08, wacc_terminal=0.08, terminal_growth=0.0)
    d.update(kw)
    return DCFInputs(**d)


def test_no_growth_is_a_perpetuity():
    # FCFF = EBIT(1-t) = 1000*10%*75% = 75 forever -> value = 75/8% = 937.5
    res = run_dcf(base())
    assert res.pv_operating == pytest.approx(75 / 0.08, rel=1e-12)


@pytest.mark.parametrize("g", [0.02, 0.04])
def test_constant_growth_matches_gordon(g):
    # With growth g in every year and terminal ROC consistent with the
    # sales-to-capital ratio, the whole thing collapses to FCFF1/(WACC-g).
    s2c, m, t = 2.0, 0.10, 0.25
    roc = s2c * m * (1 - t)
    inp = base(revenue_growth_y1=g, revenue_growth_y2_5=g, terminal_growth=g,
               riskfree_rate=g, terminal_roc=roc)
    res = run_dcf(inp)
    rev1 = 1000 * (1 + g)
    fcff1 = rev1 * m * (1 - t) - rev1 * g / s2c
    assert res.pv_operating == pytest.approx(fcff1 / (0.08 - g), rel=1e-10)


def test_growth_fades_linearly_to_terminal():
    res = run_dcf(base(revenue_growth_y1=0.20, revenue_growth_y2_5=0.15,
                       terminal_growth=0.03, riskfree_rate=0.03))
    g = res.table.loc["Revenue growth", "1":"10"].to_numpy(float)
    assert g[0] == pytest.approx(0.20)
    assert np.allclose(g[1:5], 0.15)
    assert np.allclose(np.diff(g[4:]), -0.024)
    assert g[9] == pytest.approx(0.03)


def test_margin_converges_by_convergence_year():
    res = run_dcf(base(op_margin_y1=0.05, target_margin=0.20, convergence_year=4))
    m = res.table.loc["Operating margin", "1":"10"].to_numpy(float)
    assert m[0] == pytest.approx(0.05)
    assert np.allclose(m[3:], 0.20)
    assert np.all(np.diff(m[:4]) > 0)


def test_tax_moves_to_marginal_and_wacc_to_terminal():
    res = run_dcf(base(effective_tax_rate=0.15, marginal_tax_rate=0.25,
                       wacc_initial=0.10, wacc_terminal=0.07))
    t = res.table
    assert np.allclose(t.loc["Tax rate", "1":"5"].astype(float), 0.15)
    assert t.loc["Tax rate", "10"] == pytest.approx(0.25)
    assert np.allclose(t.loc["Cost of capital", "1":"5"].astype(float), 0.10)
    assert t.loc["Cost of capital", "10"] == pytest.approx(0.07)


def test_nol_shields_tax():
    # NOL of 250 covers EBIT of 100 for 2.5 years
    res = run_dcf(base(nol_carryforward=250))
    eat = res.table.loc["EBIT(1-t)", "1":"4"].to_numpy(float)
    assert eat[0] == pytest.approx(100) and eat[1] == pytest.approx(100)
    assert eat[2] == pytest.approx(100 - 50 * 0.25)   # 50 of taxable income left
    assert eat[3] == pytest.approx(75)
    # Losses add to the NOL and are not taxed
    res = run_dcf(base(op_margin_y1=-0.05))
    assert res.table.loc["EBIT(1-t)", "1"] == pytest.approx(-50)
    assert res.table.loc["NOL", "1"] == pytest.approx(50)


def test_reinvestment_lag_options():
    inp = base(revenue_growth_y1=0.10, revenue_growth_y2_5=0.10)
    lag1 = run_dcf(inp).table.loc["Reinvestment", "1"]
    inp.reinvestment_lag = 0
    lag0 = run_dcf(inp).table.loc["Reinvestment", "1"]
    assert lag0 == pytest.approx(100 / 2.0)          # (1100-1000)/2
    assert lag1 == pytest.approx(110 / 2.0)          # (1210-1100)/2


def test_equity_bridge():
    res = run_dcf(base(debt=200, cash=50, minority_interests=10,
                       non_operating_assets=20, options_value=5))
    expect = 937.5 - 200 + 50 - 10 + 20 - 5
    assert res.equity_common == pytest.approx(expect)
    assert res.value_per_share == pytest.approx(expect / 10)


def test_failure_probability():
    res = run_dcf(base(prob_failure=0.2, failure_proceeds_pct=0.5))
    assert res.value_operating == pytest.approx(937.5 * (0.8 + 0.2 * 0.5))


def test_vectorised_matches_scalar():
    inp = base(revenue_growth_y1=0.08, revenue_growth_y2_5=0.06, terminal_growth=0.03,
               riskfree_rate=0.03, wacc_initial=0.09, wacc_terminal=0.075)
    waccs = np.array([0.08, 0.09, 0.10])
    vec = project(inp, wacc_initial=waccs)["value_per_share"]
    for w, v in zip(waccs, vec):
        inp2 = base(**{**inp.to_dict(), "wacc_initial": w})
        assert run_dcf(inp2).value_per_share == pytest.approx(v)


def test_sensitivity_centre_is_base_case():
    inp = base(terminal_growth=0.02, riskfree_rate=0.02, wacc_terminal=0.07)
    grid = sensitivity_grid(inp)
    assert grid.iloc[2, 2] == pytest.approx(run_dcf(inp).value_per_share)
    # Value falls as WACC rises
    assert np.all(np.diff(grid.iloc[:, 2].to_numpy()) < 0)


def test_monte_carlo_is_reproducible_and_centred():
    inp = base(revenue_growth_y1=0.05, revenue_growth_y2_5=0.05, terminal_growth=0.02,
               riskfree_rate=0.02)
    a = monte_carlo(inp, MCSpec(sims=5000, seed=1))
    b = monte_carlo(inp, MCSpec(sims=5000, seed=1))
    assert np.array_equal(a.values, b.values)
    base_v = run_dcf(inp).value_per_share
    assert abs(a.summary()["median"] / base_v - 1) < 0.1


def test_tornado_brackets_base():
    t = tornado(base(revenue_growth_y2_5=0.05, terminal_growth=0.02, riskfree_rate=0.02))
    wacc = t[t.input == "Initial WACC"].iloc[0]
    assert wacc.value_low > wacc.base > wacc.value_high


def test_terminal_wacc_must_exceed_growth():
    with pytest.raises(ValueError):
        run_dcf(base(terminal_growth=0.09, wacc_terminal=0.08))


def test_synthetic_rating_bands():
    assert synthetic_rating(100, 5)[0] == "AAA"       # 20x
    assert synthetic_rating(100, 35)[0] == "BBB"      # 2.86x
    assert synthetic_rating(100, 0)[0] == "AAA"
    assert synthetic_rating(-10, 5)[0] == "D"


def test_recommendation_rules():
    assert recommendation(130, 100, 0.8)[0] == "BUY"
    assert recommendation(70, 100, 0.2)[0] == "SELL"
    assert recommendation(110, 100, 0.6)[0] == "HOLD"
    assert recommendation(130, 100, 0.5)[0] == "HOLD"   # value up, but sims disagree


def test_validation_harness_round_trip():
    inp = base(revenue_growth_y1=0.08, revenue_growth_y2_5=0.06, terminal_growth=0.03,
               riskfree_rate=0.03, wacc_terminal=0.075)
    res = run_dcf(inp)
    spec = dict(inputs=inp.to_dict(),
                expected=dict(fcff=res.table.loc["FCFF", "1":"10"].tolist(),
                              value_per_share=res.value_per_share))
    ok, _ = compare(spec)
    assert ok
    spec["expected"]["value_per_share"] *= 1.01
    ok, _ = compare(spec)
    assert not ok


# --------------------------------------------------------------------------- #
# Data layer, with a fake yfinance (no network needed)
# --------------------------------------------------------------------------- #
class FakeTicker:
    def __init__(self, sym):
        self.sym = sym
        q = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
        a = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
        self.quarterly_income_stmt = pd.DataFrame(
            {d: v for d, v in zip(q, [
                [2.6e9, 0.39e9, 0.38e9, 0.08e9, 0.02e9],
                [2.5e9, 0.37e9, 0.36e9, 0.08e9, 0.02e9],
                [2.5e9, 0.37e9, 0.36e9, 0.08e9, 0.02e9],
                [2.4e9, 0.37e9, 0.36e9, 0.07e9, 0.02e9]])},
            index=["Total Revenue", "Operating Income", "Pretax Income",
                   "Tax Provision", "Interest Expense"])
        self.income_stmt = pd.DataFrame(
            {d: v for d, v in zip(a, [[9.6e9, 1.4e9], [8.8e9, 1.2e9],
                                      [8.0e9, 1.0e9], [7.5e9, 0.9e9]])},
            index=["Total Revenue", "Operating Income"])
        self.quarterly_balance_sheet = pd.DataFrame(
            {q[0]: [3.0e9, 1.0e9, 5.0e9, 0.1e9, 500e6]},
            index=["Total Debt", "Cash Cash Equivalents And Short Term Investments",
                   "Stockholders Equity", "Minority Interest", "Ordinary Shares Number"])
        self.balance_sheet = self.quarterly_balance_sheet
        self.info = dict(currentPrice=30.0, sharesOutstanding=500e6, beta=1.1,
                         currency="USD", financialCurrency="USD", longName="Fake Corp")

    def history(self, period="5d"):
        return pd.DataFrame({"Close": [4.10, 4.15]})


class FakeYF:
    Ticker = FakeTicker


def test_fetch_and_build_inputs_from_fake_yfinance():
    f = fetch_fundamentals("fake", yf=FakeYF)
    assert f.revenue_ttm == pytest.approx(10_000)          # 2.6+2.5+2.5+2.4 bn
    assert f.ebit_ttm == pytest.approx(1_500)
    assert f.shares == pytest.approx(500)
    assert f.riskfree_rate == pytest.approx(0.0415)
    inp, prov, w = build_inputs(f, {"target_margin": 0.17})
    assert inp.target_margin == 0.17
    assert prov["target_margin"].startswith("Config")
    assert inp.revenue_growth_y1 == pytest.approx(9.6 / 8.8 - 1)
    assert inp.revenue_growth_y2_5 == pytest.approx((9.6 / 7.5) ** (1 / 3) - 1)
    assert inp.sales_to_capital == pytest.approx(10_000 / (5_000 + 3_000 - 1_000))
    assert w.rating == "AAA"                                 # 1500/80 = 18.75x
    assert run_dcf(inp).value_per_share > 0
