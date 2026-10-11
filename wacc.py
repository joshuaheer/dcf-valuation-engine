"""
Cost of capital: CAPM cost of equity + synthetic-rating cost of debt.

The synthetic rating maps interest coverage (EBIT / interest expense) to a bond
rating and default spread, following Damodaran's ratings.xls table for large
non-financial firms. Coverage bands are stable year to year; the SPREADS move
with credit markets, so refresh them each January from his site.
"""

from __future__ import annotations

from dataclasses import dataclass

# (min coverage, rating, default spread). Spreads approximate Damodaran's
# January 2025 large-firm table. Update annually.
RATING_TABLE = [
    (8.50, "AAA", 0.0059),
    (6.50, "AA", 0.0078),
    (5.50, "A+", 0.0108),
    (4.25, "A", 0.0122),
    (3.00, "A-", 0.0133),
    (2.50, "BBB", 0.0171),
    (2.25, "BB+", 0.0231),
    (2.00, "BB", 0.0277),
    (1.75, "B+", 0.0416),
    (1.50, "B", 0.0492),
    (1.25, "B-", 0.0594),
    (0.80, "CCC", 0.0904),
    (0.65, "CC", 0.1148),
    (0.20, "C", 0.1508),
    (-1e9, "D", 0.1900),
]


def synthetic_rating(ebit: float, interest_expense: float) -> tuple[str, float, float]:
    """Return (rating, default spread, interest coverage)."""
    interest = abs(interest_expense or 0.0)
    coverage = float("inf") if interest == 0 else ebit / interest
    for floor, rating, spread in RATING_TABLE:
        if coverage >= floor:
            return rating, spread, coverage
    return RATING_TABLE[-1][1], RATING_TABLE[-1][2], coverage


@dataclass
class WACCBreakdown:
    riskfree_rate: float
    beta: float
    erp: float
    cost_of_equity: float
    rating: str
    interest_coverage: float
    default_spread: float
    pretax_cost_of_debt: float
    after_tax_cost_of_debt: float
    market_cap: float
    debt: float
    weight_equity: float
    weight_debt: float
    wacc: float


def estimate_wacc(*, riskfree_rate: float, beta: float, erp: float,
                  ebit: float, interest_expense: float, market_cap: float,
                  debt: float, marginal_tax_rate: float) -> WACCBreakdown:
    ke = riskfree_rate + beta * erp
    rating, spread, cov = synthetic_rating(ebit, interest_expense)
    kd = riskfree_rate + spread
    kd_at = kd * (1 - marginal_tax_rate)
    total = market_cap + debt
    we, wd = market_cap / total, debt / total
    return WACCBreakdown(
        riskfree_rate=riskfree_rate, beta=beta, erp=erp, cost_of_equity=ke,
        rating=rating, interest_coverage=cov, default_spread=spread,
        pretax_cost_of_debt=kd, after_tax_cost_of_debt=kd_at,
        market_cap=market_cap, debt=debt, weight_equity=we, weight_debt=wd,
        wacc=we * ke + wd * kd_at,
    )
