"""
Run a full valuation and write the PDF report.

  python run.py AAPL                                  # live data via yfinance
  python run.py AAPL --config configs/example_override.json   # your story on top
  python run.py --input examples/sample_company.json  # offline, all inputs given
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dcf import MCSpec, monte_carlo, run_dcf, sensitivity_grid, tornado
from dcf.analysis import recommendation
from dcf.data import build_inputs, fetch_fundamentals, inputs_from_file, load_json
from dcf.report import build_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker", nargs="?", help="e.g. AAPL, MSFT, ULVR.L")
    ap.add_argument("--input", help="JSON with every model input (no internet)")
    ap.add_argument("--config", help="JSON of overrides applied on top of live data")
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)

    prov, wacc, notes, as_of = None, None, [], None
    if a.input:
        inp = inputs_from_file(a.input)
        notes.append(f"All inputs read from {Path(a.input).name}.")
    elif a.ticker:
        f = fetch_fundamentals(a.ticker)
        overrides = load_json(a.config) if a.config else {}
        inp, prov, wacc = build_inputs(f, overrides)
        notes = f.notes + ["Financial data: Yahoo Finance via yfinance."]
        as_of = f.as_of
    else:
        ap.error("give a ticker or --input")

    res = run_dcf(inp)
    mc = monte_carlo(inp, MCSpec(sims=a.sims, seed=a.seed))
    grid = sensitivity_grid(inp)
    tor = tornado(inp)
    rec, why = recommendation(res.value_per_share, inp.price, mc.prob_undervalued)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = (inp.ticker or inp.company).replace(" ", "_").replace(".", "_")
    pdf = build_report(res, mc, grid, tor, out / f"{stem}_DCF_report.pdf",
                       provenance=prov, wacc=wacc, data_notes=notes, as_of=as_of)
    res.table.to_csv(out / f"{stem}_projection.csv")
    grid.to_csv(out / f"{stem}_sensitivity.csv")

    s = mc.summary()
    print(f"\n{inp.company} ({inp.ticker})")
    print(f"  Value per share : {inp.currency} {res.value_per_share:,.2f}")
    print(f"  Share price     : {inp.currency} {inp.price:,.2f}  ({res.upside:+.1%})")
    print(f"  Monte Carlo     : 5th {s['p5']:,.2f} | median {s['median']:,.2f} | "
          f"95th {s['p95']:,.2f} | P(value>price) {s['prob_undervalued']:.0%}")
    print(f"  Call            : {rec} - {why}")
    print(f"  Report          : {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
