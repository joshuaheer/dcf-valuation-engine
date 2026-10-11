"""
Validate the Python engine against Damodaran's fcffsimpleginzu.xlsx.

Workflow:
  1. Open fcffsimpleginzu.xlsx, fill the "Input sheet" for a company.
  2. Copy those inputs, and the outputs from the "Valuation output" sheet,
     into a JSON file shaped like validation/template.json.
  3. Run:  python -m dcf.validate validation/my_company.json

The script runs the same inputs through the engine and prints a line-by-line
comparison (FCFF per year, terminal value, PV, value per share). If a single
row is off, that tells you exactly which mechanic differs.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .data import load_json
from .model import DCFInputs, run_dcf

ROW_MAP = {  # JSON key in "expected" -> row name in result table
    "revenue": "Revenue",
    "ebit": "EBIT",
    "ebit_after_tax": "EBIT(1-t)",
    "reinvestment": "Reinvestment",
    "fcff": "FCFF",
    "cost_of_capital": "Cost of capital",
    "pv_fcff": "PV(FCFF)",
}
SCALARS = {
    "terminal_value": "terminal_value",
    "pv_terminal": "pv_terminal",
    "sum_pv_fcff": "sum_pv_fcff",
    "value_operating": "value_operating",
    "equity_value": "equity_value",
    "value_per_share": "value_per_share",
}


def compare(spec: dict, rel_tol: float = 1e-4) -> tuple[bool, list[str]]:
    inputs = {k: v for k, v in spec["inputs"].items() if not k.startswith("_")}
    res = run_dcf(DCFInputs.from_dict(inputs))
    exp = spec["expected"]
    lines, ok_all = [], True

    def check(label, mine, theirs):
        nonlocal ok_all
        diff = mine - theirs
        rel = abs(diff) / max(abs(theirs), 1e-9)
        ok = rel <= rel_tol
        ok_all &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'}  {label:<28} python={mine:>14,.4f}  "
                     f"excel={theirs:>14,.4f}  diff={diff:>+12,.4f} ({rel:.4%})")

    for key, row in ROW_MAP.items():
        if key in exp:
            mine = res.table.loc[row, "1":"10"].to_numpy(dtype=float)
            theirs = np.asarray(exp[key], dtype=float)
            for i, (a, b) in enumerate(zip(mine, theirs), start=1):
                check(f"{row} yr {i}", a, b)
    for key, attr in SCALARS.items():
        if key in exp:
            check(key, getattr(res, attr), float(exp[key]))
    return ok_all, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="JSON with 'inputs' and 'expected'")
    ap.add_argument("--tol", type=float, default=1e-4, help="relative tolerance")
    a = ap.parse_args(argv)
    try:
        ok, lines = compare(load_json(a.spec), a.tol)
    except (ZeroDivisionError, ValueError) as e:
        print(f"Could not run the inputs ({e}). Have you filled in every input "
              "from the spreadsheet? Revenue, shares and price must be non-zero.")
        return 2
    print("\n".join(lines))
    print("\nRESULT:", "MATCH" if ok else "MISMATCH (see FAIL rows)")
    if not ok:
        print("Tip: if only Reinvestment/FCFF differ, try flipping reinvestment_lag (0 <-> 1).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
