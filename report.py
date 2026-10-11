"""
Charts (matplotlib) and the PDF equity research report (reportlab).
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from .analysis import MCResult, recommendation
from .model import DCFResult, sanity_checks

# --- Palette (validated reference palette; see README) ---
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
DARK_BLUE = "#184f95"

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "Inter", "DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def money(x, dp=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = f"{abs(x):,.{dp}f}"
    return f"({s})" if x < 0 else s


def pct(x, dp=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x * 100:.{dp}f}%"


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _img(buf, width_mm) -> Image:
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(buf).getSize()
    buf.seek(0)
    width = width_mm * mm
    return Image(buf, width=width, height=width * h / w)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_bridge(res: DCFResult):
    """Waterfall from operating value to equity value per share."""
    items = res.bridge()
    labels, starts, heights, kinds = [], [], [], []
    run = 0.0
    for i, (k, v) in enumerate(items):
        labels.append(k.replace("Less: ", "– ").replace("Plus: ", "+ "))
        starts.append(run if v >= 0 else run + v)
        heights.append(abs(v))
        kinds.append("pos" if v >= 0 else "neg")
        run += v
        if i == 1:  # subtotal after the two PV pieces
            labels.append("Operating assets")
            starts.append(0)
            heights.append(run)
            kinds.append("total")
    labels.append("Equity value")
    starts.append(0)
    heights.append(run)
    kinds.append("total")

    col = {"pos": BLUE, "neg": RED, "total": INK2}
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, heights, bottom=starts, color=[col[k] for k in kinds], width=0.62,
           edgecolor="white", linewidth=1.2)
    top = max(s + h for s, h in zip(starts, heights))
    for xi, s, h, k in zip(x, starts, heights, kinds):
        sign = "–" if k == "neg" else ""
        ax.text(xi, s + h + top * 0.015, f"{sign}{h:,.0f}", ha="center", va="bottom",
                fontsize=7, color=INK)
    import textwrap
    ax.set_xticks(x, ["\n".join(textwrap.wrap(l, 13)) for l in labels], fontsize=6.5)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, top * 1.12)
    ax.set_title(f"Value bridge ({res.inputs.currency} {res.inputs.units})")
    return _png(fig)


def chart_operations(res: DCFResult):
    t = res.table
    yrs = ["Base"] + [str(i) for i in range(1, 11)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.3))
    rev = t.loc["Revenue", yrs].astype(float)
    a1.bar(range(len(yrs)), rev, color=BLUE, width=0.62)
    a1.set_xticks(range(len(yrs)), ["B"] + yrs[1:])
    a1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.set_title("Revenue")
    m = t.loc["Operating margin", yrs].astype(float)
    a2.plot(range(len(yrs)), m, color=BLUE, linewidth=2, marker="o", markersize=4)
    a2.set_xticks(range(len(yrs)), ["B"] + yrs[1:])
    a2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    lo, hi = m.min(), m.max()
    pad = max((hi - lo) * 0.3, 0.01)
    a2.set_ylim(lo - pad, hi + pad)
    a2.set_title("Operating margin")
    a2.annotate(f"{m.iloc[-1]:.1%}", (len(yrs) - 1, m.iloc[-1]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7, color=INK)
    fig.tight_layout(w_pad=3)
    return _png(fig)


def chart_fcff(res: DCFResult):
    t = res.table
    yrs = [str(i) for i in range(1, 11)]
    f = t.loc["FCFF", yrs].astype(float).to_numpy()
    pv = t.loc["PV(FCFF)", yrs].astype(float).to_numpy()
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.bar(x - 0.18, f, width=0.34, color=BLUE, label="FCFF")
    ax.bar(x + 0.18, pv, width=0.34, color=ORANGE, label="Present value of FCFF")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(x, yrs)
    ax.set_xlabel("Year")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left", fontsize=7, ncols=2)
    ax.set_title("Free cash flow to the firm")
    return _png(fig)


def chart_mc(mc: MCResult, base_value: float, currency: str):
    v = mc.values
    lo, hi = np.percentile(v, [0.5, 99.5])
    lo, hi = min(lo, mc.price, base_value), max(hi, mc.price, base_value)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    counts, _, _ = ax.hist(v, bins=60, range=(lo, hi), color=BLUE, edgecolor="white",
                           linewidth=0.4)
    ytop = counts.max() * 1.32
    ax.set_ylim(0, ytop)
    p5, p95 = mc.pct(5), mc.pct(95)
    ax.axvspan(p5, p95, color=NEUTRAL, zorder=0)
    # Labels sit in the headroom above the bars, pushed apart if they'd collide
    left_first = mc.price <= base_value
    for x, txt, c, ls, side in (
            (mc.price, f"Price {mc.price:,.2f}", INK, "--", "right" if left_first else "left"),
            (base_value, f"Base case {base_value:,.2f}", ORANGE, "-",
             "left" if left_first else "right")):
        ax.axvline(x, color=c, linestyle=ls, linewidth=1.4)
        ax.text(x, ytop * 0.97, f" {txt} " , color=INK, fontsize=7, va="top",
                ha="left" if side == "left" else "right")
    ax.set_xlabel(f"Value per share ({currency}); shaded band = 5th to 95th percentile; "
                  "outer 1% of runs not drawn")
    ax.set_ylabel("Simulations")
    ax.set_title(f"Monte Carlo: {len(v):,} valuations")
    return _png(fig)


def chart_tornado(tor: pd.DataFrame, currency: str):
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    base = tor.base.iloc[0]
    y = np.arange(len(tor))
    for yi, r in zip(y, tor.itertuples()):
        for val, lab in ((r.value_low, r.low_input), (r.value_high, r.high_input)):
            c = BLUE if val >= base else RED
            ax.barh(yi, val - base, left=base, color=c, height=0.55)
            ha = "left" if val >= base else "right"
            ax.text(val, yi, f" {lab} → {val:,.2f} ", va="center", ha=ha, fontsize=6.5,
                    color=INK)
    ax.axvline(base, color=INK2, linewidth=1)
    ax.set_yticks(y, tor.input, fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=GRID)
    span = (tor[["value_low", "value_high"]].to_numpy() - base)
    m = np.abs(span).max() * 1.55
    ax.set_xlim(base - m, base + m)
    ax.set_xlabel(f"Value per share ({currency}); blue = raises value, red = lowers it")
    ax.set_title("What moves the value: one input at a time")
    return _png(fig)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.5,
                          leading=12, textColor=colors.HexColor(INK))
    return dict(
        title=ParagraphStyle("t", parent=base, fontName="Helvetica-Bold", fontSize=18,
                             leading=22),
        sub=ParagraphStyle("s", parent=base, textColor=colors.HexColor(INK2), fontSize=9),
        h=ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=11,
                         leading=15, spaceBefore=8, spaceAfter=4),
        body=base,
        small=ParagraphStyle("sm", parent=base, fontSize=7, leading=9.5,
                             textColor=colors.HexColor(INK2)),
        cell=ParagraphStyle("c", parent=base, fontSize=7.5, leading=9.5, alignment=TA_LEFT),
        tile_label=ParagraphStyle("tl", parent=base, fontSize=7,
                                  textColor=colors.HexColor(INK2)),
        tile_value=ParagraphStyle("tv", parent=base, fontName="Helvetica-Bold",
                                  fontSize=14, leading=17),
    )


def _grid_style(header_rows=1, font=7.5):
    return TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", font),
        ("FONT", (0, 0), (-1, header_rows - 1), "Helvetica-Bold", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor(NEUTRAL)),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, colors.HexColor(AXIS)),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ])


def _mix(c1, c2, t):
    a = np.array(colors.HexColor(c1).rgb())
    b = np.array(colors.HexColor(c2).rgb())
    return colors.Color(*((1 - t) * a + t * b))


def _story(r, res: DCFResult) -> str:
    cur = r.currency
    g_txt = (f"revenue growing {pct(r.revenue_growth_y1)} next year and "
             f"{pct(r.revenue_growth_y2_5)} a year in years 2-5, fading to "
             f"{pct(r.terminal_growth)} in perpetuity")
    m_now = r.ebit / r.revenue
    m_txt = (f"operating margin moving from {pct(m_now)} today to "
             f"{pct(r.target_margin)} by year {int(r.convergence_year)}")
    s_txt = (f"growth funded by {r.sales_to_capital:.2f} of revenue per {cur}1 "
             f"of capital invested")
    w_txt = (f"a cost of capital of {pct(r.wacc_initial)} falling to "
             f"{pct(r.wacc_terminal)} as the firm matures")
    return (f"The valuation assumes {g_txt}; {m_txt}; {s_txt}; and {w_txt}. "
            f"The terminal value accounts for {res.terminal_share:.0%} of operating "
            f"asset value.")


def build_report(res: DCFResult, mc: MCResult, grid: pd.DataFrame, tor: pd.DataFrame,
                 out_path: str | Path, provenance: dict | None = None,
                 wacc=None, data_notes: list[str] | None = None,
                 as_of: str | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    S = _styles()
    r = res.inputs.resolved()
    cur, units = r.currency, r.units
    summ = mc.summary()
    rec, why = recommendation(res.value_per_share, r.price, summ["prob_undervalued"])
    as_of = as_of or dt.date.today().strftime("%d %b %Y")
    W = 180  # usable width in mm
    story = []

    # ---------------- Page 1: the call ----------------
    name = f"{r.company} ({r.ticker})" if r.ticker else r.company
    story += [Paragraph(name, S["title"]),
              Paragraph(f"Equity research · Intrinsic valuation (FCFF DCF) · {as_of}",
                        S["sub"]), Spacer(1, 8)]

    def tile(label, value, note=""):
        cells = [Paragraph(label, S["tile_label"]), Paragraph(value, S["tile_value"])]
        if note:
            cells.append(Paragraph(note, S["tile_label"]))
        return cells

    up = res.upside
    tiles = [
        tile("Recommendation", rec, "15% band + simulation agreement"),
        tile("Intrinsic value / share", f"{cur} {res.value_per_share:,.2f}", "Base case"),
        tile("Share price", f"{cur} {r.price:,.2f}", "At time of run"),
        tile("Upside / (downside)", f"{up:+.1%}", "Value vs price"),
    ]
    tiles2 = [
        tile("Monte Carlo 5th-95th", f"{summ['p5']:,.0f} – {summ['p95']:,.0f}",
             f"{len(mc.values):,} runs"),
        tile("P(value > price)", f"{summ['prob_undervalued']:.0%}", "Share of simulations"),
        tile("Cost of capital", pct(r.wacc_initial), f"Terminal {pct(r.wacc_terminal)}"),
        tile("Terminal value share", f"{res.terminal_share:.0%}", "Of operating value"),
    ]
    tt = Table([tiles, tiles2], colWidths=[W / 4 * mm] * 4)
    tt.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(AXIS)),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(NEUTRAL)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [tt, Spacer(1, 8)]

    story += [Paragraph("Investment view", S["h"]),
              Paragraph(f"<b>{rec}.</b> {why}", S["body"]), Spacer(1, 3),
              Paragraph(_story(r, res), S["body"])]

    # Key drivers table
    prov = provenance or {}
    drivers = [
        ("Revenue growth, year 1", pct(r.revenue_growth_y1), "revenue_growth_y1"),
        ("Revenue growth, years 2-5", pct(r.revenue_growth_y2_5), "revenue_growth_y2_5"),
        ("Operating margin, year 1", pct(r.op_margin_y1), "op_margin_y1"),
        ("Target operating margin", f"{pct(r.target_margin)} by yr {int(r.convergence_year)}",
         "target_margin"),
        ("Sales-to-capital (yrs 1-5 / 6-10)",
         f"{r.sales_to_capital:.2f} / {r.sales_to_capital_y6_10:.2f}", "sales_to_capital"),
        ("Tax rate (effective → marginal)",
         f"{pct(r.effective_tax_rate)} → {pct(r.marginal_tax_rate)}", "effective_tax_rate"),
        ("Cost of capital (initial → terminal)",
         f"{pct(r.wacc_initial)} → {pct(r.wacc_terminal)}", "wacc_initial"),
        ("Risk-free rate / terminal growth",
         f"{pct(r.riskfree_rate, 2)} / {pct(r.terminal_growth, 2)}", "riskfree_rate"),
        ("Terminal return on capital", pct(r.terminal_roc), "terminal_roc"),
    ]
    rows = [["Driver", "Assumption", "Source"]] + [
        [d, v, Paragraph(prov.get(k, "Input file" if provenance is None else "Default"),
                         S["cell"])] for d, v, k in drivers]
    t = Table(rows, colWidths=[62 * mm, 38 * mm, 80 * mm])
    st = _grid_style()
    st.add("ALIGN", (2, 0), (2, -1), "LEFT")
    t.setStyle(st)
    story += [Paragraph("Key drivers", S["h"]), t]

    story += [Spacer(1, 6), _img(chart_bridge(res), W),
              Paragraph(f"Value per share = {money(res.equity_common)} ÷ "
                        f"{r.shares_outstanding:,.1f}m shares = "
                        f"<b>{cur} {res.value_per_share:,.2f}</b>", S["small"])]
    story.append(PageBreak())

    # ---------------- Page 2: the model ----------------
    story.append(Paragraph(f"Ten-year projection ({cur} {units})", S["h"]))
    tbl = res.table
    cols = list(tbl.columns)
    fmt_rows = [
        ("Revenue growth", pct), ("Revenue", money), ("Operating margin", pct),
        ("EBIT", money), ("Tax rate", pct), ("EBIT(1-t)", money),
        ("Reinvestment", money), ("FCFF", money), ("Cost of capital", pct),
        ("Discount factor", lambda x: "" if np.isnan(x) else f"{x:.3f}"),
        ("PV(FCFF)", money),
    ]
    if "ROIC" in tbl.index:
        fmt_rows.append(("ROIC", pct))
    data = [[""] + ["Base"] + cols[1:-1] + ["Term."]]
    bold_rows = []
    for name_, f in fmt_rows:
        data.append([name_] + [f(float(tbl.loc[name_, c])) for c in cols])
        if name_ in ("FCFF",):
            bold_rows.append(len(data) - 1)
    cw = [27 * mm] + [12.75 * mm] * 12
    pt = Table(data, colWidths=cw)
    st = _grid_style(font=6.8)
    for br in bold_rows:
        st.add("FONT", (0, br), (-1, br), "Helvetica-Bold", 6.8)
    pt.setStyle(st)
    story += [pt, Spacer(1, 3)]

    bridge_rows = [
        ["Terminal value (year 11 FCFF ÷ (WACC − g))", money(res.terminal_value)],
        ["PV of terminal value", money(res.pv_terminal)],
        ["PV of FCFF, years 1-10", money(res.sum_pv_fcff)],
        ["Value of operating assets", money(res.value_operating)],
        ["Equity value attributable to common shares", money(res.equity_common)],
        ["Value per share", f"{res.value_per_share:,.2f}"],
    ]
    bt = Table([["Summary", f"{cur} {units}"]] + bridge_rows, colWidths=[110 * mm, 40 * mm])
    st = _grid_style()
    st.add("FONT", (0, len(bridge_rows)), (-1, len(bridge_rows)), "Helvetica-Bold", 7.5)
    bt.setStyle(st)
    story += [bt, Spacer(1, 6), _img(chart_operations(res), W), Spacer(1, 4),
              _img(chart_fcff(res), W)]
    story.append(PageBreak())

    # ---------------- Page 3: risk ----------------
    story += [Paragraph("Risk: how wrong could this be?", S["h"]),
              Paragraph(
                  "Each simulation redraws revenue growth (years 2-5), the target margin "
                  "(correlated with growth), the initial cost of capital and the "
                  "sales-to-capital ratio, then reruns the full ten-year model. The spread "
                  "of outcomes is the honest answer to 'what is it worth?'.", S["body"]),
              _img(chart_mc(mc, res.value_per_share, cur), W)]
    srows = [["Mean", "Median", "Std dev", "5th pct", "25th pct", "75th pct", "95th pct",
              "P(value > price)"],
             [f"{summ['mean']:,.2f}", f"{summ['median']:,.2f}", f"{summ['std']:,.2f}",
              f"{summ['p5']:,.2f}", f"{summ['p25']:,.2f}", f"{summ['p75']:,.2f}",
              f"{summ['p95']:,.2f}", f"{summ['prob_undervalued']:.0%}"]]
    s_t = Table(srows, colWidths=[W / 8 * mm] * 8)
    st = _grid_style()
    st.add("ALIGN", (0, 0), (-1, -1), "CENTER")
    s_t.setStyle(st)
    story += [s_t, Spacer(1, 6)]

    # Sensitivity grid, coloured against the share price
    story.append(Paragraph(
        f"Sensitivity: value per share ({cur}) by cost of capital and terminal growth", S["h"]))
    g = grid
    gdata = [[f"WACC ↓  /  g →"] + list(g.columns)]
    for idx, row in g.iterrows():
        gdata.append([idx] + ["n/a" if np.isnan(v) else f"{v:,.2f}" for v in row])
    gt = Table(gdata, colWidths=[36 * mm] + [28 * mm] * len(g.columns))
    st = _grid_style(font=8)
    st.add("ALIGN", (0, 0), (-1, -1), "CENTER")
    st.add("FONT", (0, 1), (0, -1), "Helvetica-Bold", 8)
    st.add("BACKGROUND", (0, 1), (0, -1), colors.HexColor(NEUTRAL))
    vals = g.to_numpy()
    span = np.nanmax(np.abs(vals / r.price - 1))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            d = v / r.price - 1
            tint = min(abs(d) / max(span, 1e-9), 1) * 0.85
            c = _mix(NEUTRAL, DARK_BLUE if d >= 0 else RED, tint)
            st.add("BACKGROUND", (j + 1, i + 1), (j + 1, i + 1), c)
            if tint > 0.55:
                st.add("TEXTCOLOR", (j + 1, i + 1), (j + 1, i + 1), colors.white)
    mid_i, mid_j = vals.shape[0] // 2 + 1, vals.shape[1] // 2 + 1
    st.add("BOX", (mid_j, mid_i), (mid_j, mid_i), 1.4, colors.HexColor(INK))
    gt.setStyle(st)
    neutral_growth = abs(r.terminal_roc - r.wacc_terminal) < 1e-9
    note = (" Terminal growth barely moves the value because the terminal return on "
            "capital equals the cost of capital: growth that earns exactly its cost of "
            "capital creates no value." if neutral_growth else "")
    story += [gt, Paragraph(
        f"Blue cells are above the current price ({cur} {r.price:,.2f}), red below; "
        "the boxed cell is the base case. The WACC shift moves the whole cost-of-capital "
        "path. Damodaran caps terminal growth at the risk-free rate, so columns to the "
        "right of the base are deliberately aggressive." + note, S["small"]),
        Spacer(1, 6)]
    story.append(_img(chart_tornado(tor, cur), W))
    story.append(PageBreak())

    # ---------------- Page 4: checks, method, data ----------------
    story.append(Paragraph("Sanity checks", S["h"]))
    ck = [["", "Check"]] + [[("OK" if s == "ok" else "FLAG"), Paragraph(m, S["cell"])]
                            for s, m in sanity_checks(res)]
    ct = Table(ck, colWidths=[14 * mm, 166 * mm])
    st = _grid_style()
    st.add("ALIGN", (0, 0), (-1, -1), "LEFT")
    for i, (s, _) in enumerate(sanity_checks(res), start=1):
        if s != "ok":
            st.add("FONT", (0, i), (0, i), "Helvetica-Bold", 7.5)
    ct.setStyle(st)
    story.append(ct)

    if wacc is not None:
        story.append(Paragraph("Cost of capital build", S["h"]))
        wrows = [
            ["Risk-free rate", pct(wacc.riskfree_rate, 2)],
            ["Beta", f"{wacc.beta:.2f}"],
            ["Equity risk premium", pct(wacc.erp, 2)],
            ["Cost of equity (CAPM)", pct(wacc.cost_of_equity, 2)],
            ["Interest coverage → synthetic rating",
             f"{wacc.interest_coverage:,.1f}x → {wacc.rating}"
             if np.isfinite(wacc.interest_coverage) else f"no interest → {wacc.rating}"],
            ["Pre-tax cost of debt (rf + spread)", pct(wacc.pretax_cost_of_debt, 2)],
            ["After-tax cost of debt", pct(wacc.after_tax_cost_of_debt, 2)],
            ["Weights: equity / debt (market values)",
             f"{wacc.weight_equity:.0%} / {wacc.weight_debt:.0%}"],
            ["WACC", pct(wacc.wacc, 2)],
        ]
        wt = Table([["Component", "Value"]] + wrows, colWidths=[110 * mm, 40 * mm])
        st = _grid_style()
        st.add("FONT", (0, len(wrows)), (-1, len(wrows)), "Helvetica-Bold", 7.5)
        wt.setStyle(st)
        story.append(wt)

    story += [Paragraph("Methodology", S["h"]), Paragraph(
        "A ten-year free-cash-flow-to-the-firm model following the structure of "
        "Aswath Damodaran's <i>fcffsimpleginzu</i> spreadsheet. Revenue grows at a year-1 "
        "rate, then a years 2-5 rate, then fades linearly to the terminal rate. The "
        "operating margin converges linearly to a target. Taxes move from the effective "
        "to the marginal rate over years 6-10, with operating losses carried forward. "
        "Reinvestment is the change in revenue divided by the sales-to-capital ratio, so "
        "growth always has a cost. Cash flows are discounted at a cost of capital that "
        "drifts toward a mature-firm rate; the terminal value assumes reinvestment of "
        "g ÷ ROC, so stable growth is only valuable if returns exceed the cost of "
        "capital. Operating value is bridged to equity by subtracting debt and minority "
        "interests and adding cash and non-operating assets.", S["body"])]

    notes = list(data_notes or [])
    if notes:
        story += [Paragraph("Data notes", S["h"])] + [
            Paragraph("• " + n, S["small"]) for n in notes]
    story += [Spacer(1, 8), Paragraph(
        "Prepared with an open-source Python DCF engine for educational purposes. Not "
        "investment advice. Inputs are estimates and the output is only as good as the "
        "story behind them.", S["small"])]

    def on_page(canv, doc):
        canv.saveState()
        canv.setFont("Helvetica", 7)
        canv.setFillColor(colors.HexColor(MUTED))
        canv.drawString(15 * mm, 10 * mm, f"{name} · DCF valuation · {as_of}")
        canv.drawRightString(195 * mm, 10 * mm, f"Page {doc.page}")
        canv.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=15 * mm,
                            rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"{name} DCF valuation", author="Python DCF engine")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out_path
