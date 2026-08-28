#!/usr/bin/env python3
"""
02_index.py — construct the data-centre construction cost index (DCCI).

Honest naming. Without period-by-period quantity data there is no way to form a
true Fisher index, so this is a **fixed-weight Laspeyres-type input cost
index** - a cost model, not a superlative index number. Saying otherwise would
not survive a referee who knows index theory. The weights are the single most
contestable choice in the paper, so they get an explicit sensitivity block
rather than a footnote.

Labour needs care. Every PPI here is Not Seasonally Adjusted and monthly. The
Employment Cost Index is the conceptually right labour price - it holds
occupation and industry mix fixed, which matters precisely because a
data-centre boom shifts the craft mix toward electrical trades - but it is
quarterly. It is therefore used as a benchmark and interpolated to monthly by
proportional Denton, with average hourly earnings as the indicator series.
Raw AHE is retained as a robustness variant.

Also recorded here: WPU1017 (steel mill products) is NOT discontinued. FRED's
HTML data page caps at 1000 rows, which truncates it at 1939-01 + 1000 months
= 2022-04. WPU101704 (hot rolled bars, plates and structural shapes) is used
instead - live to 2026-07, and the structurally relevant sub-index anyway.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"data"/"interim"; TAB = ROOT/"tables"
X = pd.read_parquet(OUT/"aligned_monthly.parquet")

# --- basket -----------------------------------------------------------------
# Weights reflect a hyperscale build: electrical plant dominates, the shell is
# comparatively cheap, labour is roughly a third of installed cost.
BASKET = {
    "PCU335313335313": ("Switchgear and switchboard apparatus", 0.14),
    "PCU335311335311": ("Power and specialty transformers",     0.06),
    "WPU10260332":     ("Power wire and cable",                 0.08),
    "PCU333618333618": ("Engine gensets",                       0.07),
    "PCU333415333415": ("Cooling and air-handling plant",       0.13),
    "PCU332312332312": ("Fabricated structural metal",          0.08),
    "WPU101704":       ("Hot rolled steel shapes",              0.05),
    "PCU327320327320": ("Ready-mix concrete",                   0.06),
    "LABOUR":          ("Construction labour (ECI-benchmarked)",0.33),
}
BENCH = {"PCU236211236211":"New industrial building construction",
         "PCU236221236221":"New warehouse building construction",
         "PCU236223236223":"New office building construction",
         "WPU801":         "New nonresidential building construction"}

# --- monthly labour price by proportional Denton -----------------------------
def denton_proportional(indicator: pd.Series, benchmark_q: pd.Series) -> pd.Series:
    """Distribute a quarterly benchmark over months, preserving the indicator's
    month-to-month shape and matching the benchmark's quarterly average."""
    ind = indicator.dropna()
    q = ind.groupby(pd.PeriodIndex(ind.index, freq="Q")).mean()
    bm = benchmark_q.dropna()
    bm.index = pd.PeriodIndex(bm.index, freq="Q")
    common = q.index.intersection(bm.index)
    ratio = (bm[common] / q[common]).reindex(q.index)
    # smooth the benchmark-to-indicator ratio across quarters, then apply monthly
    ratio = ratio.interpolate(limit_direction="both")
    r_m = pd.Series(ratio.values, index=ratio.index.to_timestamp(how="s")) \
            .reindex(ind.index, method=None).interpolate(limit_direction="both")
    return ind * r_m

ahe = X["CEU2000000003"].dropna()                 # monthly NSA, matches PPI basis
eci = X["CIU2012300000000I"].dropna()             # quarterly NSA, mix-fixed
lab = denton_proportional(ahe, eci).rename("LABOUR")
X = X.join(lab)
X["LABOUR_AHE"] = ahe                             # robustness variant
X.to_parquet(OUT/"aligned_with_labour.parquet")   # downstream scripts read this

# --- index construction ------------------------------------------------------
START = "2005-01-01"; BASE = "2015-01-01"

def build(weights: dict, labour_col="LABOUR") -> pd.Series:
    cols = {k: (labour_col if k == "LABOUR" else k) for k in weights}
    sub = X[list(cols.values())].dropna()
    sub = sub[sub.index >= START]
    rel = sub / sub.loc[sub.index[sub.index.get_indexer([pd.Timestamp(BASE)], method="nearest")[0]]]
    w = pd.Series({cols[k]: v for k, v in weights.items()})
    w = w / w.sum()
    return (rel * w).sum(axis=1) * 100

W = {k: v[1] for k, v in BASKET.items()}
dcci = build(W).rename("DCCI")
dcci_ahe = build(W, "LABOUR_AHE").rename("DCCI_AHE")

idx = pd.DataFrame({"DCCI": dcci, "DCCI_AHE": dcci_ahe})
for b in BENCH:
    s = X[b].dropna()
    if len(s):
        base = s.reindex([pd.Timestamp(BASE)], method="nearest").iloc[0]
        idx[b] = (s / base * 100)
idx = idx[idx.index >= START]
idx.to_parquet(OUT/"dcci.parquet"); idx.to_csv(TAB/"dcci.csv")

print(f"DCCI: {idx.index.min():%Y-%m} .. {idx.index.max():%Y-%m}   n={idx['DCCI'].notna().sum()}"
      f"   (base {BASE[:7]}=100)\n")
print("=== index levels at key dates ===")
show = [d for d in ["2005-01","2010-01","2015-01","2019-01","2021-01","2022-06","2024-01","2026-07"]
        if pd.Timestamp(d) in idx.index]
print(idx.loc[show].round(1).to_string())

print("\n=== cumulative escalation, Jan 2015 -> latest ===")
last = idx.dropna(subset=["DCCI"]).iloc[-1]
for c in idx.columns:
    if pd.notna(last.get(c)):
        print(f"  {c:<18} {last[c]-100:+6.1f}%   ({BENCH.get(c, 'this paper')[:52]})")

print("\n=== annualised growth, 2020-01 -> latest (the AI-buildout window) ===")
sub = idx[idx.index >= "2020-01-01"].dropna(subset=["DCCI"])
yrs = (sub.index[-1] - sub.index[0]).days / 365.25
for c in idx.columns:
    s = sub[c].dropna()
    if len(s) > 12:
        print(f"  {c:<18} {100*((s.iloc[-1]/s.iloc[0])**(1/yrs)-1):+6.2f}% / yr")

print("\n=== component contributions to DCCI growth since 2020-01 ===")
sub2 = X[X.index >= "2020-01-01"]
rows = []
for k, (label, w) in BASKET.items():
    col = "LABOUR" if k == "LABOUR" else k
    s = sub2[col].dropna()
    if len(s) < 12: continue
    g = s.iloc[-1]/s.iloc[0] - 1
    rows.append({"component": label, "weight": w, "growth_%": 100*g,
                 "contribution_pp": 100*w*g})
c = pd.DataFrame(rows).sort_values("contribution_pp", ascending=False)
print(c.round(2).to_string(index=False))
print(f"  total contribution: {c['contribution_pp'].sum():.2f} pp")
