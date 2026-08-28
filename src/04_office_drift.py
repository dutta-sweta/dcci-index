#!/usr/bin/env python3
"""
04_office_drift.py — has the office construction PPI quietly become a
data-centre index?

Census split 'Data center' out of private Office in 2024 because it had grown
too large to leave buried. It is now 59.7% of private office construction, up
from 4.6% in 2014. BLS did not follow: PCU236223236223 is still labelled 'New
Office Building Construction' and is what any analyst deflating a data-centre
project would reach for.

If the office PPI is increasingly priced by data-centre work, then as the
data-centre share of office construction rose, the office PPI should have
drifted away from its peer building-type indices and toward a data-centre input
cost index. Testable directly.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
ROOT=Path(__file__).resolve().parent.parent
X=pd.read_parquet(ROOT/"data"/"interim"/"aligned_with_labour.parquet")
idx=pd.read_parquet(ROOT/"data"/"interim"/"dcci.parquet")

d=pd.DataFrame({
    "dcci":idx["DCCI"], "office":idx["PCU236223236223"],
    "industrial":idx["PCU236211236211"], "warehouse":idx["PCU236221236221"],
    "nonres":idx["WPU801"],
    "dc_share":(X["nsa_data_center"]/X["nsa_office"]),
}).dropna(subset=["office","dcci"])
d["peer"]=d[["industrial","warehouse"]].mean(axis=1)

g=d.apply(lambda s: np.log(s)).diff(12)*100          # 12-month log growth, %
g["dc_share"]=d["dc_share"]
g=g.dropna(subset=["office","peer","dcci"])

print(f"sample {g.index.min():%Y-%m}..{g.index.max():%Y-%m}  n={len(g)}")
print(f"data-centre share of private office construction: "
      f"{d['dc_share'].dropna().iloc[0]:.1%} ({d['dc_share'].dropna().index[0]:%Y}) -> "
      f"{d['dc_share'].dropna().iloc[-1]:.1%} ({d['dc_share'].dropna().index[-1]:%Y-%m})")

print("\n=== rolling 36-month correlation of 12m growth rates ===")
for lo,hi,lab in [("2010","2015","2010-2015"),("2016","2020","2016-2020"),("2021","2026","2021-2026")]:
    s=g.loc[lo:hi]
    if len(s)<24: continue
    print(f"  {lab}: corr(office, DCCI) = {s['office'].corr(s['dcci']):+.3f} | "
          f"corr(office, peer building types) = {s['office'].corr(s['peer']):+.3f} | n={len(s)}")

print("\n=== does the office-minus-peer spread move with the data-centre share? ===")
h=g.dropna(subset=["dc_share"]).copy()
h["spread"]=h["office"]-h["peer"]
h["dcci_minus_peer"]=h["dcci"]-h["peer"]
for dv,label in [("spread","office PPI growth - peer building-type growth"),
                 ("office","office PPI growth")]:
    Xr=sm.add_constant(h[["dc_share"]])
    m=sm.OLS(h[dv],Xr).fit(cov_type="HAC",cov_kwds={"maxlags":12})
    print(f"  {label}:")
    print(f"    beta(dc_share) {m.params['dc_share']:+.3f}  se {m.bse['dc_share']:.3f}  "
          f"p {m.pvalues['dc_share']:.4f}  R2 {m.rsquared:.3f}  n {int(m.nobs)}")

print("\n=== which index best tracks DCCI, by era? (RMSE of 12m growth, pp) ===")
for lo,hi,lab in [("2010","2015","2010-2015"),("2016","2020","2016-2020"),("2021","2026","2021-2026")]:
    s=g.loc[lo:hi].dropna(subset=["dcci"])
    if len(s)<24: continue
    r={c:float(np.sqrt(((s["dcci"]-s[c])**2).mean())) for c in ["office","industrial","warehouse","nonres"]}
    best=min(r,key=r.get)
    print(f"  {lab}: " + "  ".join(f"{k} {v:.2f}" for k,v in r.items()) + f"   <- closest: {best}")
