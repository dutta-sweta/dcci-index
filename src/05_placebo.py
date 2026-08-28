#!/usr/bin/env python3
"""
05_placebo.py — try to kill Finding 2.

Finding 2 says the office construction PPI has drifted toward a data-centre
input index as data centres came to dominate the office category. Three ways
that could be spurious, tested here in ascending order of seriousness:

  1. PLACEBO. If the data-centre share also "explains" spreads between building
     types that have nothing to do with data centres (industrial vs warehouse),
     it is picking up something common, not composition.
  2. TREND. dc_share rises almost monotonically from 4.6% to 59.7%. Any variable
     trending over the same window will load on it. Does the coefficient survive
     a linear time trend? This is the one most likely to kill it.
  3. COMMON INPUT SHOCK. Office and data-centre construction share an electrical
     supply chain. If electrical input growth is controlled for and dc_share
     still loads, composition is doing work beyond the shared cost shock.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
ROOT=Path(__file__).resolve().parent.parent
X=pd.read_parquet(ROOT/"data"/"interim"/"aligned_with_labour.parquet")
idx=pd.read_parquet(ROOT/"data"/"interim"/"dcci.parquet")

d=pd.DataFrame({"dcci":idx["DCCI"],"office":idx["PCU236223236223"],
                "industrial":idx["PCU236211236211"],"warehouse":idx["PCU236221236221"],
                "nonres":idx["WPU801"],
                "elec":X["PCU335313335313"],           # switchgear, the electrical bellwether
                "dc_share":(X["nsa_data_center"]/X["nsa_office"])}).dropna(subset=["office"])
g=d[["dcci","office","industrial","warehouse","nonres","elec"]].apply(np.log).diff(12)*100
g["dc_share"]=d["dc_share"]
g["peer"]=g[["industrial","warehouse"]].mean(axis=1)
g["t"]=np.arange(len(g))
g=g.dropna(subset=["office","industrial","warehouse","dc_share","elec"])

def hac(dv, rhs, data=g):
    dd=data.dropna(subset=[dv]+rhs)
    m=sm.OLS(dd[dv], sm.add_constant(dd[rhs])).fit(cov_type="HAC",cov_kwds={"maxlags":12})
    return m

def line(tag, m, key="dc_share"):
    b,se,p=m.params[key],m.bse[key],m.pvalues[key]
    star="***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
    print(f"  {tag:<48}{b:>+8.3f}{se:>8.3f}{p:>9.4f}{m.rsquared:>8.3f}{int(m.nobs):>6}  {star}")

print(f"sample {g.index.min():%Y-%m}..{g.index.max():%Y-%m}  n={len(g)}")
print(f"\n{'specification':<48}{'beta':>8}{'se':>8}{'p':>9}{'R2':>8}{'n':>6}")

print("\n-- 1. PLACEBO: spreads that should not respond to data-centre share --")
for dv,lab in [("ind_minus_ware","industrial - warehouse"),
               ("ware_minus_nonres","warehouse - nonresidential"),
               ("ind_minus_nonres","industrial - nonresidential")]:
    a,b_=lab.split(" - ")
    key={"industrial":"industrial","warehouse":"warehouse","nonresidential":"nonres"}
    g[dv]=g[key[a]]-g[key[b_]]
    line(f"{lab:<28} ~ dc_share", hac(dv,["dc_share"]))

print("\n-- the real one, for comparison --")
g["office_minus_peer"]=g["office"]-g["peer"]
line("office - peer                 ~ dc_share", hac("office_minus_peer",["dc_share"]))

print("\n-- 2. TREND: does it survive a linear time trend? --")
line("office - peer  ~ dc_share + t", hac("office_minus_peer",["dc_share","t"]))
m=hac("office_minus_peer",["dc_share","t"])
print(f"      (trend coefficient {m.params['t']:+.4f}, p {m.pvalues['t']:.4f};"
      f" corr(dc_share, t) = {g['dc_share'].corr(g['t']):+.3f})")

print("\n-- 3. COMMON INPUT SHOCK: control for electrical input growth --")
line("office - peer  ~ dc_share + elec", hac("office_minus_peer",["dc_share","elec"]))
line("office - peer  ~ dc_share + elec + t", hac("office_minus_peer",["dc_share","elec","t"]))

print("\n-- 4. same battery on the placebo, to be fair to it --")
line("industrial - warehouse ~ dc_share + t", hac("ind_minus_ware",["dc_share","t"]))

print("\n-- 5. does dc_share beat a trend on its own terms? (nested F-test) --")
dd=g.dropna(subset=["office_minus_peer","dc_share","t"])
full=sm.OLS(dd["office_minus_peer"], sm.add_constant(dd[["dc_share","t"]])).fit()
rest=sm.OLS(dd["office_minus_peer"], sm.add_constant(dd[["t"]])).fit()
f=((rest.ssr-full.ssr)/1)/(full.ssr/full.df_resid)
from scipy import stats as st
print(f"  trend-only R2 {rest.rsquared:.4f} -> +dc_share R2 {full.rsquared:.4f}   "
      f"F={f:.3f}, p={1-st.f.cdf(f,1,full.df_resid):.4f}")
