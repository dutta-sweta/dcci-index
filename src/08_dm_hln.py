#!/usr/bin/env python3
"""
08_dm_hln.py — Diebold-Mariano with the Harvey-Leybourne-Newbold correction.

The DM statistic over-rejects in small samples, and with ~50 rolling origins
and multi-step horizons this sample is small. HLN (1997) rescale the statistic
by sqrt[(n + 1 - 2h + h(h-1)/n)/n] and compare against t(n-1) rather than the
standard normal. Reporting the uncorrected statistic alone invites the obvious
referee question, so both are reported.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
ROOT=Path(__file__).resolve().parent.parent
r=pd.read_csv(ROOT/"tables"/"forecast_results.csv").dropna(subset=["err"])
BASE={"Random walk (no change)","Drift (24m mean)","ARIMA(1,1,1)"}

out=[]
for h in sorted(r["h"].unique()):
    s=r[r["h"]==h]
    rmse=s.groupby("model")["err"].apply(lambda e: float(np.sqrt((e**2).mean()))).sort_values()
    best_base=[m for m in rmse.index if m in BASE][0]
    be=s[s["model"]==best_base].set_index("split")["err"]
    for mdl in rmse.index:
        if mdl==best_base: continue
        me=s[s["model"]==mdl].set_index("split")["err"]
        j=be.index.intersection(me.index)
        if len(j)<8: continue
        dm_series=(be[j]**2)-(me[j]**2)
        n=len(j)
        t_dm=dm_series.mean()/(dm_series.std(ddof=1)/np.sqrt(n))
        adj=np.sqrt((n+1-2*h+h*(h-1)/n)/n)
        t_hln=t_dm*adj
        p_dm=2*(1-st.norm.cdf(abs(t_dm)))
        p_hln=2*(1-st.t.cdf(abs(t_hln), n-1))
        out.append({"h":h,"model":mdl,"baseline":best_base,"n_origins":n,
                    "RMSE":rmse[mdl],"RMSE_base":rmse[best_base],
                    "pct_vs_base":100*(rmse[best_base]-rmse[mdl])/rmse[best_base],
                    "DM_t":t_dm,"DM_p":p_dm,"HLN_t":t_hln,"HLN_p":p_hln})
d=pd.DataFrame(out)
d.to_csv(ROOT/"tables"/"dm_hln.csv",index=False)
pd.set_option("display.width",220)
for h in sorted(d["h"].unique()):
    s=d[d["h"]==h]
    print(f"\n=== h = {h} months   (baseline: {s['baseline'].iat[0]}, "
          f"{s['n_origins'].iat[0]} origins) ===")
    print(s[["model","RMSE","pct_vs_base","DM_t","DM_p","HLN_t","HLN_p"]]
          .round(4).to_string(index=False))
