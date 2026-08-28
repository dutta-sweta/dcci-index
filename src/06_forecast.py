#!/usr/bin/env python3
"""
06_forecast.py — the model demonstration.

Target: cumulative log growth of DCCI over the next h months. Growth rather
than level, because the level is near-unit-root and a level RMSE would flatter
every model equally.

Baselines are the point of this exercise, not the machine learning. A cost
index is highly persistent, so a random walk and a drift model are hard to
beat; any paper claiming a fancy model wins without showing those two is not
worth reading. Diebold-Mariano tests each model against the best baseline.

One invocation per horizon (each shell here has ~45 s), results cached.
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

ROOT=Path(__file__).resolve().parent.parent; SEED=20260827
ap=argparse.ArgumentParser(); ap.add_argument("--h",type=int,default=0)
ap.add_argument("--report",action="store_true"); A=ap.parse_args()
CACHE=ROOT/"tables"/"forecast_results.csv"

X=pd.read_parquet(ROOT/"data"/"interim"/"aligned_with_labour.parquet")
idx=pd.read_parquet(ROOT/"data"/"interim"/"dcci.parquet")
INPUTS=["PCU335313335313","PCU335311335311","WPU10260332","PCU333618333618",
        "PCU333415333415","PCU332312332312","WPU101704","PCU327320327320","LABOUR"]

d=pd.DataFrame({"dcci":idx["DCCI"]}).join(X[INPUTS]).dropna()
ld=np.log(d)
F=pd.DataFrame(index=d.index)
F["g1"]=ld["dcci"].diff(); F["g3"]=ld["dcci"].diff(3); F["g12"]=ld["dcci"].diff(12)
for c in INPUTS:
    F[f"{c}_g1"]=ld[c].diff(); F[f"{c}_g12"]=ld[c].diff(12)
F["mom"]=ld["dcci"].diff().rolling(6).mean()
F["vol"]=ld["dcci"].diff().rolling(12).std()
m=F.index.month
F["sin"]=np.sin(2*np.pi*m/12); F["cos"]=np.cos(2*np.pi*m/12)

def models(seed):
    return {"Ridge":make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,25))),
            "SVR (RBF)":make_pipeline(StandardScaler(),SVR(C=1.0,epsilon=.005,gamma="scale")),
            "Random forest":RandomForestRegressor(n_estimators=200,min_samples_leaf=3,
                                                  random_state=seed,n_jobs=-1),
            "Gradient boosting":GradientBoostingRegressor(n_estimators=200,max_depth=2,
                                                          learning_rate=.05,subsample=.8,
                                                          random_state=seed)}

def run(h):
    y=(ld["dcci"].shift(-h)-ld["dcci"])*100          # % cumulative growth over h months
    data=F.join(y.rename("y")).dropna()
    feats=[c for c in F.columns]
    months=data.index
    cuts=months[int(len(months)*0.55)::2]
    rows=[]
    for rep,cut in enumerate(cuts):
        tr=data[data.index<cut]; te=data[(data.index>=cut)&(data.index<cut+pd.DateOffset(months=2))]
        if len(te)<1 or len(tr)<80: continue
        yte=te["y"].values
        # --- baselines ---
        rows.append({"h":h,"split":str(cut)[:7],"model":"Random walk (no change)",
                     "pred":0.0,"actual":yte.mean(),"err":yte.mean()-0.0})
        drift=tr["y"].tail(24).mean()
        rows.append({"h":h,"split":str(cut)[:7],"model":"Drift (24m mean)",
                     "pred":drift,"actual":yte.mean(),"err":yte.mean()-drift})
        try:
            a=ARIMA(ld["dcci"][ld.index<cut]*100,order=(1,1,1)).fit()
            fc=a.forecast(h); ap_=float(fc.iloc[-1]-ld["dcci"][ld.index<cut].iloc[-1]*100)
        except Exception: ap_=np.nan
        rows.append({"h":h,"split":str(cut)[:7],"model":"ARIMA(1,1,1)",
                     "pred":ap_,"actual":yte.mean(),"err":yte.mean()-ap_})
        # --- learned ---
        for name,mdl in models(SEED+rep).items():
            mdl.fit(tr[feats].values,tr["y"].values)
            p=float(mdl.predict(te[feats].values).mean())
            rows.append({"h":h,"split":str(cut)[:7],"model":name,
                         "pred":p,"actual":yte.mean(),"err":yte.mean()-p})
    new=pd.DataFrame(rows)
    if CACHE.exists():
        old=pd.read_csv(CACHE); old=old[old["h"]!=h]; new=pd.concat([old,new],ignore_index=True)
    new.to_csv(CACHE,index=False); print(f"h={h}: {len(rows)} rows cached")

if A.h: run(A.h)
if not A.report: raise SystemExit
r=pd.read_csv(CACHE)
from scipy import stats as st
print("=== forecast accuracy, rolling origin (target: % growth of DCCI over h months) ===")
for h in sorted(r["h"].unique()):
    s=r[(r["h"]==h)].dropna(subset=["err"])
    print(f"\n  --- h = {h} months ---   origins: {s['split'].nunique()}")
    tab=(s.groupby("model")
           .apply(lambda x: pd.Series({
               "MAE":x["err"].abs().mean(),
               "RMSE":float(np.sqrt((x["err"]**2).mean())),
               "R2":r2_score(x["actual"],x["pred"]),
               "n":len(x)}), include_groups=False)
           .sort_values("RMSE"))
    print(tab.round(4).to_string())
    best_base=tab.loc[[m for m in tab.index if m in
                       ("Random walk (no change)","Drift (24m mean)","ARIMA(1,1,1)")]].index[0]
    print(f"    best baseline: {best_base}")
    be=s[s["model"]==best_base].set_index("split")["err"]
    for mdl in tab.index:
        if mdl==best_base: continue
        me=s[s["model"]==mdl].set_index("split")["err"]
        j=be.index.intersection(me.index)
        if len(j)<8: continue
        dm=(be[j]**2)-(me[j]**2)
        t=dm.mean()/(dm.std(ddof=1)/np.sqrt(len(j)))
        p=2*(1-st.t.cdf(abs(t),len(j)-1))
        verdict = "beats baseline" if t>0 and p<.1 else ("worse" if t<0 and p<.1 else "no difference")
        print(f"    DM vs {best_base}: {mdl:<24} t={t:+.2f} p={p:.4f}  {verdict}")
