#!/usr/bin/env python3
"""
03_divergence.py — how much does the basket matter, and where does the money go?

Three things a referee will demand:
  1. The weights are asserted, so show what happens when they move. One-at-a-time
     perturbation plus a Dirichlet draw over the whole simplex.
  2. DCCI is an *input* price index; the BLS building-construction PPIs are
     *output* price indices, which embed contractor margin and productivity.
     They are not the same object and the paper must say so. What is comparable
     is the growth differential, reported here for the 2020+ window.
  3. The decomposition: which inputs actually drove escalation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
rng = np.random.default_rng(20260827)
ROOT = Path(__file__).resolve().parent.parent
X = pd.read_parquet(ROOT/"data"/"interim"/"aligned_with_labour.parquet")
idx = pd.read_parquet(ROOT/"data"/"interim"/"dcci.parquet")
TAB = ROOT/"tables"

BASKET = {"PCU335313335313":("Switchgear",0.14),"PCU335311335311":("Transformers",0.06),
          "WPU10260332":("Power cable",0.08),"PCU333618333618":("Gensets",0.07),
          "PCU333415333415":("Cooling plant",0.13),"PCU332312332312":("Fab. structural metal",0.08),
          "WPU101704":("Steel shapes",0.05),"PCU327320327320":("Ready-mix concrete",0.06),
          "LABOUR":("Labour",0.33)}
GROUP = {"Switchgear":"Electrical","Transformers":"Electrical","Power cable":"Electrical",
         "Gensets":"Electrical","Cooling plant":"Mechanical","Fab. structural metal":"Structure",
         "Steel shapes":"Structure","Ready-mix concrete":"Structure","Labour":"Labour"}
BASE="2015-01-01"; START="2020-01-01"

cols={k:("LABOUR" if k=="LABOUR" else k) for k in BASKET}
sub=X[list(cols.values())].dropna(); sub=sub[sub.index>=pd.Timestamp("2005-01-01")]
rel=sub/sub.loc[sub.index[sub.index.get_indexer([pd.Timestamp(BASE)],method="nearest")[0]]]

def index_from(w: dict) -> pd.Series:
    ws=pd.Series({cols[k]:v for k,v in w.items()}); ws=ws/ws.sum()
    return (rel*ws).sum(axis=1)*100

W={k:v[1] for k,v in BASKET.items()}
base_idx=index_from(W)
def cagr(s, start=START):
    s=s[s.index>=pd.Timestamp(start)].dropna()
    return 100*((s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1)

print(f"=== baseline DCCI CAGR since {START[:7]}: {cagr(base_idx):.2f}%/yr ===")

print("\n=== 1. one-at-a-time weight perturbation (+/-50% of each weight, renormalised) ===")
rows=[]
for k,(label,w) in BASKET.items():
    for mult,tag in [(0.5,"-50%"),(1.5,"+50%")]:
        W2=dict(W); W2[k]=w*mult
        rows.append({"component":label,"shift":tag,"weight":round(w*mult,3),
                     "CAGR_%":cagr(index_from(W2))})
oat=pd.DataFrame(rows).pivot(index="component",columns="shift",values="CAGR_%")
oat["range_pp"]=(oat["+50%"]-oat["-50%"]).abs()
print(oat.round(3).sort_values("range_pp",ascending=False).to_string())

print("\n=== 2. Dirichlet sensitivity over the whole simplex (2,000 draws, alpha=8) ===")
w0=np.array([v[1] for v in BASKET.values()]); keys=list(BASKET)
draws=rng.dirichlet(w0*8, size=2000)
cg=np.array([cagr(index_from(dict(zip(keys,d)))) for d in draws])
print(f"  baseline {cagr(base_idx):.2f}%/yr | draws: mean {cg.mean():.2f}, "
      f"sd {cg.std():.2f}, 5th {np.percentile(cg,5):.2f}, 95th {np.percentile(cg,95):.2f}")
for b,name in [("WPU801","nonresidential"),("PCU236211236211","industrial"),
               ("PCU236221236221","warehouse"),("PCU236223236223","office")]:
    bc=cagr(idx[b]); share=100*(cg>bc).mean()
    print(f"  P(DCCI CAGR > {name} PPI {bc:.2f}%/yr) = {share:.1f}% of draws")

print("\n=== 3. escalation decomposition since 2020-01, grouped ===")
s2=X[X.index>=pd.Timestamp(START)]
rec=[]
for k,(label,w) in BASKET.items():
    s=s2[cols[k]].dropna()
    g=s.iloc[-1]/s.iloc[0]-1
    rec.append({"group":GROUP[label],"component":label,"weight":w,
                "price_growth_%":100*g,"contribution_pp":100*w*g})
d=pd.DataFrame(rec)
tot=d["contribution_pp"].sum()
g=(d.groupby("group")[["weight","contribution_pp"]].sum()
     .assign(share_of_escalation=lambda x:100*x["contribution_pp"]/tot)
     .sort_values("contribution_pp",ascending=False))
print(g.round(2).to_string())
print(f"\n  total input-cost escalation since {START[:7]}: {tot:.1f}%")
print(f"  electrical plant is {g.loc['Electrical','weight']:.0%} of the basket "
      f"but {g.loc['Electrical','share_of_escalation']:.0f}% of the escalation")

print("\n=== 4. growth differential vs official output PPIs, 2020-01 -> latest ===")
bd=cagr(base_idx)
for b,name in [("WPU801","new nonresidential building"),("PCU236211236211","new industrial building"),
               ("PCU236221236221","new warehouse building"),("PCU236223236223","new office building")]:
    bc=cagr(idx[b]); yrs=(idx[b].dropna().index[-1]-pd.Timestamp(START)).days/365.25
    gap=((1+bd/100)**yrs)/((1+bc/100)**yrs)-1
    print(f"  vs {name:<28} {bd-bc:+.2f} pp/yr  ->  {100*gap:+.1f}% cumulative over {yrs:.1f} yrs")

pd.concat([oat,],axis=1).to_csv(TAB/"weight_sensitivity.csv")
d.to_csv(TAB/"escalation_decomposition.csv",index=False)
