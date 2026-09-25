#!/usr/bin/env python3
"""
07_figures.py — publication figures.

Figures are built greyscale-safe: printed monochrome or photocopied, colour is
the first thing to go. So every series carries identity three ways - hue, line style, and marker - and
the palette is chosen for *luminance* separation, which is what greyscale
actually preserves. That separation is asserted below rather than eyeballed:
if two series would collapse in print the script fails instead of shipping.

Other constraints: sized for a two-column journal page and placed at native
resolution, text no smaller than 6 pt, 300 dpi, no dual axes anywhere.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT=Path(__file__).resolve().parent.parent
FIG=ROOT/"figures"; FIG.mkdir(exist_ok=True)
X=pd.read_parquet(ROOT/"data"/"interim"/"aligned_with_labour.parquet")
idx=pd.read_parquet(ROOT/"data"/"interim"/"dcci.parquet")

CM=1/2.54
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":7,"axes.labelsize":7.5,
    "axes.titlesize":8,"legend.fontsize":6.5,"xtick.labelsize":6.5,
    "ytick.labelsize":6.5,"axes.linewidth":.6,"grid.linewidth":.4,
    "lines.linewidth":1.2,"figure.dpi":300,"savefig.dpi":300,
    "savefig.bbox":"tight","axes.grid":True,"grid.alpha":.25,
    "axes.spines.top":False,"axes.spines.right":False,
})

def lum(hexc):
    r,g,b=[int(hexc[i:i+2],16)/255 for i in (1,3,5)]
    f=lambda c: c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4
    return .2126*f(r)+.7152*f(g)+.0722*f(b)

def ramp(base_rgb, ts):
    """Blend a base hue toward white at fixed fractions. Greyscale separation is
    then a property of the construction rather than of taste, and is asserted
    below."""
    out=[]
    for t in ts:
        rgb=[base_rgb[i]+(1.0-base_rgb[i])*t for i in range(3)]
        out.append("#{:02X}{:02X}{:02X}".format(*[int(round(c*255)) for c in rgb]))
    return out

NAVY=(0.055,0.105,0.165)
C=ramp(NAVY,[0.00,0.28,0.48,0.66,0.82])
SERIES={
 "DCCI (this paper)":      (C[0],"-",  "o"),
 "New office building":    (C[1],"--", "s"),
 "New nonresidential":     (C[2],"-.", "^"),
 "New industrial building":(C[3],(0,(4,1,1,1)),"D"),
 "New warehouse building": (C[4],":",  "v"),
}
L=sorted(lum(c) for c,_,_ in SERIES.values())
gaps=[b-a for a,b in zip(L,L[1:])]
assert min(gaps)>=0.05, f"greyscale separation too small: {min(gaps):.3f} in {[round(x,3) for x in L]}"
print("palette:", ", ".join(C))
print(f"greyscale luminance: {[round(x,3) for x in L]}   min gap {min(gaps):.3f}  OK")

COLMAP={"DCCI (this paper)":"DCCI","New office building":"PCU236223236223",
        "New nonresidential":"WPU801","New industrial building":"PCU236211236211",
        "New warehouse building":"PCU236221236221"}

# ---- Fig 1: index levels ----------------------------------------------------
fig,ax=plt.subplots(figsize=(17*CM,7*CM))
for name,(c,ls,mk) in SERIES.items():
    s=idx[COLMAP[name]].dropna(); s=s[s.index>="2007-01-01"]
    ax.plot(s.index,s.values,color=c,linestyle=ls,label=name,
            marker=mk,markevery=24,markersize=3,markerfacecolor="white",
            markeredgewidth=.7,linewidth=1.6 if name.startswith("DCCI") else 1.1)
ax.axhline(100,color="#999999",lw=.5,ls=(0,(1,3)))
ax.annotate("base: Jan 2015 = 100",xy=(pd.Timestamp("2007-06-01"),101),fontsize=6,color="#666666")
ax.set_ylabel("Index (Jan 2015 = 100)"); ax.set_xlabel("")
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.legend(frameon=False,ncol=3,loc="upper left")
fig.savefig(FIG/"fig1_index_levels.png"); plt.close(fig)

# ---- Fig 2: escalation decomposition ---------------------------------------
dec=pd.read_csv(ROOT/"tables"/"escalation_decomposition.csv").sort_values("contribution_pp")
GREY={"Electrical":"#1B1B1B","Structure":"#6E6E6E","Labour":"#9E9E9E","Mechanical":"#C4C4C4"}
HATCH={"Electrical":"","Structure":"//","Labour":"\\\\","Mechanical":".."}
fig,ax=plt.subplots(figsize=(9.2*CM,7.5*CM))
for _,r in dec.iterrows():
    ax.barh(r["component"],r["contribution_pp"],color=GREY[r["group"]],
            hatch=HATCH[r["group"]],edgecolor="white",linewidth=.8,height=.72)
    ax.text(r["contribution_pp"]+.25,r["component"],f"{r['contribution_pp']:.1f}",
            va="center",fontsize=6.2,color="#333333")
ax.set_xlabel("Contribution to escalation since Jan 2020 (pp)")
ax.set_xlim(0,max(dec["contribution_pp"])*1.18); ax.grid(axis="y",visible=False)
handles=[plt.Rectangle((0,0),1,1,facecolor=GREY[g],hatch=HATCH[g],edgecolor="white")
         for g in ["Electrical","Structure","Labour","Mechanical"]]
ax.legend(handles,["Electrical","Structure","Labour","Mechanical"],
          frameon=False,loc="lower right",ncol=1,
          bbox_to_anchor=(1.0,0.02))
fig.savefig(FIG/"fig2_decomposition.png"); plt.close(fig)

# ---- Fig 3: the compositional drift ----------------------------------------
share=(X["nsa_data_center"]/X["nsa_office"]).dropna()*100
spend=X["nsa_data_center"].dropna().rolling(12).sum()/1000
fig,axes=plt.subplots(1,2,figsize=(17*CM,6.2*CM))
fig.subplots_adjust(wspace=0.42)
a=axes[0]
a.plot(share.index,share.values,color="#111111",lw=1.6)
a.fill_between(share.index,0,share.values,color="#111111",alpha=.10)
a.set_ylabel("Data centres as share of\nprivate office construction (%)")
a.set_ylim(0,70)
a.annotate(f"{share.iloc[-1]:.0f}%",xy=(share.index[-1],share.iloc[-1]),
           xytext=(-34,-16),textcoords="offset points",fontsize=7,weight="bold")
a.annotate(f"{share.iloc[0]:.0f}% in 2014",xy=(share.index[0],share.iloc[0]),
           xytext=(6,10),textcoords="offset points",fontsize=6.2,color="#555555")
b=axes[1]
g=idx[["DCCI","PCU236223236223","PCU236211236211","PCU236221236221"]].apply(np.log).diff(12)*100
g=g.dropna()
peer=g[["PCU236211236211","PCU236221236221"]].mean(axis=1)
roll_dc=g["PCU236223236223"].rolling(36).corr(g["DCCI"])
roll_pe=g["PCU236223236223"].rolling(36).corr(peer)
b.plot(roll_dc.index,roll_dc.values,color="#111111",ls="-",lw=1.6,
       label="office PPI vs DCCI",marker="o",markevery=18,markersize=3,
       markerfacecolor="white",markeredgewidth=.7)
b.plot(roll_pe.index,roll_pe.values,color="#8A8A8A",ls="--",lw=1.2,
       label="office PPI vs industrial + warehouse",marker="^",markevery=18,
       markersize=3,markerfacecolor="white",markeredgewidth=.7)
b.axhline(0,color="#999999",lw=.5); b.set_ylim(-1,1.05)
b.set_ylabel("36-month rolling correlation\nof 12-month growth rates")
b.legend(frameon=False,loc="lower right")
fig.savefig(FIG/"fig3_office_drift.png"); plt.close(fig)

# ---- Fig 4: forecast accuracy ----------------------------------------------
r=pd.read_csv(ROOT/"tables"/"forecast_results.csv").dropna(subset=["err"])
acc=(r.groupby(["h","model"])["err"].apply(lambda e: float(np.sqrt((e**2).mean())))
       .rename("RMSE").reset_index())
order=["Random walk (no change)","Drift (24m mean)","ARIMA(1,1,1)","Ridge",
       "SVR (RBF)","Random forest","Gradient boosting"]
SH={"Random walk (no change)":("#D6D6D6","xx"),"Drift (24m mean)":("#B0B0B0",".."),
    "ARIMA(1,1,1)":("#8A8A8A","//"),"Ridge":("#6E6E6E","\\\\"),
    "SVR (RBF)":("#4F4F4F","--"),"Random forest":("#2E2E2E","++"),
    "Gradient boosting":("#111111","")}
fig,ax=plt.subplots(figsize=(17*CM,6.5*CM))
hs=sorted(acc["h"].unique()); w=.115
for i,m in enumerate(order):
    v=[acc[(acc["h"]==h)&(acc["model"]==m)]["RMSE"].mean() for h in hs]
    c,ht=SH[m]
    ax.bar(np.arange(len(hs))+(i-3)*w,v,width=w*.9,color=c,hatch=ht,
           edgecolor="white",linewidth=.6,label=m)
ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"h = {h} month" + ("" if h==1 else "s") for h in hs])
ax.set_ylabel("RMSE of forecast growth (pp)"); ax.grid(axis="x",visible=False)
ax.set_ylim(0, acc["RMSE"].max()*1.34)
ax.legend(frameon=False,ncol=4,fontsize=6,loc="upper left",
          bbox_to_anchor=(0,1.02))
fig.savefig(FIG/"fig4_forecast_rmse.png"); plt.close(fig)

print("wrote:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
for p in sorted(FIG.glob("*.png")): print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")
