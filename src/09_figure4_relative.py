#!/usr/bin/env python3
"""
09_figure4_relative.py — Figure 4 for the SJST submission.

SJST forbids presenting the same data in both a table and a figure. Table 3
carries the absolute three-month accuracy (RMSE, MAE, R-squared, HLN p), so
Figure 4 cannot repeat RMSE levels. It plots forecast error *relative to
ARIMA(1,1,1)* across all four horizons instead: the figure then carries the
horizon dimension and the table carries the levels, and no value appears twice.

Style (palette, hatching, rcParams, sizing) follows 07_figures.py so the four
figures remain one visual set. Greyscale-safe by construction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

CM = 1/2.54
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":7,"axes.labelsize":7.5,
    "axes.titlesize":8,"legend.fontsize":6.5,"xtick.labelsize":6.5,
    "ytick.labelsize":6.5,"axes.linewidth":.6,"grid.linewidth":.4,
    "lines.linewidth":1.2,"figure.dpi":300,"savefig.dpi":300,
    "savefig.bbox":"tight","axes.grid":True,"grid.alpha":.25,
    "axes.spines.top":False,"axes.spines.right":False,
})

r = pd.read_csv(ROOT/"tables"/"forecast_results.csv").dropna(subset=["err"])
acc = (r.groupby(["h","model"])["err"]
         .apply(lambda e: float(np.sqrt((e**2).mean())))
         .rename("RMSE").reset_index())

BENCH = "ARIMA(1,1,1)"
base = acc[acc["model"]==BENCH].set_index("h")["RMSE"]
acc["ratio"] = acc.apply(lambda x: x["RMSE"]/base[x["h"]], axis=1)

order = ["Random walk (no change)","Drift (24m mean)","Ridge",
         "SVR (RBF)","Random forest","Gradient boosting"]
SH = {"Random walk (no change)":("#D6D6D6","xx"),"Drift (24m mean)":("#B0B0B0",".."),
      "Ridge":("#6E6E6E","\\\\"),"SVR (RBF)":("#4F4F4F","--"),
      "Random forest":("#2E2E2E","++"),"Gradient boosting":("#111111","")}

fig, ax = plt.subplots(figsize=(17*CM, 6.5*CM))
hs = sorted(acc["h"].unique()); w = .135
for i, m in enumerate(order):
    v = [float(acc[(acc["h"]==h)&(acc["model"]==m)]["ratio"].mean()) for h in hs]
    c, ht = SH[m]
    ax.bar(np.arange(len(hs))+(i-2.5)*w, v, width=w*.9, color=c, hatch=ht,
           edgecolor="white", linewidth=.6, label=m)

ax.axhline(1.0, color="#111111", lw=.9, ls=(0,(5,2)), zorder=3)
ax.text(len(hs)-.42, 1.012, "ARIMA(1,1,1) = 1.00", fontsize=6,
        ha="right", va="bottom", color="#111111")
ax.set_xticks(range(len(hs)))
ax.set_xticklabels([f"h = {h} month" + ("" if h==1 else "s") for h in hs])
ax.set_ylabel("RMSE relative to ARIMA(1,1,1)\n(below 1.00 = lower error)")
ax.grid(axis="x", visible=False)
ax.set_ylim(0, float(acc["ratio"].max())*1.30)
ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper left", bbox_to_anchor=(0,1.02))
fig.savefig(FIG/"fig4_forecast_skill.png"); plt.close(fig)

print("wrote figures/fig4_forecast_skill.png")
print(acc.pivot(index="model", columns="h", values="ratio").round(3).to_string())
