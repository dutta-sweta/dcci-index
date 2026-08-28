#!/usr/bin/env python3
"""
01_load.py — parse and align every input series.

FRED .txt files carry a self-describing header block (Title, Series ID, Units,
Frequency, Seasonal Adjustment, Last Updated) followed by DATE/VALUE rows. That
metadata is retained rather than discarded: the paper has to state the base
period and adjustment basis of every component, and a mixed SA/NSA basket is a
real error that this parser makes visible.

Census: 'Data center' sits at column 9 of the private monthly workbooks, under
Office, alongside General (8) and Financial (10) - verified 27 Aug 2026, not
inferred. It is nominal spending in millions of dollars, SAAR in privsatime,
plain monthly in privtime.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np, pandas as pd, openpyxl

ROOT = Path(__file__).resolve().parent.parent
FRED = ROOT/"data"/"raw"/"fred"; CEN = ROOT/"data"/"raw"/"census"
OUT = ROOT/"data"/"interim"; OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT/"tables"; TAB.mkdir(exist_ok=True)

# FRED's /data/<ID>.txt endpoint now serves an HTML page rather than plain text
# (checked 27 Aug 2026). The observations are still there, in a table with
# id="data-table-observations", and the header block is a metadata table above
# it - so the file is parsed rather than re-downloaded.
META_RE = re.compile(r'<th scope="row"[^>]*>([^<]+)</th>\s*<td[^>]*>(.*?)</td>',
                     re.S)
OBS_RE = re.compile(r'<th scope="row"[^>]*>(\d{4}-\d{2}-\d{2})</th>\s*'
                    r'<td[^>]*>\s*([-\d.]+|\.)\s*</td>', re.S)

def parse_fred(path: Path):
    raw = path.read_text(errors="replace")
    if not raw.lstrip().startswith("<"):                 # plain-text fallback
        meta, rows, in_data = {}, [], False
        for line in raw.splitlines():
            if not in_data:
                if re.match(r"^DATE\s+VALUE", line.strip()): in_data = True; continue
                if ":" in line:
                    k, v = line.split(":", 1); meta[k.strip()] = v.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                try: rows.append((pd.Timestamp(parts[0]), float(parts[1])))
                except ValueError: pass
        return pd.Series(dict(rows), name=path.stem).sort_index(), meta

    meta = {}
    for k, v in META_RE.findall(raw):
        v = re.sub(r"<[^>]+>", " ", v)
        meta.setdefault(k.strip(), " ".join(v.split())[:200])
    rows = [(pd.Timestamp(d), float(v)) for d, v in OBS_RE.findall(raw) if v != "."]
    return pd.Series(dict(rows), name=path.stem).sort_index(), meta

series, metas = {}, []
for f in sorted(FRED.glob("*.txt")):
    s, m = parse_fred(f)
    if s.empty: print(f"  EMPTY {f.stem}"); continue
    series[f.stem] = s
    metas.append({"series_id": f.stem,
                  "title": m.get("Title", "")[:70],
                  "units": m.get("Units", ""),
                  "adjustment": m.get("Seasonal Adjustment", ""),
                  "frequency": m.get("Frequency", ""),
                  "start": s.index.min().date(), "end": s.index.max().date(),
                  "n": len(s)})
meta = pd.DataFrame(metas).sort_values("series_id")
meta.to_csv(TAB/"series_register.csv", index=False)

print(f"parsed {len(series)} FRED series\n")
print(meta[["series_id","frequency","adjustment","start","end","n"]].to_string(index=False))

print("\n=== adjustment basis (a mixed SA/NSA basket is an error) ===")
print(meta["adjustment"].value_counts().to_string())
print("\n=== frequency ===")
print(meta["frequency"].value_counts().to_string())

# ---------------- Census -----------------------------------------------------
def census_monthly(fname, label):
    wb = openpyxl.load_workbook(CEN/fname, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    hdr = next(ws.iter_rows(min_row=4, max_row=4, values_only=True))
    def col(name):
        for i, h in enumerate(hdr):
            if h and str(h).replace("_x000D_"," ").strip().lower() == name.lower():
                return i
        raise KeyError(name)
    ci = {n: col(n) for n in ["Date","Nonresidential","Office","General","Data center","Financial"]}
    rec = []
    for row in ws.iter_rows(min_row=5, values_only=True):
        d = row[ci["Date"]]
        if d is None: continue
        ds = str(d).strip().rstrip("pr")           # 'Jun-26p', 'May-26r'
        try: ts = pd.to_datetime(ds, format="%b-%y")
        except ValueError:
            try: ts = pd.to_datetime(ds)
            except Exception: continue
        rec.append({"date": ts,
                    **{k.replace(" ","_").lower(): row[v]
                       for k, v in ci.items() if k != "Date"}})
    wb.close()
    df = pd.DataFrame(rec).dropna(subset=["date"]).set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = [f"{label}_{c}" for c in df.columns]
    return df

sa  = census_monthly("privsatime.xlsx", "sa")
nsa = census_monthly("privtime.xlsx",  "nsa")
cen = sa.join(nsa, how="outer")

dc = cen["nsa_data_center"].dropna()
print(f"\n=== Census 'Data center', NSA monthly ===")
print(f"  {dc.index.min():%Y-%m} .. {dc.index.max():%Y-%m}   n={len(dc)}   $m/month")
ann = dc.groupby(dc.index.year).sum()
print("  annual totals ($m):")
print("   " + "  ".join(f"{y}:{v:,.0f}" for y, v in ann.items()))
sh = (cen["nsa_data_center"]/cen["nsa_office"]).dropna()
print(f"  share of private Office: {sh.iloc[-1]:.1%} latest, {sh.loc['2014'].mean():.1%} in 2014")

# ---------------- align ------------------------------------------------------
# Keep quarterly series too - the ECI is quarterly and is the labour benchmark.
# On a monthly index they simply carry NaN in non-quarter months.
X = pd.DataFrame(series)
X = X[X.index >= "1995-01-01"]
X.index.name = "date"
panel = X.join(cen, how="left")
panel.to_parquet(OUT/"aligned_monthly.parquet")

full = X.dropna()
print(f"\n=== aligned monthly panel ===")
print(f"  {panel.shape[0]} rows x {panel.shape[1]} cols, {panel.index.min():%Y-%m}..{panel.index.max():%Y-%m}")
print(f"  all PPI/labour series jointly available from {full.index.min():%Y-%m} "
      f"({len(full)} months)")
binding = X.apply(lambda c: c.first_valid_index()).sort_values(ascending=False)
print(f"  binding constraint: {binding.index[0]} starts {binding.iloc[0]:%Y-%m}; "
      f"next {binding.index[1]} {binding.iloc[1]:%Y-%m}")
