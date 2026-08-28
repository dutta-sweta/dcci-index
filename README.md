# A construction cost escalation index for data centres in the United States

Code and data for the paper of that title, submitted to *Tehnički vjesnik –
Technical Gazette*.

Everything here runs on free public data from the U.S. Census Bureau and the
Bureau of Labor Statistics. No proprietary or employer data is used anywhere in
the pipeline.

## What this builds

A monthly fixed-weight Laspeyres input-cost index for data-centre construction,
January 2005 to July 2026, base January 2015 = 100, assembled from 23 producer
price and labour-cost series. The index itself is in `tables/dcci.csv`; if that
is all you need, you can stop reading here.

The pipeline also reproduces the three results in the paper: the escalation
decomposition, the office-index drift test with its placebo battery, and the
forecast benchmark against ARIMA, drift, random-walk, Ridge, SVR, random forest
and gradient boosting.

## Running it

```bash
python3 -m pip install -r requirements.txt
bash src/00_fetch.sh          # optional - raw data is already committed
python3 src/01_load.py        # parse and align every series
python3 src/02_index.py       # build the index
python3 src/03_divergence.py  # weight sensitivity and decomposition
python3 src/04_office_drift.py
python3 src/05_placebo.py     # placebo, trend and common-shock battery
python3 src/06_forecast.py --h 1 && python3 src/06_forecast.py --h 3
python3 src/06_forecast.py --h 6 && python3 src/06_forecast.py --h 12
python3 src/06_forecast.py --report
python3 src/08_dm_hln.py      # Diebold-Mariano with the HLN small-sample correction
python3 src/07_figures.py     # the four paper figures
```

Scripts resolve paths relative to themselves, so the repository can sit
anywhere. `00_fetch.sh` re-downloads the raw inputs and is only needed to
refresh the data; a snapshot retrieved on 27 August 2026 is committed under
`data/raw/`.

## Changing the weights

The basket weights are the most contestable choice in the paper, and they are
one dictionary at the top of `src/02_index.py`. Edit `BASKET`, re-run
`02_index.py` and `03_divergence.py`, and every downstream number moves with
them. `03_divergence.py` also reports what happens across 2,000 randomised
weight vectors, which is the honest width of the estimate.

## Four data traps

Each of these will silently corrupt a replication, so they are recorded here
rather than left to be rediscovered.

- **`WPU1017` is not discontinued.** FRED's HTML data page caps output at 1,000
  rows. The series starts in January 1939, so the page truncates it at April
  2022 and it looks retired. `WPU101704` is used instead.
- **`WPU1332` is concrete pipe**, not ready-mix concrete. It is a common
  misattribution. Ready-mix is `PCU327320327320`.
- **`PCU3353113353111`** stopped publishing in February 2023. Use `WPU117409`.
- **`WPU10226` does not exist.**

Also: FRED's `/data/<ID>.txt` endpoint now returns HTML rather than plain text.
`01_load.py` parses both forms.

## Layout

```
src/         nine scripts, in execution order
data/raw/    FRED series and Census workbooks, retrieved 27 Aug 2026
data/interim/  parsed and aligned panels (parquet)
tables/      dcci.csv is the index; the rest are results tables
figures/     the four figures as they appear in the paper
```

## Licence

Code is MIT. The derived index and results tables are CC BY 4.0. The underlying
Census and BLS source data are U.S. Government works in the public domain.

## Citation

See `CITATION.cff`. If you use the index, please cite the paper once it appears.
