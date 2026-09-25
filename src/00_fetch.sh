#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# src/00_fetch.sh — inputs for the data-centre construction cost index
#
# RUN IN YOUR OWN macOS Terminal. fred.stlouisfed.org and census.gov are both
# blocked by the egress proxy that Claude's sandbox and its Linux VM share.
#
#   cd "$HOME/Documents/dcci-index"
#   bash src/00_fetch.sh
#
# Small: about 10 MB total, under a minute. Resumable.
#
# Every FRED/BLS identifier below was verified live on 27 Aug 2026. Three
# identifiers that circulate in the literature are deliberately NOT used:
#   WPU10226              does not exist (404)
#   WPU1332               is concrete PIPE, not ready-mix - common misattribution
#   PCU3353113353111      transformers, stopped publishing Feb 2023
# ---------------------------------------------------------------------------
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DF="$ROOT/data/raw/fred"; DC="$ROOT/data/raw/census"
mkdir -p "$DF" "$DC"
UA="Mozilla/5.0 (Macintosh; academic research; ORCID 0009-0007-2162-3564)"
log(){ printf "  %s\n" "$*"; }
head2(){ printf "\n\033[1m%s\033[0m\n" "$*"; }

get_fred () {  # get_fred <series_id> <description>
  local id="$1" desc="$2" dest="$DF/$1.txt"
  if [[ -s "$dest" ]] && [[ $(wc -c <"$dest") -gt 400 ]]; then
    log "skip   $id  ($desc)"; return 0; fi
  if curl -fsSL --retry 2 --connect-timeout 15 -A "$UA" \
       -o "$dest.part" "https://fred.stlouisfed.org/data/${id}.txt"; then
    if [[ $(wc -c <"$dest.part") -gt 400 ]] && head -1 "$dest.part" | grep -qi .; then
      mv "$dest.part" "$dest"
      log "got    $id  $(awk 'END{print NR}' "$dest") lines  ($desc)"; return 0; fi
  fi
  rm -f "$dest.part"; log "MISS   $id  ($desc)  <- not available, note it"; return 1
}

head2 "1/3  PPI: construction output benchmarks"
get_fred PCU236211236211 "new industrial building construction"
get_fred PCU236221236221 "new warehouse building construction"
get_fred PCU236223236223 "new office building construction"
get_fred WPU801          "new nonresidential building construction"

head2 "2/3  PPI: input basket"
get_fred PCU335313335313 "switchgear and switchboard apparatus"
get_fred WPU1175         "switchgear, switchboard, industrial controls"
get_fred PCU3353133353133 "low-voltage panelboards"
get_fred WPU10260314     "copper wire and cable"
get_fred WPU10260332     "power wire and cable"
get_fred WPU1017         "steel mill products"
get_fred WPU101704       "hot rolled bars, plates, structural shapes"
get_fred PCU332312332312 "fabricated structural metal"
get_fred PCU327320327320 "ready-mix concrete"
get_fred PCU335311335311 "electric power and specialty transformers"
get_fred WPU117409       "power and distribution transformers"
get_fred PCU333415333415 "air-conditioning and refrigeration equipment"
get_fred PCU333618333618 "other engine equipment (gensets)"
get_fred PCU333611333611 "turbine and turbine generator sets"

head2 "3/3  Labour cost"
get_fred CES2000000003   "avg hourly earnings, construction, monthly SA"
get_fred CEU2000000003   "avg hourly earnings, construction, monthly NSA"
get_fred AHECONS         "avg hourly earnings, production workers, construction"
get_fred CIU2012300000000I "ECI total compensation, construction, quarterly"
get_fred ECICONWAG       "ECI wages and salaries, construction, quarterly SA"

head2 "Census: Value of Construction Put in Place"
# The 'Data center' line (private Office -> Data center, back to Jan 2014) is
# NOT in the monthly release tables. It lives only in these historical
# time-series workbooks.
for f in privsatime.xlsx privtime.xlsx private.xlsx privateha2.xlsx \
         privatepower.xlsx release.xlsx; do
  dest="$DC/$f"
  if [[ -s "$dest" ]]; then log "skip   $f"; continue; fi
  if curl -fsSL --retry 2 -A "$UA" -o "$dest.part" \
       "https://www.census.gov/construction/c30/xlsx/$f" \
     && [[ $(wc -c <"$dest.part") -gt 20000 ]]; then
    mv "$dest.part" "$dest"; log "got    $f ($(du -h "$dest" | cut -f1))"
  else rm -f "$dest.part"; log "MISS   $f"; fi
done

head2 "Done."
log "FRED series : $(ls -1 "$DF"/*.txt 2>/dev/null | wc -l | tr -d ' ')"
log "Census files: $(ls -1 "$DC"/*.xlsx 2>/dev/null | wc -l | tr -d ' ')"
log "On disk     : $(du -sh "$ROOT/data/raw" | cut -f1)"
printf "\nTell Claude it's finished.\n\n"
