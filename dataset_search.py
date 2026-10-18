"""Documented search for public organoid datasets that fit the de-escalation benchmark.

A dataset qualifies when it holds (1) human tumor organoids from at least three
patients, (2) at least two burden levels of one agent or regimen within each patient
(dose, duration, or schedule), (3) a per-well state read before the arms diverge and
a final efficacy endpoint on the same wells, and (4) well-level data that can be
downloaded without a request to the authors.

The script runs every query below and writes the query log (queries.csv) and the
deduplicated records (records.csv) to out/search. Abstracts and dataset descriptions
go to a cache for screening and are not committed.

Screening (run of 2026-10-17/18) went title -> abstract -> full text for articles and
description -> deposit and article for repository records. Claude agents applied the
codes below; records sharing a title (preprint and journal versions) took one full-text
decision, and agents judged data files from listings and descriptions without opening
them. C4 was verified directly for Schuster (prep.py reads its per-well series from the
source-data workbook), Wells 2026 (prep.py reads its per-organoid archive), and the SCOPE
preprint (its dose_response_sn38_5fu_analysis.mat holds per-organoid live/dead status
and area on days 0, 3, and 5). screening.csv holds one row per record with the stage it
left at; studies.csv groups the full-text records by study.
  title and abstract  E1 not human tumor organoids (animal, normal tissue, stem-cell
                      derived, cell lines, slices, xenografts); E2 no treatment-response
                      experiment; E3 response read once per culture, or repeated readings
                      from separate destructive samples; E4 no original screen (review,
                      perspective, methods chapter, editorial, erratum); E5 dataset or
                      repository without treatment-response data (code, models, images
                      for segmentation)
  full text and data  C1-C4 above, each judged met, unmet, or unclear
"""
import argparse
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
# Literature: titles and abstracts in Europe PMC (PubMed, PMC, and preprint servers).
# Organoid cultures from adult stem cells date from 2009.
MODEL = "TITLE_ABS:(organoid* OR tumoroid*)"
CANCER = ("TITLE_ABS:(cancer* OR tumor* OR tumour* OR carcinoma* OR neoplas* OR malignan* OR oncolog* "
          "OR adenocarcinoma* OR glioblastoma* OR glioma* OR sarcoma* OR melanoma* OR leukem* OR lymphoma*)")
TREATMENT = ("TITLE_ABS:(drug* OR chemotherap* OR treatment* OR therap* OR dose* OR dosing OR schedul* "
             "OR inhibitor* OR compound* OR agent*)")
REPEATED = ('TITLE_ABS:("time-lapse" OR "time lapse" OR timelapse OR "live-cell" OR "live cell" OR "live imaging" '
            'OR longitudinal* OR kinetic* OR "real-time" OR "real time" OR "time course" OR "time-course" '
            'OR "time-resolved" OR timepoint* OR "time point" OR "time points" OR track* OR dynamic* OR temporal* '
            'OR "non-destructive" OR nondestructive OR "label-free" OR "over time" OR sequential* OR microfluidic* '
            'OR "on-a-chip" OR "on-chip")')
YEARS = "PUB_YEAR:[2009 TO 2026]"
LITERATURE = f"{MODEL} AND {CANCER} AND {TREATMENT} AND {REPEATED} AND {YEARS}"
# Full text: papers that state a public data location next to a repeated-readout assay.
FULL_TEXT = ('(organoid* OR tumoroid*) AND ("patient-derived" OR "patient derived") '
             'AND ("time-lapse" OR "time lapse" OR incucyte OR "live-cell imaging" OR "live cell imaging" OR kinetic) '
             'AND ("source data" OR zenodo OR figshare OR "github.com" OR dryad OR "mendeley data" '
             'OR "image data resource" OR "bioimage archive") AND (drug* OR chemotherap*) ' + f"AND {YEARS}")
# The two included studies seed forward citation search and serve as the recall check.
INCLUDED = {"MED:33077832": "Schuster 2020", "MED:34993134": "Spiller 2021"}
ZENODO = ("(organoid OR organoids OR tumoroid OR tumoroids) "
          "AND (drug OR drugs OR chemotherapy OR treatment OR dose OR inhibitor)")
# Figshare search handles two-term conjunctions reliably (three-term ones return nothing), and
# journal supplements it mirrors are covered through the articles.
FIGSHARE = ('"organoid" AND "drug"', '"organoid" AND "chemotherapy"', '"organoid" AND "dose"', '"tumoroid" AND "drug"')
BIOSTUDIES = '"organoid" AND "drug" AND ("time-lapse" OR "time course" OR "live")'
GITHUB = ("organoid drug", "organoid response", "organoid viability", "organoid time-lapse",
          "organoid tracking", "tumoroid")


def get_json(url: str, params: dict | None = None, body: dict | None = None):
    if params:
        url = f"{url}?{urlencode(params)}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": "ooc-dataset-search", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(5):
        try:
            with urlopen(Request(url, data=data, headers=headers), timeout=120) as r:
                return json.load(r)
        except Exception:  # transient network or rate-limit errors
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


def plain(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def article(r: dict, source: str) -> dict:
    key = f"{r.get('source')}:{r.get('id')}"
    return {"key": key, "kind": "article", "source": source, "doi": r.get("doi", ""), "year": r.get("pubYear", ""),
            "title": plain(r.get("title")),
            "venue": r.get("journalInfo", {}).get("journal", {}).get("title") or r.get("bookOrReportDetails", {}).get("publisher", ""),
            "url": f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}", "text": plain(r.get("abstractText"))}


def epmc_search(query: str, source: str) -> tuple[int, list[dict]]:
    out, cursor, hits = [], "*", 0
    while True:
        d = get_json(f"{EPMC}/search", {"query": query, "format": "json", "resultType": "core",
                                         "pageSize": 1000, "cursorMark": cursor})
        hits = d["hitCount"]
        out += [article(r, source) for r in d["resultList"]["result"]]
        if d.get("nextCursorMark") in (None, cursor) or len(out) >= hits:
            return hits, out
        cursor = d["nextCursorMark"]


def epmc_citing(key: str) -> tuple[int, list[dict]]:
    src, ext = key.split(":")
    ids, page = [], 1
    while True:
        d = get_json(f"{EPMC}/{src}/{ext}/citations", {"format": "json", "pageSize": 1000, "page": page})
        ids += [(c["source"], c["id"]) for c in d["citationList"]["citation"]]
        if len(ids) >= d["hitCount"] or not d["citationList"]["citation"]:
            break
        page += 1
    out = []
    for i in range(0, len(ids), 50):
        q = " OR ".join(f"(EXT_ID:{e} AND SRC:{s})" for s, e in ids[i:i + 50])
        out += epmc_search(q, f"cites {INCLUDED[key]}")[1]
    return len(ids), out


def dataset(source: str, rid, title, text, doi="", year="", url="") -> dict:
    return {"key": f"{source}:{rid}", "kind": "dataset", "source": source, "doi": doi or "", "year": str(year or "")[:4],
            "title": plain(title), "venue": source, "url": url, "text": plain(text)[:4000]}


def zenodo() -> tuple[int, list[dict]]:
    out, page = [], 1
    while True:
        d = get_json("https://zenodo.org/api/records", {"q": ZENODO, "size": 25, "page": page})  # anonymous page limit
        recs = d["hits"]["hits"]
        out += [dataset("zenodo", r["id"], r["metadata"].get("title"), r["metadata"].get("description"),
                        r.get("doi"), r["metadata"].get("publication_date"), r["links"].get("self_html", ""))
                for r in recs]
        if not recs or len(out) >= d["hits"]["total"]:
            return d["hits"]["total"], out
        page += 1


def figshare() -> tuple[int, list[dict]]:
    seen = {}
    for q in FIGSHARE:
        page = 1
        while True:
            recs = get_json("https://api.figshare.com/v2/articles/search", body={"search_for": q, "page_size": 100, "page": page})
            seen |= {r["id"]: r for r in recs}
            if len(recs) < 100:
                break
            page += 1
    out = []
    for rid, r in seen.items():
        full = get_json(f"https://api.figshare.com/v2/articles/{rid}")
        out.append(dataset("figshare", rid, r.get("title"), full.get("description"), r.get("doi"),
                           r.get("published_date"), r.get("url_public_html")))
    return len(out), out


def dryad() -> tuple[int, list[dict]]:
    out, page = [], 1
    while True:
        d = get_json("https://datadryad.org/api/v2/search", {"q": "organoid", "per_page": 100, "page": page})
        recs = d.get("_embedded", {}).get("stash:datasets", [])
        out += [dataset("dryad", r.get("identifier"), r.get("title"), r.get("abstract"), r.get("identifier", "").replace("doi:", ""),
                        r.get("publicationDate"), f"https://datadryad.org/dataset/{r.get('identifier')}") for r in recs]
        if not recs or len(out) >= d.get("total", 0):
            return d.get("total", len(out)), out
        page += 1


def biostudies(collection: str, query: str) -> tuple[int, list[dict]]:
    base = "https://www.ebi.ac.uk/biostudies/api/v1" + (f"/{collection}" if collection else "")
    out, page = [], 1
    while True:
        d = get_json(f"{base}/search", {"query": query, "pageSize": 100, "page": page})
        hits = d.get("hits", [])
        out += [dataset("bioimage_archive" if collection else "biostudies", h["accession"], h.get("title"), h.get("content"),
                        year=h.get("release_date"), url=f"https://www.ebi.ac.uk/biostudies/studies/{h['accession']}")
                for h in hits]
        if not hits or len(out) >= d.get("totalHits", 0):
            return d.get("totalHits", len(out)), out
        page += 1


def idr() -> tuple[int, list[dict]]:
    studies = []
    for kind in ("screens", "projects"):
        studies += get_json(f"https://idr.openmicroscopy.org/api/v0/m/{kind}/", {"limit": 500})["data"]
    hits = [s for s in studies if "organoid" in (s.get("Name", "") + " " + s.get("Description", "")).lower()]
    return len(hits), [dataset("idr", s["@id"], s.get("Name"), s.get("Description"),
                               url=f"https://idr.openmicroscopy.org/search/?query=Name:{s.get('Name', '')}") for s in hits]


def github() -> tuple[int, list[dict]]:
    seen = {}
    for q in GITHUB:
        d = get_json("https://api.github.com/search/repositories", {"q": q, "per_page": 100})
        for r in d.get("items", []):
            seen[r["full_name"]] = dataset("github", r["full_name"], r["full_name"], r.get("description"),
                                           year=r.get("created_at"), url=r.get("html_url"))
        time.sleep(7)  # unauthenticated search allows 10 requests per minute
    return len(seen), list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out/search")
    parser.add_argument("--cache", required=True, help="where abstracts and descriptions are written for screening")
    args = parser.parse_args()
    outdir, cache = Path(args.outdir), Path(args.cache)
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    runs = {
        ("Europe PMC titles and abstracts", LITERATURE): lambda: epmc_search(LITERATURE, "europepmc_tiabs"),
        ("Europe PMC full text", FULL_TEXT): lambda: epmc_search(FULL_TEXT, "europepmc_fulltext"),
        **{(f"Europe PMC citations of {name}", key): (lambda k=key: epmc_citing(k)) for key, name in INCLUDED.items()},
        ("Zenodo", ZENODO): zenodo,
        ("Figshare", " | ".join(FIGSHARE)): figshare,
        ("Dryad", "organoid"): dryad,
        ("BioImage Archive", "organoid"): lambda: biostudies("BioImages", "organoid"),
        ("BioStudies", BIOSTUDIES): lambda: biostudies("", BIOSTUDIES),
        ("Image Data Resource", "organoid in study name or description"): idr,
        ("GitHub repositories", " | ".join(GITHUB)): github,
    }
    log, records = [], []
    for (source, query), run in runs.items():
        hits, recs = run()
        log.append({"source": source, "query": query, "run_date": date.today().isoformat(), "hits": hits, "retrieved": len(recs)})
        records += recs
        print(f"{source}: {hits} hits, {len(recs)} retrieved")

    df = pd.DataFrame(records)
    merged = (df.groupby("key", sort=False)
              .agg(kind=("kind", "first"), sources=("source", lambda s: ";".join(dict.fromkeys(s))), doi=("doi", "first"),
                   year=("year", "first"), title=("title", "first"), venue=("venue", "first"), url=("url", "first"),
                   text=("text", "first"))
              .reset_index())
    recall = {name: key in set(merged.loc[merged["sources"].str.contains("europepmc_tiabs"), "key"])
              for key, name in INCLUDED.items()}
    print("recall check (titles and abstracts):", recall)
    pd.DataFrame(log).to_csv(outdir / "queries.csv", index=False)
    merged.drop(columns="text").to_csv(outdir / "records.csv", index=False)
    merged.to_json(cache / "records_text.jsonl", orient="records", lines=True)
    print(f"{len(merged)} unique records ({(merged['kind'] == 'article').sum()} articles, "
          f"{(merged['kind'] == 'dataset').sum()} datasets)")


if __name__ == "__main__":
    main()
