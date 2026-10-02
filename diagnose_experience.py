"""
Tests whether JSearch's experience filter (job_requirements) really works on
the /search-v2 endpoint, and how it changes the results. Never prints your key.
Uses 3 of your monthly requests; can take a minute or two.

    py -3.12 diagnose_experience.py
"""
import os
import sys
import time

import requests

key = os.environ.get("JSEARCH_API_KEY", "")
if not key:
    sys.exit("JSEARCH_API_KEY missing in this terminal.")

URL = "https://jsearch.p.rapidapi.com/search-v2"
HEADERS = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
BASE = {"query": "software engineer jobs in dubai", "country": "ae", "num_pages": "1"}

trials = [
    ("No filter (baseline)", None),
    ("Fresher: job_requirements=no_experience", "no_experience"),
    ("Entry: job_requirements=under_3_years_experience", "under_3_years_experience"),
]

for label, req in trials:
    params = dict(BASE)
    if req:
        params["job_requirements"] = req
    print(f"\n{label}")
    start = time.time()
    try:
        r = requests.get(URL, headers=HEADERS, params=params, timeout=90)
    except Exception as e:
        print(f"  FAILED after {time.time() - start:.1f}s: {type(e).__name__}: {e}")
        continue
    elapsed = time.time() - start
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} in {elapsed:.1f}s -> body: {r.text[:200]}")
        continue
    payload = r.json().get("data")
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else (payload or [])
    print(f"  HTTP 200 in {elapsed:.1f}s | jobs returned: {len(jobs)}")
    for j in jobs[:5]:
        print("   -", (j.get("job_title") or "")[:70], "@", (j.get("employer_name") or "")[:30])
