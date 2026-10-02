"""
Measures how long JSearch actually takes for a few query variants, so we
pick a timeout from data instead of guessing. Never prints your API key.
Uses 3 of your 200 monthly requests. Can take a couple of minutes.

    py -3.12 diagnose_timing.py
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

trials = [
    ("A) lowercase query (what diagnose_jsearch.py used)",
     {"query": "software engineer jobs in dubai", "country": "ae", "num_pages": "1"}),
    ("B) mixed-case query (what the app sends)",
     {"query": "Software Engineer jobs in Dubai", "country": "ae", "num_pages": "1"}),
    ("C) lowercase + fields projection (smaller payload)",
     {"query": "data analyst jobs in dubai", "country": "ae", "num_pages": "1",
      "fields": "job_title,employer_name,job_apply_link,job_city,job_country,job_location"}),
]

for label, params in trials:
    print(f"\n{label}")
    start = time.time()
    try:
        r = requests.get(URL, headers=HEADERS, params=params, timeout=90)
        elapsed = time.time() - start
        body = r.json() if r.status_code == 200 else {}
        payload = body.get("data")
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else (payload or [])
        print(f"  HTTP {r.status_code} in {elapsed:.1f}s | jobs returned: {len(jobs)} | response size: {len(r.content)//1024} KB")
        if r.status_code != 200:
            print("  body:", r.text[:200])
    except Exception as e:
        print(f"  FAILED after {time.time() - start:.1f}s: {type(e).__name__}: {e}")
