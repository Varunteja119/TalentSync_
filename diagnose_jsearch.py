"""
Finds which JSearch search path actually works for your subscription and
shows the response structure. Never prints your API key.
Uses up to 2 of your 200 monthly requests.

    py -3.12 diagnose_jsearch.py
"""
import os
import sys

import requests

key = os.environ.get("JSEARCH_API_KEY", "")
print(f"JSEARCH_API_KEY set: {bool(key)}  (length: {len(key)})")
if not key:
    sys.exit("Key missing in this terminal. setx, then open a NEW terminal.")

headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
params = {"query": "software engineer jobs in dubai", "country": "ae", "num_pages": "1"}

for path in ("/search-v2", "/search"):
    url = "https://jsearch.p.rapidapi.com" + path
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
    except Exception as e:
        print(f"{path}: request failed: {type(e).__name__}: {e}")
        continue

    print(f"\n{path}: HTTP {r.status_code}")
    if r.status_code != 200:
        print("  body:", r.text[:200])
        continue

    body = r.json()
    print("  top-level keys:", list(body.keys()))
    data = body.get("data")
    if isinstance(data, list):
        print(f"  data is a list of {len(data)} jobs")
        first = data[0] if data else None
    elif isinstance(data, dict):
        print("  data is a dict with keys:", list(data.keys()))
        first = None
        for v in data.values():
            if isinstance(v, list) and v:
                first = v[0]
                print(f"  found a list of {len(v)} items inside data")
                break
    else:
        first = None
    if isinstance(first, dict):
        print("  first job keys:", sorted(first.keys())[:25])
        print("  title:", first.get("job_title"), "| employer:", first.get("employer_name"))
        print("  apply link present:", bool(first.get("job_apply_link")))
    print(f"\nUse {path}")
    break
