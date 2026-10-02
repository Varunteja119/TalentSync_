"""
Quick standalone check that the JSearch integration in job_redirect.py
actually works with REAL credentials -- run this once after setup, before
trusting it inside the full app.

Setup:
    pip install requests
    Sign up free at https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
    Subscribe to the free BASIC plan (200 requests/month, hard-limited,
    no credit card required). Copy your RapidAPI key from the app's
    dashboard.

    setx JSEARCH_API_KEY "your-rapidapi-key"      (Windows, new terminal after)
    -- or for just this session --
    $env:JSEARCH_API_KEY = "your-rapidapi-key"

    Optional: $env:JSEARCH_COUNTRY = "in"   (defaults to "ae" for UAE)

Run:
    py -3.12 verify_jsearch_fallback.py
"""
import os
import sys

if not os.environ.get("JSEARCH_API_KEY"):
    print("ERROR: JSEARCH_API_KEY is not set.")
    print("Set it, then re-run this script.")
    sys.exit(1)

try:
    import requests  # noqa: F401
except ImportError:
    print("ERROR: the 'requests' package isn't installed.")
    print("Run: pip install requests")
    sys.exit(1)

import job_redirect as jr

print(f"Using JSearch country: {jr.JSEARCH_COUNTRY}")
print("Fetching live listings for 'Software Engineer' in Dubai...")
print()

print("(the live API is slow -- this can take up to ~45 seconds)")
listings = jr.fetch_live_job_listings("Software Engineer", location="Dubai", limit=10)

if listings:
    print(f"SUCCESS: found {len(listings)} live listing(s):")
    for job in listings:
        print(f"  - {job['title']} @ {job['company'] or 'unknown company'} "
              f"({job['location'] or 'location not specified'})")
        print(f"    {job['url']}")
    print()
    print("Your real JSearch integration is working.")
else:
    print("WARNING: no listings returned.")
    print(f"Actual reason reported by the code: {jr.LAST_ERROR or 'none (API answered but had zero results)'}")
    print()
    print("Possible causes:")
    print("  - Invalid or expired RapidAPI key")
    print("  - Free tier's 200 requests/month already used up this cycle")
    print("  - JSEARCH_COUNTRY set to something invalid (must be a valid "
          "ISO 3166-1 alpha-2 code, e.g. 'ae', 'in', 'us')")
    print("Note: this is NOT a crash -- the app is designed to fall back to "
          "static search links when this happens, which is correct behavior, "
          "but it means the live-listings feature isn't actually active yet.")
