import datetime
import os
import re
import time
import urllib.parse

try:
    import requests
except ImportError:
    requests = None


# ==========================================
# Live job listings (JSearch, via RapidAPI)
# ==========================================
# Originally built against Adzuna, but Adzuna covers a fixed list of 19
# countries and has no UAE endpoint on any plan -- a hard coverage gap.
# JSearch (Google for Jobs aggregation) accepts any ISO 3166-1 alpha-2
# country code, UAE ("ae") included.
#
# Facts confirmed against the live API (not just docs):
#   * search lives at /search-v2 (plain /search returns 404 "Endpoint does
#     not exist" for current subscriptions)
#   * response shape is {"data": {"jobs": [...], "cursor": ...}}
#   * calls are SLOW and variable (observed ~5s to >25s), so the timeout
#     must be generous
#   * one page = up to 10 jobs = 1 request credit, so we show all 10
#   * free tier is 200 requests/month, so calls must be on-demand and
#     cached -- never fired per-role on every page render
#
# Design: static search links are always available and free. Live listings
# are opt-in (see fetch_live_job_listings, called on demand from the UI)
# and fetched only
# when the UI asks for one specific role.

JSEARCH_API_KEY = os.environ.get("JSEARCH_API_KEY")
JSEARCH_COUNTRY = os.environ.get("JSEARCH_COUNTRY", "ae")  # UAE by default
JSEARCH_HOST = "jsearch.p.rapidapi.com"
JSEARCH_URL = f"https://{JSEARCH_HOST}/search-v2"
JSEARCH_TIMEOUT = 45  # seconds; latency is highly variable (observed ~5s to >25s)
CACHE_TTL_SECONDS = 6 * 3600
# Fetch two pages per role to broaden results without making requests on reruns.
JSEARCH_NUM_PAGES = max(1, int(os.environ.get("JSEARCH_NUM_PAGES", "2")))

# Set to a human-readable reason whenever a live fetch fails, so callers
# and verification scripts can see WHY instead of just getting [].
LAST_ERROR = None

_cache = {}  # (role, location, country, limit) -> (timestamp, listings)


# JSearch's `job_requirements` filter only offers these buckets (there is no
# exact "0-2 years"; the nearest is "under 3 years"). The filter relies on
# how Google for Jobs parsed each posting, so postings that don't state
# requirements may be excluded -- expect fewer than 10 results.
EXPERIENCE_LEVELS = {
    "Any experience": None,
    "Fresher / no experience": "no_experience",
    "Entry level (under 3 years)": "under_3_years_experience",
    "Experienced (3+ years)": "more_than_3_years_experience",
}

# The experience filter is applied on Google's parsing of each posting and is
# noisy: a live test let "Senior Full Stack Developer" through the
# no-experience filter. For fresher/entry levels we drop titles that are
# clearly senior. Deliberately NOT included: "lead" and "staff", because
# "Staff Accountant" and "Lead Generation Intern" are genuine entry-level jobs.
_SENIOR_TITLE_RE = re.compile(
    r"\b(senior|sr|principal|director|vp|manager|expert)\b|\bhead of\b",
    re.IGNORECASE,
)
_JUNIOR_FILTERS = ("no_experience", "under_3_years_experience")
_GENERIC_JOB_TITLE_WORDS = {
    "junior", "jr", "senior", "sr", "principal", "lead", "staff",
    "intern", "trainee", "associate", "entry", "level", "expert",
    "engineer", "developer", "analyst", "scientist", "specialist",
    "manager", "consultant", "architect", "officer", "executive",
}
_ROLE_TITLE_ALIASES = {
    "ai": {"ai", "artificial", "intelligence"},
    "ml": {"ml", "machine", "learning"},
    "genai": {"genai", "generative", "ai", "artificial", "intelligence"},
    "llm": {"llm", "large", "language", "model", "models"},
}


def _role_title_is_relevant(role, title):
    """Reject search-engine results whose title is about a different domain.

    Matching is intentionally based on title terms, not the description, so
    incidental mentions of AI in an unrelated posting do not make it relevant.
    """
    role_words = set(re.findall(r"[a-z0-9]+", (role or "").lower()))
    title_words = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    role_domain = role_words - _GENERIC_JOB_TITLE_WORDS
    title_domain = title_words - _GENERIC_JOB_TITLE_WORDS
    if not role_domain:
        return True

    role_expanded = set(role_domain)
    title_expanded = set(title_domain)
    for acronym, aliases in _ROLE_TITLE_ALIASES.items():
        if role_domain & aliases:
            role_expanded |= aliases
        if title_domain & aliases:
            title_expanded |= aliases
    # Preserve role families such as Analyst vs Scientist: sharing only
    # the word "data" is not enough to treat those as the same job.
    role_family = role_words & {"analyst", "scientist", "manager", "developer", "engineer"}
    title_family = title_words & {"analyst", "scientist", "manager", "developer", "engineer"}
    if role_family and title_family and not (role_family & title_family):
        return False
    # Require a meaningful domain overlap; one domain term permits common
    # title variations (e.g. "AI Engineer" for "Generative AI Engineer").
    return bool(role_expanded & title_expanded)

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _parse_year_month(text):
    """'05/2025', 'May 2025', 'September 2022', '2024' or 'Present' -> (year, month)."""
    raw = (text or "").strip().lower()
    if not raw:
        return None
    if raw in ("present", "current", "now", "ongoing"):
        today = datetime.date.today()
        return today.year, today.month
    m = re.fullmatch(r"(0?[1-9]|1[0-2])/((?:19|20)\d{2})", raw)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.fullmatch(r"([a-z]{3,9})\.?,?\s+((?:19|20)\d{2})", raw)
    if m and m.group(1)[:3] in _MONTHS:
        return int(m.group(2)), _MONTHS[m.group(1)[:3]]
    m = re.fullmatch(r"((?:19|20)\d{2})", raw)
    if m:
        return int(m.group(1)), 1
    return None


def estimate_experience_years(experience):
    """
    Total years of work experience across a profile's experience entries.
    Understands both shapes the app stores: {"from": "05/2025", "to": "Present"}
    and {"date": "May 2025 - July 2025"}. Overlapping entries are merged so
    two concurrent jobs don't double-count. Unparseable entries are skipped.
    """
    intervals = []
    for exp in experience or []:
        start = end = None
        if exp.get("from"):
            start, end = _parse_year_month(exp.get("from")), _parse_year_month(exp.get("to"))
        if (start is None or end is None) and exp.get("date"):
            parts = re.split(r"\s*(?:-|\u2013|\u2014|\bto\b)\s*", str(exp["date"]), maxsplit=1)
            if len(parts) == 2:
                start, end = _parse_year_month(parts[0]), _parse_year_month(parts[1])
        if start and end:
            a, b = start[0] * 12 + start[1], end[0] * 12 + end[1]
            if b >= a:
                intervals.append((a, b))

    intervals.sort()
    total_months, cur_start, cur_end = 0, None, None
    for a, b in intervals:
        if cur_end is None or a > cur_end:
            if cur_end is not None:
                total_months += cur_end - cur_start
            cur_start, cur_end = a, b
        else:
            cur_end = max(cur_end, b)
    if cur_end is not None:
        total_months += cur_end - cur_start
    return total_months / 12.0


def suggest_experience_label(years):
    """Default selector label for an estimated number of years (user can override)."""
    if years < 1:
        return "Fresher / no experience"
    if years < 3:
        return "Entry level (under 3 years)"
    return "Experienced (3+ years)"


def live_listings_configured():
    """True if live listings can be attempted (package + API key present)."""
    return requests is not None and bool(JSEARCH_API_KEY)


def _parse_jobs(data, limit, experience=None, role=None):
    # /search-v2 returns {"data": {"jobs": [...], "cursor": ...}}; older
    # versions returned {"data": [...]}. Accept either.
    payload = data.get("data")
    if isinstance(payload, dict):
        jobs = payload.get("jobs") or []
    elif isinstance(payload, list):
        jobs = payload
    else:
        jobs = []

    listings = []
    seen = set()
    for job in jobs:
        if len(listings) >= limit:
            break
        title = (job.get("job_title") or "").strip()
        company = (job.get("employer_name") or "").strip()
        job_url = (job.get("job_apply_link") or "").strip()
        city = job.get("job_city") or ""
        country_code = job.get("job_country") or ""
        job_location = (job.get("job_location") or "").strip() or ", ".join(
            part for part in [city, country_code] if part
        )
        if not (title and job_url):
            continue
        if experience in _JUNIOR_FILTERS and _SENIOR_TITLE_RE.search(title):
            continue
        if role and not _role_title_is_relevant(role, title):
            continue
        dedupe_key = (title.lower(), company.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        listings.append({
            "title": title,
            "company": company,
            "url": job_url,
            "location": job_location,
        })
    return listings


def fetch_live_job_listings(role, location="", limit=10, experience=None):
    """
    Fetch real, current job postings from JSearch for one role.

    `experience` is a JSearch job_requirements value (see EXPERIENCE_LEVELS)
    or None for no filter.

    Returns [{"title", "company", "url", "location"}, ...]. Never raises:
    on any failure it returns [] and records the reason in LAST_ERROR.
    Successful results (including legitimately empty ones) are cached for
    CACHE_TTL_SECONDS so repeat requests don't spend free-tier quota;
    failures are not cached, so a retry actually retries.
    """
    global LAST_ERROR
    LAST_ERROR = None

    if requests is None:
        LAST_ERROR = "the 'requests' package is not installed"
        return []
    if not JSEARCH_API_KEY:
        LAST_ERROR = "JSEARCH_API_KEY is not set in this terminal"
        return []

    role_clean = role.strip()
    if not role_clean:
        return []

    location_clean = location.strip() if location else ""
    cache_key = (role_clean.lower(), location_clean.lower(), JSEARCH_COUNTRY, limit, experience)
    cached = _cache.get(cache_key)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    query = f"{role_clean} jobs in {location_clean}" if location_clean else f"{role_clean} jobs"
    headers = {
        "X-RapidAPI-Key": JSEARCH_API_KEY,
        "X-RapidAPI-Host": JSEARCH_HOST,
    }
    # JSearch's parsed experience buckets can exclude suitable junior postings.
    # For fresher/entry-level searches, broaden the API query and enforce only
    # the clear senior-title exclusion locally. Other experience selections
    # retain the requested API filter.
    params = {
        "query": query,
        "country": JSEARCH_COUNTRY,
        "num_pages": str(JSEARCH_NUM_PAGES),
    }
    if experience and experience not in _JUNIOR_FILTERS:
        params["job_requirements"] = experience

    try:
        response = requests.get(
            JSEARCH_URL, headers=headers, params=params, timeout=JSEARCH_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return []

    listings = _parse_jobs(data, limit, experience, role=role_clean)
    _cache[cache_key] = (time.time(), listings)
    return listings


# LinkedIn's job search URL supports an experience-level filter (f_E),
# publicly documented and stable: 1=Internship, 2=Entry level,
# 3=Associate, 4=Mid-Senior level, 5=Director, 6=Executive. Multiple
# values are comma-separated. Reuses the SAME job_requirements bucket
# values already used for JSearch (see EXPERIENCE_LEVELS above) so one
# dropdown choice drives both the live listings and this static link,
# rather than maintaining two separate experience vocabularies.
_LINKEDIN_EXPERIENCE_CODES = {
    "no_experience": "1,2",
    "under_3_years_experience": "2,3",
    "more_than_3_years_experience": "3,4",
}


def generate_job_links(role, location="India", experience=None):
    """
    Static job-search links for major platforms -- free and instant.

    `location` and `experience` (a job_requirements value from
    EXPERIENCE_LEVELS, or None) are folded into the LinkedIn link, since
    LinkedIn's search URL supports both. The other platforms only
    support a location-scoped slug/query, no experience filter.

    Live JSearch listings are NOT fetched here on purpose: match_roles()
    calls this once per qualifying role (up to ~39), and the live call is
    slow and quota-limited. Fetch live listings for a single role on
    demand via fetch_live_job_listings().
    """
    role_clean = role.strip()
    location_clean = location.strip()

    role_encoded = urllib.parse.quote(role_clean)
    search_query = urllib.parse.quote(f"{role_clean} {location_clean}".strip())

    def _slugify(value):
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
        return slug.strip("-")

    role_slug = _slugify(role_clean)
    location_slug = _slugify(location_clean)

    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={search_query}"
    if location_clean:
        linkedin_url += f"&location={urllib.parse.quote(location_clean)}"
    if experience in _LINKEDIN_EXPERIENCE_CODES:
        linkedin_url += f"&f_E={_LINKEDIN_EXPERIENCE_CODES[experience]}"

    links = {
        "linkedin": linkedin_url,
        "naukri": f"https://www.naukri.com/{role_slug}-jobs-in-{location_slug}",
        "glassdoor": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={role_encoded}",
        "wellfound": f"https://wellfound.com/jobs?query={role_encoded}",
        "internshala": f"https://internshala.com/jobs/{role_slug}-jobs",
        "live_listings": [],
    }

    return links
