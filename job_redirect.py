import re
import urllib.parse


def generate_job_links(role, location="India"):
    """
    Generate job search URLs for multiple platforms
    based on the recommended role and location.
    """

    role_clean = role.strip()
    location_clean = location.strip()

    role_encoded = urllib.parse.quote(role_clean)
    location_encoded = urllib.parse.quote(location_clean)

    search_query = urllib.parse.quote(f"{role_clean} {location_clean}")

    # Naukri and Internshala use hyphenated, lowercase URL slugs
    # ("data-scientist-jobs-in-bangalore"), not raw percent-encoded spaces
    # or stray punctuation. Percent-encoding a multi-word role (urllib.parse
    # .quote turns a space into "%20") produced a slug like
    # "Data%20Scientist-jobs-in-bangalore", which doesn't match either
    # site's URL convention. A role containing a slash (e.g. "AI/ML
    # Engineer") also needs that slash removed — left as-is, it becomes an
    # extra "/" in the URL path and splits the slug into the wrong segments.
    def _slugify(value):
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
        return slug.strip("-")

    role_slug = _slugify(role_clean)
    location_slug = _slugify(location_clean)

    links = {
        "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={search_query}",
        
        "naukri": f"https://www.naukri.com/{role_slug}-jobs-in-{location_slug}",
        
        "glassdoor": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={role_encoded}",
        
        "wellfound": f"https://wellfound.com/jobs?query={role_encoded}",
        
        "internshala": f"https://internshala.com/jobs/{role_slug}-jobs"
    }

    return links