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

    links = {
        "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={search_query}",
        
        "naukri": f"https://www.naukri.com/{role_encoded}-jobs-in-{location_encoded.lower()}",
        
        "glassdoor": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={role_encoded}",
        
        "wellfound": f"https://wellfound.com/jobs?query={role_encoded}",
        
        "internshala": f"https://internshala.com/jobs/{role_encoded}-jobs"
    }

    return links
