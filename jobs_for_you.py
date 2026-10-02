"""Unified resume-based job discovery for TalentSync."""
import streamlit as st


def _combined_search_links(roles, location, experience, job_redirect):
    """Create one search per board using the user's overall role fit."""
    # Search query uses the best-matching role titles together, not a separate
    # set of links/cards for every recommended role.
    query = " OR ".join(roles[:4])
    return job_redirect.generate_job_links(query, location, experience)


def render_jobs_for_you(profile, safe_match_roles, render_live_listings, job_redirect):
    st.subheader("Jobs for You")
    st.caption("Job searches tailored to your resume, experience, and preferred location.")

    experience_entries = profile.get("experience", []) or []
    years = job_redirect.estimate_experience_years(experience_entries) if job_redirect else 0.0
    levels = list(job_redirect.EXPERIENCE_LEVELS.keys()) if job_redirect else [
        "Any experience", "Fresher / no experience", "Entry level (under 3 years)", "Experienced (3+ years)"
    ]
    suggested = job_redirect.suggest_experience_label(years) if job_redirect else "Any experience"
    level = st.selectbox(
        "Experience level", levels,
        index=levels.index(suggested) if suggested in levels else 0,
        key="jobs_for_you_experience",
        help=f"Estimated from dated resume experience: about {years:.1f} years. Adjust if needed.",
    )
    experience_filter = job_redirect.EXPERIENCE_LEVELS[level] if job_redirect else None
    location = st.text_input(
        "Job location", value=str(profile.get("location") or "UAE").strip() or "UAE",
        key="jobs_for_you_location",
    ).strip()

    if not any(profile.get(k) for k in ("skills", "roles", "tools")):
        st.info("Add skills, tools, or target roles to your profile to get tailored job searches.")
        return

    with st.spinner("Matching jobs to your resume..."):
        recs = safe_match_roles(
            profile.get("skills", []), profile.get("roles", []), profile.get("tools", []),
            location=profile.get("location", ""), experience=None,
        )
    recs = [r for r in (recs or []) if str(r.get("role", "")).strip()]
    if not recs:
        st.info("No role recommendations are available yet. Update your profile and try again.")
        return

    top_roles = [str(r["role"]).strip() for r in recs[:4]]
    st.markdown("### Recommended job profile")
    st.write("Based on your resume, search across these related roles:")
    st.write(" · ".join(top_roles))
    st.caption(f"Resume experience estimate: {years:.1f} years · Location: {location or 'Not specified'}")

    if job_redirect is None:
        st.warning("Job search links are unavailable. Check job_redirect.py.")
        return

    links = _combined_search_links(top_roles, location, experience_filter, job_redirect)
    st.markdown("### Find matching openings")
    st.caption("Each link searches across your recommended role profile instead of splitting results by role.")
    cols = st.columns(3)
    for col, (label, key) in zip(cols, [("LinkedIn", "linkedin"), ("Naukri", "naukri"), ("Glassdoor", "glassdoor")]):
        url = links.get(key)
        if url:
            col.markdown(f"[{label} — view matching jobs]({url})")

    if job_redirect.live_listings_configured():
        if st.button("Fetch live matching jobs", key="jobs_for_you_fetch_live"):
            st.session_state["jobs_for_you_live_requested"] = True
        if st.session_state.get("jobs_for_you_live_requested"):
            # A single explicit user action; combine and deduplicate listings
            # from a small set of high-fit roles into one results list.
            all_jobs, seen = [], set()
            with st.spinner("Fetching live openings for your recommended roles..."):
                for role in top_roles[:4]:
                    for job in job_redirect.fetch_live_job_listings(
                        role, location, limit=15, experience=experience_filter
                    ):
                        identity = (job.get("title", "").lower(), job.get("company", "").lower(), job.get("url", ""))
                        if identity not in seen:
                            seen.add(identity)
                            all_jobs.append(job)
            if all_jobs:
                st.markdown(f"### Live matching openings ({len(all_jobs)})")
                for job in all_jobs[:30]:
                    company = f" @ {job['company']}" if job.get("company") else ""
                    place = f" ({job['location']})" if job.get("location") else ""
                    st.markdown(f"- [{job['title']}{company}]({job['url']}){place}")
            else:
                reason = getattr(job_redirect, "LAST_ERROR", None)
                st.caption(f"No live listings returned. {reason or 'Try the job-board links above.'}")
            st.session_state["jobs_for_you_live_requested"] = False
    else:
        st.caption("Live listings are not configured; the job-board links above are available without an API key.")
