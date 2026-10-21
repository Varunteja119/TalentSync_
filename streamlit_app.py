import streamlit as st
import os
import json
import re
import datetime
from auth import login, signup, update_profile
import importlib

# Load feature-specific modules only when their functionality is first used.
# This keeps the initial login render from importing NLP/ML and document tooling.
def _lazy_call(module_name, function_name, *args, **kwargs):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(*args, **kwargs)

def extract_basic_info(*args, **kwargs):
    return _lazy_call("utils", "extract_basic_info", *args, **kwargs)

def extract_education_experience(*args, **kwargs):
    return _lazy_call("utils", "extract_education_experience", *args, **kwargs)

def extract_text(*args, **kwargs):
    return _lazy_call("text_extractor", "extract_text", *args, **kwargs)

def generate_resume(*args, **kwargs):
    return _lazy_call("resume_generator", "generate_resume", *args, **kwargs)

def convert_profile_to_resume_format(*args, **kwargs):
    return _lazy_call("resume_generator", "convert_profile_to_resume_format", *args, **kwargs)

def extract_entities(*args, **kwargs):
    return _lazy_call("skill_extractor", "extract_entities", *args, **kwargs)

def match_roles(*args, **kwargs):
    return _lazy_call("role_matcher", "match_roles", *args, **kwargs)

def run_pipeline(*args, **kwargs):
    return _lazy_call("pipeline", "run_pipeline", *args, **kwargs)

try:
    import job_redirect
except Exception:
    job_redirect = None

try:
    from jobs_for_you import render_jobs_for_you
except Exception:
    render_jobs_for_you = None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="TalentSync", layout="wide")

# =========================
# STYLING (NO LOGIC CHANGE)
# =========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #141a24 0%, #0E1117 45%, #0B0E13 100%);
    color: #E5E7EB;
}

h1, h2, h3 {
    color: #E5E7EB;
    letter-spacing: 0.2px;
}

.stMetric {
    background: rgba(28, 31, 38, 0.95);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #4F8CFF 0%, #3B82F6 100%);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #3D79E8 0%, #2F6FD9 100%);
}

.stTextInput>div>div>input {
    background: #1C1F26;
    color: #E5E7EB;
    border: 1px solid #2D3748;
    border-radius: 8px;
}

.stTextArea textarea {
    background: #1C1F26;
    color: #E5E7EB;
    border: 1px solid #2D3748;
    border-radius: 8px;
}

.stProgress > div > div > div > div {
    background-color: #22C55E;
}

.stSelectbox label, .stTextInput label, .stTextArea label, .stRadio label, .stCaption {
    color: #C9D1D9 !important;
}

.ts-brand {
    padding-top: 0.8rem;
}

.ts-logo {
    font-size: 2.2rem;
    font-weight: 800;
    color: #E5E7EB;
    margin-bottom: 0.6rem;
}

.ts-headline {
    font-size: 1.4rem;
    line-height: 1.25;
    font-weight: 700;
    font-style: italic;
    color: #E5E7EB;
    margin-bottom: 0.9rem;
}

.ts-body {
    color: #B8C1CC;
    font-size: 1.02rem;
    line-height: 1.7;
    max-width: 760px;
    margin-bottom: 1rem;
}

.ts-bullets {
    color: #D1D5DB;
    line-height: 1.8;
    font-size: 0.98rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #1C1F26;
    border: 1px solid #2D3748;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "view" not in st.session_state:
    st.session_state.view = "Dashboard"
if "workspace_nav" not in st.session_state:
    st.session_state.workspace_nav = st.session_state.view
if "username" not in st.session_state:
    st.session_state.username = ""
if "profile_edit_mode" not in st.session_state:
    st.session_state.profile_edit_mode = False


def safe_match_roles(skills, roles, tools, location="", experience=None):
    """Return cached role matches so page navigation doesn't rerun SBERT."""

    # Include all inputs that affect the matcher. Dashboard and Role Matches
    # can then reuse the same result, while profile/experience edits invalidate it.
    cache_key = json.dumps(
        {
            "skills": skills or [],
            "roles": roles or [],
            "tools": tools or [],
            "location": location or "",
            "experience": experience,
        },
        sort_keys=True,
        default=str,
    )
    cache = st.session_state.setdefault("_role_match_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    try:
        if not st.session_state.get("_matcher_warmed_up"):
            with st.spinner("Loading the matching model (first time only, can take a minute)..."):
                result = match_roles(
                    skills or [], roles or [], tools or [],
                    candidate_location=location,
                    candidate_experience=experience,
                )
            st.session_state["_matcher_warmed_up"] = True
        else:
            result = match_roles(
                skills or [], roles or [], tools or [],
                candidate_location=location,
                candidate_experience=experience,
            )
        # Cache only successful results; keep memory bounded for edited profiles.
        if result is not None:
            cache[cache_key] = result
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
            return result
    except Exception as exc:
        st.warning(f"Role matching couldn't be completed: {exc}")
    return []

def _sync_workspace_view():
    """Keep the rendered page aligned with the keyed sidebar radio."""
    st.session_state.view = st.session_state.workspace_nav


def _navigate_workspace(view_name):
    """Programmatic navigation, invoked as a widget callback before rerun."""
    st.session_state.workspace_nav = view_name
    st.session_state.view = view_name


def logout_user():
    for key in ["logged_in", "profile", "view", "workspace_nav", "username", "profile_edit_mode"]:
        st.session_state.pop(key, None)
    st.session_state.logged_in = False
    st.session_state.view = "Dashboard"


def render_experience_selector(profile, scope):
    """One experience-level selector per page (not per role).

    Defaults to a level estimated from the resume's experience dates; the
    user can override it. Returns the JSearch job_requirements value, or
    None for "any experience" / when live listings aren't available.
    """
    if job_redirect is None or not job_redirect.live_listings_configured():
        return None

    labels = list(job_redirect.EXPERIENCE_LEVELS.keys())
    years = job_redirect.estimate_experience_years(profile.get("experience", []))
    default_label = job_redirect.suggest_experience_label(years)
    choice = st.selectbox(
        "Experience level for live openings",
        labels,
        index=labels.index(default_label),
        key=f"live_exp_{scope}",
        help=(
            f"Suggested from your resume: about {years:.1f} years of experience. "
            "Job boards only offer these buckets (there is no exact 0-2 years), "
            "and postings that don't state requirements may be filtered out."
        ),
    )
    return job_redirect.EXPERIENCE_LEVELS[choice]


def render_live_listings(role, location, key, experience=None):
    """On-demand live job listings for ONE role.

    Deliberately button-triggered: the live API is slow (up to ~45s) and the
    free tier is 200 requests/month, so it must never fire automatically for
    every role on every Streamlit rerun. Results are cached in job_redirect.
    The experience level is part of the state key, so changing the level
    closes old results and requires a fresh click rather than silently
    re-fetching every role that was already opened.
    """
    if job_redirect is None or not job_redirect.live_listings_configured():
        return

    state_key = f"live_open_{key}_{experience}"
    if st.button("Show live openings", key=f"live_btn_{key}_{experience}"):
        st.session_state[state_key] = True

    if st.session_state.get(state_key):
        with st.spinner("Fetching live listings (can take up to ~45s)..."):
            jobs = job_redirect.fetch_live_job_listings(
                role, location or "UAE", experience=experience
            )
        if jobs:
            st.write("**Open roles right now:**")
            for job in jobs:
                company_part = f" @ {job['company']}" if job.get("company") else ""
                location_part = f" ({job['location']})" if job.get("location") else ""
                st.markdown(f"- [{job['title']}{company_part}]({job['url']}){location_part}")
        else:
            reason = job_redirect.LAST_ERROR
            if reason:
                msg = f"No live listings came back. ({reason}) Click the button again to retry."
            elif experience:
                msg = ("No openings matched this experience filter right now. "
                       "Try 'Any experience', or use the search links below.")
            else:
                msg = "No live listings came back. Try the search links below."
            st.caption(msg)
            # Failures aren't cached, so leaving the flag on would re-fetch
            # (another wait and another credit) on EVERY Streamlit rerun.
            # Require an explicit click to try again.
            st.session_state[state_key] = False


def render_dashboard(profile):
    display_username = st.session_state.get("username", profile.get("name", "Candidate"))
    st.subheader(f"Welcome back, {display_username}")

    # Experience selector runs BEFORE matching (not after, as it did
    # originally) so the chosen level can actually inform the job_links
    # baked into each recommendation, not just the on-demand live listings.
    live_experience = render_experience_selector(profile, "dash")

    recommendation_data = safe_match_roles(
        profile.get("skills", []),
        profile.get("roles", []),
        profile.get("tools", []),
        location=profile.get("location", ""),
        experience=live_experience,
    )

    total_skills = len(profile.get("skills", []))
    total_roles = len(profile.get("roles", []))
    total_tools = len(profile.get("tools", []))
    total_education = len(profile.get("education", []))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Skills", total_skills)
    metric_cols[1].metric("Roles", total_roles)
    metric_cols[2].metric("Tools", total_tools)
    metric_cols[3].metric("Education", total_education)

    completion_fields = ["name", "email", "phone", "location", "linkedin"]
    completion_score = int(
        sum(1 for field in completion_fields if profile.get(field)) / len(completion_fields) * 100
    )

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown("### Recommended Roles")
        for rec in recommendation_data[:3]:
            with st.container(border=True):
                row_1, row_2 = st.columns([3, 1])
                with row_1:
                    st.markdown(f"**{rec['role']}**")
                    st.write(rec["fit_label"])
                with row_2:
                    st.metric("Score", f"{int(rec['final_score'] * 100)}%")
                st.progress(rec["final_score"])
                st.caption(
                    f"Core matched: {rec['core_matched']}/{rec['core_total']} | "
                    f"Optional matched: {rec['optional_matched']}/{rec['optional_total']}"
                )
                # Display hybrid scoring components
                score_cols = st.columns(3)
                with score_cols[0]:
                    st.metric("Semantic", f"{int(rec['semantic_similarity']*100)}%", delta=None)
                with score_cols[1]:
                    st.metric("Skill Overlap", f"{int(rec['skill_overlap']*100)}%", delta=None)
                with score_cols[2]:
                    st.metric("Role Bonus", f"{int(rec['role_bonus']*100)}%", delta=None)
                if rec.get("matched_core_skills"):
                    st.write("**Matched core skills:** " + ", ".join(rec["matched_core_skills"]))
                if rec.get("matched_optional_skills"):
                    st.write("**Matched optional skills:** " + ", ".join(rec["matched_optional_skills"]))
                render_live_listings(rec["role"], profile.get("location", ""), f"dash_{rec['role']}", live_experience)
                st.markdown(f"[Apply on LinkedIn]({rec['job_links']['linkedin']})")

    with right_col:
        st.markdown("### Profile Strength")
        st.progress(completion_score / 100)
        st.caption(f"Profile completion: {completion_score}%")

        st.markdown("### Quick Snapshot")
        st.write(f"**Location:** {profile.get('location', 'Not set')}")
        st.write(f"**LinkedIn:** {profile.get('linkedin', 'Not set')}")
        st.write(f"**Education entries:** {len(profile.get('education', []))}")
        st.write(f"**Experience entries:** {len(profile.get('experience', []))}")

        st.button(
            "Open Resume Builder",
            on_click=_navigate_workspace,
            args=("Resume Builder",),
        )


def render_profile_view(profile):
    st.subheader("Your Profile")

    edit_col, _ = st.columns([1, 5])
    with edit_col:
        if not st.session_state.profile_edit_mode:
            if st.button("✏️ Edit Profile"):
                st.session_state.profile_edit_mode = True
                st.rerun()
        else:
            if st.button("❌ Cancel Edit"):
                st.session_state.profile_edit_mode = False
                st.rerun()

    if st.session_state.profile_edit_mode:
        st.markdown("### Edit Profile")

        with st.form("profile_edit_form"):
            st.markdown("#### Basic Info")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", value=profile.get("name", ""))
                email = st.text_input("Email", value=profile.get("email", ""))
                phone = st.text_input("Phone", value=profile.get("phone", ""))
            with col2:
                location = st.text_input("Location", value=profile.get("location", ""))
                linkedin = st.text_input("LinkedIn", value=profile.get("linkedin", ""))
                dob = st.text_input("DOB", value=str(profile.get("dob", "")))

            st.markdown("#### Professional Summary")
            summary = st.text_area(
                "Summary",
                value=profile.get("summary", ""),
                placeholder="Write a short professional summary to use in your resume.",
            )

            st.markdown("#### Skills, Roles, Tools")
            skills_csv = st.text_area("Skills (comma-separated)", value=", ".join(profile.get("skills", [])))
            roles_csv = st.text_area("Roles (comma-separated)", value=", ".join(profile.get("roles", [])))
            tools_csv = st.text_area("Tools (comma-separated)", value=", ".join(profile.get("tools", [])))

            st.markdown("#### Education")
            education = profile.get("education", [])
            edu_count = st.number_input(
                "Education entries",
                min_value=0,
                max_value=10,
                value=len(education),
                step=1,
                key="edit_edu_count",
            )
            edited_education = []
            for i in range(edu_count):
                current = education[i] if i < len(education) else {}
                st.markdown(f"**Education {i + 1}**")
                degree = st.text_input(f"Degree {i+1}", value=current.get("degree", ""), key=f"edit_degree_{i}")
                field = st.text_input(f"Field {i+1}", value=current.get("field", ""), key=f"edit_field_{i}")
                college = st.text_input(f"College {i+1}", value=current.get("college", ""), key=f"edit_college_{i}")
                date = st.text_input(
                    f"Year/Duration {i+1}",
                    value=current.get("date", "") or f"{current.get('from', '')} - {current.get('to', '')}".strip(" -"),
                    key=f"edit_date_{i}",
                )
                edited_education.append({
                    "degree": degree,
                    "field": field,
                    "college": college,
                    "date": date,
                })

            st.markdown("#### Experience")
            experience = profile.get("experience", [])
            exp_count = st.number_input(
                "Experience entries",
                min_value=0,
                max_value=20,
                value=len(experience),
                step=1,
                key="edit_exp_count",
            )
            edited_experience = []
            for i in range(exp_count):
                current = experience[i] if i < len(experience) else {}
                st.markdown(f"**Experience {i + 1}**")
                role = st.text_input(f"Role {i+1}", value=current.get("role", ""), key=f"edit_role_{i}")
                company = st.text_input(f"Company {i+1}", value=current.get("company", ""), key=f"edit_company_{i}")
                date = st.text_input(
                    f"Date {i+1}",
                    value=current.get("date", "") or f"{current.get('from', '')} - {current.get('to', '')}".strip(" -"),
                    key=f"edit_exp_date_{i}",
                )
                description = st.text_area(
                    f"Description {i+1}",
                    value=current.get("description", ""),
                    key=f"edit_desc_{i}",
                )
                edited_experience.append({
                    "role": role,
                    "company": company,
                    "date": date,
                    "description": description,
                })

            submitted = st.form_submit_button("💾 Save Profile")

        if submitted:
            def parse_csv(value):
                return [item.strip() for item in value.split(",") if item.strip()]

            manual_skills = parse_csv(skills_csv)
            manual_roles = parse_csv(roles_csv)
            manual_tools = parse_csv(tools_csv)

            # Re-run entity extraction on the latest skill text so matching always reflects edits.
            extracted_entities = extract_entities(", ".join(manual_skills)) if manual_skills else {"skills": [], "roles": [], "tools": []}

            merged_skills = sorted(set((manual_skills or []) + extracted_entities.get("skills", [])))
            merged_roles = sorted(set((manual_roles or []) + extracted_entities.get("roles", [])))
            merged_tools = sorted(set((manual_tools or []) + extracted_entities.get("tools", [])))

            updated_profile = {
                "name": name,
                "dob": dob,
                "summary": summary,
                "phone": phone,
                "email": email,
                "location": location,
                "linkedin": linkedin,
                "education": edited_education,
                "experience": edited_experience,
                "projects": profile.get("projects", []),
                "skills": merged_skills,
                "roles": merged_roles,
                "tools": merged_tools,
            }

            if update_profile(st.session_state.username, updated_profile):
                st.session_state.profile = updated_profile
                st.session_state.profile_edit_mode = False
                st.success("Profile updated successfully ✅")
                st.rerun()
            else:
                st.error("Could not save profile. Please login again.")
        return

    info_cols = st.columns(3)
    info_cols[0].metric("Skills", len(profile.get("skills", [])))
    info_cols[1].metric("Roles", len(profile.get("roles", [])))
    info_cols[2].metric("Tools", len(profile.get("tools", [])))

    with st.container(border=True):
        st.markdown("### Basic Info")
        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.write(f"**Name:** {profile.get('name', '')}")
            st.write(f"**Email:** {profile.get('email', '')}")
            st.write(f"**Phone:** {profile.get('phone', '')}")
        with detail_cols[1]:
            st.write(f"**Location:** {profile.get('location', '')}")
            st.write(f"**LinkedIn:** {profile.get('linkedin', '')}")
            st.write(f"**DOB:** {profile.get('dob', '')}")

    with st.container(border=True):
        st.markdown("### Education")
        if profile.get("education"):
            for idx, edu in enumerate(profile.get("education", []), 1):
                degree = edu.get("degree", "").strip()
                field = edu.get("field", "").strip()
                college = edu.get("college", "").strip()
                date_text = edu.get("date", "").strip()
                if not date_text:
                    from_d = str(edu.get("from", "")).strip()
                    to_d = str(edu.get("to", "")).strip()
                    date_text = f"{from_d} - {to_d}".strip(" -") if (from_d or to_d) else ""

                # Compose a clear degree title from available fields.
                if degree and field:
                    degree_title = f"{degree} in {field}"
                else:
                    degree_title = degree or field or "Education"

                with st.container(border=True):
                    st.markdown(f"**Entry {idx}: {degree_title}**")
                    st.write(f"**College:** {college if college else 'Not provided'}")
                    st.write(f"**Year/Duration:** {date_text if date_text else 'Not provided'}")
        else:
            st.write("No education entries yet.")

    with st.container(border=True):
        st.markdown("### Professional Summary")
        summary_text = str(profile.get("summary", "")).strip()
        if summary_text:
            st.write(summary_text)
        else:
            st.write("No summary added yet.")

    with st.container(border=True):
        st.markdown("### Skills & Expertise")

        skills = profile.get("skills", [])
        roles = profile.get("roles", [])
        tools = profile.get("tools", [])

        if skills:
            st.write("**Skills:** " + ", ".join(skills))
        else:
            st.write("**Skills:** Not available")

        if roles:
            st.write("**Roles:** " + ", ".join(roles))
        else:
            st.write("**Roles:** Not available")

        if tools:
            st.write("**Tools:** " + ", ".join(tools))
        else:
            st.write("**Tools:** Not available")

    with st.container(border=True):
        st.markdown("### Experience")
        if profile.get("experience"):
            for exp in profile.get("experience", []):
                with st.expander(f"{exp.get('role', 'Role')} - {exp.get('company', 'Company')}"):
                    st.write(f"**Date:** {exp.get('date', exp.get('years', ''))}")
                    st.write(exp.get("description", ""))
        else:
            st.write("No experience entries yet.")


def render_matches_view(profile):
    st.subheader("Role Matches")

    # Experience selector runs BEFORE matching (not after, as it did
    # originally) so the chosen level can actually inform the job_links
    # baked into each recommendation, not just the on-demand live listings.
    live_experience = render_experience_selector(profile, "match")

    recommendation_data = safe_match_roles(
        profile.get("skills", []),
        profile.get("roles", []),
        profile.get("tools", []),
        location=profile.get("location", ""),
        experience=live_experience,
    )

    for rec in recommendation_data:
        with st.container(border=True):
            header_col, score_col = st.columns([3, 1])
            with header_col:
                st.markdown(f"### {rec['role']}")
                st.write(rec["fit_label"])
            with score_col:
                st.metric("Fit", f"{int(rec['final_score'] * 100)}%")
            st.progress(rec["final_score"])
            
            # Hybrid scoring breakdown
            st.markdown("#### Scoring Components")
            score_cols = st.columns(3)
            with score_cols[0]:
                st.metric("Semantic Similarity", f"{int(rec['semantic_similarity']*100)}%")
            with score_cols[1]:
                st.metric("Skill Overlap", f"{int(rec['skill_overlap']*100)}%")
            with score_cols[2]:
                st.metric("Role Bonus", f"{int(rec['role_bonus']*100)}%")
            
            st.caption(
                f"Core matched: {rec['core_matched']}/{rec['core_total']} | "
                f"Optional matched: {rec['optional_matched']}/{rec['optional_total']}"
            )
            if rec.get("matched_core_skills"):
                st.write("**Matched core skills:** " + ", ".join(rec["matched_core_skills"]))
            if rec.get("matched_optional_skills"):
                st.write("**Matched optional skills:** " + ", ".join(rec["matched_optional_skills"]))
            render_live_listings(rec["role"], profile.get("location", ""), f"match_{rec['role']}", live_experience)
            st.markdown(f"[LinkedIn Apply Link]({rec['job_links']['linkedin']})")


def render_resume_tools(profile):
    st.subheader("ATS-Friendly Resume")
    st.caption("Generate an ATS-friendly DOCX resume from your current profile.")
    st.info("Want to improve this resume? Update your details in the Profile page, then regenerate.")

    formatted_data = convert_profile_to_resume_format(profile)

    with st.expander("Preview ATS Resume", expanded=True):
        st.markdown(f"### {formatted_data.get('name', '')}")
        contact_parts = [
            formatted_data.get("email", ""),
            formatted_data.get("phone", ""),
            formatted_data.get("location", ""),
            formatted_data.get("linkedin", ""),
            formatted_data.get("github", ""),
        ]
        contact_parts = [p for p in contact_parts if str(p).strip()]
        if contact_parts:
            st.write(" | ".join(contact_parts))

        st.markdown("#### Professional Summary")
        if profile.get("summary", "").strip():
            st.write(profile.get("summary", "").strip())
        else:
            st.write("Summary is auto-generated in the resume output. Add your own summary in Profile for a personalized version.")

        st.markdown("#### Technical Skills")
        if formatted_data.get("skills"):
            st.write("Core Skills: " + ", ".join(formatted_data.get("skills", [])))
        if formatted_data.get("tools"):
            st.write("Tools/Platforms: " + ", ".join(formatted_data.get("tools", [])))
        if formatted_data.get("roles"):
            st.write("Target Roles: " + ", ".join(formatted_data.get("roles", [])))

        st.markdown("#### Experience")
        if formatted_data.get("experience"):
            for exp in formatted_data.get("experience", []):
                title_parts = [exp.get("role", ""), exp.get("company", ""), exp.get("date", "")]
                title = " | ".join([p for p in title_parts if str(p).strip()])
                if title:
                    st.markdown(f"**{title}**")
                desc = str(exp.get("description", "")).strip()
                if desc:
                    bullets = [
                        line.strip(" -•\t")
                        for line in desc.splitlines()
                        if line.strip(" -•\t")
                    ]
                    if bullets:
                        for bullet in bullets:
                            st.markdown(f"- {bullet}")
                    else:
                        st.markdown(f"- {desc}")
                st.write("")
        else:
            st.write("No experience entries yet.")

        st.markdown("#### Education")
        if formatted_data.get("education"):
            for edu in formatted_data.get("education", []):
                degree = edu.get("degree", "")
                field = edu.get("field", "")
                college = edu.get("college", "")
                date = edu.get("date", "")
                degree_line = f"{degree} in {field}" if degree and field else (degree or field)
                line = " | ".join([p for p in [degree_line, college, date] if str(p).strip()])
                st.markdown(f"- {line}")
        else:
            st.write("No education entries yet.")

        st.markdown("#### Projects")
        projects = formatted_data.get("projects", []) or []
        if projects:
            for project in projects:
                if isinstance(project, dict):
                    project_name = str(project.get("name") or project.get("title") or "Project").strip()
                    if project_name:
                        st.markdown(f"**{project_name}**")
                    description = project.get("description", "")
                    if isinstance(description, list):
                        for bullet in description:
                            if str(bullet).strip():
                                st.markdown(f"- {str(bullet).strip()}")
                    elif str(description).strip():
                        for bullet in str(description).splitlines():
                            if bullet.strip():
                                st.markdown(f"- {bullet.strip().lstrip('-•* ')}")
                    technologies = project.get("technologies", project.get("tools", []))
                    if technologies:
                        tech_text = technologies if isinstance(technologies, str) else ", ".join(map(str, technologies))
                        st.caption(f"Technologies: {tech_text}")
                elif str(project).strip():
                    st.markdown(f"- {project}")
        else:
            st.write("No projects added yet. Add projects below to include them in your ATS resume.")

    st.markdown("### Manage Resume Projects")
    existing_projects = profile.get("projects", []) or []
    with st.form("resume_projects_form"):
        project_count = st.number_input(
            "Number of projects", min_value=0, max_value=10,
            value=min(len(existing_projects), 10), step=1, key="resume_project_count"
        )
        edited_projects = []
        for i in range(int(project_count)):
            current = existing_projects[i] if i < len(existing_projects) and isinstance(existing_projects[i], dict) else {}
            st.markdown(f"**Project {i + 1}**")
            project_name = st.text_input("Project name", value=current.get("name", current.get("title", "")), key=f"resume_project_name_{i}")
            project_description = st.text_area("Description / achievements (one per line)", value=current.get("description", "") if isinstance(current.get("description", ""), str) else "\n".join(map(str, current.get("description", []))), key=f"resume_project_description_{i}")
            project_technologies = current.get("technologies", current.get("tools", []))
            if not isinstance(project_technologies, str):
                project_technologies = ", ".join(map(str, project_technologies or []))
            technologies = st.text_input("Technologies (comma-separated)", value=project_technologies, key=f"resume_project_technologies_{i}")
            edited_projects.append({"name": project_name.strip(), "description": project_description.strip(), "technologies": [x.strip() for x in technologies.split(",") if x.strip()]})
        save_projects = st.form_submit_button("Save Projects to Resume")

    if save_projects:
        updated_profile = dict(profile)
        updated_profile["projects"] = [project for project in edited_projects if project["name"] or project["description"] or project["technologies"]]
        if update_profile(st.session_state.username, updated_profile):
            st.session_state.profile = updated_profile
            st.success("Projects saved. They will be included in the resume preview and generated DOCX.")
            st.rerun()
        else:
            st.error("Could not save projects. Please log in again and retry.")

    if st.button("Generate ATS Resume"):
        file_path = "ats_resume.docx"
        generate_resume(formatted_data, file_path)
        with open(file_path, "rb") as f:
            st.download_button("Download ATS Resume", f, file_name=file_path)


# =========================
# SIDEBAR
# =========================
menu = "Login"

st.sidebar.title("Navigation")

if st.session_state.logged_in:
    sidebar_username = st.session_state.get("username", "User")
    st.sidebar.success(f"Signed in as {sidebar_username}")
    workspace_pages = ["Dashboard", "Profile", "Role Matches", "Jobs for You", "Resume Builder"]
    # A dedicated widget key prevents radio state from drifting away from the
    # page selector used below. The callback synchronizes state before rerun.
    if st.session_state.workspace_nav not in workspace_pages:
        st.session_state.workspace_nav = "Dashboard"
    selected_view = st.sidebar.radio(
        "Workspace",
        workspace_pages,
        key="workspace_nav",
        on_change=_sync_workspace_view,
    )
    # Keep page routing tied to the radio's returned value; avoid writing back
    # to the radio's own keyed state during the same render.
    st.session_state.view = selected_view
    if st.sidebar.button("🚪 Logout"):
        logout_user()
        st.rerun()
else:
    st.sidebar.caption("Please sign in from the main panel.")

# =========================
# AUTH / LANDING UI
# =========================
if not st.session_state.logged_in:
    intro_col, form_col = st.columns([3, 2])

    with intro_col:
        st.markdown("""
        <div class="ts-brand">
            <div class="ts-logo">🧭 TalentSync</div>
            <div class="ts-headline">Smarter hiring starts with better matching.</div>
            <div class="ts-body">
                Stop guessing your next move. TalentSync transforms your profile into a complete career toolkit — generating tailored resumes, analyzing your skills, and matching you with roles that actually fit. Everything happens in one seamless, intelligent workspace designed to remove friction and deliver clarity.
            </div>
            <div class="ts-bullets">
                ✓ AI-powered role matching based on real skill similarity<br>
                ✓ Instantly generate structured, ATS-friendly resumes<br>
                ✓ Manage your profile, experience, and skills in one place<br>
                ✓ Explore personalized job matches — not random listings<br>
                ✓ Continuously refine and optimize your career positioning
            </div>
        </div>
        """, unsafe_allow_html=True)

    with form_col:
        with st.container(border=True):
            st.markdown("### Welcome Back")
            menu = st.radio("Access", ["Login", "Signup"], horizontal=True)

            if menu == "Login":
                col1, col2 = st.columns(2)
                with col1:
                    username = st.text_input("Username")
                with col2:
                    password = st.text_input("Password", type="password")

# =========================
# LOGIN
# =========================
    if menu == "Login":
        if st.button("🔐 Login"):
            profile = login(username, password)

            if profile:
                st.session_state.logged_in = True
                st.session_state.profile = profile
                st.session_state.username = username
                st.session_state.profile_edit_mode = False
                st.session_state.view = "Dashboard"
                st.success("Logged in successfully ✅")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

        if not st.session_state.get("logged_in"):
            st.stop()

# =========================
# SIGNUP (UI IMPROVED ONLY)
# =========================
    elif menu == "Signup":

        st.subheader("📝 Create Your Profile")

        option = st.radio("Choose Input Method", ["Upload Resume", "Enter Manually"])

        profile = {}
        basic = {"name": "", "email": "", "phone": "", "location": ""}
        extracted = {"skills": [], "roles": [], "tools": []}
        name = ""
        dob = datetime.date(2000, 1, 1)
        email = ""
        phone = ""
        location = ""
        linkedin = ""
        raw_skills = ""
        education_list = []
        experience_list = []
        extracted_edu_exp = {"education": [], "experience": []}

        if option == "Upload Resume":
            uploaded_file = st.file_uploader(
                "Upload Resume",
                type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "tif", "tiff", "webp"],
            )

            if uploaded_file:
                import tempfile

                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.read())
                    path = tmp.name

                text = extract_text(path)
                basic = extract_basic_info(text)
                extracted_edu_exp = extract_education_experience(text)

                extracted = {"skills": [], "roles": [], "tools": []}
                if callable(extract_entities):
                    try:
                        if not st.session_state.get("_nlp_warmed_up"):
                            with st.spinner("Loading NLP model (first time only)..."):
                                extracted = extract_entities(text)
                            st.session_state["_nlp_warmed_up"] = True
                        else:
                            extracted = extract_entities(text)
                    except Exception:
                        extracted = {"skills": [], "roles": [], "tools": []}

                if callable(run_pipeline):
                    try:
                        with st.spinner("Processing resume..."):
                            result = run_pipeline(path)
                        pipeline_entities = result.get("raw", {})
                        extracted = {
                            "skills": sorted(set(extracted.get("skills", []) + pipeline_entities.get("skills", []))),
                            "roles": sorted(set(extracted.get("roles", []) + pipeline_entities.get("roles", []))),
                            "tools": sorted(set(extracted.get("tools", []) + pipeline_entities.get("tools", []))),
                        }
                    except Exception:
                        pass

                st.success("Resume processed successfully ✅")
                st.info(
                    f"Detected {len(extracted_edu_exp.get('education', []))} education and "
                    f"{len(extracted_edu_exp.get('experience', []))} experience entries. "
                    "You can edit them before signup."
                )

        st.markdown("### 👤 Basic Info")
        name = st.text_input("Full Name", basic.get("name", ""))
        dob = st.date_input(
            "Date of Birth",
            value=datetime.date(2000, 1, 1),
            min_value=datetime.date(1901, 1, 1),
            max_value=datetime.date(2015, 12, 31)
        )
        email = st.text_input("Email", basic.get("email", ""))
        phone = st.text_input("Phone", basic.get("phone", ""))
        location = st.text_input("Location", basic.get("location", ""))
        linkedin = st.text_input("LinkedIn", basic.get("linkedin", ""))
        summary = st.text_area(
            "Professional Summary",
            value=basic.get("summary", ""),
            placeholder="Add a profile summary (optional). If left empty, resume generator will auto-create one.",
        )
        raw_skills = st.text_area("Skills / Competencies", ", ".join(extracted["skills"]), key="signup_skills")
        raw_roles = st.text_input("Target Roles", ", ".join(extracted["roles"]), key="signup_roles")
        raw_tools = st.text_input("Tools / Platforms", ", ".join(extracted["tools"]), key="signup_tools")

        st.markdown("### 🎓 Education")
        prefilled_education = extracted_edu_exp.get("education", [])
        num_edu = st.number_input(
            "How many education entries?",
            min_value=1,
            max_value=10,
            value=max(1, len(prefilled_education)),
            step=1,
        )
        for i in range(num_edu):
            pre_edu = prefilled_education[i] if i < len(prefilled_education) else {}
            degree = st.text_input(
                f"Degree Name {i+1}",
                value=pre_edu.get("degree", ""),
                placeholder="e.g. B.Tech, B.E., M.Sc, MBA",
                key=f"deg_manual_{i}",
            )
            field = st.text_input(
                f"Specialization/Field {i+1}",
                value=pre_edu.get("field", ""),
                placeholder="e.g. Computer Science Engineering",
                key=f"field_manual_{i}",
            )
            college = st.text_input(f"College {i+1}", value=pre_edu.get("college", ""), key=f"college_manual_{i}")
            pre_date = pre_edu.get("date", "")
            if not pre_date:
                from_d = pre_edu.get("from", "")
                to_d = pre_edu.get("to", "")
                pre_date = f"{from_d} - {to_d}".strip(" -") if (from_d or to_d) else ""
            edu_date = st.text_input(
                f"Date {i+1}",
                value=pre_date,
                placeholder="e.g. 2018 - 2022",
                key=f"edu_date_manual_{i}",
            )
            education_list.append({"degree": degree, "field": field, "college": college, "date": edu_date})

        st.markdown("### 💼 Experience")
        prefilled_experience = extracted_edu_exp.get("experience", [])
        num_exp = st.number_input(
            "How many experience entries?",
            min_value=0,
            max_value=20,
            value=len(prefilled_experience),
            step=1,
        )
        for i in range(num_exp):
            pre_exp = prefilled_experience[i] if i < len(prefilled_experience) else {}
            role = st.text_input(f"Role {i+1}", value=pre_exp.get("role", ""), key=f"role_manual_{i}")
            company = st.text_input(f"Company {i+1}", value=pre_exp.get("company", ""), key=f"comp_manual_{i}")
            pre_exp_date = pre_exp.get("date", "")
            if not pre_exp_date:
                from_d = pre_exp.get("from", "")
                to_d = pre_exp.get("to", "")
                pre_exp_date = f"{from_d} - {to_d}".strip(" -") if (from_d or to_d) else ""
            exp_date = st.text_input(
                f"Date {i+1}",
                value=pre_exp_date,
                placeholder="e.g. Jan 2023 - Mar 2026",
                key=f"exp_date_manual_{i}",
            )
            work_summary = st.text_area(
                f"Work Description {i+1}",
                value=pre_exp.get("description", ""),
                key=f"work_desc_manual_{i}",
            )
            experience_list.append({"role": role, "company": company, "date": exp_date, "description": work_summary})

        # Skills come from whatever the user has in the editable text box
        # (seeded from the resume, but editable before signup).
        # Roles/tools are kept from the full-resume extraction above, since
        # there's no equivalent editable field for them pre-signup and a
        # short comma-separated skills string won't contain role/tool terms.
        def parse_terms(value):
            return list(dict.fromkeys(part.strip() for part in re.split(r"[,;\n|]", value or "") if part.strip()))

        manual_skills = parse_terms(raw_skills)
        manual_roles = parse_terms(raw_roles)
        manual_tools = parse_terms(raw_tools)
        entities = {
            "skills": manual_skills,
            "roles": manual_roles,
            "tools": manual_tools,
        }

        # Always build profile (no conditional)
        profile = {
            "name": name,
            "dob": str(dob) if dob else "",
            "summary": summary,
            "phone": phone,
            "email": email,
            "location": location,
            "linkedin": linkedin,
            "education": education_list,
            "experience": experience_list,
            "skills": entities["skills"],
            "roles": entities["roles"],
            "tools": entities["tools"]
        }

        st.markdown("---")
        # Keep credential entry inside a form so typing does not trigger a
        # Streamlit rerun on every keystroke. The profile fields above remain
        # independently editable and are read from the current session state.
        with st.form("signup_credentials_form", clear_on_submit=False):
            cred_col1, cred_col2 = st.columns(2)
            with cred_col1:
                username = st.text_input("Choose a Username", key="signup_username")
            with cred_col2:
                password = st.text_input("Choose a Password", type="password", key="signup_password")
            signup_submitted = st.form_submit_button("🚀 Signup")

        if signup_submitted:
            if not username.strip() or not password.strip():
                st.error("Please enter a username and password above before signing up ❌")
            elif not name.strip():
                st.error("Please fill in at least your name before signing up ❌")
            else:
                if signup(username, password, profile):
                    st.success("Account created successfully ✅")
                    st.stop()
                else:
                    st.error("User already exists ❌")

        st.stop()

# =========================
# MAIN APP AFTER LOGIN
# =========================
if st.session_state.logged_in:
    profile = st.session_state.profile

    st.title("TalentSync")
    st.caption("Resume Analysis • Job Matching • Resume Builder")

    if st.session_state.view == "Dashboard":
        render_dashboard(profile)
    elif st.session_state.view == "Profile":
        render_profile_view(profile)
    elif st.session_state.view == "Role Matches":
        render_matches_view(profile)
    elif st.session_state.view == "Jobs for You":
        if render_jobs_for_you is not None:
            render_jobs_for_you(profile, safe_match_roles, render_live_listings, job_redirect)
        else:
            st.error("Jobs for You page could not be loaded. Ensure jobs_for_you.py is in the app directory.")
    elif st.session_state.view == "Resume Builder":
        render_resume_tools(profile)