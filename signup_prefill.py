"""Helpers for seeding editable signup fields from a newly uploaded resume."""


def build_signup_prefill_state(basic, extracted, education, experience, projects):
    """Build widget state for a fresh resume upload."""
    basic, extracted = basic or {}, extracted or {}
    education, experience, projects = education or [], experience or [], projects or []
    state = {
        "signup_name": basic.get("name", ""),
        "signup_email": basic.get("email", ""),
        "signup_phone": basic.get("phone", ""),
        "signup_location": basic.get("location", ""),
        "signup_linkedin": basic.get("linkedin", ""),
        "signup_summary": basic.get("summary", ""),
        "signup_dob": None,  # Never invent a DOB that wasn't extracted.
        "signup_skills": ", ".join(extracted.get("skills", [])),
        "signup_roles": ", ".join(extracted.get("roles", [])),
        "signup_tools": ", ".join(extracted.get("tools", [])),
        "signup_education_count": max(1, len(education)),
        "signup_experience_count": len(experience),
        "signup_project_count": min(len(projects), 10),
    }
    for i, item in enumerate(education[:10]):
        state[f"deg_manual_{i}"] = item.get("degree", "")
        state[f"field_manual_{i}"] = item.get("field", "")
        state[f"college_manual_{i}"] = item.get("college", "")
        date_value = item.get("date", "")
        if not date_value:
            start, end = item.get("from", ""), item.get("to", "")
            date_value = f"{start} - {end}".strip(" -") if (start or end) else ""
        state[f"edu_date_manual_{i}"] = date_value
    for i, item in enumerate(experience[:20]):
        state[f"role_manual_{i}"] = item.get("role", "")
        state[f"comp_manual_{i}"] = item.get("company", "")
        date_value = item.get("date", "")
        if not date_value:
            start, end = item.get("from", ""), item.get("to", "")
            date_value = f"{start} - {end}".strip(" -") if (start or end) else ""
        state[f"exp_date_manual_{i}"] = date_value
        state[f"work_desc_manual_{i}"] = item.get("description", "")
    for i, item in enumerate(projects[:10]):
        state[f"signup_project_name_{i}"] = item.get("name", "")
        description = item.get("description", "")
        if isinstance(description, list):
            description = "\n".join(map(str, description))
        state[f"signup_project_description_{i}"] = description
        technologies = item.get("technologies", [])
        state[f"signup_project_technologies_{i}"] = (
            technologies if isinstance(technologies, str) else ", ".join(technologies or [])
        )
    return state
