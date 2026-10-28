"""Small, UI-independent cache for signup resume entity extraction."""


_EMPTY_ENTITIES = {"skills": [], "roles": [], "tools": []}


def get_or_extract_entities(session_state, fingerprint, resume_text, extractor):
    """Extract once per uploaded-resume fingerprint within a Streamlit session.

    A new fingerprint triggers one extractor call and replaces the cached value.
    Re-renders for the same fingerprint reuse the cached entities.
    """
    if session_state.get("_signup_processed_resume_hash") == fingerprint:
        return session_state.get("_signup_extracted_entities", _EMPTY_ENTITIES.copy())

    try:
        extracted = extractor(resume_text) if callable(extractor) else _EMPTY_ENTITIES.copy()
    except Exception:
        extracted = _EMPTY_ENTITIES.copy()

    if not isinstance(extracted, dict):
        extracted = _EMPTY_ENTITIES.copy()
    normalized = {
        category: list(extracted.get(category, []) or [])
        for category in ("skills", "roles", "tools")
    }
    session_state["_signup_extracted_entities"] = normalized
    session_state["_signup_processed_resume_hash"] = fingerprint
    return normalized
