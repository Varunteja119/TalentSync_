import json
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from job_redirect import generate_job_links


# Loaded lazily (see _get_model), not at import time. The old eager
# `model = SentenceTransformer(...)` here ran the instant this module was
# imported -- which happens at the very top of streamlit_app.py, before
# any UI renders, before login, before anything. That made every app
# launch (or restart) pay a one-to-two-minute blank-screen cost up front,
# even for a user who was only there to log in. Deferring it means the
# cost only lands on the first actual match_roles() call, where a
# Streamlit spinner can show the user what's happening instead of a
# silent frozen page.
_model = None


def _get_model():
    global _model
    if _model is None:
        # Deferred here, not at the top of the file: `import sentence_
        # transformers` alone pulls in its full dependency chain (torch,
        # transformers, etc.), which can be slow on its own, separate
        # from actually instantiating a specific model. Keeping the
        # import out of the module's top level means merely importing
        # role_matcher.py (which streamlit_app.py does before any UI
        # renders) doesn't pay that cost either.
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model



BASE_DIR = os.path.dirname(__file__)
ROLE_PROFILE_PATH = os.path.join(BASE_DIR, "role_profiles", "role_skill_level_profiles.json")

with open(ROLE_PROFILE_PATH, "r", encoding="utf-8") as f:
    ROLE_PROFILES = json.load(f)

WEIGHT_SEMANTIC = 0.55
WEIGHT_SKILL_OVERLAP = 0.35
WEIGHT_ROLE_BONUS = 0.10


def _build_role_text(role_name, role_profile):
    role_skills = role_profile["core_skills"] + role_profile.get("optional_skills", [])
    return (
        f"The {role_name} role requires strong capabilities in "
        f"{', '.join(role_skills)}, including execution, coordination, and decision making."
    )


def _skill_rarity_rank(candidate_skills):
    """Sort skills rarest-first among those that actually appear in some
    role profile. Uses the same domain-word+acronym normalization as
    compute_role_match_bonus so "nlp" correctly matches a profile's
    "natural language processing". Skills with zero profile matches are
    kept (so nothing is dropped from the text) but pushed to the end --
    a zero count means "not in this app's role vocabulary", not "rare
    and distinctive", and shouldn't be emphasized as a specialization."""
    counts = {}
    for profile in ROLE_PROFILES.values():
        for s in profile.get("core_skills", []) + profile.get("optional_skills", []):
            key = frozenset(_domain_words(s))
            if key:
                counts[key] = counts.get(key, 0) + 1

    def rarity(skill):
        return counts.get(frozenset(_domain_words(skill)), 0)

    # count==1 is too thin to trust as a specialization signal (often one
    # unrelated profile happens to list it); require it shared by >=2
    # profiles, a real cluster, not noise.
    with_signal = [s for s in candidate_skills if rarity(s) >= 2]
    without_signal = [s for s in candidate_skills if rarity(s) < 2]
    return sorted(with_signal, key=rarity) + without_signal


def _build_candidate_text(candidate_skills, candidate_roles, candidate_tools):
    ranked = _skill_rarity_rank(candidate_skills)
    specialization = f"This candidate specializes in {', '.join(ranked[:5])}. " if ranked else ""
    return (
        f"{specialization}"
        f"A candidate with strong experience in {', '.join(ranked)}, "
        f"targeting roles such as {', '.join(candidate_roles) if candidate_roles else 'generalist positions'}, "
        f"and using tools like {', '.join(candidate_tools) if candidate_tools else 'standard tools'}, "
        f"demonstrating leadership, execution, problem-solving, and coordination."
    )


# Role embeddings only depend on ROLE_PROFILES, which is loaded once above
# and never changes at runtime. They're computed once, lazily, on first
# use (not at import time -- see _get_model above for why), and cached
# here for every match_roles() call after that.
_role_embeddings = None


def _get_role_embeddings():
    global _role_embeddings
    if _role_embeddings is None:
        model = _get_model()
        _role_embeddings = {
            role_name: model.encode(_build_role_text(role_name, role_profile))
            for role_name, role_profile in ROLE_PROFILES.items()
        }
    return _role_embeddings


def compute_semantic_similarity(candidate_skills, candidate_roles, candidate_tools, role_name, role_profile, candidate_embedding=None):
    """
    Compute semantic similarity using SBERT embeddings and cosine similarity.
    
    Formula: Similarity(A, B) = (A · B) / (||A|| * ||B||)
    
    where A = candidate embedding, B = role embedding

    `candidate_embedding` lets a caller (match_roles) pass in an embedding
    already computed once for this candidate, instead of re-encoding the
    same candidate text for every role. The role embedding is looked up
    from the module-level cache built once at import time; if a role isn't
    in the cache for some reason, it's encoded on demand as a fallback.
    """
    candidate_text = _build_candidate_text(candidate_skills, candidate_roles, candidate_tools)

    if candidate_embedding is None:
        candidate_embedding = _get_model().encode(candidate_text)  # Dense vector

    role_embedding = _get_role_embeddings().get(role_name)
    if role_embedding is None:
        role_embedding = _get_model().encode(_build_role_text(role_name, role_profile))  # Dense vector

    similarity = cosine_similarity(
        [candidate_embedding],
        [role_embedding]
    )[0][0]

    return round(float(similarity), 4)

def compute_skill_overlap(candidate_skills, role_profile):
    """
    Calculate skill overlap ratio:
    
    Skill Overlap = Number of Matched Skills / Total Required Skills
    
    Only core skills are considered for overlap calculation.
    """
    candidate_skills_set = set(s.lower() for s in candidate_skills)
    core_skills = set(s.lower() for s in role_profile["core_skills"])
    
    core_matched = candidate_skills_set & core_skills
    total_required = len(core_skills)
    
    if total_required == 0:
        return 0.0
    
    skill_overlap = len(core_matched) / total_required
    return round(skill_overlap, 4)


_GENERIC_TITLE_WORDS = {
    "engineer", "developer", "intern", "analyst", "manager", "specialist",
    "executive", "associate", "lead", "officer", "coordinator",
    "administrator", "scientist", "architect", "consultant", "supervisor",
    "trainee", "senior", "junior", "sr", "jr",
}

# Narrow, targeted acronym<->expansion pairs for the acronyms that actually
# appear in this app's own role names (AI, NLP, LLM, IoT, ERP). NOT a
# general synonym system -- just enough that "nlp" and "natural language
# processing" (the literal skill string produced by skill_extractor.py)
# are recognized as the same thing, since a purely literal word-match
# would otherwise treat an acronym and its spelled-out form as unrelated.
_ACRONYM_EXPANSIONS = {
    "ai": {"artificial", "intelligence"},
    "ml": {"machine", "learning"},
    "nlp": {"natural", "language", "processing"},
    "llm": {"large", "language", "model", "models"},
    "iot": {"internet", "things"},
    "erp": {"enterprise", "resource", "planning"},
}


def _domain_words(text):
    """Words in a role/skill string with generic title/level words (engineer,
    intern, manager, senior, ...) stripped out, leaving the DOMAIN-specific
    words (e.g. "generative ai engineer" -> {"generative", "ai"}). Each
    domain word that has a known acronym/expansion pair is expanded to
    include both forms, so e.g. "nlp" and "natural language processing"
    overlap correctly regardless of which form a resume actually uses."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    domain = {w for w in words if w not in _GENERIC_TITLE_WORDS and len(w) > 1}
    expanded = set(domain)
    for w in domain:
        if w in _ACRONYM_EXPANSIONS:
            expanded |= _ACRONYM_EXPANSIONS[w]
        for acronym, expansion_words in _ACRONYM_EXPANSIONS.items():
            if w in expansion_words:
                expanded.add(acronym)
    return expanded


def compute_role_match_bonus(candidate_skills, candidate_roles, role_name):
    """
    Role Match Bonus rewards a candidate who has already stated (or had
    inferred) an interest in or experience with a role close to the
    target profile's name.

    This does NOT require an exact string match, on purpose. A resume
    almost never phrases a job title identically to a profile's
    canonical name -- "Generative AI Intern" vs "Generative AI Engineer"
    is a title-suffix difference (Intern vs Engineer), not a different
    domain. A strict-equality check meant the vast majority of role
    profiles could realistically never receive this bonus regardless of
    how well the candidate actually matched, since the exact profile
    name almost never appears verbatim on a real resume.

    Approach: strip generic title/level words from the role name,
    leaving its domain-specific words (expanded for known acronyms, see
    _domain_words), then require ALL of those domain words to appear
    somewhere in the candidate's extracted skills or roles -- not just a
    majority. A majority threshold sounds reasonable but isn't: for a
    2-word role name like "Business Systems Analyst", a candidate who is
    merely a "business owner" shares one of two words and would clear a
    50% bar, awarding full credit for a role that has nothing to do with
    systems analysis. Requiring every domain word keeps this a real
    signal ("you said something in this SPECIFIC domain") rather than a
    coincidental-single-word-overlap detector.

    This is deliberately NOT a semantic/embedding comparison -- that
    would substantially duplicate the semantic_similarity component
    (55% of the final score, same SBERT mechanism), just at a smaller
    weight, without adding much real signal. Keeping this check
    literal-but-tolerant (via generic-word stripping and acronym
    expansion) preserves it as a distinct, interpretable signal rather
    than a second, correlated copy of the semantic score.
    """
    candidate_terms = set()
    for r in candidate_roles:
        candidate_terms |= _domain_words(r)
    for s in candidate_skills:
        candidate_terms |= _domain_words(s)

    role_terms = _domain_words(role_name)
    if not role_terms:
        return 0.0

    return 1.0 if role_terms.issubset(candidate_terms) else 0.0


def compute_final_score(semantic_similarity, skill_overlap, role_bonus):
    """
    Compute final weighted score using the hybrid formula:
    
    Final Score = w1 * Semantic Similarity + w2 * Skill Overlap + w3 * Role Match Bonus
    
    where:
    - w1 = 0.55 (highest weight: semantic understanding is most important)
    - w2 = 0.35 (skill match is important)
    - w3 = 0.10 (role alignment provides a tiebreaker)
    
    The result is then scaled to percentage (0-100%).
    """
    final_score = (
        WEIGHT_SEMANTIC * semantic_similarity +
        WEIGHT_SKILL_OVERLAP * skill_overlap +
        WEIGHT_ROLE_BONUS * role_bonus
    )
    
    final_score = max(0.0, min(1.0, final_score))
    
    return round(final_score, 4)


def check_qualification(skill_overlap, semantic_similarity):
    """
    Screen candidates: must meet minimum thresholds.
    
    - Skill overlap must be at least 15%
    - Semantic similarity must be at least 0.25
    """
    if skill_overlap < 0.15:
        return False
    if semantic_similarity < 0.25:
        return False
    return True



def assign_fit_label(final_score):
    """
    Assign qualitative label based on final score percentage.
    
    final_score is already in [0, 1] range.
    """
    if final_score >= 0.70:
        return "Strong Fit"
    elif final_score >= 0.50:
        return "Good Fit"
    elif final_score >= 0.35:
        return "Potential Fit"
    else:
        return "Low Fit"


def match_roles(candidate_skills, candidate_roles=None, candidate_tools=None, candidate_level=1,
                 candidate_location="", candidate_experience=None):
    """
    Main function to match candidate profile to all available job roles.
    
    Process:
    1. For each role, compute semantic similarity via SBERT
    2. Compute skill overlap ratio
    3. Compute role match bonus
    4. Aggregate using weighted hybrid formula
    5. Apply qualification thresholds
    6. Rank by final score in descending order
    
    Returns:
        list: Recommendations sorted by final_score (descending)
    """
    if candidate_roles is None:
        candidate_roles = []
    if candidate_tools is None:
        candidate_tools = []

    recommendations = []

    candidate_text = _build_candidate_text(candidate_skills, candidate_roles, candidate_tools)
    candidate_embedding = _get_model().encode(candidate_text)

    for role_name, role_profile in ROLE_PROFILES.items():

        semantic_similarity = compute_semantic_similarity(
            candidate_skills,
            candidate_roles,
            candidate_tools,
            role_name,
            role_profile,
            candidate_embedding=candidate_embedding
        )

        skill_overlap = compute_skill_overlap(candidate_skills, role_profile)

        role_bonus = compute_role_match_bonus(candidate_skills, candidate_roles, role_name)

        if not check_qualification(skill_overlap, semantic_similarity):
            continue

        final_score = compute_final_score(semantic_similarity, skill_overlap, role_bonus)

        fit_label = assign_fit_label(final_score)

        job_links = generate_job_links(
            role_name,
            location=candidate_location or "India",
            experience=candidate_experience,
        )

        recommendations.append({
            "role": role_name,
            "final_score": final_score,
            "semantic_similarity": semantic_similarity,
            "skill_overlap": skill_overlap,
            "role_bonus": role_bonus,
            "fit_label": fit_label,
            "job_links": job_links,
            "core_matched": len(set(s.lower() for s in candidate_skills) & set(s.lower() for s in role_profile["core_skills"])),
            "core_total": len(role_profile["core_skills"]),
            "optional_matched": len(set(s.lower() for s in candidate_skills) & set(s.lower() for s in role_profile.get("optional_skills", []))),
            "optional_total": len(role_profile.get("optional_skills", [])),
            "matched_core_skills": sorted(set(s.lower() for s in candidate_skills) & set(s.lower() for s in role_profile["core_skills"])),
            "matched_optional_skills": sorted(set(s.lower() for s in candidate_skills) & set(s.lower() for s in role_profile.get("optional_skills", [])))
        })

   
    recommendations.sort(
        key=lambda x: (
            -x["final_score"],
            -x["skill_overlap"],
            -x["semantic_similarity"],
            x["role"].lower()
        )
    )

    return recommendations
