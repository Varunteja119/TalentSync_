import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from job_redirect import generate_job_links


model = SentenceTransformer('all-MiniLM-L6-v2')

ROLE_PROFILE_PATH = r"E:\Capstone Project\role_profiles\role_skill_level_profiles.json"

with open(ROLE_PROFILE_PATH, "r", encoding="utf-8") as f:
    ROLE_PROFILES = json.load(f)

WEIGHT_SEMANTIC = 0.55
WEIGHT_SKILL_OVERLAP = 0.35
WEIGHT_ROLE_BONUS = 0.10


def compute_semantic_similarity(candidate_skills, candidate_roles, candidate_tools, role_name, role_profile):
    """
    Compute semantic similarity using SBERT embeddings and cosine similarity.
    
    Formula: Similarity(A, B) = (A · B) / (||A|| * ||B||)
    
    where A = candidate embedding, B = role embedding
    """
    role_skills = role_profile["core_skills"] + role_profile.get("optional_skills", [])

    candidate_text = (
        f"A candidate with strong experience in {', '.join(candidate_skills)}, "
        f"targeting roles such as {', '.join(candidate_roles) if candidate_roles else 'generalist positions'}, "
        f"and using tools like {', '.join(candidate_tools) if candidate_tools else 'standard tools'}, "
        f"demonstrating leadership, execution, problem-solving, and coordination."
    )

    role_text = (
        f"The {role_name} role requires strong capabilities in "
        f"{', '.join(role_skills)}, including execution, coordination, and decision making."
    )

    candidate_embedding = model.encode(candidate_text)  # Dense vector
    role_embedding = model.encode(role_text)  # Dense vector

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


def compute_role_match_bonus(candidate_skills, candidate_roles, role_name):
    """
    Calculate role match bonus:
    
    Role Match Bonus = 1.0 if the target role is found in candidate's extracted roles/skills
                      = 0.0 otherwise
    
    This gives a boost when the candidate has already indicated they're
    interested in or experienced with this specific role.
    """
    candidate_skills_set = set(s.lower() for s in candidate_skills)
    candidate_roles_set = set(r.lower() for r in candidate_roles)
    
    if role_name.lower() in candidate_skills_set or role_name.lower() in candidate_roles_set:
        return 1.0
    else:
        return 0.0


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


def match_roles(candidate_skills, candidate_roles=None, candidate_tools=None, candidate_level=1):
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

    for role_name, role_profile in ROLE_PROFILES.items():

        semantic_similarity = compute_semantic_similarity(
            candidate_skills,
            candidate_roles,
            candidate_tools,
            role_name,
            role_profile
        )

        skill_overlap = compute_skill_overlap(candidate_skills, role_profile)

        role_bonus = compute_role_match_bonus(candidate_skills, candidate_roles, role_name)

        if not check_qualification(skill_overlap, semantic_similarity):
            continue

        final_score = compute_final_score(semantic_similarity, skill_overlap, role_bonus)

        fit_label = assign_fit_label(final_score)

        job_links = generate_job_links(role_name)

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
