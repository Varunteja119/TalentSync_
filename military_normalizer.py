import os
import json

NER_OUTPUT = r"E:\Capstone Project\ner_output"
MILITARY_MAP = r"E:\Capstone Project\military_mapping\military_to_civilian_skills.json"
OUTPUT_ROOT = r"E:\Capstone Project\ex_servicemen_dataset"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

with open(MILITARY_MAP, "r", encoding="utf-8") as f:
    military_map = json.load(f)

COMPLEX_SKILL_KEYWORDS = {
    "audit", "compliance", "security", "architecture",
    "system design", "data modeling", "cloud",
    "governance", "risk", "operations planning"
}

RESPONSIBILITY_KEYWORDS = {
    "management", "planning", "coordination",
    "supervision", "leadership", "project management"
}

CODING_SKILLS = {
    "python", "sql", "java", "c++", "javascript"
}

def compute_level(skills):
    score = 0
    skills = [s.lower() for s in skills]
    skill_count = len(skills)

    if skill_count <= 3:
        score += 1
    elif skill_count <= 6:
        score += 2
    elif skill_count <= 10:
        score += 3
    else:
        score += 4

    for s in skills:
        for kw in COMPLEX_SKILL_KEYWORDS:
            if kw in s:
                score += 1
                break

    if any(kw in s for s in skills for kw in RESPONSIBILITY_KEYWORDS):
        score += 2

    score += 1

    if score <= 2:
        level = 1
    elif score <= 4:
        level = 2
    elif score <= 6:
        level = 3
    elif score <= 8:
        level = 4
    else:
        level = 5

    if any(code in s for s in skills for code in CODING_SKILLS):
        level = max(level, 2)

    has_complexity = any(kw in s for s in skills for kw in COMPLEX_SKILL_KEYWORDS)
    has_responsibility = any(kw in s for s in skills for kw in RESPONSIBILITY_KEYWORDS)

    if "python" in skills and "sql" in skills and (has_complexity or has_responsibility):
        level = max(level, 3)


    return level

def normalize_skills(skills):
    normalized = set(s.lower() for s in skills)

    for mapping in military_map.values():
        military_terms = [t.lower() for t in mapping["military_terms"]]
        civilian_skills = mapping["civilian_skills"]

        if any(term in normalized for term in military_terms):
            for cs in civilian_skills:
                normalized.add(cs.lower())

    return sorted(normalized)

for folder in os.listdir(NER_OUTPUT):
    in_folder = os.path.join(NER_OUTPUT, folder)
    if not os.path.isdir(in_folder):
        continue

    out_folder = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(out_folder, exist_ok=True)

    print(f"\n[FORMALISING] {folder}")

    for fname in os.listdir(in_folder):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(in_folder, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_skills = data["entities"].get("skills", [])
        normalized = normalize_skills(raw_skills)
        level = compute_level(normalized)

        output = {
            "file": data["file"],
            "skills": normalized,
            "skill_count": len(normalized),
            "professional_level": level,
            "experience_context": "military_organizational_role",
            "skill_provenance": "military",
            "work_environment": {
                "regulated": True,
                "hierarchical": True,
                "high_accountability": True
            }
        }

        with open(os.path.join(out_folder, fname), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"[OK] {folder}/{fname}")
