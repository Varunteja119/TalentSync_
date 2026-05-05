import os
import re

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:
    spacy = None
    PhraseMatcher = None

# ==========================================
# SBERT + NER Integration
# ==========================================
# SBERT is used in role_matcher.py for semantic similarity
# NER (spaCy PhraseMatcher) is used here for entity extraction
# Both algorithms are active in the project:
# - NER extracts skills, roles, tools from text
# - SBERT matches candidates to job roles semantically

BASE_DIR = os.path.dirname(__file__)

def load_terms(file_name):
    """Load vocabulary from NER resource files"""
    path = os.path.join(BASE_DIR, "ner_resources", file_name)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

# Load vocabulary for NER
SKILLS = load_terms("skills.txt")
ROLES = load_terms("roles.txt")
TOOLS = load_terms("tools.txt")

# ==========================================
# Initialize spaCy NER Pipeline
# ==========================================
try:
    nlp = spacy.load("en_core_web_sm") if spacy else None
except OSError:
    print("[WARN] spaCy model not installed. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize PhraseMatcher for NER entity extraction
matcher = None
if nlp:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILL", [nlp.make_doc(s) for s in SKILLS])
    matcher.add("ROLE", [nlp.make_doc(r) for r in ROLES])
    matcher.add("TOOL", [nlp.make_doc(t) for t in TOOLS])

# 🔥 Skill Inference Rules (fallback when NER misses patterns)
INFERENCE_RULES = {
    "leadership": ["led", "managed", "supervised", "commanded"],
    "team management": ["team of", "personnel", "crew"],
    "crisis management": ["under pressure", "critical situation", "high-pressure"],
    "operations management": ["operations", "mission", "deployment"],
    "logistics": ["logistics", "supply", "transport"],
    "training": ["trained", "training", "instructed"],
    "communication": ["coordinated", "liaison", "communicated"],
    "security operations": ["security", "surveillance", "patrol"],
    "technical troubleshooting": ["troubleshoot", "debug", "repair", "maintenance"],
}

# 🔥 Role Inference Rules (fallback when NER misses patterns)
ROLE_INFERENCE = {
    "software engineer": ["developed", "built", "implemented", "coded"],
    "data analyst": ["analyzed", "analysis", "insights"],
    "network engineer": ["network", "communication systems", "signal"],
    "operations manager": ["operations", "managed operations", "coordination"],
    "team leader": ["led team", "managed team", "supervised"]
}

# Extra hints to improve extraction on noisy, real-world resumes.
SKILL_HINTS = {
    "python": ["python", "pandas", "numpy", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "ecmascript"],
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node", "node.js", "nodejs"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "ci/cd": ["ci/cd", "cicd", "jenkins", "gitlab ci", "github actions"],
    "sql": ["sql", "mysql", "postgresql", "postgres"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "tensorflow", "pytorch"],
    "account reconciliation": ["account reconciliation", "reconciliation"],
    "accounts payable": ["accounts payable", "ap process"],
    "general ledger accounting": ["general ledger", "ledger accounting"],
}

ROLE_HINTS = {
    "software engineer": ["software engineer", "application developer", "backend developer"],
    "software developer": ["software developer", "developer"],
    "full stack developer": ["full stack", "frontend and backend"],
    "devops engineer": ["devops", "site reliability", "sre"],
    "data analyst": ["data analyst", "business analyst"],
    "data scientist": ["data scientist", "ai engineer", "ml engineer"],
    "accounts executive": ["accounts executive", "accountant", "finance executive"],
}

TOOL_HINTS = {
    "git": ["git", "github", "gitlab", "bitbucket"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "jira": ["jira"],
    "excel": ["excel", "spreadsheets"],
    "sap": ["sap", "sap erp"],
    "power bi": ["power bi"],
}


def _contains_term(text: str, term: str) -> bool:
    # Algorithm: Use regex boundary matching to detect whole terms while tolerating separators like spaces, hyphens, and slashes.
    """Boundary-aware term matching for keyword fallback"""
    term_pattern = re.escape(term).replace(r"\ ", r"[\s\-/&]+")
    pattern = rf"(?<![a-z0-9]){term_pattern}(?![a-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _extract_section_tokens(text: str):
    # Heuristic: Capture candidate terms from resume skill sections and stop when non-skill sections begin.
    """Extract tokens from skills/expertise sections"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    section_headers = [
        "skills",
        "technical skills",
        "core competencies",
        "expertise",
        "tools",
    ]
    stop_headers = [
        "experience",
        "education",
        "projects",
        "summary",
        "profile",
        "certifications",
    ]

    tokens = set()
    capture = False
    collected_lines = 0

    for line in lines:
        lower = line.lower().strip(" :")

        if any(lower == h or lower.startswith(h + " ") for h in section_headers):
            capture = True
            collected_lines = 0
            continue

        if capture and any(lower.startswith(h) for h in stop_headers):
            break

        if capture:
            parts = re.split(r"[,;|/•\u2022]+", line)
            for part in parts:
                token = re.sub(r"\s+", " ", part).strip(" -:\t")
                if 2 <= len(token) <= 50:
                    tokens.add(token.lower())
            collected_lines += 1
            if collected_lines >= 12:
                break

    return tokens


def _extract_entities_ner(text: str):
    # Algorithm: Apply spaCy PhraseMatcher-based NER to extract known skills, roles, and tools from text.
    """
    PRIMARY METHOD: Extract entities using spaCy NER (PhraseMatcher)
    This is the NER algorithm used in the project
    """
    if not matcher or not nlp:
        return {"skills": [], "roles": [], "tools": []}

    doc = nlp(text.lower())
    matches = matcher(doc)

    extracted = {
        "skills": set(),
        "roles": set(),
        "tools": set()
    }

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        value = doc[start:end].text.strip().lower()

        if label == "SKILL":
            extracted["skills"].add(value)
        elif label == "ROLE":
            extracted["roles"].add(value)
        elif label == "TOOL":
            extracted["tools"].add(value)

    return {k: list(v) for k, v in extracted.items()}


def _extract_entities_fallback(text: str):
    # Algorithm: Use keyword lookup plus context inference rules to recover entities missed by NER.
    """
    FALLBACK METHOD: Keyword matching when NER is unavailable
    or for context-based inference
    """
    text = text.lower()
    normalized_text = re.sub(r"\s+", " ", text)
    section_tokens = _extract_section_tokens(text)

    found_skills = set()
    found_roles = set()
    found_tools = set()

    # 🔹 Keyword matching against vocabulary
    for skill in SKILLS:
        if _contains_term(normalized_text, skill) or skill in section_tokens:
            found_skills.add(skill)

    for role in ROLES:
        if _contains_term(normalized_text, role) or role in section_tokens:
            found_roles.add(role)

    for tool in TOOLS:
        if _contains_term(normalized_text, tool) or tool in section_tokens:
            found_tools.add(tool)

    # 🔥 Skill inference from context
    for skill, patterns in INFERENCE_RULES.items():
        for pattern in patterns:
            if _contains_term(normalized_text, pattern):
                found_skills.add(skill)
                break

    # 🔥 Role inference from context
    for role, patterns in ROLE_INFERENCE.items():
        for pattern in patterns:
            if _contains_term(normalized_text, pattern):
                found_roles.add(role)
                break

    # Additional domain and technical hints for noisy resumes.
    for skill, patterns in SKILL_HINTS.items():
        for pattern in patterns:
            if _contains_term(normalized_text, pattern):
                found_skills.add(skill)
                break

    for role, patterns in ROLE_HINTS.items():
        for pattern in patterns:
            if _contains_term(normalized_text, pattern):
                found_roles.add(role)
                break

    for tool, patterns in TOOL_HINTS.items():
        for pattern in patterns:
            if _contains_term(normalized_text, pattern):
                found_tools.add(tool)
                break

    return {
        "skills": list(found_skills),
        "roles": list(found_roles),
        "tools": list(found_tools)
    }


def extract_entities(text: str):
    # Algorithm: Build final entities by merging NER-first extraction with fallback keyword/inference extraction.
    """
    Extract skills, roles, and tools from text using NER + Fallback
    
    Algorithm flow:
    1. PRIMARY: Use spaCy NER (PhraseMatcher) to extract entities
    2. FALLBACK: Enhance with keyword matching & context inference
    3. MERGE: Combine both for comprehensive coverage
    
    Returns:
        dict: {"skills": [...], "roles": [...], "tools": [...]}
    """
    
    # Extract using NER (primary)
    ner_results = _extract_entities_ner(text)
    
    # Extract using keyword/inference (fallback)
    fallback_results = _extract_entities_fallback(text)
    
    # Merge results: NER + Fallback for best coverage
    merged = {
        "skills": list(set(ner_results.get("skills", []) + fallback_results.get("skills", []))),
        "roles": list(set(ner_results.get("roles", []) + fallback_results.get("roles", []))),
        "tools": list(set(ner_results.get("tools", []) + fallback_results.get("tools", [])))
    }

    # Low-signal boost: if extraction is sparse, infer extra entities from section tokens.
    if len(merged["skills"]) < 5:
        normalized_text = re.sub(r"\s+", " ", text.lower())
        section_tokens = _extract_section_tokens(text)
        searchable = set(section_tokens)
        searchable.add(normalized_text)

        for skill, patterns in SKILL_HINTS.items():
            if any(any(_contains_term(candidate, p) for p in patterns) for candidate in searchable):
                merged["skills"].append(skill)

        for role, patterns in ROLE_HINTS.items():
            if any(any(_contains_term(candidate, p) for p in patterns) for candidate in searchable):
                merged["roles"].append(role)

        for tool, patterns in TOOL_HINTS.items():
            if any(any(_contains_term(candidate, p) for p in patterns) for candidate in searchable):
                merged["tools"].append(tool)

        merged = {
            "skills": list(set(merged["skills"])),
            "roles": list(set(merged["roles"])),
            "tools": list(set(merged["tools"])),
        }
    
    return merged
