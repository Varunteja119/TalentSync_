import os
import re
import json

try:
    from groq import Groq
except ImportError:
    Groq = None

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
    if not os.path.isfile(path):
        # Resource files are optional in lightweight deployments; the
        # deterministic hint dictionaries below remain available.
        fallback = os.path.join(BASE_DIR, file_name)
        if file_name == "roles.txt" and os.path.isfile(fallback):
            path = fallback
        else:
            return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

# Load vocabulary for NER
SKILLS = load_terms("skills.txt")
ROLES = load_terms("roles.txt")
TOOLS = load_terms("tools.txt")

# ==========================================
# spaCy NER Pipeline (loaded lazily -- see _get_nlp_and_matcher)
# ==========================================
# Previously `nlp = spacy.load(...)` and the PhraseMatcher setup ran at
# module import time -- which happens the instant streamlit_app.py does
# `from skill_extractor import extract_entities`, before any UI renders.
# Combined with role_matcher.py's own eager SBERT load (now also fixed to
# be lazy), this meant every app launch paid for loading a spaCy model
# AND compiling ~240 phrase patterns (97 skills + 73 roles + 72 tools)
# before the login page could even appear. Deferring both to first actual
# use means the app itself opens instantly; the one-time cost still
# happens, just at the first real extract_entities() call instead of at
# import time.
_nlp = None
_matcher = None
_nlp_load_attempted = False


def _get_nlp_and_matcher():
    global _nlp, _matcher, _nlp_load_attempted
    if _nlp_load_attempted:
        return _nlp, _matcher
    _nlp_load_attempted = True

    if not spacy:
        return None, None
    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[WARN] spaCy model not installed. Install with: python -m spacy download en_core_web_sm")
        return None, None

    _matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    _matcher.add("SKILL", [_nlp.make_doc(s) for s in SKILLS])
    _matcher.add("ROLE", [_nlp.make_doc(r) for r in ROLES])
    _matcher.add("TOOL", [_nlp.make_doc(t) for t in TOOLS])
    return _nlp, _matcher

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
    # NOTE: the bare "ml" pattern was removed on purpose. It used to be
    # boundary-matched against things like "AI/ML:" (a section-header
    # abbreviation meaning "AI and ML" as an umbrella label), which
    # silently inferred a "machine learning" skill for candidates who
    # never actually stated it -- e.g. a resume whose AI/ML line only
    # lists RAG, LLM Integration, NLP, Semantic Search, Prompt
    # Engineering, MCP (no ML methodology at all). That false positive
    # inflated skill_overlap for ML/NLP-adjacent roles in role_matcher.py
    # to the same 1.0 as a role the candidate is actually a precise match
    # for (e.g. Generative AI Engineer), erasing the very signal that's
    # supposed to differentiate them. "scikit-learn" alone is similarly
    # weak (often used for a non-ML data pipeline), so a real "machine
    # learning" skill now requires one of the stronger, harder-to-fake
    # signals below -- the literal phrase, or a deep-learning-specific
    # framework name.
    "machine learning": ["machine learning", "tensorflow", "pytorch"],
    "account reconciliation": ["account reconciliation", "reconciliation"],
    "accounts payable": ["accounts payable", "ap process"],
    "general ledger accounting": ["general ledger", "ledger accounting"],
    "natural language processing": ["natural language processing", "nlp"],
    "retrieval augmented generation": ["retrieval augmented generation", "retrieval-augmented generation", "rag"],
    "llm integration": ["llm", "llms", "large language model", "large language models", "llm integration"],
    "prompt engineering": ["prompt engineering", "prompt design"],
    "semantic search": ["semantic search", "vector search", "similarity search"],
    "model context protocol": ["model context protocol", "mcp"],
    "workflow automation": ["workflow automation", "process automation"],
    "task automation": ["task automation", "automated tasks", "automation pipeline"],
}

ROLE_HINTS = {
    "software engineer": ["software engineer", "application developer", "backend developer"],
    "software developer": ["software developer", "developer"],
    "full stack developer": ["full stack", "frontend and backend"],
    "devops engineer": ["devops", "site reliability", "sre"],
    "data analyst": ["data analyst", "business analyst"],
    # Keep AI/ML Engineer titles from being generalized into Data Scientist.
    # A data-science role should be inferred from an explicit title only.
    "data scientist": ["data scientist"],
    "accounts executive": ["accounts executive", "accountant", "finance executive"],
}

# Common cross-domain methods that should be retained when explicitly stated.
# This is an evidence list, not a closed vocabulary: LLM and section-token terms
# remain open-vocabulary.
EXPLICIT_ROLE_PHRASES = (
    "junior site engineer", "trainee civil engineer", "civil engineer",
    "entry-level data analyst", "data analyst", "junior data analyst",
    "junior software engineer", "software engineer", "software developer",
    "junior business analyst", "business analyst", "business analysis intern",
)


EXPLICIT_SKILL_PHRASES = (
    "structural analysis", "construction planning", "site execution",
    "technical drawing interpretation", "quantity estimation",
    "embankment filling", "field density testing", "compaction",
    "exploratory data analysis", "data cleaning", "kpi reporting",
    "dashboard design", "statistical analysis", "data visualization",
    "rest api design", "unit testing", "debugging",
    "object-oriented programming", "asynchronous task processing",
    "containerization", "requirements gathering", "requirements elicitation",
    "stakeholder interviews", "user stories", "acceptance criteria",
    "process mapping", "gap analysis", "uat coordination",
    "business process documentation", "process improvement",
)


TOOL_HINTS = {
    "git": ["git", "github", "gitlab", "bitbucket"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "jira": ["jira"],
    "excel": ["excel", "spreadsheets"],
    "sap": ["sap", "sap erp"],
    "power bi": ["power bi"],
    "langchain": ["langchain"],
    "chromadb": ["chromadb", "chroma db", "chroma"],
    "openai": ["openai", "open ai", "gpt", "chatgpt"],
    "huggingface": ["huggingface", "hugging face", "transformers"],
    "postgresql": ["postgresql", "postgres"],
    "pandas": ["pandas"],
    "streamlit": ["streamlit"],
    "beautifulsoup": ["beautifulsoup", "beautiful soup", "bs4"],
}


def _contains_term(text: str, term: str) -> bool:
    # Algorithm: Use regex boundary matching to detect whole terms while tolerating separators like spaces, hyphens, and slashes.
    """Boundary-aware term matching for keyword fallback"""
    term_pattern = re.escape(term).replace(r"\ ", r"[\s\-/&]+")
    pattern = rf"(?<![a-z0-9]){term_pattern}(?![a-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _extract_section_tokens(text: str):
    """Extract individual items from skill/tool sections without category labels.

    Resume PDFs commonly put labels and values on the same line, for example
    "Design & Structural Analysis Tools: AutoCAD, STAAD-Pro". Treating the
    whole left-hand label as a skill pollutes the profile with phrases that
    are not skills. Conversely, treating every line beginning with "Tools:"
    as a section heading drops the actual tool names. This handles both.
    """
    lines = [re.sub(r"\\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    section_headers = {
        "skills", "technical skills", "core competencies", "expertise", "tools",
        "technical expertise", "key skills", "areas of expertise",
    }
    stop_headers = {
        "experience", "professional experience", "work experience", "work history",
        "employment", "employment history", "education", "educational qualifications",
        "academic qualifications", "projects", "academic projects", "personal projects",
        "summary", "professional summary", "profile", "certifications",
        "certifications & specialized training", "internship", "languages",
    }

    def header_and_payload(line):
        if ":" in line:
            left, right = line.split(":", 1)
            return re.sub(r"\\s+", " ", left.lower()).strip(), right.strip(), True
        return re.sub(r"\\s+", " ", line.lower()).strip(), "", False

    def is_header(line, vocabulary):
        label, payload, has_colon = header_and_payload(line.strip().strip("•▪▫*- "))
        return label in vocabulary and not payload

    tokens = set()
    capture = False
    collected_lines = 0
    category_labels = {
        "tools", "tool", "software", "software tools", "design tools",
        "design & structural analysis tools", "structural analysis tools",
        "office & documentation", "documentation", "technical tools",
        "programming languages", "languages", "frameworks", "libraries",
        "platforms", "technologies", "technology stack", "tech stack",
        "skills", "technical skills", "core competencies", "expertise",
        "methodologies", "databases", "cloud platforms",
    }

    for raw_line in lines:
        line = raw_line.strip().strip("•▪▫*- ")
        label, payload, has_colon = header_and_payload(line)

        if is_header(line, section_headers):
            capture = True
            collected_lines = 0
            continue

        if capture and is_header(line, stop_headers):
            break

        # Support a section title and values on the same line, such as
        # "Technical Skills: Python, SQL".
        if not capture and has_colon and label in section_headers and payload:
            capture = True
            collected_lines = 0
            line = payload
            has_colon = False
        elif capture and has_colon:
            # Category labels are metadata; only the right-hand values are
            # candidate terms. Unknown colon labels are handled the same way.
            if label in category_labels or len(label.split()) <= 6:
                line = payload

        if not capture:
            continue

        parts = re.split(r"[,;|•●▪▫]+", line)
        for part in parts:
            token = re.sub(r"\\s+", " ", part).strip(" -:\\t")
            token = re.sub(r"^(?:and|or)\\s+", "", token, flags=re.I).strip()
            if not (2 <= len(token) <= 50):
                continue
            if token.lower() in category_labels or len(token.split()) > 7:
                continue
            if re.match(r"^(?:tools?|skills?|software|office|documentation)\\s*:", token, re.I):
                continue
            tokens.add(token.lower())

        collected_lines += 1
        if collected_lines >= 20:
            break

    return tokens

def _extract_entities_ner(text: str):
    # Algorithm: Apply spaCy PhraseMatcher-based NER to extract known skills, roles, and tools from text.
    """
    PRIMARY METHOD: Extract entities using spaCy NER (PhraseMatcher)
    This is the NER algorithm used in the project
    """
    nlp, matcher = _get_nlp_and_matcher()
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

    # Preserve explicitly listed, open-vocabulary items from resume skill/tool
    # sections. Known tools and titles are removed from the skill category later
    # by _normalize_entity_categories, while novel methods remain available.
    found_skills.update(section_tokens)
    for phrase in EXPLICIT_SKILL_PHRASES:
        if _contains_term(normalized_text, phrase):
            found_skills.add(phrase)

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


# ==========================================
# LLM Fallback (open-vocabulary extraction)
# ==========================================
# The NER path and the keyword-fallback path above both share the same
# ceiling: they can only ever find terms that already exist in
# skills.txt / roles.txt / tools.txt (or the SKILL_HINTS / ROLE_HINTS /
# TOOL_HINTS dictionaries). A resume using a framework or domain term
# that isn't in those lists yet -- a new library, a niche specialty --
# will silently under-extract, no matter how good the matching regex is.
#
# This fallback breaks that ceiling by asking an LLM to extract entities
# directly from the text with no fixed vocabulary. It's deliberately the
# LAST resort in extract_entities()'s tiered flow (list-match -> hint
# boost -> LLM), since it's slower and costs an API call, unlike the
# free/instant paths above it -- it should only fire when the resume's
# terminology has genuinely outrun the static vocabulary.

LOW_SIGNAL_THRESHOLD = 5

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# ^ Groq periodically deprecates model names (this project has already
# been bitten once during development -- llama-3.1-70b-versatile, then
# its replacement llama-3.3-70b-versatile, were both retired within the
# span of this project). Configurable via env var so a future deprecation
# is a one-line env change, not a code edit. Check
# https://console.groq.com/docs/deprecations for the current status.

_groq_client_cache = {"client": None, "checked": False}


def _get_groq_client():
    """Lazily build (and cache) a Groq client from GROQ_API_KEY.
    Returns None if the groq package isn't installed or no key is set --
    callers must treat that as "LLM fallback unavailable" and degrade
    gracefully, not raise."""
    if _groq_client_cache["checked"]:
        return _groq_client_cache["client"]

    _groq_client_cache["checked"] = True
    if Groq is None:
        return None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    _groq_client_cache["client"] = Groq(api_key=api_key)
    return _groq_client_cache["client"]


LLM_EXTRACTION_PROMPT = """You are extracting structured career information from resume text.
Read the resume text below and return ONLY a JSON object (no markdown fences, no commentary) with this exact shape:
{{
  "skills": ["...", "..."],
  "roles": ["...", "..."],
  "tools": ["...", "..."]
}}

Rules:
- "skills" = methodologies, techniques, and domains of expertise (e.g. "retrieval augmented generation", "prompt engineering", "supply chain management").
- "tools" = named software, frameworks, libraries, platforms, or products (e.g. "LangChain", "PostgreSQL", "Power BI").
- "roles" = job titles or role archetypes the candidate has held or is targeting (e.g. "software engineer", "data analyst").
- Lowercase every value.
- Do not invent anything not implied by the text.
- If a category has nothing, return an empty list for it.

Resume text:
---
{resume_text}
---
"""


def _clean_llm_list(values):
    if not isinstance(values, list):
        return []
    return sorted(set(str(v).strip().lower() for v in values if str(v).strip()))


MAX_CHUNK_CHARS = 6000
MAX_LLM_CHUNKS = 4  # safety cap: an unusually huge "resume" (garbage OCR
                     # text, a bad upstream extraction) shouldn't trigger
                     # unbounded API spend. Real resumes fit in 1-2 chunks.


def _chunk_text_for_llm(text, max_chars=MAX_CHUNK_CHARS):
    """Split text into chunks that respect paragraph boundaries (blank
    lines) instead of cutting at a fixed character count -- a straight
    text[:max_chars] truncation silently drops whatever comes after the
    cutoff (e.g. a candidate's most recent job or their education section)
    with no indication anything was lost. Falls back to a hard split only
    for a single paragraph that's itself longer than max_chars, so this
    always terminates."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) > max_chars:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
            current = ""
        else:
            current = para
    if current:
        chunks.append(current)
    return chunks


def _extract_entities_llm(text: str):
    """
    FALLBACK METHOD 2 (last resort): LLM-based open-vocabulary extraction.

    Only reached when the vocabulary-based paths above are still thin.
    Unlike list-matching, this has no fixed-vocabulary ceiling -- at the
    cost of being slower, non-deterministic, and dependent on an external
    API call. The resume is processed in paragraph-aware chunks (see
    _chunk_text_for_llm) rather than truncated, so nothing past an
    arbitrary character cutoff is silently dropped; results from every
    chunk are merged. Any single chunk's failure (malformed response,
    transient API error) is skipped rather than failing the whole
    extraction -- whatever the other chunks found is still kept. If no
    client is configured at all, this degrades to empty results, since
    this is an enhancement layer, not a required one.
    """
    client = _get_groq_client()
    if client is None:
        return {"skills": [], "roles": [], "tools": []}

    chunks = _chunk_text_for_llm(text)[:MAX_LLM_CHUNKS]

    merged = {"skills": [], "roles": [], "tools": []}
    for chunk in chunks:
        prompt = LLM_EXTRACTION_PROMPT.format(resume_text=chunk)
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # Models sometimes wrap JSON in ```json ... ``` fences despite
            # being told not to -- strip that defensively rather than fail.
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
        except Exception:
            continue  # this chunk failed; keep whatever other chunks yielded

        merged["skills"].extend(_clean_llm_list(parsed.get("skills")))
        merged["roles"].extend(_clean_llm_list(parsed.get("roles")))
        merged["tools"].extend(_clean_llm_list(parsed.get("tools")))

    return {
        "skills": sorted(set(merged["skills"])),
        "roles": sorted(set(merged["roles"])),
        "tools": sorted(set(merged["tools"])),
    }


def _normalize_entity_categories(entities):
    """Deduplicate and prevent known tools/roles from leaking into skills."""
    clean = {}
    for key in ("skills", "roles", "tools"):
        vals = entities.get(key, []) or []
        clean[key] = sorted({re.sub(r"\s+", " ", str(v)).strip().lower() for v in vals if str(v).strip()})
    # Explicit tool vocabulary and hint aliases take precedence over generic skill labels.
    tool_terms = set(TOOLS) | set(TOOL_HINTS)
    tool_terms.update(alias.lower() for aliases in TOOL_HINTS.values() for alias in aliases)
    def norm(v):
        return re.sub(r"[^a-z0-9]+", "", v.lower())
    tool_norms = {norm(t) for t in tool_terms}
    role_norms = {norm(r) for r in ROLES} | {norm(r) for r in ROLE_HINTS}
    clean["tools"] = sorted(set(clean["tools"]))
    tool_norms.update(norm(t) for t in clean["tools"])
    clean["skills"] = [v for v in clean["skills"] if norm(v) not in tool_norms and norm(v) not in role_norms]
    clean["roles"] = [v for v in clean["roles"] if norm(v) not in tool_norms]
    # Exact overlaps are assigned to the most specific category.
    skill_norms = {norm(v) for v in clean["skills"]}
    clean["roles"] = [v for v in clean["roles"] if norm(v) not in skill_norms]
    return clean


def _merge_entity_lists(*entity_sets):
    """Merge categories deterministically while preserving open-vocabulary terms."""
    return {
        category: sorted({
            re.sub(r"\\s+", " ", str(value)).strip().lower()
            for entities in entity_sets
            for value in (entities.get(category, []) or [])
            if str(value).strip()
        })
        for category in ("skills", "roles", "tools")
    }


def _filter_roles_to_resume_evidence(roles, text):
    """Reject role predictions that are not explicitly evidenced by the resume.

    Job titles are especially prone to LLM hallucination from transferable
    skills (e.g. calling a civil engineer a data analyst because of generic
    words such as 'analysis'). Only keep a predicted title when its wording,
    or a conservative equivalent, occurs in the source text.
    """
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    kept = []
    for role in roles:
        normalized_role = re.sub(r"[^a-z0-9]+", " ", str(role).lower()).strip()
        if not normalized_role:
            continue
        # Require the title itself to appear as a phrase in the resume.
        if re.search(r"(?<![a-z0-9])" + re.escape(normalized_role) + r"(?![a-z0-9])", normalized_text):
            kept.append(str(role).lower().strip())
    return sorted(set(kept))


def extract_entities(text: str):
    """Extract resume entities from both the LLM and deterministic evidence paths.

    LLM extraction adds open-vocabulary terms; deterministic matching recovers
    explicitly mentioned skills/tools the model can omit. Role predictions are
    evidence-filtered to prevent unsupported inferred job titles.
    """
    empty = {"skills": [], "roles": [], "tools": []}
    if not isinstance(text, str) or not text.strip():
        return empty

    try:
        llm_results = _extract_entities_llm(text) or empty
    except Exception:
        # API/network/provider failures must not break signup extraction.
        llm_results = empty

    try:
        ner_results = _extract_entities_ner(text) or empty
    except Exception:
        ner_results = empty
    try:
        fallback_results = _extract_entities_fallback(text) or empty
    except Exception:
        fallback_results = empty

    result = _merge_entity_lists(llm_results, ner_results, fallback_results)
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    for phrase in EXPLICIT_ROLE_PHRASES:
        normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
        if re.search(r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])", normalized_text):
            result["roles"].append(phrase)
    result["roles"] = sorted(set(result["roles"]))

    # Model/heuristic role predictions must be grounded in a title explicitly
    # present in the resume. This intentionally prefers precision over guesses.
    result["roles"] = _filter_roles_to_resume_evidence(result["roles"], text)

    # Avoid inferring ML methodology from an umbrella title alone.
    explicit_ml_evidence = any(
        _contains_term(text.lower(), term)
        for term in ("machine learning", "tensorflow", "pytorch")
    )
    if not explicit_ml_evidence:
        result["skills"] = [
            skill for skill in result["skills"]
            if skill.lower() != "machine learning"
        ]

    return _normalize_entity_categories(result)
