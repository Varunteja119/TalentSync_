# normalizer.py
import re

# ==========================================
# MILITARY RANK NORMALIZATION
# ==========================================
RANK_MAP = {
    # Officers (O-1 to O-10)
    "second lieutenant": "senior professional",
    "2lt": "senior professional",
    "first lieutenant": "senior professional",
    "1lt": "senior professional",
    "captain": "senior professional",
    "capt": "senior professional",
    "major": "manager",
    "maj": "manager",
    "lieutenant colonel": "senior manager",
    "ltc": "senior manager",
    "colonel": "director",
    "col": "director",
    "brigadier general": "executive director",
    "major general": "vice president",
    "lieutenant general": "senior vice president",
    "general": "c-tier executive",
    
    # Enlisted (E-1 to E-9)
    "private": "junior professional",
    "pvt": "junior professional",
    "private first class": "junior professional",
    "pfc": "junior professional",
    "corporal": "professional",
    "cpl": "professional",
    "sergeant": "senior professional",
    "sgt": "senior professional",
    "staff sergeant": "senior professional",
    "ssgt": "senior professional",
    "sergeant first class": "supervisor",
    "sfc": "supervisor",
    "master sergeant": "supervisor",
    "msgt": "supervisor",
    "first sergeant": "team lead",
    "1sg": "team lead",
    "master sergeant major": "manager",
    "sergeant major": "manager",
    "command sergeant major": "senior manager",
    "csm": "senior manager",
    "sergeant major of the army": "director",
}

# ==========================================
# COMPREHENSIVE MILITARY ROLE MAPPINGS
# ==========================================
ROLE_MAP = {
    # Command & Leadership
    "platoon commander": "team leader",
    "squad leader": "team leader",
    "section leader": "team lead",
    "company commander": "operations manager",
    "battalion commander": "operations director",
    "regiment commander": "senior director",
    "first sergeant": "team lead",
    "command sergeant major": "operations manager",
    
    # Communications & Signals
    "signals officer": "network engineer",
    "communications specialist": "network engineer",
    "radio operator": "communication systems technician",
    "signal operations": "network operations",
    "telecommunications": "network engineer",
    "information systems": "systems administrator",
    
    # Intelligence & Surveillance
    "intelligence officer": "data analyst",
    "intelligence specialist": "business analyst",
    "military intelligence": "data intelligence analyst",
    "surveillance officer": "security analyst",
    "counterintelligence specialist": "security researcher",
    "analyst": "data analyst",
    
    # Logistics & Supply
    "logistics officer": "supply chain manager",
    "supply officer": "supply chain manager",
    "quartermaster": "inventory manager",
    "supply specialist": "supply chain specialist",
    "warehouse management": "logistics coordinator",
    "transport coordinator": "logistics coordinator",
    "procurement officer": "procurement specialist",
    
    # Engineering & Maintenance
    "combat engineer": "civil engineer",
    "military engineer": "engineer",
    "maintenance officer": "maintenance engineer",
    "maintenance technician": "equipment technician",
    "equipment operator": "operations technician",
    "vehicle maintenance": "fleet maintenance technician",
    "aircraft maintenance": "aerospace technician",
    
    # Medical
    "combat medic": "healthcare technician",
    "medical officer": "healthcare manager",
    "hospital corpsman": "medical technician",
    "field medic": "emergency medical technician",
    
    # Training & Education
    "training officer": "training manager",
    "instructor": "training specialist",
    "curriculum developer": "instructional designer",
    "military trainer": "corporate trainer",
    
    # Security & Law Enforcement
    "security officer": "security manager",
    "military police": "law enforcement officer",
    "security specialist": "security analyst",
    "physical security": "security coordinator",
    "cybersecurity": "cybersecurity analyst",
    
    # Administration
    "administrative officer": "office manager",
    "personnel specialist": "human resources specialist",
    "human resources": "hr specialist",
    "records manager": "documentation specialist",
    "payroll specialist": "payroll coordinator",
    
    # Operations & Planning
    "operations officer": "operations manager",
    "tactical planner": "project manager",
    "mission planner": "project coordinator",
    "strategic planner": "strategic analyst",
    
    # Aviation
    "pilot": "pilot",
    "air traffic controller": "air traffic controller",
    "flight engineer": "flight engineer",
    "crew chief": "operations supervisor",
    
    # Explosives & Specialized
    "explosive ordnance disposal": "hazmat technician",
    "eod specialist": "specialized technician",
    "bomb technician": "hazmat specialist",
}

# ==========================================
# COMPREHENSIVE MILITARY SKILL MAPPINGS
# ==========================================
SKILL_MAP = {
    # ===== TACTICAL & STRATEGIC =====
    "tactical planning": "strategic planning",
    "tactical operations": "operations management",
    "tactical assessment": "risk assessment",
    "operational planning": "project planning",
    "strategic planning": "strategic planning",
    "mission planning": "project planning",
    "crisis management": "incident management",
    "emergency response": "incident response",
    
    # ===== LEADERSHIP & MANAGEMENT =====
    "command and control": "team leadership",
    "team leadership": "team leadership",
    "team management": "team leadership",
    "personnel supervision": "team management",
    "team supervision": "team management",
    "leader development": "talent development",
    "performance management": "performance management",
    "decision making": "decision making",
    "conflict resolution": "conflict resolution",
    
    # ===== COMMUNICATION =====
    "radio communication": "communication systems",
    "communication systems": "communication systems",
    "signal operations": "network operations",
    "liaison operations": "stakeholder management",
    "coordinated communication": "cross-functional communication",
    "briefing": "presentation skills",
    "report writing": "technical writing",
    
    # ===== COMBAT & COMBAT ARMS =====
    "combat operations": "field operations",
    "weapon systems": "systems operation",
    "vehicle operation": "equipment operation",
    "marksmanship": "precision execution",
    "land navigation": "field navigation",
    "military tactics": "operational planning",
    "battlefield coordination": "operations coordination",
    
    # ===== INTELLIGENCE & SECURITY =====
    "threat analysis": "risk analysis",
    "threat assessment": "security assessment",
    "surveillance operations": "monitoring",
    "counterintelligence": "security operations",
    "intelligence gathering": "data collection",
    "security clearance": "background verification",
    "information security": "cybersecurity",
    "data classification": "information management",
    
    # ===== LOGISTICS & SUPPLY CHAIN =====
    "supply chain management": "supply chain management",
    "inventory control": "inventory management",
    "warehouse management": "warehouse operations",
    "procurement": "purchasing",
    "material handling": "logistics operations",
    "transportation management": "transport coordination",
    "resource allocation": "resource management",
    "demand planning": "forecasting",
    
    # ===== TECHNICAL & MAINTENANCE =====
    "equipment maintenance": "preventive maintenance",
    "vehicle maintenance": "fleet maintenance",
    "technical maintenance": "technical support",
    "technical troubleshooting": "troubleshooting",
    "mechanical repair": "equipment repair",
    "electrical systems": "electrical engineering",
    "hydraulic systems": "systems engineering",
    "preventive maintenance": "preventive maintenance",
    
    # ===== OPERATIONS & COORDINATION =====
    "operations management": "operations",
    "operations coordination": "operations coordination",
    "scheduling": "scheduling",
    "dispatch": "logistics coordination",
    "resource management": "resource management",
    "process optimization": "process optimization",
    "workflow management": "process management",
    
    # ===== TRAINING & DEVELOPMENT =====
    "training and development": "training",
    "training": "training",
    "instruction": "instructional delivery",
    "curriculum development": "curriculum design",
    "performance evaluation": "performance assessment",
    "mentoring": "mentoring",
    "coaching": "coaching",
    
    # ===== ADMINISTRATION =====
    "documentation": "technical documentation",
    "record keeping": "records management",
    "report preparation": "report writing",
    "data entry": "data management",
    "office administration": "office administration",
    "office management": "office management",
    "payroll coordination": "payroll administration",
    "scheduling": "calendar management",
    
    # ===== COMPLIANCE & REGULATIONS =====
    "compliance": "compliance management",
    "regulatory compliance": "regulatory affairs",
    "safety protocols": "safety management",
    "occupational safety": "workplace safety",
    "environmental compliance": "environmental management",
    "quality assurance": "quality assurance",
    "audit": "audit",
    
    # ===== PROBLEM SOLVING & ANALYTICAL =====
    "problem solving": "problem-solving",
    "analytical thinking": "data analysis",
    "critical thinking": "critical analysis",
    "situation assessment": "assessment",
    "root cause analysis": "root cause analysis",
    
    # ===== PHYSICAL & SPECIALIZED =====
    "physical fitness": "physical training",
    "field operations": "field operations",
    "parachute operations": "specialized operations",
    "diving operations": "specialized operations",
    "survival training": "field training",
    
    # ===== GENERAL CLEANUP =====
    "military operations": "operations management",
    "military training": "training",
    "armed forces": "professional experience",
    "military service": "professional experience",
}

# ==========================================
# SERVICE BRANCH CONTEXT
# ==========================================
BRANCH_CONTEXT = {
    "army": "land operations and logistics",
    "navy": "maritime operations and engineering",
    "air force": "aviation and systems engineering",
    "marines": "assault operations and leadership",
    "coast guard": "maritime security and operations",
    "space force": "space operations and technology",
}

# ==========================================
# DEPARTMENT OF DEFENSE CONTRACTORS
# ==========================================
DOD_INDUSTRIES = [
    "defense contractor",
    "aerospace",
    "defense technology",
    "defense logistics",
    "government contracting",
    "federal contractor",
]

def normalize_entities(entities: dict):
    """
    Normalize military terminology to civilian equivalents.
    
    Process:
    1. Normalize roles using ROLE_MAP
    2. Normalize skills using SKILL_MAP
    3. Infer additional civilian skills from military context
    4. Preserve tools (already civilian)
    
    Returns: Normalized entity dict with civilian terminology
    """
    normalized_roles = []
    normalized_skills = []

    # ===== NORMALIZE ROLES =====
    for role in entities.get("roles", []):
        role_lower = role.lower().strip()
        # Direct mapping
        if role_lower in ROLE_MAP:
            normalized_roles.append(ROLE_MAP[role_lower])
        # Partial matching for compound terms
        else:
            matched = False
            for military_role, civilian_role in ROLE_MAP.items():
                if military_role in role_lower or role_lower in military_role:
                    normalized_roles.append(civilian_role)
                    matched = True
                    break
            if not matched:
                normalized_roles.append(role)

    # ===== NORMALIZE SKILLS =====
    for skill in entities.get("skills", []):
        skill_lower = skill.lower().strip()
        # Direct mapping
        if skill_lower in SKILL_MAP:
            normalized_skills.append(SKILL_MAP[skill_lower])
        # Partial matching
        else:
            matched = False
            for military_skill, civilian_skill in SKILL_MAP.items():
                if military_skill in skill_lower or skill_lower in military_skill:
                    normalized_skills.append(civilian_skill)
                    matched = True
                    break
            if not matched:
                normalized_skills.append(skill)

    # ===== NORMALIZE RANKS (IF PRESENT) =====
    for rank in [r.lower() for r in entities.get("roles", [])]:
        if rank in RANK_MAP:
            normalized_roles.append(RANK_MAP[rank])

    return {
        "skills": list(set(normalized_skills)),  # Remove duplicates
        "roles": list(set(normalized_roles)),     # Remove duplicates
        "tools": entities.get("tools", []),
        "military_context": {
            "is_veteran": True if entities.get("roles") else False,
            "normalized": True
        }
    }