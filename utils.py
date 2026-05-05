import re


def _clean_lines(text):
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        # Fix OCR/PDF extraction artifacts like "2025AI" -> "2025 AI"
        line = re.sub(r"((?:19|20)\d{2})([A-Za-z])", r"\1 \2", line)
        if line:
            lines.append(line)
    return lines


def _find_section(lines, section_keys, all_section_keys):
    start = None
    for idx, line in enumerate(lines):
        upper = line.upper()
        if any(key in upper for key in section_keys):
            start = idx + 1
            break

    if start is None:
        return []

    end = len(lines)
    for idx in range(start, len(lines)):
        upper = lines[idx].upper()
        if any(key in upper for key in all_section_keys):
            end = idx
            break

    return lines[start:end]


def _looks_like_education_line(line):
    lower = line.lower()
    edu_markers = [
        "bachelor", "master", "phd", "diploma", "b.tech", "m.tech", "b.e", "m.e",
        "b.sc", "m.sc", "mba", "college", "university", "institute", "school",
        "cgpa", "gpa", "percentage", "sslc", "hsc", "12th", "10th",
    ]
    noisy_markers = [
        "designed", "developed", "implemented", "delivered", "managed", "led", "trained",
        "compliance", "module", "software", "project", "intern", "engineer",
    ]

    if any(marker in lower for marker in noisy_markers):
        return False

    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", line))
    has_marker = any(marker in lower for marker in edu_markers)
    return has_marker or has_year


def _parse_role_company(text):
    value = text.strip(" -|,:")
    if not value:
        return "", ""
    if " at " in value.lower():
        parts = re.split(r"\s+at\s+", value, flags=re.IGNORECASE, maxsplit=1)
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    if "|" in value:
        parts = [p.strip() for p in value.split("|", 1)]
        return parts[0], parts[1] if len(parts) > 1 else ""
    return value, ""


def _normalize_month_year(value):
    if not value:
        return ""

    raw = value.strip()
    if raw.lower() == "present":
        return "Present"

    mm_yyyy = re.fullmatch(r"(0?[1-9]|1[0-2])/(19|20)\d{2}", raw)
    if mm_yyyy:
        month = int(mm_yyyy.group(1))
        year = raw.split("/")[1]
        return f"{month:02d}/{year}"

    month_names = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    month_year = re.fullmatch(r"([A-Za-z]{3,9})\s+((?:19|20)\d{2})", raw)
    if month_year:
        mon = month_year.group(1).lower()[:3]
        yr = month_year.group(2)
        if mon in month_names:
            return f"{month_names[mon]}/{yr}"

    month_dash_year = re.fullmatch(r"([A-Za-z]{3,9})\s*[-/]\s*((?:19|20)\d{2})", raw)
    if month_dash_year:
        mon = month_dash_year.group(1).lower()[:3]
        yr = month_dash_year.group(2)
        if mon in month_names:
            return f"{month_names[mon]}/{yr}"

    year_only = re.fullmatch(r"((?:19|20)\d{2})", raw)
    if year_only:
        return f"01/{year_only.group(1)}"

    return raw


def _extract_date_range(text):
    month_piece = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    date_piece = rf"(?:{month_piece}\s*(?:-|/)?\s*(?:19|20)\d{{2}}|(?:0?[1-9]|1[0-2])/(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
    range_re = re.compile(
        rf"({date_piece})\s*(?:-|–|to)\s*({date_piece}|Present|Current)",
        re.IGNORECASE,
    )
    match = range_re.search(text)
    if not match:
        return "", ""
    return _normalize_month_year(match.group(1)), _normalize_month_year(match.group(2))


def _looks_like_contact_line(line):
    lower = line.lower()
    markers = ["email", "phone", "linkedin", "github", "portfolio", "@"]
    return any(m in lower for m in markers)


def _looks_like_education_text(line):
    lower = line.lower()
    edu_markers = [
        "b.tech", "bachelor", "master", "phd", "diploma", "cgpa", "gpa", "percentage",
        "secondary", "higher secondary", "school", "university", "institute", "college",
    ]
    return any(m in lower for m in edu_markers)


def _looks_like_job_text(line):
    lower = line.lower()
    job_markers = [
        "engineer", "developer", "intern", "analyst", "manager", "lead", "consultant",
        "specialist", "officer", "associate", "executive", "architect", "scientist",
        "trainee", "administrator", "coordinator", "supervisor",
    ]
    company_markers = ["pvt", "ltd", "llp", "inc", "corp", "technologies", "solutions", "systems"]
    return any(m in lower for m in job_markers + company_markers)


def _is_mostly_date_line(line):
    cleaned = re.sub(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*", "", line, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:19|20)\d{2}", "", cleaned)
    cleaned = re.sub(r"[-–to|,:/]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", "", cleaned)
    return len(cleaned) <= 3


def _extract_experience_from_lines(lines):
    month_piece = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    date_piece = rf"(?:{month_piece}\s*(?:-|/)?\s*(?:19|20)\d{{2}}|(?:0?[1-9]|1[0-2])/(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
    range_re = re.compile(
        rf"({date_piece})\s*(?:-|–|to)\s*({date_piece}|Present|Current)",
        re.IGNORECASE,
    )
    experience = []

    for idx, line in enumerate(lines):
        if _looks_like_contact_line(line):
            continue

        candidate_line = line
        match = range_re.search(candidate_line)
        used_next_line_for_range = False
        if not match and idx + 1 < len(lines):
            candidate_line = f"{line} {lines[idx + 1]}"
            match = range_re.search(candidate_line)
            used_next_line_for_range = bool(match)
        if not match:
            continue

        from_date = match.group(1).strip()
        to_date = match.group(2).strip()
        from_date = _normalize_month_year(from_date)
        to_date = _normalize_month_year(to_date)

        line_wo_range = range_re.sub("", candidate_line).strip(" -|,")
        prev_line = lines[idx - 1] if idx > 0 and not _looks_like_contact_line(lines[idx - 1]) else ""
        role_source = line_wo_range if line_wo_range else prev_line
        role, company = _parse_role_company(role_source)

        if not role and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if not _looks_like_contact_line(next_line):
                role, company = _parse_role_company(next_line)

        combined = f"{role} {company}".strip()
        if not combined:
            continue
        if _looks_like_contact_line(combined) or _looks_like_education_text(combined):
            continue
        if not _looks_like_job_text(combined):
            continue

        description = ""
        desc_idx = idx + 2 if used_next_line_for_range else idx + 1
        if desc_idx < len(lines):
            next_line = lines[desc_idx].strip()
            if next_line and len(next_line.split()) > 3 and len(next_line) < 180 and not _looks_like_contact_line(next_line):
                description = next_line

        if not company and description:
            desc_lower = description.lower()
            company_markers = ["pvt", "ltd", "llp", "inc", "corp", "technologies", "solutions", "systems"]
            if any(m in desc_lower for m in company_markers) or "," in description:
                company = description
                description = ""

        experience.append(
            {
                "role": role,
                "company": company,
                "from": from_date,
                "to": to_date,
                "description": description,
            }
        )

    return experience[:5]


def _infer_degree(line):
    lower = line.lower()
    if "class xii" in lower or "higher secondary" in lower or "senior secondary" in lower:
        return "Senior Secondary"
    if "class x" in lower or ("secondary" in lower and "senior" not in lower):
        return "Secondary"
    if "phd" in lower or "doctor" in lower:
        return "PhD"
    if "master" in lower or "m.tech" in lower or "mba" in lower or "m.sc" in lower:
        return "Masters"
    if "bachelor" in lower or "b.tech" in lower or "b.e" in lower or "b.sc" in lower:
        return "Bachelors"
    if "diploma" in lower:
        return "Diploma"
    return ""


def _extract_structured_education(edu_lines):
    month = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    date_only_re = re.compile(rf"^({month}\s*[-/]\s*(?:19|20)\d{{2}})\s*[-–]?$", re.IGNORECASE)
    date_with_text_re = re.compile(rf"^({month}\s*[-/]\s*(?:19|20)\d{{2}})\s*(.*)$", re.IGNORECASE)

    start_indices = [idx for idx, line in enumerate(edu_lines) if date_only_re.search(line.strip())]
    if not start_indices:
        return []

    entries = []
    boundaries = start_indices + [len(edu_lines)]

    for pos in range(len(start_indices)):
        start = boundaries[pos]
        end = boundaries[pos + 1]
        chunk = edu_lines[start:end]
        if not chunk:
            continue

        from_match = date_only_re.search(chunk[0].strip())
        from_raw = from_match.group(1) if from_match else ""
        edu_from = _normalize_month_year(from_raw)
        edu_to = ""
        detail_lines = []

        if len(chunk) > 1:
            to_match = date_with_text_re.search(chunk[1].strip())
            if to_match:
                edu_to = _normalize_month_year(to_match.group(1))
                trailing = to_match.group(2).strip(" -")
                if trailing:
                    detail_lines.append(trailing)
                detail_lines.extend(chunk[2:])
            else:
                detail_lines.extend(chunk[1:])

        degree_line = next((ln for ln in detail_lines if _infer_degree(ln)), "")
        degree = _infer_degree(degree_line) or "Bachelors"

        college = next(
            (ln for ln in detail_lines if any(k in ln.lower() for k in ["college", "university", "institute", "school"])),
            "",
        )

        field = ""
        for ln in detail_lines:
            lower = ln.lower()
            if ln == college:
                continue
            if "cgpa" in lower or "gpa" in lower or "percentage" in lower:
                continue
            if _is_mostly_date_line(ln):
                continue
            field = ln
            break

        year = edu_to.split("/")[-1] if edu_to else (edu_from.split("/")[-1] if edu_from else "")

        if field or college or year:
            entries.append(
                {
                    "degree": degree,
                    "field": field,
                    "college": college,
                    "year": year,
                    "from": edu_from,
                    "to": edu_to,
                }
            )

    return entries[:5]


def extract_education_experience(text):
    lines = _clean_lines(text)
    all_keys = [
        "EDUCATION",
        "EDUCATIONAL QUALIFICATION",
        "ACADEMIC QUALIFICATION",
        "ACADEMICS",
        "ACADEMIC",
        "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "WORK EXPERIENCE",
        "WORK HISTORY",
        "EMPLOYMENT",
        "EMPLOYMENT HISTORY",
        "PROJECT",
        "SKILLS",
        "CERTIFICATION",
        "SUMMARY",
        "PROFILE",
    ]

    edu_lines = _find_section(
        lines,
        ["EDUCATION", "EDUCATIONAL QUALIFICATION", "ACADEMIC QUALIFICATION", "ACADEMIC", "ACADEMICS"],
        all_keys,
    )
    exp_lines = _find_section(
        lines,
        ["EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "WORK HISTORY", "EMPLOYMENT", "EMPLOYMENT HISTORY"],
        all_keys,
    )

    year_re = re.compile(r"\b(?:19|20)\d{2}\b")
    education = _extract_structured_education(edu_lines)
    current_edu = None

    if not education:
        for line in edu_lines:
            if not _looks_like_education_line(line):
                continue

            degree = _infer_degree(line)
            year_match = year_re.search(line)
            year = year_match.group(0) if year_match else ""
            edu_from, edu_to = _extract_date_range(line)
            has_date_range = bool(edu_from or edu_to)

            if has_date_range and current_edu and (current_edu.get("field") or current_edu.get("college")):
                if current_edu:
                    education.append(current_edu)
                current_edu = {
                    "degree": degree or "Bachelors",
                    "field": "",
                    "college": "",
                    "year": year,
                    "from": edu_from,
                    "to": edu_to,
                }
                continue

            if not current_edu:
                current_edu = {
                    "degree": degree or "Bachelors",
                    "field": "",
                    "college": "",
                    "year": year,
                    "from": edu_from,
                    "to": edu_to,
                }

            if degree and not current_edu["degree"]:
                current_edu["degree"] = degree

            if year and not current_edu["year"]:
                current_edu["year"] = year
            if edu_to:
                current_edu["year"] = edu_to.split("/")[-1]

            if edu_from and not current_edu["from"]:
                current_edu["from"] = edu_from
            if edu_to and not current_edu["to"]:
                current_edu["to"] = edu_to

            if year and not current_edu["from"]:
                current_edu["from"] = _normalize_month_year(year)
            if year and not current_edu["to"]:
                current_edu["to"] = _normalize_month_year(year)

            lower = line.lower()
            if any(k in lower for k in ["college", "university", "institute", "school"]) and not current_edu["college"]:
                current_edu["college"] = line
                continue

            if any(k in lower for k in ["cgpa", "gpa", "percentage", "%", "field", "major", "specialization"]) and not current_edu["field"]:
                current_edu["field"] = line
                continue

            if _is_mostly_date_line(line):
                continue

            if not current_edu["field"] and len(line.split()) <= 18:
                current_edu["field"] = line
            elif not current_edu["college"] and len(line.split()) <= 18:
                current_edu["college"] = line

        if current_edu:
            education.append(current_edu)

    if not education:
        fallback_edu = []
        for line in lines:
            if _looks_like_contact_line(line):
                continue
            if _looks_like_job_text(line):
                continue
            if not _looks_like_education_line(line):
                continue

            degree = _infer_degree(line) or "Bachelors"
            year_match = year_re.search(line)
            year = year_match.group(0) if year_match else ""
            edu_from, edu_to = _extract_date_range(line)

            entry = {
                "degree": degree,
                "field": "",
                "college": "",
                "year": year,
                "from": edu_from,
                "to": edu_to,
            }

            lower = line.lower()
            if any(k in lower for k in ["college", "university", "institute", "school"]):
                entry["college"] = line
            else:
                entry["field"] = line

            fallback_edu.append(entry)

        education = fallback_edu[:5]

    experience = _extract_experience_from_lines(exp_lines)
    if not experience:
        # Fallback for resumes with weak/missing section headers.
        experience = _extract_experience_from_lines(lines)

    return {
        "education": [e for e in education[:5] if e.get("field") or e.get("college") or e.get("year")],
        "experience": experience[:5],
    }


def extract_basic_info(text):
    email_matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    phone_matches = re.findall(
        r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?)?[\d\s\-]{8,14}\d",
        text,
    )
    phone = ""
    for raw in phone_matches:
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 15:
            phone = raw.strip()
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = ""
    for line in lines[:8]:
        lower = line.lower()
        if "powered by" in lower or "resume" in lower and "gemini" in lower:
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,60}", line):
            name = line
            break
    if not name and lines:
        name = lines[0]

    location = ""
    loc_match = re.search(r"(?:location|address)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if loc_match:
        location = loc_match.group(1).strip()

    return {
        "name": name,
        "email": email_matches[0] if email_matches else "",
        "phone": phone,
        "location": location,
    }