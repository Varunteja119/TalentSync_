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
    def _has_whole_word_key(line_upper, keys):
        # Allow a simple trailing plural ("PROJECT" -> also matches
        # "PROJECTS", "CERTIFICATION" -> also matches "CERTIFICATIONS")
        # so a resume's actual header wording doesn't slip past the
        # boundary check just because it's plural.
        return any(re.search(rf"\b{re.escape(key)}S?\b", line_upper) for key in keys)

    def _is_section_header_line(line, keys):
        # A real section header is short — just the heading itself
        # ("Experience", "Work Experience"), not a keyword occurring
        # incidentally inside a longer sentence of body text (e.g. a
        # Summary paragraph mentioning "hands-on experience").
        if len(line.split()) > 4:
            return False
        return _has_whole_word_key(line.upper(), keys)

    start = None
    for idx, line in enumerate(lines):
        if _is_section_header_line(line, section_keys):
            start = idx + 1
            break

    if start is None:
        return []

    end = len(lines)
    for idx in range(start, len(lines)):
        if _is_section_header_line(lines[idx], all_section_keys):
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

    # Whole-word match only. Plain substring containment let "engineer"
    # match inside "Engineering" (e.g. "B.Tech in Computer Science and
    # Engineering"), silently discarding a real degree line as if it were
    # job-description text. Word boundaries keep "Software Engineer" (an
    # actual job-title fragment) filtered out while leaving "Engineering"
    # (an academic field) alone.
    if any(re.search(rf"\b{re.escape(marker)}\b", lower) for marker in noisy_markers):
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
    # Common resume format: "Role – Company" / "Role - Company", using an
    # en dash, em dash, or hyphen as the separator.
    dash_match = re.split(r"\s+[-–—]\s+", value, maxsplit=1)
    if len(dash_match) == 2:
        return dash_match[0].strip(), dash_match[1].strip()
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


def _strip_date_range(text):
    """Remove a matched date-range substring from text, so it doesn't bleed
    into a college/field value that shares a line with it (common with
    PDF text extraction, where columns collapse onto one line)."""
    month_piece = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    date_piece = rf"(?:{month_piece}\s*(?:-|/)?\s*(?:19|20)\d{{2}}|(?:0?[1-9]|1[0-2])/(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
    range_re = re.compile(
        rf"({date_piece})\s*(?:-|–|to)\s*({date_piece}|Present|Current)",
        re.IGNORECASE,
    )
    cleaned = range_re.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip(" -–,")


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
    # Lines already used to build an entry (either as the title/role line
    # itself, or as a lookahead line that supplied the date range) must not
    # be re-examined as the start of a *new* entry — otherwise a date range
    # sitting on its own line (e.g. title on one line, "Jun 2023 - Aug 2023"
    # on the next) gets matched twice: once via lookahead from the title
    # line, and again when the loop naturally reaches the date line itself,
    # which then falls back to the *same* previous line for role/company
    # and manufactures a duplicate entry.
    consumed_indices = set()

    for idx, line in enumerate(lines):
        if idx in consumed_indices:
            continue

        if _looks_like_contact_line(line):
            continue

        # A bullet/description line is never itself the start of a job
        # entry — treating one as a candidate causes it to get stitched
        # together with the *next* entry's title whenever that next line
        # happens to carry the date range, producing a duplicate, garbled
        # entry. Bullet text still gets picked up correctly as a
        # `description` value by the entries that legitimately match below.
        if re.match(r"^\s*[•\-\*]", line):
            continue

        # A bullet that wraps across two physical PDF lines only has the
        # marker on its first line — the continuation line has none, but
        # is still body text, not a new entry. Continuation lines almost
        # always start mid-sentence (lowercase), while a real role/company
        # title line starts capitalized — use that to filter it out too.
        first_alpha = next((c for c in line if c.isalpha()), "")
        if first_alpha and first_alpha.islower():
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

        consumed_indices.add(idx)
        if used_next_line_for_range:
            consumed_indices.add(idx + 1)

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
            first_desc_line = lines[desc_idx].strip()
            if (
                first_desc_line
                and len(first_desc_line.split()) > 3
                and not _looks_like_contact_line(first_desc_line)
            ):
                # Collect every bullet under this entry, not just the first.
                # A bullet line (marker) or its wrapped continuation
                # (starts lowercase, mid-sentence) is body text belonging
                # to this job; keep consuming those. Stop only when we hit
                # something that isn't body text for this entry: a blank
                # or contact line, the next entry's date range, or a line
                # that looks like a new entry's title (capitalized start,
                # not a bullet, not a continuation).
                first_is_bullet = bool(re.match(r"^\s*[•\-\*]", first_desc_line))
                desc_entries = [(first_desc_line, first_is_bullet)]
                consumed_indices.add(desc_idx)
                scan_idx = desc_idx + 1
                while scan_idx < len(lines):
                    candidate = lines[scan_idx].strip()
                    if not candidate or _looks_like_contact_line(candidate):
                        break
                    if range_re.search(candidate):
                        break  # this line carries a date — it's the next entry
                    is_bullet = bool(re.match(r"^\s*[•\-\*]", candidate))
                    cand_first_alpha = next((c for c in candidate if c.isalpha()), "")
                    is_continuation = bool(cand_first_alpha) and cand_first_alpha.islower()
                    if not (is_bullet or is_continuation):
                        # A wrapped bullet can resume with a capitalized
                        # word (a tool/product name, e.g. "...a production-
                        # ready" / "Streamlit dashboard..."), which looks
                        # identical to a new entry's title line by
                        # capitalization alone. Only treat it as a real
                        # new entry — and stop — when it actually reads
                        # like a job/education title; otherwise keep
                        # accumulating it as body text.
                        if _looks_like_job_text(candidate) or _looks_like_education_text(candidate):
                            break

                    desc_entries.append((candidate, is_bullet))
                    consumed_indices.add(scan_idx)
                    scan_idx += 1
                    if sum(len(p) for p, _ in desc_entries) > 800:
                        break

                # Join distinct bullets with a newline (so downstream
                # consumers like resume_generator.py's _split_bullets can
                # still tell them apart and render a proper bullet list),
                # but glue a wrapped continuation line onto the bullet it
                # belongs to with a plain space.
                pieces = []
                for i, (text, starts_bullet) in enumerate(desc_entries):
                    cleaned = re.sub(r"^[•\-\*]\s*", "", text).strip()
                    if not cleaned:
                        continue
                    if i == 0:
                        pieces.append(cleaned)
                    elif starts_bullet:
                        pieces.append("\n" + cleaned)
                    else:
                        pieces.append(" " + cleaned)
                description = "".join(pieces).strip()

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
    if "class xii" in lower or "higher secondary" in lower or "senior secondary" in lower or "12th" in lower or "grade 12" in lower or "hsc" in lower:
        return "Senior Secondary"
    if "class x" in lower or "10th" in lower or "grade 10" in lower or "ssc" in lower or ("secondary" in lower and "senior" not in lower):
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
        degree = _infer_degree(degree_line)

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
                education.append(current_edu)
                current_edu = {
                    "degree": degree,
                    "field": "",
                    "college": "",
                    "year": year,
                    "from": edu_from,
                    "to": edu_to,
                }
                # Fall through (no `continue`) so this same line can still
                # contribute its college/field text below — otherwise a line
                # like "Our Own English High School... Mar 2021 – Apr 2022"
                # would have its school name silently dropped.

            if not current_edu:
                current_edu = {
                    "degree": degree,
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

            cleaned_line = _strip_date_range(line) if has_date_range else line
            lower = cleaned_line.lower()

            if any(k in lower for k in ["college", "university", "institute", "school"]) and not current_edu["college"]:
                current_edu["college"] = cleaned_line
                continue

            if any(k in lower for k in ["cgpa", "gpa", "percentage", "%", "field", "major", "specialization"]) and not current_edu["field"]:
                current_edu["field"] = cleaned_line
                continue

            if _is_mostly_date_line(cleaned_line):
                continue

            if not current_edu["field"] and len(cleaned_line.split()) <= 18:
                current_edu["field"] = cleaned_line
            elif not current_edu["college"] and len(cleaned_line.split()) <= 18:
                current_edu["college"] = cleaned_line

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

            degree = _infer_degree(line)
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

    linkedin = ""
    linkedin_match = re.search(
        r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?",
        text,
        flags=re.IGNORECASE,
    )
    if linkedin_match:
        linkedin = linkedin_match.group(0)
        if not linkedin.lower().startswith("http"):
            linkedin = "https://" + linkedin

    return {
        "name": name,
        "email": email_matches[0] if email_matches else "",
        "phone": phone,
        "location": location,
        "linkedin": linkedin,
    }