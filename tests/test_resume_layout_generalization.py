"""General layout-variation regressions for resume extraction.

These examples exercise layout patterns, not a single person's resume.
"""
import re

from skill_extractor import _extract_entities_fallback
from utils import extract_education_experience, extract_projects


def test_education_qualification_lines_are_not_used_as_specializations():
    resume = """EDUCATION
B.Tech Jun 2019 - Aug 2023
Example University, Thanjavur 8.2 CGPA
Intermediate Jun 2017 - Apr 2019
Sample Junior College, Vijayawada 9.9 CGPA
High School Jun 2016 - Apr 2017
Northside High School, Kurnool 9.8 GPA
TECHNICAL SKILLS
Python, SQL
"""
    education = extract_education_experience(resume)["education"]
    assert len(education) == 3, education
    assert [item["degree"] for item in education] == ["Bachelors", "Intermediate", "High School"]
    assert "Example University" in education[0]["college"]
    assert "Sample Junior College" in education[1]["college"]
    assert "Northside High School" in education[2]["college"]
    assert all(not item["field"] for item in education), education
    assert all(not re.search(r"\b(?:CGPA|GPA)\b", item["college"], re.I) for item in education)


def test_wrapped_project_title_with_date_on_continuation_line_stays_one_project():
    resume = """ACADEMIC PROJECTS
Evaluating Urban Heat Patterns and Their Relationship With Satellite-Derived
For The Past Three Decades In a Regional Study. Mar 2022 - Jul 2022
Analyzed historical temperature trends using satellite imagery.
Derived land-surface temperature and land-use indices using GeoTool, MapSuite, and RasterLab.
EXPERIENCE
Site Engineer Jan 2024 - May 2025
"""
    projects = extract_projects(resume)
    assert len(projects) == 1, projects
    project = projects[0]
    assert project["name"] == (
        "Evaluating Urban Heat Patterns and Their Relationship With Satellite-Derived "
        "For The Past Three Decades In a Regional Study"
    ), project
    assert "Project period: Mar 2022 – Jul 2022" in project["description"]
    assert "Analyzed historical temperature trends" in project["description"]


def test_tool_extraction_uses_resume_labels_not_a_closed_product_list():
    resume = """TECHNICAL SKILLS
Design Tools: GeoTool, MapSuite, RasterLab
Software: PlanBoard, FieldSuite
EXPERIENCE
Site Engineer Jan 2024 - May 2025
"""
    result = _extract_entities_fallback(resume)
    assert {"geotool", "mapsuite", "rasterlab", "planboard", "fieldsuite"} <= set(result["tools"]), result


def test_role_and_employer_are_separated_from_a_colon_heading():
    resume = """PROFESSIONAL EXPERIENCE
Bridge construction at Example City : Site Engineer Jan 2024 – May 2025
• Coordinated field execution and quality inspections.
"""
    experience = extract_education_experience(resume)["experience"]
    assert len(experience) == 1, experience
    assert experience[0]["role"] == "Site Engineer", experience[0]
    assert experience[0]["company"] == "Bridge construction at Example City", experience[0]
