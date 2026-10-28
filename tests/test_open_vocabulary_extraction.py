"""Regression tests for resume-driven open-vocabulary entity extraction."""
from pathlib import Path
import skill_extractor

FIXTURES = Path(__file__).parent / "sample_resumes"


def test_civil_engineering_open_vocabulary_terms_are_kept(monkeypatch):
    text = (FIXTURES / "civil_engineer.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(skill_extractor, "_extract_entities_llm", lambda _: {
        "skills": ["structural analysis", "embankment filling", "field density testing"],
        "roles": ["junior site engineer", "trainee civil engineer"],
        "tools": ["autocad", "staad.pro", "arcgis", "qgis", "erdas imagine 2015", "primavera p6"],
    })
    result = skill_extractor.extract_entities(text)
    assert {"structural analysis", "embankment filling", "field density testing"} <= set(result["skills"])
    assert {"autocad", "staad.pro", "arcgis", "qgis", "erdas imagine 2015", "primavera p6"} <= set(result["tools"])
    assert "junior site engineer" in result["roles"]


def test_other_professions_use_llm_result_without_static_vocabulary_gate(monkeypatch):
    expected = {
        "data_analyst.txt": {
            "skills": ["exploratory data analysis", "kpi reporting"],
            "roles": ["data analyst"],
            "tools": ["power bi", "tableau", "postgresql"],
        },
        "software_engineer.txt": {
            "skills": ["asynchronous task processing", "unit testing"],
            "roles": ["software engineer"],
            "tools": ["fastapi", "celery", "github actions"],
        },
        "business_analyst.txt": {
            "skills": ["requirements elicitation", "acceptance criteria", "gap analysis"],
            "roles": ["business analyst"],
            "tools": ["jira", "confluence", "visio"],
        },
    }
    for filename, entities in expected.items():
        text = (FIXTURES / filename).read_text(encoding="utf-8")
        monkeypatch.setattr(skill_extractor, "_extract_entities_llm", lambda _, e=entities: e)
        result = skill_extractor.extract_entities(text)
        for category in ("skills", "roles", "tools"):
            assert set(entities[category]) <= set(result[category]), (filename, category, result)


def test_fallback_is_used_when_llm_returns_no_entities(monkeypatch):
    text = (FIXTURES / "civil_engineer.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(skill_extractor, "_extract_entities_llm", lambda _: {
        "skills": [], "roles": [], "tools": []
    })
    monkeypatch.setattr(skill_extractor, "_extract_entities_ner", lambda _: {
        "skills": ["problem solving"], "roles": [], "tools": []
    })
    monkeypatch.setattr(skill_extractor, "_extract_entities_fallback", lambda _: {
        "skills": ["site execution"], "roles": ["junior site engineer"], "tools": ["autocad"]
    })
    result = skill_extractor.extract_entities(text)
    assert "problem solving" in result["skills"]
    assert "site execution" in result["skills"]
    assert "autocad" in result["tools"]


def test_empty_resume_returns_empty_categories():
    assert skill_extractor.extract_entities("   ") == {"skills": [], "roles": [], "tools": []}


def test_project_title_wrapped_over_two_lines_and_tools_in_description():
    from utils import extract_projects

    resume = """KANAMARLAPUTI BHUVANESH
SUMMARY
Civil Engineer.
EDUCATION
B.Tech
TECHNICAL SKILLS
Design & Structural Analysis Tools: AutoCAD, STAAD-Pro
Tools: ArcGIS, QGIS, ERDAS Imagine 2015, Primavera P6
ACADEMIC PROJECTS
Modelling Urban Heat Island And Urban Hotspot And Their Correlation With Satellite-Derived Land Use Indices
For The Past Three Decades In Salem, Tamil Nadu. Mar 2022 - Jul 2022
Analyzed 30 years of thermal trends in Salem, Tamil Nadu, by modeling Urban Heat Island (UHI) and Urban Hotspots using Landsat satellite imagery.
Derived Land Surface Temperature (LST) and calculated land use indices (NDVI, NDBI, NDBaI, MNDWI) using ArcGIS, QGIS, and ERDAS Imagine.
"""
    projects = extract_projects(resume)
    assert len(projects) == 1, projects
    project = projects[0]
    assert project["name"] == (
        "Modelling Urban Heat Island And Urban Hotspot And Their Correlation "
        "With Satellite-Derived Land Use Indices For The Past Three Decades "
        "In Salem, Tamil Nadu"
    ), project
    assert "For The Past Three Decades" in project["name"]
    assert "Mar 2022" not in project["name"]
    assert "Analyzed 30 years" in project["description"]
    assert "Project period: Mar 2022 – Jul 2022" in project["description"]
    assert {"ArcGIS", "QGIS", "ERDAS Imagine", "Landsat"} <= set(project["technologies"]), project
