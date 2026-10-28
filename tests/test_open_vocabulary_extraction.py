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
