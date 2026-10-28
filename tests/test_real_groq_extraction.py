"""Opt-in integration tests against the real Groq model.

These tests intentionally make real API calls and run only in GitHub Actions
when GROQ_API_KEY is configured. They verify open-vocabulary domain extraction
from representative resumes rather than mocked responses.
"""
from pathlib import Path
import os
import pytest

import skill_extractor

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY is not configured; real-model integration test skipped",
)

FIXTURES = Path(__file__).parent / "sample_resumes"


@pytest.mark.parametrize(
    ("filename", "required_skill", "required_tool", "required_role"),
    [
        ("civil_engineer.txt", "structural analysis", "autocad", "junior site engineer"),
        ("data_analyst.txt", "exploratory data analysis", "power bi", "data analyst"),
        ("software_engineer.txt", "unit testing", "fastapi", "software engineer"),
        ("business_analyst.txt", "gap analysis", "jira", "business analyst"),
    ],
)
def test_real_groq_extracts_resume_specific_entities(
    filename, required_skill, required_tool, required_role
):
    text = (FIXTURES / filename).read_text(encoding="utf-8")
    result = skill_extractor.extract_entities(text)

    assert result["skills"], f"No skills extracted from {filename}: {result}"
    assert result["tools"], f"No tools extracted from {filename}: {result}"
    assert result["roles"], f"No roles extracted from {filename}: {result}"

    # These terms are explicitly present in each fixture and test whether
    # extraction captures domain-specific terms, not merely generic keywords.
    assert required_skill in result["skills"], (filename, result)
    assert required_tool in result["tools"], (filename, result)
    assert required_role in result["roles"], (filename, result)
