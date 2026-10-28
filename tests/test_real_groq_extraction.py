"""Opt-in integration tests against the real Groq model.

Run only when GROQ_API_KEY is configured. These tests make real API calls.
"""
from pathlib import Path
import os
import unittest

import skill_extractor

FIXTURES = Path(__file__).parent / "sample_resumes"


@unittest.skipUnless(
    os.environ.get("GROQ_API_KEY"),
    "GROQ_API_KEY is not configured; real-model integration test skipped",
)
class RealGroqExtractionTests(unittest.TestCase):
    CASES = [
        ("civil_engineer.txt", "structural analysis", "autocad", "junior site engineer", {"data analyst", "software engineer"}),
        ("data_analyst.txt", "exploratory data analysis", "power bi", "data analyst", {"civil engineer", "operations manager"}),
        ("software_engineer.txt", "unit testing", "fastapi", "software engineer", {"data analyst", "civil engineer"}),
        ("business_analyst.txt", "gap analysis", "jira", "business analyst", {"data analyst", "operations manager"}),
    ]

    def test_real_groq_extracts_resume_specific_entities(self):
        for filename, required_skill, required_tool, required_role, unsupported_roles in self.CASES:
            with self.subTest(resume=filename):
                text = (FIXTURES / filename).read_text(encoding="utf-8")
                result = skill_extractor.extract_entities(text)

                self.assertTrue(result["skills"], f"No skills extracted: {result}")
                self.assertTrue(result["tools"], f"No tools extracted: {result}")
                self.assertTrue(result["roles"], f"No roles extracted: {result}")
                self.assertIn(required_skill, result["skills"], result)
                self.assertIn(required_tool, result["tools"], result)
                self.assertIn(required_role, result["roles"], result)
                self.assertFalse(unsupported_roles.intersection(result["roles"]), result)


if __name__ == "__main__":
    unittest.main()
