import unittest

from utils import extract_projects


class ProjectExtractionTests(unittest.TestCase):
    def test_extracts_multiple_projects_and_stops_at_skills(self):
        resume_text = """
Projects
TalentSync – AI Career Intelligence & Role Matching Platform
• Developed a Streamlit-based career intelligence platform that extracts skills, roles, and tools from resumes using NLP and document-processing pipelines.
• Implemented military-to-civilian skill normalization and hybrid job-role matching using SentenceTransformer embeddings, semantic similarity, and skill-overlap scoring.
RetailPulse – E-Commerce Sales Analytics
• Built an end-to-end sales analytics pipeline using Python ETL, data cleaning, and PostgreSQL to process and analyze e-commerce data.
• Created SQL views and interactive Power BI dashboards using DAX to analyze sales performance, customer trends, and business KPIs.
Technical Skills & Certifications
Languages: Python, SQL, JavaScript
"""
        projects = extract_projects(resume_text)
        self.assertEqual(len(projects), 2)
        self.assertEqual(projects[0]["name"], "TalentSync – AI Career Intelligence & Role Matching Platform")
        self.assertIn("military-to-civilian skill normalization", projects[0]["description"])
        self.assertEqual(projects[1]["name"], "RetailPulse – E-Commerce Sales Analytics")
        self.assertIn("Power BI dashboards", projects[1]["description"])
        self.assertNotIn("Languages: Python", projects[1]["description"])

    def test_returns_empty_when_no_projects_section(self):
        self.assertEqual(extract_projects("Education\nB.Tech in Computer Science\nSkills\nPython, SQL"), [])


if __name__ == "__main__":
    unittest.main()
