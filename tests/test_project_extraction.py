import unittest

from utils import extract_projects


class ExtractProjectsTests(unittest.TestCase):
    def test_extracts_multiple_projects_and_stops_at_skills(self):
        resume = """SUMMARY
        Data science graduate
        PROJECTS
        TalentSync – AI Career Intelligence & Role Matching Platform
        • Developed a Streamlit-based career intelligence platform that extracts skills and roles.
        • Implemented hybrid job-role matching using SentenceTransformer embeddings.
        Technologies: Python, Streamlit, SentenceTransformers
        RetailPulse – E-Commerce Sales Analytics
        • Built an end-to-end analytics pipeline using Python and PostgreSQL.
        • Created SQL views and Power BI dashboards to analyze business KPIs.
        TECHNICAL SKILLS
        Python, SQL, Power BI
        EXPERIENCE
        Data Analyst Intern
        """
        projects = extract_projects(resume)
        self.assertEqual(len(projects), 2)
        self.assertEqual(projects[0]["name"], "TalentSync – AI Career Intelligence & Role Matching Platform")
        self.assertIn("hybrid job-role matching", projects[0]["description"])
        self.assertEqual(projects[0]["technologies"], ["Python", "Streamlit", "SentenceTransformers"])
        self.assertEqual(projects[1]["name"], "RetailPulse – E-Commerce Sales Analytics")
        self.assertIn("Power BI dashboards", projects[1]["description"])
        self.assertNotIn("TECHNICAL SKILLS", str(projects))

    def test_bullet_without_space_is_not_misread_as_project_title(self):
        resume = """Projects
        TalentSync – AI Career Intelligence & Role Matching Platform
        •Developed a Streamlit platform for resume matching.
        •Implemented military-to-civilian skill normalization using SentenceTransformer embeddings, semantic similarity, and skill-overlap scoring.
        RetailPulse – E-Commerce Sales Analytics
        •Built a sales analytics pipeline with PostgreSQL.
        """
        projects = extract_projects(resume)
        self.assertEqual(len(projects), 2)
        self.assertEqual(
            projects[0]["name"],
            "TalentSync – AI Career Intelligence & Role Matching Platform",
        )
        self.assertIn("Implemented military-to-civilian", projects[0]["description"])
        self.assertEqual(
            projects[1]["name"],
            "RetailPulse – E-Commerce Sales Analytics",
        )

    def test_private_use_pdf_bullets_are_not_misread_as_project_titles(self):
        resume = """PROJECTS
        TalentSync - AI Career Intelligence & Role Matching Platform
        \uf0b7Developed a platform that extracts skills and roles using NLP and document processing
        \uf0b7Implemented hybrid job-role matching using SentenceTransformer
        embeddings, semantic similarity, and skill-overlap scoring.
        \uf0b7Built recommendation and resume-generation workflows to convert
        unstructured resumes into actionable career insights.
        RetailPulse - E-Commerce Sales Analytics
        \uf0b7Built a sales analytics pipeline with Python and PostgreSQL.
        """
        projects = extract_projects(resume)
        self.assertEqual(len(projects), 2)
        self.assertEqual(projects[0]["name"], "TalentSync - AI Career Intelligence & Role Matching Platform")
        self.assertIn("Implemented hybrid job-role matching", projects[0]["description"])
        self.assertIn("embeddings, semantic similarity", projects[0]["description"])
        self.assertIn("unstructured resumes into actionable career insights", projects[0]["description"])
        self.assertEqual(projects[1]["name"], "RetailPulse - E-Commerce Sales Analytics")

    def test_missing_projects_section_returns_empty_list(self):
        self.assertEqual(extract_projects("SUMMARY\nA graduate\nSKILLS\nPython, SQL"), [])

    def test_bullet_lines_do_not_become_project_titles(self):
        resume = """Projects
        My Project
        • Built a dashboard.
        """
        projects = extract_projects(resume)
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["name"], "My Project")
        self.assertEqual(projects[0]["description"], "Built a dashboard.")


if __name__ == "__main__":
    unittest.main()
