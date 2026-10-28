import unittest

from skill_extractor import _extract_section_tokens


class SkillSectionTokenizationTests(unittest.TestCase):
    def test_category_labels_are_not_returned_as_skills(self):
        resume = """TECHNICAL SKILLS
        Design & Structural Analysis Tools: AutoCAD, STAAD-Pro
        Tools: ArcGIS, QGIS, ERDAS Imagine 2015, Primavera P6
        Office & Documentation: MS Office, MS Excel
        PROFESSIONAL EXPERIENCE
        Construction of a road using compaction and field density tests.
        """
        tokens = _extract_section_tokens(resume)

        for expected in (
            "autocad", "staad-pro", "arcgis", "qgis",
            "erdas imagine 2015", "primavera p6", "ms office", "ms excel",
        ):
            self.assertIn(expected, tokens)

        for unwanted in (
            "design & structural analysis tools: autocad",
            "tools: arcgis",
            "office & documentation: ms office",
            "construction of a road using compaction and field density tests.",
        ):
            self.assertNotIn(unwanted, tokens)

    def test_inline_section_heading_keeps_its_values(self):
        tokens = _extract_section_tokens(
            "SUMMARY\nAnalyst\nTechnical Skills: Python, SQL, Power BI\n"
            "EXPERIENCE\nBuilt a dashboard with Python."
        )
        self.assertTrue({"python", "sql", "power bi"}.issubset(tokens))
        self.assertNotIn("technical skills: python", tokens)

    def test_generic_category_labels_do_not_depend_on_one_resume(self):
        resume = """CORE COMPETENCIES
        Data Analysis: Excel, SQL, Tableau
        Cloud Platforms: AWS, Google Cloud
        PROJECTS
        Retail analytics dashboard
        """
        tokens = _extract_section_tokens(resume)
        self.assertTrue(
            {"excel", "sql", "tableau", "aws", "google cloud"}.issubset(tokens)
        )
        self.assertNotIn("data analysis: excel", tokens)
        self.assertNotIn("cloud platforms: aws", tokens)


if __name__ == "__main__":
    unittest.main()
