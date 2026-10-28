import unittest

from utils import extract_education_experience, extract_projects


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

    def test_internship_employer_is_not_replaced_by_work_description(self):
        resume = """WORK EXPERIENCE
        INTERNSHIP South Central Railway : Trainee Civil Engineer
        01/2023 - 04/2023
        • Involved in initial stages of 6-lane highway construction, focusing on site clearance and setting out work.
        • Executed excavation, levelling, and embankment filling as per approved design profiles and MoRTH specifications.
        • Monitored earthwork formation including compaction, moisture control, and layer-wise quality checks using field density tests.
        """
        result = extract_education_experience(resume)
        self.assertEqual(len(result["experience"]), 1)
        entry = result["experience"][0]
        self.assertEqual(entry["role"], "Trainee Civil Engineer")
        self.assertEqual(entry["company"], "South Central Railway")
        self.assertIn("6-lane highway construction", entry["description"])
        self.assertIn("field density tests", entry["description"])
        self.assertNotIn("highway construction", entry["company"])

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


    def test_civil_engineer_resume_preserves_project_and_experience_fields(self):
        # Regression fixture transcribed from the real one-page Civil Engineer PDF.
        resume = """KANAMARLAPUTI BHUVANESH
        SUMMARY
        Detail-oriented and enthusiastic Civil Engineer, with a strong foundation in structural analysis, construction planning,
        and site execution. Proficient in industry-standard tools like AutoCAD, STAAD Pro, and MS Office. Skilled in interpreting
        technical drawings, estimating quantities, and coordinating with multidisciplinary teams. Known for strong problem-
        solving ability, on-site adaptability, and commitment to delivering quality and safety in every phase of construction.
        Eager to contribute to impactful infrastructure projects with a forward-thinking engineering firm.
        EDUCATION
        B.Tech Jun 2019 - Aug 2023
        SASTRA Deemed to be University, Thanjavur, TamilNadu 7.8 CGPA
        Intermediate Jun 2017 - April 2019
        Sri Chaitanya Junior College, Vijayawada, Andhra Pradesh 9.94 CGPA
        High School Jun 2016 - April 2017
        Indus High School, Kurnool, Andhra Pradesh 9.8 GPA
        TECHNICAL SKILLS
        Design & Structural Analysis Tools: AutoCAD, STAAD-Pro
        Tools: ArcGIS, QGIS, ERDAS Imagine 2015, Primavera P6
        Office & Documentation: MS Office, MS Excel
        PROFESSIONAL EXPERIENCE
        Construction of 6 Lane ring road at Yadagirigutta : Junior Site Engineer Jan 2024 – May 2025
        • Involved in initial stages of 6-lane highway construction, focusing on site clearance and setting out work.
        • Executed excavation, levelling, and embankment filling as per approved design profiles and MoRTH specifications.
        • Monitored earthwork formation including compaction, moisture control, and layer-wise quality checks using field
        density tests.
        INTERNSHIP
        South Central Railway : Trainee Civil Engineer Jan 2023 – Apr 2023
        • Worked on the Project Title: STUDY ON EXECUTION OF RUB USING BOX PUSHING TECHNIQUE.
        • Studied the execution of a Rail Under Bridge (RUB) using the Box Pushing Technique for a canal crossing in
        Guntur, Andhra Pradesh.
        • Involved in casting of RCC thrust beds, shields, and segmental boxes, followed by hydraulic jacking operations
        under active railway lines.
        • Gained hands-on exposure to large-scale infrastructure execution without traffic disruption, IS/IRS codes, and site
        management practices.
        CERTIFICATIONS & SPECIALIZED TRAINING
        Primavera P6 Project Management Training
        • Completed comprehensive training using Primavera P6 software.
        • Covered modules on: Work Breakdown Structure (WBS), Activities, Relationships, Project Calendars and
        Codes, Scheduling and Constraints.
        ACADEMIC PROJECTS
        Modelling Urban Heat Island And Urban Hotspot And Their Correlation With Satellite-Derived Land Use Indices
        For The Past Three Decades In Salem, Tamil Nadu. Mar 2022 - Jul 2022
        • Analyzed 30 years of thermal trends in Salem, Tamil Nadu, by modeling Urban Heat Island (UHI) and Urban
        Hotspots using Landsat satellite imagery.
        • Derived Land Surface Temperature (LST) and calculated land use indices (NDVI, NDBI, NDBaI, MNDWI) using
        ArcGIS, QGIS, and ERDAS Imagine.
        """
        projects = extract_projects(resume)
        self.assertEqual(len(projects), 1, projects)
        project = projects[0]
        self.assertEqual(
            project["name"],
            "Modelling Urban Heat Island And Urban Hotspot And Their Correlation With Satellite-Derived Land Use Indices For The Past Three Decades In Salem, Tamil Nadu",
        )
        self.assertIn("Mar 2022 – Jul 2022", project["description"])
        self.assertIn("30 years of thermal trends", project["description"])
        self.assertIn("NDVI, NDBI, NDBaI, MNDWI", project["description"])
        for tool in ("ArcGIS", "QGIS", "ERDAS Imagine", "Landsat"):
            self.assertIn(tool, project["technologies"])

        result = extract_education_experience(resume)
        education = result["education"]
        self.assertEqual(len(education), 3, education)
        self.assertEqual(education[0]["degree"], "Bachelors")
        self.assertIn("SASTRA Deemed to be University", education[0]["college"])
        self.assertNotIn("7.8 CGPA", education[0]["college"])
        self.assertEqual(education[1]["degree"], "Intermediate")
        self.assertIn("Sri Chaitanya Junior College", education[1]["college"])
        self.assertNotIn("9.94 CGPA", education[1]["college"])
        self.assertEqual(education[2]["degree"], "High School")
        self.assertIn("Indus High School", education[2]["college"])
        self.assertNotIn("9.8 GPA", education[2]["college"])
        experience = result["experience"]
        self.assertEqual(len(experience), 2, experience)
        junior = next(item for item in experience if "Junior Site Engineer" in item["role"])
        self.assertEqual(junior["company"], "Construction of 6 Lane ring road at Yadagirigutta")
        self.assertIn("6-lane highway construction", junior["description"])
        self.assertIn("field density tests", junior["description"])
        trainee = next(item for item in experience if "Trainee Civil Engineer" in item["role"])
        self.assertEqual(trainee["company"], "South Central Railway")
        self.assertIn("Box Pushing Technique", trainee["description"])
        self.assertIn("hydraulic jacking operations", trainee["description"])

if __name__ == "__main__":
    unittest.main()
