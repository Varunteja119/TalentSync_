import unittest

from signup_prefill import build_signup_prefill_state


class SignupPrefillStateTests(unittest.TestCase):
    def test_new_resume_seeds_sections_and_leaves_unknown_dob_blank(self):
        state = build_signup_prefill_state(
            {"name": "Varun", "email": "varun@example.com", "summary": "Summary"},
            {"skills": ["Python", "RAG"], "roles": ["AI Engineer"], "tools": ["Docker"]},
            [{"degree": "B.Tech", "field": "Computer Science", "college": "VIT",
              "from": "09/2022", "to": "04/2026"}],
            [{"role": "AI Intern", "company": "Example", "date": "05/2026 - Present",
              "description": "Built an AI tool."}],
            [{"name": "TalentSync", "description": ["Built a matcher."],
              "technologies": ["Python", "Streamlit"]}],
        )
        self.assertEqual(state["signup_name"], "Varun")
        self.assertEqual(state["signup_skills"], "Python, RAG")
        self.assertIsNone(state["signup_dob"])
        self.assertEqual(state["signup_education_count"], 1)
        self.assertEqual(state["deg_manual_0"], "B.Tech")
        self.assertEqual(state["edu_date_manual_0"], "09/2022 - 04/2026")
        self.assertEqual(state["signup_experience_count"], 1)
        self.assertEqual(state["work_desc_manual_0"], "Built an AI tool.")
        self.assertEqual(state["signup_project_name_0"], "TalentSync")
        self.assertEqual(state["signup_project_description_0"], "Built a matcher.")
        self.assertEqual(state["signup_project_technologies_0"], "Python, Streamlit")

    def test_empty_extraction_does_not_create_fake_entries(self):
        state = build_signup_prefill_state({}, {}, [], [], [])
        self.assertEqual(state["signup_education_count"], 1)
        self.assertEqual(state["signup_experience_count"], 0)
        self.assertEqual(state["signup_project_count"], 0)
        self.assertIsNone(state["signup_dob"])


if __name__ == "__main__":
    unittest.main()
