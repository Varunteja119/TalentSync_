import json
import os
import unittest


class AIJobRoleCoverageTests(unittest.TestCase):
    def test_ai_and_generative_ai_roles_are_available(self):
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "role_profiles", "role_skill_level_profiles.json")
        with open(path, encoding="utf-8") as handle:
            profiles = json.load(handle)

        expected = {
            "AI Engineer",
            "Generative AI Engineer",
            "LLM Application Engineer",
            "AI Automation Engineer",
            "NLP Engineer",
        }
        self.assertTrue(expected.issubset(profiles.keys()))
        for role in expected:
            self.assertTrue(profiles[role]["core_skills"], f"{role} should have core skills")
            self.assertTrue(profiles[role]["optional_skills"], f"{role} should have optional skills")


if __name__ == "__main__":
    unittest.main()
