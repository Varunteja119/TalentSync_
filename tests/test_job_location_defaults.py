import unittest
from urllib.parse import parse_qs, urlparse

from job_redirect import generate_job_links


class JobLocationDefaultTests(unittest.TestCase):
    def test_linkedin_defaults_to_uae(self):
        links = generate_job_links("AI Engineer")
        parsed = urlparse(links["linkedin"])
        query = parse_qs(parsed.query)
        self.assertEqual(query.get("location"), ["UAE"])
        self.assertIn("UAE", query.get("keywords", [""])[0])

    def test_blank_location_falls_back_to_uae(self):
        links = generate_job_links("AI Engineer", location="")
        query = parse_qs(urlparse(links["linkedin"]).query)
        self.assertEqual(query.get("location"), ["UAE"])

    def test_none_location_falls_back_to_uae(self):
        links = generate_job_links("AI Engineer", location=None)
        query = parse_qs(urlparse(links["linkedin"]).query)
        self.assertEqual(query.get("location"), ["UAE"])

    def test_whitespace_location_falls_back_to_uae(self):
        links = generate_job_links("AI Engineer", location="   ")
        query = parse_qs(urlparse(links["linkedin"]).query)
        self.assertEqual(query.get("location"), ["UAE"])

    def test_linkedin_experience_levels_change_filter(self):
        fresher = parse_qs(urlparse(generate_job_links("AI Engineer", experience="no_experience")["linkedin"]).query)
        entry = parse_qs(urlparse(generate_job_links("AI Engineer", experience="under_3_years_experience")["linkedin"]).query)
        experienced = parse_qs(urlparse(generate_job_links("AI Engineer", experience="more_than_3_years_experience")["linkedin"]).query)
        self.assertEqual(fresher.get("f_E"), ["2"])
        self.assertEqual(entry.get("f_E"), ["2,3"])
        self.assertEqual(experienced.get("f_E"), ["4,5,6"])
        self.assertNotEqual(fresher["f_E"], experienced["f_E"])
        # LinkedIn should receive its actual entry-level checkbox value for a fresher.
        self.assertIn("f_E=2", generate_job_links("AI Engineer", experience="no_experience")["linkedin"])

    def test_explicit_location_overrides_uae_default(self):
        links = generate_job_links("AI Engineer", location="Dubai, United Arab Emirates")
        parsed = urlparse(links["linkedin"])
        query = parse_qs(parsed.query)
        self.assertEqual(query.get("location"), ["Dubai, United Arab Emirates"])


if __name__ == "__main__":
    unittest.main()
