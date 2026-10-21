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

    def test_explicit_location_overrides_uae_default(self):
        links = generate_job_links("AI Engineer", location="Dubai, United Arab Emirates")
        parsed = urlparse(links["linkedin"])
        query = parse_qs(parsed.query)
        self.assertEqual(query.get("location"), ["Dubai, United Arab Emirates"])


if __name__ == "__main__":
    unittest.main()
