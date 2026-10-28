"""Tests ensuring signup extraction runs once per uploaded resume."""
import unittest
from signup_extraction_cache import get_or_extract_entities


class SignupExtractionCacheTests(unittest.TestCase):
    def test_same_resume_reuses_result_without_duplicate_extractor_call(self):
        state = {}
        calls = []

        def extractor(text):
            calls.append(text)
            return {"skills": ["unit testing"], "roles": ["software engineer"], "tools": ["fastapi"]}

        first = get_or_extract_entities(state, "resume-hash-a", "resume text", extractor)
        second = get_or_extract_entities(state, "resume-hash-a", "resume text", extractor)

        self.assertEqual(first, second)
        self.assertEqual(calls, ["resume text"], "same uploaded resume must not trigger duplicate extraction")

    def test_new_resume_fingerprint_triggers_one_new_extraction(self):
        state = {}
        calls = []

        def extractor(text):
            calls.append(text)
            return {"skills": [text], "roles": [], "tools": []}

        first = get_or_extract_entities(state, "hash-a", "resume A", extractor)
        second = get_or_extract_entities(state, "hash-b", "resume B", extractor)

        self.assertEqual(first["skills"], ["resume A"])
        self.assertEqual(second["skills"], ["resume B"])
        self.assertEqual(calls, ["resume A", "resume B"])

    def test_extractor_failure_returns_empty_result_and_does_not_crash(self):
        state = {}

        def broken_extractor(_):
            raise RuntimeError("provider unavailable")

        result = get_or_extract_entities(state, "hash-a", "resume", broken_extractor)
        self.assertEqual(result, {"skills": [], "roles": [], "tools": []})
        self.assertEqual(state["_signup_processed_resume_hash"], "hash-a")


if __name__ == "__main__":
    unittest.main()
