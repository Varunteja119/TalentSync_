import ast
import re
import unittest
from pathlib import Path


class SkillNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "role_matcher.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        wanted = {"_SKILL_ALIASES", "_SKILL_QUALIFIERS", "_skill_key"}
        nodes = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
                 and any(
                     (isinstance(target, ast.Name) and target.id in wanted)
                     for target in (node.targets if isinstance(node, ast.Assign) else [node])
                 )]
        namespace = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        cls.normalize = staticmethod(namespace["_skill_key"])

    def test_rag_variants_normalize_to_same_skill(self):
        self.assertEqual(self.normalize("RAG"), "rag")
        self.assertEqual(self.normalize("retrieval-augmented generation"), "rag")
        self.assertEqual(self.normalize("retrieval augmented generation"), "rag")

    def test_nlp_variants_normalize_to_same_skill(self):
        self.assertEqual(self.normalize("NLP"), "natural language processing")
        self.assertEqual(self.normalize("natural language processing"), "natural language processing")

    def test_llm_variants_normalize_to_llm_integration(self):
        self.assertEqual(self.normalize("LLM integration"), "llm integration")
        self.assertEqual(self.normalize("large language models"), "llm integration")

    def test_qualifier_words_are_removed(self):
        self.assertEqual(self.normalize("Advanced NLP"), "natural language processing")


if __name__ == "__main__":
    unittest.main()
