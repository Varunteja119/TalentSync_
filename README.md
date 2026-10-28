# TalentSync

TalentSync is a resume-to-role matching engine that combines vocabulary-based entity extraction, rule-based parsing, LLM-assisted open-vocabulary extraction, and transformer-based semantic similarity. It parses unstructured resumes, extracts skills/roles/tools, and ranks candidate-role fit using a hybrid scoring model — with a validation and sensitivity-analysis pipeline used to tune and justify the scoring weights rather than picking them arbitrarily.

This project exists to demonstrate applied NLP/ML engineering: entity extraction, embedding-based similarity, hybrid scoring design, tiered fallback architecture, and a real evaluation methodology — not just orchestrating a single LLM call.

## What this demonstrates

- **Hybrid entity extraction, not a single method.** The current `extract_entities()` implementation calls the Groq LLM first, then spaCy `PhraseMatcher` against curated skill/role/tool vocabularies, then the rule-based fallback. Their outputs are merged and filtered. Important: despite comments describing the LLM as a last resort, the current runtime code calls it unconditionally when extraction receives non-empty text (unless the Groq client/key is unavailable).
- **Hybrid semantic + symbolic scoring, not pure vibes-based LLM ranking.** Role fit is a weighted combination of SBERT cosine similarity (semantic fit), literal skill-overlap ratio (symbolic, explainable), and a role-match bonus — each with a documented formula and rationale, not a black-box prompt.
- **An actual evaluation methodology.** `validation_test_suite.py` runs labeled test cases against the matcher and reports pass/partial/fail rates; `sensitivity_analysis.py` sweeps the scoring formula's weight configuration across multiple combinations and reports which setting performs best, with results written to a JSON report and visualized via `graph.py`. This is the same validate → tune → re-validate loop used for any ML system's hyperparameters — on the 29-case suite the current 0.55 / 0.35 / 0.10 weighting sits on a plateau (7 of 9 swept configurations tie at 96.6%, the two skill-heavy ones score 94.8%), so the sweep supports these weights without sharply identifying them.
- **Engineering discipline beyond the ML pieces.** Passwords are hashed with PBKDF2-HMAC-SHA256 with transparent migration of any legacy plaintext account on next login; resume parsing has dedicated unit-style test coverage for its regex/heuristic edge cases; performance-sensitive paths (like SBERT embeddings) are cached rather than recomputed per request.

## How it works

1. **Extraction** (`text_extractor.py`) — pulls text from PDF, DOCX, TXT, or image resumes, falling back to OCR (Tesseract + PyMuPDF rendering) when direct text extraction yields nothing meaningful.
2. **Entity extraction** (`skill_extractor.py`) — three complementary methods whose results are merged:
   - Groq LLM extraction is called first for open-vocabulary skills, roles, and tools. It uses a prompt-defined JSON schema and processes long resumes in paragraph-aware chunks (up to four chunks).
   - spaCy `PhraseMatcher` runs against curated skill/role/tool phrase lists loaded from `ner_resources/skills.txt`, `roles.txt`, and `tools.txt` (with a root-level fallback for `roles.txt`). `PhraseMatcher` is dictionary/phrase matching, not spaCy's trained general-purpose NER component.
   - Keyword/regex and section-parsing fallback uses vocabulary hints and context-inference rules to recover additional evidence.
   - The outputs are merged, then role predictions and selected skill inferences are filtered. Note: the implementation currently calls the LLM before the deterministic methods; `LOW_SIGNAL_THRESHOLD` is defined but does not gate the LLM call in `extract_entities()`.
3. **Role matching** (`role_matcher.py`) — computes, for every role profile:
   - Semantic similarity (SBERT embedding cosine similarity between candidate profile and role description)
   - Skill overlap ratio (matched core skills / required core skills)
   - Role match bonus (explicit signal if the candidate already named this role)
   - Combines them via a documented weighted formula, applies qualification thresholds, and ranks results
4. **Output** — ranked role recommendations with a transparent score breakdown, plus an ATS-formatted resume generator (`resume_generator.py`) that turns a structured profile into a downloadable `.docx`.

## Architecture decisions worth calling out

**Why combine vocabulary matching, rules, and an LLM?** PhraseMatcher is fast and consistent for known terms, the rule-based path handles resume-section patterns and useful hints, and the LLM can recognize terminology outside the curated lists. In the current implementation, however, the LLM is called first on each non-empty extraction request rather than being gated by the deterministic results. A true cost-saving fallback would require changing `extract_entities()` to run deterministic methods first and consult the LLM only when a defined signal/coverage threshold is not met.

**Why hybrid semantic + skill-overlap scoring instead of asking an LLM to just rank the roles?** Skill overlap is fully explainable — you can point to exactly which required skills matched. Semantic similarity captures fit that a literal keyword match would miss (e.g. recognizing that RAG/LLM-application experience is relevant to a "Generative AI Engineer" role even if the exact core-skill strings don't line up perfectly). Combining both, with weights tuned via the sensitivity analysis pipeline rather than picked arbitrarily, gives a scoring system that's both accurate and auditable.

**Why cache embeddings?** SBERT inference is the most expensive step in the pipeline. Role profile embeddings are static (they only depend on the role definitions, not the candidate), so they're computed once and cached rather than recomputed on every single match request.

## Tech stack

- **App**: Streamlit
- **Phrase matching**: spaCy `PhraseMatcher` over curated vocabularies (not the built-in statistical NER component)
- **Semantic similarity**: `sentence-transformers` (SBERT, `all-MiniLM-L6-v2`), scikit-learn cosine similarity
- **LLM extraction**: Groq API (default model configured as `openai/gpt-oss-120b` via `GROQ_MODEL`; model name can be overridden with an environment variable)
- **Resume parsing**: PyPDF2, PyMuPDF, pytesseract (OCR), python-docx
- **Output generation**: python-docx (ATS-formatted resume export)
- **Evaluation**: custom validation suite + weight sensitivity analysis, visualized with matplotlib

## Evaluation & tuning

Run the validation suite against the current scoring weights:

```bash
py -3.12 validation_test_suite.py
```

Sweep multiple weight configurations and see which performs best on the same test cases:

```bash
py -3.12 sensitivity_analysis.py
```

This writes `sensitivity_report.json`, which `graph.py` turns into comparison charts (validation outcome breakdown and top weight configurations). Two further charts, spot-check performance and failure reasons, are drawn only when a `spot_check_report.json` from a real-resume spot check is present; no script in this repo generates it, so `graph.py` skips them otherwise.

## Run locally

```bash
git clone https://github.com/Varunteja119/TalentSync_.git
cd TalentSync_
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

You'll also need Tesseract OCR installed as a system binary (not just a pip package) and added to your PATH, for image/scanned-PDF resume support.

If you want the LLM fallback tier active, set a Groq API key as an environment variable:

```bash
setx GROQ_API_KEY "your-key-here"
```

Without it, the LLM method returns empty lists and extraction can still use PhraseMatcher and the keyword/rule-based path.

Then run:

```bash
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (`http://localhost:8501` by default).

## Deploy publicly

- **Streamlit Cloud**: connect this GitHub repo, select the `master` branch, set the main file to `streamlit_app.py`, and add `GROQ_API_KEY` under the app's secrets if you want the LLM fallback active in production.
- **Heroku / Railway / other Python hosts**: same repo, same dependencies, plus the Tesseract system binary if OCR support is needed.

## Project structure

- `streamlit_app.py` — main user interface
- `text_extractor.py` — resume text extraction and OCR handling
- `skill_extractor.py` — merged entity extraction (Groq LLM → spaCy PhraseMatcher → keyword/rule fallback in the current call order)
- `role_matcher.py` — hybrid semantic + skill-overlap scoring engine
- `resume_generator.py` — ATS-formatted `.docx` resume builder from a structured profile
- `auth.py` — user accounts with PBKDF2 password hashing and legacy-plaintext auto-migration
- `job_redirect.py` — job search link generation
- `pipeline.py` — orchestrates extraction → matching end to end
- `validation_test_suite.py` — labeled test cases for the matching engine
- `sensitivity_analysis.py` — weight-configuration sweep and effectiveness comparison
- `graph.py` — visualizes validation and sensitivity results
- `role_profiles/` — role definitions and required-skill profiles
- `ner_resources/` — curated skill/role/tool phrase vocabularies used by PhraseMatcher and deterministic matching
- `ml/` — Phase 3: model fine-tuning and evaluation code and results (see `ml/README.md`)

## Roadmap

- **Fine-tuned matching model (evaluated, not adopted).** I fine-tuned a MiniLM bi-encoder on 851 GPT-4o-scored resume-JD pairs and benchmarked it with grouped cross-validation, a human-labeled held-out set and this repo's validation suite. It improved JD-ranking accuracy in cross-validation (pairwise 0.58 -> 0.74-0.78), but about two-thirds of that gain is reproducible by a per-JD score lookup and it did not transfer to the app, so `all-MiniLM-L6-v2` remains the default. See [`ml/README.md`](ml/README.md).
- **Live job listings (implemented).** `jobs_for_you.py` fetches current listings from the JSearch API; `job_redirect.py` still generates job-search links.

## Notes

- The password-hashing migration is transparent: any legacy plaintext account gets hashed automatically on its next successful login. No manual migration step required.
- This repo's git history was rewritten to remove user data (`users.json`) that had been committed by mistake earlier in development — that file is now gitignored and generated fresh at runtime.

---

TalentSync is a working demonstration of a hybrid NLP/ML resume-matching pipeline. A fine-tuned bi-encoder was trained and benchmarked against the pretrained baseline; the evaluation is documented in [`ml/README.md`](ml/README.md).
