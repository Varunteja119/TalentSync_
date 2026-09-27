# TalentSync

TalentSync is a resume-to-role matching engine that combines NLP-based entity extraction, transformer-based semantic similarity, and a tiered LLM fallback for open-vocabulary skill recognition. It parses unstructured resumes, extracts skills/roles/tools, and ranks candidate-role fit using a hybrid scoring model — with a validation and sensitivity-analysis pipeline used to tune and justify the scoring weights rather than picking them arbitrarily.

This project exists to demonstrate applied NLP/ML engineering: entity extraction, embedding-based similarity, hybrid scoring design, tiered fallback architecture, and a real evaluation methodology — not just orchestrating a single LLM call.

## What this demonstrates

- **Hybrid entity extraction, not a single method.** spaCy NER (`PhraseMatcher` over a curated vocabulary) runs first; a keyword/regex fallback with context-inference rules catches what NER misses; an LLM-based open-vocabulary extraction step (Groq/LLaMA) only fires as a last resort when the first two are still thin. Each tier is faster and cheaper than the next, so the expensive path only runs when it's actually needed.
- **Hybrid semantic + symbolic scoring, not pure vibes-based LLM ranking.** Role fit is a weighted combination of SBERT cosine similarity (semantic fit), literal skill-overlap ratio (symbolic, explainable), and a role-match bonus — each with a documented formula and rationale, not a black-box prompt.
- **An actual evaluation methodology.** `validation_test_suite.py` runs labeled test cases against the matcher and reports pass/partial/fail rates; `sensitivity_analysis.py` sweeps the scoring formula's weight configuration across multiple combinations and reports which setting performs best, with results written to a JSON report and visualized via `graph.py`. This is the same validate → tune → re-validate loop used for any ML system's hyperparameters — the current 0.55 / 0.35 / 0.10 weighting isn't a guess, it's the sensitivity analysis's own baseline.
- **Engineering discipline beyond the ML pieces.** Passwords are hashed with PBKDF2-HMAC-SHA256 with transparent migration of any legacy plaintext account on next login; resume parsing has dedicated unit-style test coverage for its regex/heuristic edge cases; performance-sensitive paths (like SBERT embeddings) are cached rather than recomputed per request.

## How it works

1. **Extraction** (`text_extractor.py`) — pulls text from PDF, DOCX, TXT, or image resumes, falling back to OCR (Tesseract + PyMuPDF rendering) when direct text extraction yields nothing meaningful.
2. **Entity extraction** (`skill_extractor.py`) — a three-tier pipeline:
   - NER via spaCy `PhraseMatcher` against curated skill/role/tool vocabularies
   - Keyword + regex fallback with context-inference rules (e.g. inferring "leadership" from "led team of 15")
   - LLM-based open-vocabulary extraction (only triggered when the above is still low-signal), processed in paragraph-aware chunks so long resumes aren't silently truncated
3. **Role matching** (`role_matcher.py`) — computes, for every role profile:
   - Semantic similarity (SBERT embedding cosine similarity between candidate profile and role description)
   - Skill overlap ratio (matched core skills / required core skills)
   - Role match bonus (explicit signal if the candidate already named this role)
   - Combines them via a documented weighted formula, applies qualification thresholds, and ranks results
4. **Output** — ranked role recommendations with a transparent score breakdown, plus an ATS-formatted resume generator (`resume_generator.py`) that turns a structured profile into a downloadable `.docx`.

## Architecture decisions worth calling out

**Why a tiered fallback instead of "just call an LLM for everything"?** An LLM call is the most flexible option but the slowest and the only one with a per-call cost. Running it on every resume would be wasteful when a curated vocabulary already resolves most cases instantly and for free. The tiered design means the expensive path is reserved for genuinely novel terminology (a new framework, a niche domain) rather than every request.

**Why hybrid semantic + skill-overlap scoring instead of asking an LLM to just rank the roles?** Skill overlap is fully explainable — you can point to exactly which required skills matched. Semantic similarity captures fit that a literal keyword match would miss (e.g. recognizing that RAG/LLM-application experience is relevant to a "Generative AI Engineer" role even if the exact core-skill strings don't line up perfectly). Combining both, with weights tuned via the sensitivity analysis pipeline rather than picked arbitrarily, gives a scoring system that's both accurate and auditable.

**Why cache embeddings?** SBERT inference is the most expensive step in the pipeline. Role profile embeddings are static (they only depend on the role definitions, not the candidate), so they're computed once and cached rather than recomputed on every single match request.

## Tech stack

- **App**: Streamlit
- **NLP/NER**: spaCy (`PhraseMatcher`)
- **Semantic similarity**: `sentence-transformers` (SBERT, `all-MiniLM-L6-v2`), scikit-learn cosine similarity
- **LLM fallback**: Groq (LLaMA-family models)
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

This writes `sensitivity_report.json`, which `graph.py` turns into a set of comparison charts (validation outcome breakdown, top weight configurations, spot-check performance, failure reason distribution).

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

Without it, the app runs fine — that tier just no-ops and the app relies on the NER + keyword-fallback paths only.

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
- `skill_extractor.py` — tiered entity extraction (NER → keyword fallback → LLM fallback)
- `role_matcher.py` — hybrid semantic + skill-overlap scoring engine
- `resume_generator.py` — ATS-formatted `.docx` resume builder from a structured profile
- `auth.py` — user accounts with PBKDF2 password hashing and legacy-plaintext auto-migration
- `job_redirect.py` — job search link generation
- `pipeline.py` — orchestrates extraction → matching end to end
- `validation_test_suite.py` — labeled test cases for the matching engine
- `sensitivity_analysis.py` — weight-configuration sweep and effectiveness comparison
- `graph.py` — visualizes validation and sensitivity results
- `role_profiles/` — role definitions and required-skill profiles
- `ner_resources/` — skill/role/tool vocabularies for NER and keyword matching

## Roadmap

- **Fine-tuned matching model.** The current SBERT model (`all-MiniLM-L6-v2`) is used as-is, off the shelf. Planned: fine-tune a bi-encoder on labeled resume-role match/mismatch pairs, evaluate it against the pretrained baseline on a held-out set (Recall@K, MRR, NDCG), and swap it into `role_matcher.py` if it measurably outperforms — the actual "I trained and evaluated a model" deliverable, not just "I called an API."
- **Real-time job listings.** `job_redirect.py` currently generates job-search query URLs. Planned: swap in a live job-search API (Adzuna or JSearch) for real, current listings instead of static search links.

## Notes

- The password-hashing migration is transparent: any legacy plaintext account gets hashed automatically on its next successful login. No manual migration step required.
- This repo's git history was rewritten to remove user data (`users.json`) that had been committed by mistake earlier in development — that file is now gitignored and generated fresh at runtime.

---

TalentSync is a working demonstration of a hybrid NLP/ML resume-matching pipeline, built to be extended toward a genuinely fine-tuned model as the next milestone.
