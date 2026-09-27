"""
Quick standalone check that the LLM fallback in skill_extractor.py actually
works with a REAL Groq API key -- run this once after setup, before trusting
it inside the full app.

Setup:
    pip install groq
    setx GROQ_API_KEY "your-real-key-here"      (Windows, new terminal after)
    -- or --
    $env:GROQ_API_KEY = "your-real-key-here"    (current PowerShell session only)

    Optional: override the model (default is openai/gpt-oss-120b) if Groq
    has deprecated it again by the time you read this --
    check https://console.groq.com/docs/deprecations
        $env:GROQ_MODEL = "some-other-model-id"

Run:
    py -3.12 verify_llm_fallback.py
"""
import os
import sys

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY is not set in this environment.")
    print("Set it, then re-run this script.")
    sys.exit(1)

try:
    import groq  # noqa: F401
except ImportError:
    print("ERROR: the 'groq' package isn't installed.")
    print("Run: pip install groq")
    sys.exit(1)

import skill_extractor as se

# Deliberately full of terms NOT in skills.txt/tools.txt, so this only
# succeeds if the LLM fallback is actually firing and actually working --
# not just falling through to the static list-match path by coincidence.
SAMPLE_RESUME = """
Jordan Kim
jordan.kim@email.com

Experience
Data Engineer - Meridian Analytics
Jan 2023 - Present
Built data pipeline orchestration workflows using dbt for transformation
and Airflow for scheduling. Migrated the warehousing layer to Snowflake,
and built a stream processing layer with Kafka and Spark.

Education
B.Tech in Information Technology
Coastal University
2018 - 2022
"""

print("Running extract_entities() against a sample resume with terms")
print("that only exist if the REAL LLM fallback is working...")
print()

entities = se.extract_entities(SAMPLE_RESUME)

print("Skills found:", sorted(entities["skills"]))
print("Tools found: ", sorted(entities["tools"]))
print("Roles found: ", sorted(entities["roles"]))
print()

expected_tools = {"dbt", "airflow", "snowflake", "kafka", "spark"}
found_tools = set(t.lower() for t in entities["tools"])
hit = expected_tools & found_tools

if hit:
    print(f"SUCCESS: LLM fallback found {len(hit)}/{len(expected_tools)} "
          f"expected tools: {sorted(hit)}")
    print("Your real Groq integration is working.")
else:
    print("WARNING: none of the expected tools were found.")
    print("Possible causes:")
    print("  - GROQ_MODEL name in skill_extractor.py is outdated/decommissioned")
    print("  - API key is invalid or rate-limited")
    print("  - the model's JSON response didn't parse (check for stray text)")
    print("Add a print(raw) inside _extract_entities_llm's try block to debug.")
