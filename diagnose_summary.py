"""
Shows exactly what the real PyPDF2 extraction produces around the
Professional Summary section, and whether _find_section actually locates
it. Run this from inside TalentSync_cleanup, with your resume PDF in the
same folder (or give it a full path as an argument).

    py -3.12 diagnose_summary.py "Varun_Resume.pdf"
"""
import sys

from text_extractor import extract_text
from utils import _clean_lines, _find_section, _extract_summary, ALL_SECTION_KEYS

if len(sys.argv) < 2:
    sys.exit("Usage: py -3.12 diagnose_summary.py <path-to-resume-pdf-or-docx>")

path = sys.argv[1]
raw_text = extract_text(path)

print("=" * 70)
print("RAW extracted text (first 800 chars)")
print("=" * 70)
print(raw_text[:800])

print()
print("=" * 70)
print("CLEANED LINES (what utils.py actually works with)")
print("=" * 70)
lines = _clean_lines(raw_text)
for i, line in enumerate(lines):
    marker = " <-- contains 'summary'" if "summary" in line.lower() else ""
    print(f"{i:3d}: {line!r}{marker}")

print()
print("=" * 70)
print("_find_section() result for SUMMARY-type headers")
print("=" * 70)
found = _find_section(
    lines,
    ["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE", "CAREER SUMMARY", "OBJECTIVE"],
    ALL_SECTION_KEYS,
)
print(f"Lines captured: {found!r}")

print()
print("=" * 70)
print("_extract_summary() final result")
print("=" * 70)
print(repr(_extract_summary(raw_text)))
