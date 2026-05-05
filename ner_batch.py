import os
import json
import spacy
from spacy.matcher import PhraseMatcher

INPUT_ROOT = r"E:\Capstone Project\refined_text"
OUTPUT_ROOT = r"E:\Capstone Project\ner_output"
RESOURCE_DIR = r"E:\Capstone Project\ner_resources"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

def load_phrases(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

skills = load_phrases(os.path.join(RESOURCE_DIR, "skills.txt"))
roles  = load_phrases(os.path.join(RESOURCE_DIR, "roles.txt"))
tools  = load_phrases(os.path.join(RESOURCE_DIR, "tools.txt"))

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

matcher.add("SKILL", [nlp.make_doc(s) for s in skills])
matcher.add("ROLE",  [nlp.make_doc(r) for r in roles])
matcher.add("TOOL",  [nlp.make_doc(t) for t in tools])

def extract_entities(text):
    doc = nlp(text)
    matches = matcher(doc)

    extracted = {
        "skills": set(),
        "roles": set(),
        "tools": set()
    }

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        value = doc[start:end].text.strip().lower()

        if label == "SKILL":
            extracted["skills"].add(value)
        elif label == "ROLE":
            extracted["roles"].add(value)
        elif label == "TOOL":
            extracted["tools"].add(value)

    return {k: sorted(list(v)) for k, v in extracted.items()}

for role_folder in os.listdir(INPUT_ROOT):
    role_input_path = os.path.join(INPUT_ROOT, role_folder)

    if not os.path.isdir(role_input_path):
        continue

    role_output_path = os.path.join(OUTPUT_ROOT, role_folder)
    os.makedirs(role_output_path, exist_ok=True)

    print(f"\n[NER] Processing folder: {role_folder}")

    for fname in os.listdir(role_input_path):
        if not fname.endswith(".txt"):
            continue

        file_path = os.path.join(role_input_path, fname)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        entities = extract_entities(text)

        output_data = {
            "source_folder": role_folder,
            "file": fname,
            "entities": entities
        }

        output_file = os.path.join(
            role_output_path,
            fname.replace(".txt", ".json")
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"[OK] {role_folder}/{fname}")
