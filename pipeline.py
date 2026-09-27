from text_extractor import extract_text
from skill_extractor import extract_entities
from role_matcher import match_roles


def run_pipeline(file_path):

    # Step 1: Extract text
    text = extract_text(file_path)

    # Step 2: Extract entities
    entities = extract_entities(text)

    # Step 3: Role matching
    # Algorithm: Compute hybrid role scores using SBERT+cosine semantic match combined with skill overlap and role bonus.
    skills = entities.get("skills", [])
    roles = entities.get("roles", [])
    tools = entities.get("tools", [])
    recommendations = match_roles(skills, roles, tools)

    return {
        "raw": entities,
        "recommended_roles": recommendations
    }


# Optional test
if __name__ == "__main__":
    result = run_pipeline("test_resume.docx")
    print(result)