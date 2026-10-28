from skill_extractor import _clean_llm_list


def test_llm_compound_skill_output_is_split_into_atomic_items():
    raw = [
        "adaptability, and layer-wise quality checks using field, compaction, "
        "construction planning, density tests."
    ]
    skills = _clean_llm_list(raw, split_compounds=True)
    assert {"adaptability", "compaction", "construction planning", "density tests"} <= set(skills)
    assert not any("using field" in skill or skill.startswith("and ") for skill in skills)
    assert not any(len(skill.split()) > 7 for skill in skills)


def test_open_vocabulary_multiword_skills_are_preserved_when_concise():
    skills = _clean_llm_list(
        ["requirements elicitation", "quality assurance", "stakeholder communication"],
        split_compounds=True,
    )
    assert skills == ["quality assurance", "requirements elicitation", "stakeholder communication"]
