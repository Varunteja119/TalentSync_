#!/usr/bin/env python3
"""Test script to validate hybrid scoring on the user's resume."""

import sys
from text_extractor import extract_text
from skill_extractor import extract_entities
from role_matcher import match_roles

def main():
    # Extract from user's resume
    pdf_path = r'd:\Personal\Documents\Santhosh Ganesan __ FIN.pdf'
    
    print("=" * 60)
    print("HYBRID SCORING FORMULA VALIDATION TEST")
    print("=" * 60)
    print()
    
    # Extract text
    print("[1/3] Extracting text from PDF...")
    text = extract_text(pdf_path)
    if not text:
        print("ERROR: Could not extract text from PDF")
        return
    print(f"✓ Extracted {len(text)} characters")
    print()
    
    # Extract entities (NER + keywords)
    print("[2/3] Extracting entities (skills, roles, tools)...")
    entities = extract_entities(text)
    skills = entities.get('skills', [])
    roles = entities.get('roles', [])
    tools = entities.get('tools', [])
    
    print(f"✓ Found {len(skills)} skills: {skills[:5]}{'...' if len(skills) > 5 else ''}")
    print(f"✓ Found {len(roles)} roles: {roles}")
    print(f"✓ Found {len(tools)} tools: {tools[:5]}{'...' if len(tools) > 5 else ''}")
    print()
    
    # Match roles with hybrid scoring
    print("[3/3] Computing hybrid scoring formula...")
    print("Formula: Final Score = 0.55·Semantic + 0.35·SkillOverlap + 0.10·RoleBonus")
    print()
    
    recommendations = match_roles(skills, roles, tools)
    
    if not recommendations:
        print("ERROR: No recommendations generated")
        return
    
    print("=" * 60)
    print("TOP 5 ROLE MATCHES (by Final Score)")
    print("=" * 60)
    print()
    
    for idx, rec in enumerate(recommendations[:5], 1):
        print(f"{idx}. {rec['role'].upper()}")
        print(f"   Final Score:       {rec['final_score']:.1%}")
        print(f"   ├─ Semantic Sim:   {rec['semantic_similarity']:.1%} (weight: 55%)")
        print(f"   ├─ Skill Overlap:  {rec['skill_overlap']:.1%} (weight: 35%)")
        print(f"   ├─ Role Bonus:     {rec['role_bonus']:.1%} (weight: 10%)")
        print(f"   └─ Fit Label:      {rec['fit_label']}")
        print()
    
    print("=" * 60)
    print("SCORING CLASSIFICATION LEGEND")
    print("=" * 60)
    print("Strong (≥70%):   Excellent fit with high confidence")
    print("Good (≥50%):     Good candidate, worth consideration")
    print("Potential (≥35%): Could develop into role with training")
    print("Low (<35%):      Significant skill gaps")
    print()
    print("✓ Hybrid scoring formula validation complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
