#!/usr/bin/env python3
"""
Validation Test Suite for Role Matching System
Tests system accuracy with synthetic test cases covering various scenarios
"""

from role_matcher import match_roles
import json
from datetime import datetime

# ==========================================
# TEST CASES: Synthetic profiles with known correct answers
# ==========================================
TEST_CASES = [
    {
        "name": "Perfect Match - Software Engineer",
        "skills": ["Python", "Django", "FastAPI", "PostgreSQL", "REST APIs", "Git", "Docker"],
        "roles": ["Software Engineer"],
        "tools": ["PyCharm", "VS Code", "Linux"],
        "expected_role": "software engineer",
        "expected_rank": 1,  # Should be top recommendation
        "description": "Strong backend dev with all key skills"
    },
    {
        "name": "Perfect Match - Data Scientist",
        "skills": ["Python", "Machine Learning", "TensorFlow", "Pandas", "NumPy", "Statistics", "SQL"],
        "roles": ["Data Scientist"],
        "tools": ["Jupyter", "Scikit-learn", "Matplotlib"],
        "expected_role": "data scientist",
        "expected_rank": 1,
        "description": "Specialist with ML and statistical background"
    },
    {
        "name": "Perfect Match - Full Stack Developer",
        "skills": ["React", "JavaScript", "TypeScript", "HTML", "CSS", "Redux", "API Integration", "Node.js", "Backend"],
        "roles": ["Full Stack Developer"],
        "tools": ["Node.js", "NPM", "Dev Tools"],
        "expected_role": "full stack developer",
        "expected_rank": 1,
        "description": "Frontend and backend specialist with modern tooling"
    },
    {
        "name": "Perfect Match - DevOps Engineer",
        "skills": ["Kubernetes", "Docker", "AWS", "CI/CD", "Linux", "Terraform", "Monitoring"],
        "roles": ["DevOps Engineer"],
        "tools": ["Jenkins", "GitLab CI", "Prometheus"],
        "expected_role": "devops engineer",
        "expected_rank": 1,
        "description": "Infrastructure automation expert"
    },
    {
        "name": "Perfect Match - Accounts Executive",
        "skills": ["Accounts Payable", "Account Reconciliation", "General Ledger Accounting", "Excel", "SAP", "Payroll"],
        "roles": ["Accounts Executive"],
        "tools": ["SAP", "QuickBooks", "Excel"],
        "expected_role": "accounts executive",
        "expected_rank": 1,
        "description": "Finance professional with accounting expertise"
    },
    {
        "name": "Cross-Domain - Military to Backend",
        "skills": ["Java", "System Architecture", "Team Leadership", "Problem-solving", "Python basics"],
        "roles": ["Team Lead", "Software Engineer"],
        "tools": ["Linux", "Git", "Command Line"],
        "expected_role": "backend developer",
        "expected_rank": 5,  # Should rank in top 5, not #1
        "description": "Ex-serviceman with some backend exposure"
    },
    {
        "name": "Cross-Domain - Military to Data Analysis",
        "skills": ["Statistics", "Data Analysis", "Excel", "Problem-solving", "Python", "SQL"],
        "roles": ["Analyst"],
        "tools": ["Excel", "Python", "SQL"],
        "expected_role": "data analyst",
        "expected_rank": 5,
        "description": "Ex-serviceman transitioning to data field"
    },
    {
        "name": "Partial Match - Junior Software Dev",
        "skills": ["Python", "Basics", "Learning", "SQL", "Git"],
        "roles": ["Junior Developer"],
        "tools": ["VS Code"],
        "expected_role": "software developer",
        "expected_rank": 10,  # Should be in results but lower rank
        "description": "Junior with limited programming skills"
    },
    {
        "name": "Complete Mismatch - Accountant vs Backend",
        "skills": ["Accounts Payable", "Tax", "Excel", "Financial Reporting"],
        "roles": ["Accounts Executive"],
        "tools": ["QuickBooks"],
        "expected_role": "backend developer",
        "expected_rank": None,  # Might not appear or very low rank
        "description": "No tech skills, shouldn't match Backend Dev"
    },
    {
        "name": "Multi-skilled - Full Stack Developer",
        "skills": ["Python", "JavaScript", "React", "Django", "PostgreSQL", "Docker", "AWS", "Node.js"],
        "roles": ["Full Stack Developer", "Backend Developer"],
        "tools": ["VS Code", "Git", "Linux"],
        "expected_role": "full stack developer",
        "expected_rank": 2,  # Should rank very high
        "description": "Developer with backend AND frontend skills"
    },
    {
        "name": "Perfect Match - Backend Developer",
        "skills": ["Python", "API Development", "Django", "Flask", "PostgreSQL", "Microservices", "Docker"],
        "roles": ["Backend Developer"],
        "tools": ["Postman", "Git", "Linux"],
        "expected_role": "backend developer",
        "expected_rank": 1,
        "description": "Backend-focused engineering profile"
    },
    {
        "name": "Perfect Match - Software Developer",
        "skills": ["Python", "Java", "Data Structures", "Algorithms", "OOP", "SQL", "Testing"],
        "roles": ["Software Developer"],
        "tools": ["Git", "VS Code", "Jira"],
        "expected_role": "software developer",
        "expected_rank": 1,
        "description": "General software development fundamentals"
    },
    {
        "name": "Perfect Match - Data Analyst",
        "skills": ["SQL", "Excel", "Data Analysis", "Reporting", "Power BI", "Statistics", "Python"],
        "roles": ["Data Analyst"],
        "tools": ["Power BI", "Excel", "SQL"],
        "expected_role": "data analyst",
        "expected_rank": 1,
        "description": "Reporting and analytics specialist"
    },
    {
        "name": "Perfect Match - Machine Learning Engineer",
        "skills": ["Machine Learning", "Python", "TensorFlow", "Scikit-learn", "MLOps", "Model Deployment", "Data Pipelines"],
        "roles": ["Machine Learning Engineer"],
        "tools": ["Python", "Docker", "Git"],
        "expected_role": "machine learning engineer",
        "expected_rank": 1,
        "description": "Production-focused ML engineering profile"
    },
    {
        "name": "Perfect Match - Technical Support Engineer",
        "skills": ["Technical Troubleshooting", "Incident Management", "Networking", "Customer Support", "Linux", "Documentation"],
        "roles": ["Technical Support Engineer"],
        "tools": ["Jira", "ServiceNow", "Linux"],
        "expected_role": "technical support engineer",
        "expected_rank": 1,
        "description": "Support and operations troubleshooting"
    },
    {
        "name": "Perfect Match - Operations Manager",
        "skills": ["Operations Management", "Leadership", "Process Improvement", "Resource Planning", "Coordination", "Compliance"],
        "roles": ["Operations Manager"],
        "tools": ["Excel", "ERP", "Reporting"],
        "expected_role": "operations manager",
        "expected_rank": 1,
        "description": "Operational planning and team leadership"
    },
    {
        "name": "Perfect Match - Compliance Officer",
        "skills": ["Compliance", "Audit", "Risk Analysis", "Policy", "Documentation", "Governance"],
        "roles": ["Compliance Officer"],
        "tools": ["Excel", "Reporting"],
        "expected_role": "compliance officer",
        "expected_rank": 1,
        "description": "Regulatory and compliance control profile"
    },
    {
        "name": "Perfect Match - Team Lead",
        "skills": ["Leadership", "Team Management", "Coordination", "Planning", "Execution", "Communication"],
        "roles": ["Team Lead"],
        "tools": ["Jira", "Confluence"],
        "expected_role": "team lead",
        "expected_rank": 1,
        "description": "People and delivery leadership profile"
    },
    {
        "name": "Cross-Domain - Support to DevOps",
        "skills": ["Linux", "Shell Scripting", "Monitoring", "Troubleshooting", "Docker basics", "Networking"],
        "roles": ["Technical Support Engineer"],
        "tools": ["Linux", "Git", "Jenkins"],
        "expected_role": "devops engineer",
        "expected_rank": 8,
        "description": "Support engineer transitioning toward DevOps"
    },
    {
        "name": "Cross-Domain - Finance to Data Analyst",
        "skills": ["Excel", "Reporting", "Budgeting", "SQL basics", "Data Analysis", "Business Insights"],
        "roles": ["Finance Analyst"],
        "tools": ["Excel", "Power BI"],
        "expected_role": "data analyst",
        "expected_rank": 6,
        "description": "Finance profile moving into analytics"
    },
    {
        "name": "Low Information Profile",
        "skills": ["Communication", "Leadership", "Teamwork"],
        "roles": ["Generalist"],
        "tools": ["Excel"],
        "expected_role": "software engineer",
        "expected_rank": None,
        "description": "Sparse generic profile should not strongly match technical roles"
    }
]

# ==========================================
# TEST RUNNER
# ==========================================
def run_validation_suite():
    """Run all test cases and report results"""
    
    print("=" * 80)
    print("VALIDATION TEST SUITE FOR ROLE MATCHING SYSTEM")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Test Cases: {len(TEST_CASES)}")
    print()
    
    results = {
        "passed": 0,
        "partial": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }
    
    for idx, test_case in enumerate(TEST_CASES, 1):
        print("-" * 80)
        print(f"TEST {idx}/{len(TEST_CASES)}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Skills: {', '.join(test_case['skills'][:5])}{'...' if len(test_case['skills']) > 5 else ''}")
        print(f"Expected: {test_case['expected_role'].upper()} (Rank ≤ {test_case['expected_rank']})")
        print()
        
        try:
            # Run the matching algorithm
            recommendations = match_roles(
                test_case['skills'],
                test_case['roles'],
                test_case['tools']
            )
            
            if not recommendations:
                print("❌ SKIPPED: No recommendations generated")
                results["skipped"] += 1
                results["details"].append({
                    "test": test_case['name'],
                    "status": "SKIPPED",
                    "reason": "No recommendations"
                })
                print()
                continue
            
            # Find if expected role is in results
            expected_lower = test_case['expected_role'].lower()
            matching_recs = [
                (rank + 1, rec) for rank, rec in enumerate(recommendations)
                if expected_lower in rec['role'].lower()
            ]
            
            if not matching_recs:
                # Role not in results at all
                print(f"❌ FAILED: '{test_case['expected_role']}' not in recommendations")
                print(f"   Top 3 recommendations:")
                for rank, rec in enumerate(recommendations[:3], 1):
                    print(f"   {rank}. {rec['role']}: {rec['final_score']:.1%}")
                results["failed"] += 1
                results["details"].append({
                    "test": test_case['name'],
                    "status": "FAILED",
                    "reason": f"Role not found",
                    "top_result": recommendations[0]['role'] if recommendations else None
                })
            elif test_case['expected_rank'] is None:
                # Should NOT appear or low rank - check if it's low
                rank, rec = matching_recs[0]
                if rank <= 5:
                    print(f"⚠️  PARTIAL: '{test_case['expected_role']}' ranked #{rank} (expected: NOT in top)")
                    print(f"   Score: {rec['final_score']:.1%}")
                    results["partial"] += 1
                    results["details"].append({
                        "test": test_case['name'],
                        "status": "PARTIAL",
                        "reason": f"Ranked {rank}, expected low/none",
                        "actual_rank": rank
                    })
                else:
                    print(f"✓ PASSED: '{test_case['expected_role']}' ranked low (#{rank})")
                    results["passed"] += 1
                    results["details"].append({
                        "test": test_case['name'],
                        "status": "PASSED",
                        "actual_rank": rank
                    })
            else:
                # Expected rank assertion
                rank, rec = matching_recs[0]
                if rank <= test_case['expected_rank']:
                    print(f"✓ PASSED: '{test_case['expected_role']}' ranked #{rank} (expected ≤ {test_case['expected_rank']})")
                    print(f"   Score: {rec['final_score']:.1%}")
                    results["passed"] += 1
                    results["details"].append({
                        "test": test_case['name'],
                        "status": "PASSED",
                        "actual_rank": rank,
                        "score": rec['final_score']
                    })
                else:
                    print(f"⚠️  PARTIAL: '{test_case['expected_role']}' ranked #{rank} (expected ≤ {test_case['expected_rank']})")
                    print(f"   Score: {rec['final_score']:.1%}")
                    results["partial"] += 1
                    results["details"].append({
                        "test": test_case['name'],
                        "status": "PARTIAL",
                        "reason": f"Ranked {rank}, expected {test_case['expected_rank']}",
                        "actual_rank": rank,
                        "score": rec['final_score']
                    })
            
            # Show top 3 for reference
            print(f"   Top 3: ", end="")
            for rank, rec in enumerate(recommendations[:3], 1):
                print(f"{rank}. {rec['role']} ({rec['final_score']:.0%}) | ", end="")
            print()
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results["skipped"] += 1
            results["details"].append({
                "test": test_case['name'],
                "status": "ERROR",
                "error": str(e)
            })
        
        print()
    
    # ==========================================
    # SUMMARY REPORT
    # ==========================================
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    total = len(TEST_CASES)
    passed = results["passed"]
    partial = results["partial"]
    failed = results["failed"]
    skipped = results["skipped"]
    
    print(f"✓ PASSED:  {passed:2d}/{total} ({passed/total*100:5.1f}%)")
    print(f"⚠️  PARTIAL: {partial:2d}/{total} ({partial/total*100:5.1f}%)")
    print(f"❌ FAILED: {failed:2d}/{total} ({failed/total*100:5.1f}%)")
    print(f"⏭️  SKIPPED: {skipped:2d}/{total} ({skipped/total*100:5.1f}%)")
    print()
    
    effectiveness = (passed + partial * 0.5) / total * 100 if total > 0 else 0
    print(f"Overall Effectiveness Score: {effectiveness:.1f}%")
    print()
    
    if effectiveness >= 80:
        print("🟢 RESULT: System performing well!")
    elif effectiveness >= 60:
        print("🟡 RESULT: System acceptable, optimization opportunities exist")
    else:
        print("🔴 RESULT: System needs significant improvements")
    
    print()
    print("=" * 80)
    
    # Save detailed results to JSON
    report_file = "test_results.json"
    with open(report_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "skipped": skipped,
                "effectiveness": effectiveness
            },
            "details": results["details"]
        }, f, indent=2)
    
    print(f"Detailed results saved to: {report_file}")
    print()
    
    return results


if __name__ == "__main__":
    run_validation_suite()
