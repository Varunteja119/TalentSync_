#!/usr/bin/env python3
"""
Sensitivity analysis for hybrid role-matching weights.

Runs the existing validation test cases against multiple weight combinations and
reports which setting yields the best effectiveness.
"""

import json
from datetime import datetime

import role_matcher
from validation_test_suite import TEST_CASES


WEIGHT_CONFIGS = [
    # baseline
    {"name": "baseline", "semantic": 0.55, "skill": 0.35, "bonus": 0.10},
    # more skill overlap influence
    {"name": "skill_heavy_1", "semantic": 0.50, "skill": 0.40, "bonus": 0.10},
    {"name": "skill_heavy_2", "semantic": 0.45, "skill": 0.45, "bonus": 0.10},
    # more semantic influence
    {"name": "semantic_heavy_1", "semantic": 0.60, "skill": 0.30, "bonus": 0.10},
    {"name": "semantic_heavy_2", "semantic": 0.65, "skill": 0.25, "bonus": 0.10},
    # bonus-sensitive setups
    {"name": "bonus_heavy_1", "semantic": 0.50, "skill": 0.30, "bonus": 0.20},
    {"name": "bonus_heavy_2", "semantic": 0.45, "skill": 0.35, "bonus": 0.20},
    # balanced alternatives
    {"name": "balanced_1", "semantic": 0.50, "skill": 0.35, "bonus": 0.15},
    {"name": "balanced_2", "semantic": 0.58, "skill": 0.32, "bonus": 0.10},
]


def evaluate_single_case(test_case):
    """Evaluate one test case and return PASSED/PARTIAL/FAILED/SKIPPED."""
    recommendations = role_matcher.match_roles(
        test_case["skills"],
        test_case["roles"],
        test_case["tools"],
    )

    if not recommendations:
        return "SKIPPED", None, None

    expected_lower = test_case["expected_role"].lower()
    matching_recs = [
        (rank + 1, rec)
        for rank, rec in enumerate(recommendations)
        if expected_lower in rec["role"].lower()
    ]

    if not matching_recs:
        return "FAILED", None, recommendations[0]["role"] if recommendations else None

    rank, rec = matching_recs[0]
    expected_rank = test_case["expected_rank"]

    if expected_rank is None:
        # desired behavior: expected role should not be highly ranked
        if rank <= 5:
            return "PARTIAL", rank, rec["final_score"]
        return "PASSED", rank, rec["final_score"]

    if rank <= expected_rank:
        return "PASSED", rank, rec["final_score"]

    return "PARTIAL", rank, rec["final_score"]


def run_config(config):
    """Run all tests under one weight configuration."""
    role_matcher.WEIGHT_SEMANTIC = config["semantic"]
    role_matcher.WEIGHT_SKILL_OVERLAP = config["skill"]
    role_matcher.WEIGHT_ROLE_BONUS = config["bonus"]

    stats = {
        "passed": 0,
        "partial": 0,
        "failed": 0,
        "skipped": 0,
        "details": [],
    }

    for test_case in TEST_CASES:
        status, rank, extra = evaluate_single_case(test_case)
        key = status.lower()
        stats[key] += 1
        stats["details"].append(
            {
                "test": test_case["name"],
                "status": status,
                "rank": rank,
                "extra": extra,
            }
        )

    total = len(TEST_CASES)
    effectiveness = (stats["passed"] + 0.5 * stats["partial"]) / total * 100 if total else 0.0
    stats["effectiveness"] = round(effectiveness, 2)
    return stats


def print_summary_row(config, stats):
    print(
        f"{config['name']:<16} "
        f"({config['semantic']:.2f}/{config['skill']:.2f}/{config['bonus']:.2f})  "
        f"P:{stats['passed']:2d}  Pa:{stats['partial']:2d}  "
        f"F:{stats['failed']:2d}  S:{stats['skipped']:2d}  "
        f"Eff:{stats['effectiveness']:6.2f}%"
    )


def main():
    original = {
        "semantic": role_matcher.WEIGHT_SEMANTIC,
        "skill": role_matcher.WEIGHT_SKILL_OVERLAP,
        "bonus": role_matcher.WEIGHT_ROLE_BONUS,
    }

    print("=" * 90)
    print("SENSITIVITY ANALYSIS: HYBRID WEIGHT COMBINATIONS")
    print("=" * 90)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total configs: {len(WEIGHT_CONFIGS)}")
    print("Format: semantic/skill/bonus")
    print("-" * 90)

    all_results = []

    try:
        for config in WEIGHT_CONFIGS:
            stats = run_config(config)
            all_results.append({"config": config, "stats": stats})
            print_summary_row(config, stats)

        all_results.sort(key=lambda x: x["stats"]["effectiveness"], reverse=True)
        best = all_results[0]

        print("-" * 90)
        print("BEST CONFIGURATION")
        print(
            f"Name: {best['config']['name']} | "
            f"Weights: ({best['config']['semantic']:.2f}, "
            f"{best['config']['skill']:.2f}, {best['config']['bonus']:.2f}) | "
            f"Effectiveness: {best['stats']['effectiveness']:.2f}%"
        )

        baseline = next((r for r in all_results if r["config"]["name"] == "baseline"), None)
        if baseline:
            delta = best["stats"]["effectiveness"] - baseline["stats"]["effectiveness"]
            print(f"Delta vs baseline: {delta:+.2f}%")

        report = {
            "timestamp": datetime.now().isoformat(),
            "original_weights": original,
            "results": all_results,
            "best": best,
        }

        with open("sensitivity_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("Report written to sensitivity_report.json")

    finally:
        # Restore original weights to avoid side effects.
        role_matcher.WEIGHT_SEMANTIC = original["semantic"]
        role_matcher.WEIGHT_SKILL_OVERLAP = original["skill"]
        role_matcher.WEIGHT_ROLE_BONUS = original["bonus"]


if __name__ == "__main__":
    main()
