"""
Manual Validation of LLM-Detected Errors
=========================================
Randomly samples 100 LLM-detected errors from across all 300 trajectories
(6 batches x 50) for manual review.

Uses random seed 42 for reproducibility.
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
SAMPLE_SIZE = 100

BASE_DIR = Path(r"c:\Users\amita\Downloads\mp\results_1000_study")

BATCH_RUNS = {
    "batch_01": "run_20260209_174557",
    "batch_02": "run_20260212_210830",
    "batch_03": "run_20260214_084905",
    "batch_04": "run_20260215_144254",
    "batch_05": "run_20260216_111311",
    "batch_06": "run_20260217_211323",
}

LLM_ERROR_FIELDS = [
    "llm_memory_error",
    "llm_reflection_error",
    "llm_planning_error",
    "llm_action_error",
    "llm_system_error",
]


def collect_all_llm_errors():
    """Collect every LLM-detected error across all 300 trajectories."""
    all_errors = []
    files_processed = 0
    trajectories_with_errors = 0

    for batch_name, run_name in sorted(BATCH_RUNS.items()):
        individual_dir = BASE_DIR / batch_name / run_name / "experiments" / "individual"
        if not individual_dir.exists():
            print(f"WARNING: {individual_dir} does not exist")
            continue

        for json_file in sorted(individual_dir.glob("*_analysis.json")):
            files_processed += 1
            instance_id = json_file.stem.replace("_analysis", "")

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            phase1 = data.get("phase1", {})
            step_analyses = phase1.get("step_analyses", [])
            has_error = False

            for step in step_analyses:
                step_num = step.get("step_number", "?")

                # Also collect regex errors for cross-reference
                regex_errors = {}
                for field in ["regex_memory_error", "regex_reflection_error",
                              "regex_planning_error", "regex_action_error",
                              "regex_system_error"]:
                    if step.get(field) is not None:
                        regex_errors[field.replace("regex_", "")] = step[field]

                agreement = step.get("agreement", {})

                for field in LLM_ERROR_FIELDS:
                    error_obj = step.get(field)
                    if error_obj is not None and error_obj.get("has_error", True):
                        module = error_obj.get("module", field.replace("llm_", "").replace("_error", ""))
                        error_type = error_obj.get("error_type", "unknown")
                        confidence = error_obj.get("confidence", 0)
                        explanation = error_obj.get("explanation", "")

                        # Check if regex also detected an error for this module
                        module_key = field.replace("llm_", "").replace("_error", "")
                        regex_counterpart = regex_errors.get(module_key + "_error")
                        agreement_status = agreement.get(module_key, "unknown")

                        all_errors.append({
                            "batch": batch_name,
                            "instance_id": instance_id,
                            "step_number": step_num,
                            "total_steps": phase1.get("total_steps", "?"),
                            "module": module,
                            "error_type": error_type,
                            "confidence": confidence,
                            "explanation": explanation,
                            "agreement_status": agreement_status,
                            "regex_also_detected": regex_counterpart is not None,
                            "regex_error_type": regex_counterpart.get("error_type") if regex_counterpart else None,
                            "regex_evidence": regex_counterpart.get("evidence", "")[:300] if regex_counterpart else None,
                            "task_description": phase1.get("task_description", "")[:200],
                        })
                        has_error = True

            if has_error:
                trajectories_with_errors += 1

    return all_errors, files_processed, trajectories_with_errors


def main():
    print("Collecting all LLM-detected errors...")
    all_errors, files_processed, trajectories_with_errors = collect_all_llm_errors()

    print(f"\nFiles processed: {files_processed}")
    print(f"Trajectories with LLM errors: {trajectories_with_errors}")
    print(f"Total LLM-detected errors: {len(all_errors)}")

    # Distribution stats
    by_module = defaultdict(int)
    by_type = defaultdict(int)
    by_agreement = defaultdict(int)
    by_confidence = defaultdict(int)
    for e in all_errors:
        by_module[e["module"]] += 1
        by_type[e["error_type"]] += 1
        by_agreement[e["agreement_status"]] += 1
        conf_bucket = f"{e['confidence']:.1f}"
        by_confidence[conf_bucket] += 1

    print(f"\nBy module: {dict(sorted(by_module.items(), key=lambda x: -x[1]))}")
    print(f"By error type: {dict(sorted(by_type.items(), key=lambda x: -x[1]))}")
    print(f"By agreement: {dict(sorted(by_agreement.items(), key=lambda x: -x[1]))}")
    print(f"By confidence: {dict(sorted(by_confidence.items()))}")

    # Sample 100
    random.seed(SEED)
    sample = random.sample(all_errors, min(SAMPLE_SIZE, len(all_errors)))

    # Save full sample for review
    output_path = BASE_DIR.parent / "validation_sample_100_errors.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    print(f"\nSample of {len(sample)} errors saved to: {output_path}")

    # Print human-readable version
    readable_path = BASE_DIR.parent / "validation_sample_100_readable.txt"
    with open(readable_path, "w", encoding="utf-8") as f:
        for i, err in enumerate(sample):
            f.write(f"{'='*80}\n")
            f.write(f"ERROR #{i+1}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Instance:    {err['instance_id']}\n")
            f.write(f"Batch:       {err['batch']}\n")
            f.write(f"Step:        {err['step_number']} / {err['total_steps']}\n")
            f.write(f"Module:      {err['module']}\n")
            f.write(f"Error Type:  {err['error_type']}\n")
            f.write(f"Confidence:  {err['confidence']}\n")
            f.write(f"Agreement:   {err['agreement_status']}\n")
            f.write(f"Regex Also:  {err['regex_also_detected']}\n")
            if err['regex_error_type']:
                f.write(f"Regex Type:  {err['regex_error_type']}\n")
            if err['regex_evidence']:
                f.write(f"Regex Evid:  {err['regex_evidence'][:200]}\n")
            f.write(f"Task:        {err['task_description'][:150]}\n")
            f.write(f"Explanation: {err['explanation']}\n")
            f.write(f"\n")

    print(f"Readable version saved to: {readable_path}")

    # Print sample summary
    sample_by_module = defaultdict(int)
    sample_by_type = defaultdict(int)
    sample_by_agreement = defaultdict(int)
    for e in sample:
        sample_by_module[e["module"]] += 1
        sample_by_type[e["error_type"]] += 1
        sample_by_agreement[e["agreement_status"]] += 1

    print(f"\nSample by module: {dict(sorted(sample_by_module.items(), key=lambda x: -x[1]))}")
    print(f"Sample by error type: {dict(sorted(sample_by_type.items(), key=lambda x: -x[1]))}")
    print(f"Sample by agreement: {dict(sorted(sample_by_agreement.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
