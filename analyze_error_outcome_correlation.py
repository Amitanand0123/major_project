"""
Analysis Script #1: Error-Outcome Correlation
Correlates detected error patterns with trajectory success/failure.

Research Question: Do specific error types predict whether the agent succeeds or fails?
This analysis is novel - Zhu et al. did not correlate errors with outcomes.

Run after batches complete:
    python analyze_error_outcome_correlation.py --results-dir results_1000_study
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def load_trajectory_outcomes(trajectory_dir: str) -> dict:
    """Load success/failure outcomes and model names from original trajectory files"""
    outcomes = {}
    for f in Path(trajectory_dir).glob("*.json"):
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        instance_id = data.get('instance_id', '')
        outcomes[instance_id] = {
            'success': data.get('final_result', {}).get('success', False),
            'model_name': data.get('model_name', 'unknown'),
            'num_steps': len(data.get('steps', []))
        }
    return outcomes


def load_analysis_results(results_dir: str) -> list:
    """Load all individual analysis results from all batches"""
    results = []
    base = Path(results_dir)
    for batch_dir in sorted(base.glob("batch_*")):
        for run_dir in batch_dir.glob("run_*"):
            individual_dir = run_dir / "experiments" / "individual"
            if not individual_dir.exists():
                continue
            for f in sorted(individual_dir.glob("*_analysis.json")):
                with open(f, 'r', encoding='utf-8') as fh:
                    results.append(json.load(fh))
    return results


def analyze_correlation(results: list, outcomes: dict):
    """Main correlation analysis"""

    print("=" * 80)
    print("ERROR-OUTCOME CORRELATION ANALYSIS")
    print("=" * 80)

    # Split results by outcome
    success_results = []
    failure_results = []
    unmatched = 0

    for r in results:
        iid = r.get('instance_id', '')
        if iid not in outcomes:
            unmatched += 1
            continue
        if outcomes[iid]['success']:
            success_results.append(r)
        else:
            failure_results.append(r)

    print(f"\nTotal analyzed trajectories: {len(results)}")
    print(f"  Successful: {len(success_results)}")
    print(f"  Failed: {len(failure_results)}")
    if unmatched:
        print(f"  Unmatched (no outcome data): {unmatched}")
    print()

    # --- Analysis 1: Error counts ---
    print("-" * 80)
    print("1. ERROR DENSITY: Successful vs Failed Trajectories")
    print("-" * 80)

    for label, group in [("Successful", success_results), ("Failed", failure_results)]:
        if not group:
            continue
        regex_errors = []
        llm_errors = []
        steps_list = []
        for r in group:
            p1 = r.get('phase1', {}).get('summary', {})
            regex_errors.append(p1.get('regex_total_errors', p1.get('total_errors', 0)))
            llm_errors.append(p1.get('llm_total_errors', 0))
            steps_list.append(r.get('phase1', {}).get('total_steps', 0))

        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0
        avg_regex = sum(regex_errors) / len(regex_errors) if regex_errors else 0
        avg_llm = sum(llm_errors) / len(llm_errors) if llm_errors else 0
        avg_regex_per_step = sum(regex_errors) / sum(steps_list) if sum(steps_list) > 0 else 0
        avg_llm_per_step = sum(llm_errors) / sum(steps_list) if sum(steps_list) > 0 else 0

        print(f"\n  {label} (n={len(group)}):")
        print(f"    Avg steps/trajectory:       {avg_steps:.1f}")
        print(f"    Avg regex errors/trajectory: {avg_regex:.1f}")
        print(f"    Avg LLM errors/trajectory:   {avg_llm:.1f}")
        print(f"    Regex errors/step:           {avg_regex_per_step:.3f}")
        print(f"    LLM errors/step:             {avg_llm_per_step:.3f}")

    # --- Analysis 2: Error types by outcome ---
    print(f"\n{'-' * 80}")
    print("2. ERROR TYPE DISTRIBUTION: Successful vs Failed")
    print("-" * 80)

    for channel_label, type_key in [("Regex", "regex_errors_by_type"), ("LLM", "llm_errors_by_type")]:
        success_types = Counter()
        failure_types = Counter()

        for r in success_results:
            p1 = r.get('phase1', {}).get('summary', {})
            for etype, count in p1.get(type_key, p1.get('errors_by_type', {})).items():
                success_types[etype] += count

        for r in failure_results:
            p1 = r.get('phase1', {}).get('summary', {})
            for etype, count in p1.get(type_key, p1.get('errors_by_type', {})).items():
                failure_types[etype] += count

        all_types = set(list(success_types.keys()) + list(failure_types.keys()))
        if not all_types:
            continue

        print(f"\n  {channel_label} Channel Error Types:")
        print(f"  {'Error Type':<30s} {'Success':>8s} {'Failure':>8s} {'Ratio':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")

        total_s = sum(success_types.values()) or 1
        total_f = sum(failure_types.values()) or 1

        for etype in sorted(all_types, key=lambda x: -(success_types[x] + failure_types[x])):
            s_pct = (success_types[etype] / total_s) * 100
            f_pct = (failure_types[etype] / total_f) * 100
            ratio = f_pct / s_pct if s_pct > 0 else float('inf')
            ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "inf"
            print(f"  {etype:<30s} {s_pct:>7.1f}% {f_pct:>7.1f}% {ratio_str:>8s}")

    # --- Analysis 3: Error modules by outcome ---
    print(f"\n{'-' * 80}")
    print("3. ERROR MODULE DISTRIBUTION: Successful vs Failed")
    print("-" * 80)

    for channel_label, mod_key in [("Regex", "regex_errors_by_module"), ("LLM", "llm_errors_by_module")]:
        success_mods = Counter()
        failure_mods = Counter()

        for r in success_results:
            p1 = r.get('phase1', {}).get('summary', {})
            for mod, count in p1.get(mod_key, p1.get('errors_by_module', {})).items():
                success_mods[mod] += count

        for r in failure_results:
            p1 = r.get('phase1', {}).get('summary', {})
            for mod, count in p1.get(mod_key, p1.get('errors_by_module', {})).items():
                failure_mods[mod] += count

        all_mods = set(list(success_mods.keys()) + list(failure_mods.keys()))
        if not all_mods:
            continue

        total_s = sum(success_mods.values()) or 1
        total_f = sum(failure_mods.values()) or 1

        print(f"\n  {channel_label} Channel Errors by Module:")
        print(f"  {'Module':<15s} {'Success':>8s} {'Failure':>8s}")
        print(f"  {'-'*15} {'-'*8} {'-'*8}")

        for mod in ['memory', 'reflection', 'planning', 'action', 'system']:
            if mod in all_mods:
                s_pct = (success_mods[mod] / total_s) * 100
                f_pct = (failure_mods[mod] / total_f) * 100
                print(f"  {mod:<15s} {s_pct:>7.1f}% {f_pct:>7.1f}%")

    # --- Analysis 4: Critical error impact ---
    print(f"\n{'-' * 80}")
    print("4. CRITICAL ERROR (PHASE 2) vs OUTCOME")
    print("-" * 80)

    critical_success = Counter()
    critical_failure = Counter()
    no_critical_success = 0
    no_critical_failure = 0

    for r in success_results:
        ce = r.get('phase2', {}).get('critical_error')
        if ce:
            critical_success[ce.get('error_type', 'unknown')] += 1
        else:
            no_critical_success += 1

    for r in failure_results:
        ce = r.get('phase2', {}).get('critical_error')
        if ce:
            critical_failure[ce.get('error_type', 'unknown')] += 1
        else:
            no_critical_failure += 1

    print(f"\n  Trajectories with NO critical error:")
    print(f"    Successful: {no_critical_success}/{len(success_results)} ({no_critical_success/len(success_results)*100:.1f}%)" if success_results else "")
    print(f"    Failed:     {no_critical_failure}/{len(failure_results)} ({no_critical_failure/len(failure_results)*100:.1f}%)" if failure_results else "")

    all_critical = set(list(critical_success.keys()) + list(critical_failure.keys()))
    if all_critical:
        print(f"\n  Critical Error Types:")
        print(f"  {'Error Type':<30s} {'Success':>8s} {'Failure':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8}")
        for etype in sorted(all_critical, key=lambda x: -(critical_success[x] + critical_failure[x])):
            print(f"  {etype:<30s} {critical_success[etype]:>8d} {critical_failure[etype]:>8d}")

    # --- Analysis 5: First error step position ---
    print(f"\n{'-' * 80}")
    print("5. FIRST ERROR POSITION (Normalized by trajectory length)")
    print("-" * 80)

    for label, group in [("Successful", success_results), ("Failed", failure_results)]:
        if not group:
            continue
        first_error_positions = []
        for r in group:
            steps = r.get('phase1', {}).get('step_analyses', [])
            total = len(steps)
            if total == 0:
                continue
            for s in steps:
                has_regex = any(s.get(f'regex_{m}_error') is not None
                               for m in ['memory', 'reflection', 'planning', 'action', 'system'])
                if has_regex:
                    first_error_positions.append(s['step_number'] / total)
                    break

        if first_error_positions:
            avg_pos = sum(first_error_positions) / len(first_error_positions)
            print(f"\n  {label} (n={len(first_error_positions)} with errors):")
            print(f"    Avg first error position: {avg_pos:.2f} (0=start, 1=end)")
            print(f"    First quartile errors before step: {sorted(first_error_positions)[len(first_error_positions)//4]:.2f}")

    # --- Analysis 6: Agreement rate by outcome ---
    print(f"\n{'-' * 80}")
    print("6. DUAL-CHANNEL AGREEMENT BY OUTCOME")
    print("-" * 80)

    for label, group in [("Successful", success_results), ("Failed", failure_results)]:
        if not group:
            continue
        rates = []
        for r in group:
            p1 = r.get('phase1', {}).get('summary', {})
            rate = p1.get('agreement_rate', 0)
            if rate > 0:
                rates.append(rate)

        if rates:
            avg = sum(rates) / len(rates)
            print(f"\n  {label} (n={len(rates)}):")
            print(f"    Avg agreement rate: {avg:.1f}%")
            print(f"    Min: {min(rates):.1f}%, Max: {max(rates):.1f}%")

    print(f"\n{'=' * 80}")
    print("END OF CORRELATION ANALYSIS")
    print("=" * 80)

    # Save results as JSON
    return {
        'total_analyzed': len(results),
        'successful': len(success_results),
        'failed': len(failure_results),
        'unmatched': unmatched
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error-Outcome Correlation Analysis")
    parser.add_argument("--results-dir", default="results_1000_study",
                       help="Directory with batch results")
    parser.add_argument("--trajectory-dir", default="data/swebench/final_trajectories",
                       help="Directory with original trajectory files")

    args = parser.parse_args()

    print("Loading trajectory outcomes...")
    outcomes = load_trajectory_outcomes(args.trajectory_dir)
    print(f"Loaded {len(outcomes)} trajectory outcomes")
    print(f"  Success: {sum(1 for v in outcomes.values() if v['success'])}")
    print(f"  Failure: {sum(1 for v in outcomes.values() if not v['success'])}")

    print("\nLoading analysis results...")
    results = load_analysis_results(args.results_dir)
    print(f"Loaded {len(results)} analysis results")

    if not results:
        print("No results found. Run batches first.")
        sys.exit(1)

    summary = analyze_correlation(results, outcomes)

    # Save summary
    output_file = Path(args.results_dir) / "error_outcome_correlation.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_file}")
