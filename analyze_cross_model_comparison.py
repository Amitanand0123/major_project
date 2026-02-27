"""
Analysis Script #2: Cross-Model-Size Comparison
Compares error patterns across swe-agent-llama-8b, 70b, and 405b.

Research Question: Do larger models make different types of errors?
This exploits the multi-model data in our trajectories.

Run after batches complete:
    python analyze_cross_model_comparison.py --results-dir results_1000_study
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def load_trajectory_metadata(trajectory_dir: str) -> dict:
    """Load model name and outcome for each trajectory"""
    metadata = {}
    for f in Path(trajectory_dir).glob("*.json"):
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        instance_id = data.get('instance_id', '')
        metadata[instance_id] = {
            'model_name': data.get('model_name', 'unknown'),
            'success': data.get('final_result', {}).get('success', False),
            'num_steps': len(data.get('steps', []))
        }
    return metadata


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


def analyze_cross_model(results: list, metadata: dict):
    """Compare error patterns across model sizes"""

    print("=" * 80)
    print("CROSS-MODEL-SIZE COMPARISON ANALYSIS")
    print("=" * 80)

    # Group results by model
    by_model = defaultdict(list)
    unmatched = 0

    for r in results:
        iid = r.get('instance_id', '')
        if iid not in metadata:
            unmatched += 1
            continue
        model = metadata[iid]['model_name']
        by_model[model].append({
            'result': r,
            'success': metadata[iid]['success'],
            'num_steps': metadata[iid]['num_steps']
        })

    print(f"\nTotal analyzed: {len(results)}")
    print(f"Unmatched: {unmatched}")
    print(f"\nModel distribution:")
    for model in sorted(by_model.keys()):
        group = by_model[model]
        successes = sum(1 for g in group if g['success'])
        print(f"  {model}: {len(group)} trajectories ({successes} success, {len(group)-successes} failure)")

    # --- Analysis 1: Overall error rates by model ---
    print(f"\n{'-' * 80}")
    print("1. ERROR RATES BY MODEL SIZE")
    print("-" * 80)

    print(f"\n  {'Model':<25s} {'N':>5s} {'Steps':>6s} {'Regex':>7s} {'LLM':>7s} {'R/step':>7s} {'L/step':>7s} {'Agree%':>7s}")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    model_stats = {}

    for model in sorted(by_model.keys()):
        group = by_model[model]
        total_steps = 0
        total_regex = 0
        total_llm = 0
        agreement_rates = []

        for g in group:
            r = g['result']
            p1 = r.get('phase1', {}).get('summary', {})
            steps = r.get('phase1', {}).get('total_steps', 0)
            total_steps += steps
            total_regex += p1.get('regex_total_errors', p1.get('total_errors', 0))
            total_llm += p1.get('llm_total_errors', 0)
            rate = p1.get('agreement_rate', 0)
            if rate > 0:
                agreement_rates.append(rate)

        n = len(group)
        avg_steps = total_steps / n if n > 0 else 0
        avg_regex = total_regex / n if n > 0 else 0
        avg_llm = total_llm / n if n > 0 else 0
        regex_per_step = total_regex / total_steps if total_steps > 0 else 0
        llm_per_step = total_llm / total_steps if total_steps > 0 else 0
        avg_agree = sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0

        print(f"  {model:<25s} {n:>5d} {avg_steps:>6.1f} {avg_regex:>7.1f} {avg_llm:>7.1f} {regex_per_step:>7.3f} {llm_per_step:>7.3f} {avg_agree:>6.1f}%")

        model_stats[model] = {
            'n': n,
            'avg_steps': avg_steps,
            'avg_regex_errors': avg_regex,
            'avg_llm_errors': avg_llm,
            'regex_per_step': regex_per_step,
            'llm_per_step': llm_per_step,
            'avg_agreement_rate': avg_agree,
            'total_regex': total_regex,
            'total_llm': total_llm,
            'total_steps': total_steps
        }

    # --- Analysis 2: Error type distribution by model ---
    print(f"\n{'-' * 80}")
    print("2. ERROR TYPE DISTRIBUTION BY MODEL")
    print("-" * 80)

    for channel_label, type_key in [("Regex", "regex_errors_by_type"), ("LLM", "llm_errors_by_type")]:
        model_types = {}
        all_types = set()

        for model in sorted(by_model.keys()):
            model_types[model] = Counter()
            for g in by_model[model]:
                p1 = g['result'].get('phase1', {}).get('summary', {})
                for etype, count in p1.get(type_key, p1.get('errors_by_type', {})).items():
                    model_types[model][etype] += count
                    all_types.add(etype)

        if not all_types:
            continue

        print(f"\n  {channel_label} Channel - Top Error Types by Model:")
        models_sorted = sorted(by_model.keys())

        # Header
        header = f"  {'Error Type':<28s}"
        for m in models_sorted:
            short = m.replace('swe-agent-llama-', '')
            header += f" {short:>8s}"
        print(header)
        print(f"  {'-'*28}" + f" {'-'*8}" * len(models_sorted))

        # Sort by total frequency
        type_totals = Counter()
        for m in models_sorted:
            for etype, count in model_types[m].items():
                type_totals[etype] += count

        for etype, _ in type_totals.most_common(15):
            row = f"  {etype:<28s}"
            for m in models_sorted:
                total = sum(model_types[m].values()) or 1
                pct = (model_types[m][etype] / total) * 100
                row += f" {pct:>7.1f}%"
            print(row)

    # --- Analysis 3: Error module distribution by model ---
    print(f"\n{'-' * 80}")
    print("3. ERROR MODULE DISTRIBUTION BY MODEL")
    print("-" * 80)

    for channel_label, mod_key in [("Regex", "regex_errors_by_module"), ("LLM", "llm_errors_by_module")]:
        model_mods = {}

        for model in sorted(by_model.keys()):
            model_mods[model] = Counter()
            for g in by_model[model]:
                p1 = g['result'].get('phase1', {}).get('summary', {})
                for mod, count in p1.get(mod_key, p1.get('errors_by_module', {})).items():
                    model_mods[model][mod] += count

        models_sorted = sorted(by_model.keys())
        print(f"\n  {channel_label} Channel - Errors by Module:")
        header = f"  {'Module':<15s}"
        for m in models_sorted:
            short = m.replace('swe-agent-llama-', '')
            header += f" {short:>8s}"
        print(header)
        print(f"  {'-'*15}" + f" {'-'*8}" * len(models_sorted))

        for mod in ['memory', 'reflection', 'planning', 'action', 'system']:
            row = f"  {mod:<15s}"
            for m in models_sorted:
                total = sum(model_mods[m].values()) or 1
                pct = (model_mods[m][mod] / total) * 100
                row += f" {pct:>7.1f}%"
            print(row)

    # --- Analysis 4: Success rate vs error density by model ---
    print(f"\n{'-' * 80}")
    print("4. SUCCESS RATE vs ERROR DENSITY BY MODEL")
    print("-" * 80)

    for model in sorted(by_model.keys()):
        group = by_model[model]
        success_group = [g for g in group if g['success']]
        failure_group = [g for g in group if not g['success']]

        success_rate = len(success_group) / len(group) * 100 if group else 0

        print(f"\n  {model} (success rate: {success_rate:.1f}%):")

        for label, subgroup in [("Success", success_group), ("Failure", failure_group)]:
            if not subgroup:
                print(f"    {label}: no trajectories")
                continue
            regex_errors = [g['result'].get('phase1', {}).get('summary', {}).get('regex_total_errors',
                           g['result'].get('phase1', {}).get('summary', {}).get('total_errors', 0))
                           for g in subgroup]
            llm_errors = [g['result'].get('phase1', {}).get('summary', {}).get('llm_total_errors', 0)
                         for g in subgroup]
            avg_r = sum(regex_errors) / len(regex_errors) if regex_errors else 0
            avg_l = sum(llm_errors) / len(llm_errors) if llm_errors else 0
            print(f"    {label} (n={len(subgroup)}): avg regex={avg_r:.1f}, avg llm={avg_l:.1f}")

    # --- Analysis 5: Critical error types by model ---
    print(f"\n{'-' * 80}")
    print("5. CRITICAL ERROR TYPES BY MODEL")
    print("-" * 80)

    for model in sorted(by_model.keys()):
        group = by_model[model]
        critical_types = Counter()
        critical_modules = Counter()
        no_critical = 0

        for g in group:
            ce = g['result'].get('phase2', {}).get('critical_error')
            if ce:
                critical_types[ce.get('error_type', 'unknown')] += 1
                critical_modules[ce.get('module', 'unknown')] += 1
            else:
                no_critical += 1

        print(f"\n  {model} (n={len(group)}, {no_critical} without critical error):")
        print(f"    Top critical error types:")
        for etype, count in critical_types.most_common(5):
            pct = count / len(group) * 100
            print(f"      {etype:<25s}: {count:>3d} ({pct:.1f}%)")
        print(f"    Critical error modules:")
        for mod, count in critical_modules.most_common():
            pct = count / len(group) * 100
            print(f"      {mod:<15s}: {count:>3d} ({pct:.1f}%)")

    print(f"\n{'=' * 80}")
    print("END OF CROSS-MODEL COMPARISON")
    print("=" * 80)

    return model_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Model Comparison Analysis")
    parser.add_argument("--results-dir", default="results_1000_study",
                       help="Directory with batch results")
    parser.add_argument("--trajectory-dir", default="data/swebench/final_trajectories",
                       help="Directory with original trajectory files")

    args = parser.parse_args()

    print("Loading trajectory metadata...")
    metadata = load_trajectory_metadata(args.trajectory_dir)
    print(f"Loaded {len(metadata)} trajectories")

    models = Counter(v['model_name'] for v in metadata.values())
    for m, c in models.most_common():
        print(f"  {m}: {c}")

    print("\nLoading analysis results...")
    results = load_analysis_results(args.results_dir)
    print(f"Loaded {len(results)} analysis results")

    if not results:
        print("No results found. Run batches first.")
        sys.exit(1)

    model_stats = analyze_cross_model(results, metadata)

    # Save results
    output_file = Path(args.results_dir) / "cross_model_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(model_stats, f, indent=2)
    print(f"\nResults saved to: {output_file}")
