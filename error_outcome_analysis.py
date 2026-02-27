"""
Error-Outcome Correlation Analysis
Analyzes the relationship between Phase 1 error patterns and trajectory outcomes
across 300 trajectories (6 batches x 50 trajectories) from the 1000-study results.
"""

import json
import os
import glob
from collections import defaultdict
from pathlib import Path
import statistics

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"c:/Users/amita/Downloads/mp")
RESULTS_DIR = BASE_DIR / "results_1000_study"
TRAJECTORIES_DIR = BASE_DIR / "data" / "swebench" / "final_trajectories"

MODULES = ["action", "memory", "planning", "reflection", "system"]

# Batch directories mapping
BATCH_RUN_DIRS = {
    "batch_01": "run_20260209_174557",
    "batch_02": "run_20260212_210830",
    "batch_03": "run_20260214_084905",
    "batch_04": "run_20260215_144254",
    "batch_05": "run_20260216_111311",
    "batch_06": "run_20260217_211323",
}

# ============================================================
# Step 1: Build instance_id -> pass/fail mapping from trajectory files
# ============================================================
def build_outcome_map():
    """Load all 1000 trajectory files and build instance_id -> success mapping."""
    outcome_map = {}
    traj_files = list(TRAJECTORIES_DIR.glob("*.json"))
    for tf in traj_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            instance_id = data.get("instance_id", tf.stem)
            success = data.get("final_result", {}).get("success", False)
            outcome_map[instance_id] = success
        except Exception as e:
            print(f"  Warning: Could not load trajectory {tf.name}: {e}")
    return outcome_map


# ============================================================
# Step 2: Load all 300 analysis results
# ============================================================
def load_all_analyses():
    """Load all individual analysis JSON files across all batches."""
    analyses = []
    for batch_name, run_dir in BATCH_RUN_DIRS.items():
        individual_dir = RESULTS_DIR / batch_name / run_dir / "experiments" / "individual"
        if not individual_dir.exists():
            print(f"  Warning: {individual_dir} does not exist, skipping.")
            continue
        for json_file in sorted(individual_dir.glob("*_analysis.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                analyses.append(data)
            except Exception as e:
                print(f"  Warning: Could not load {json_file.name}: {e}")
    return analyses


# ============================================================
# Step 3: Extract error counts from a single analysis
# ============================================================
def extract_error_counts(analysis):
    """
    Extract per-module error counts from both regex and LLM channels,
    plus total step count and per-error-type counts.
    """
    phase1 = analysis.get("phase1", {})
    total_steps = phase1.get("total_steps", 0)
    step_analyses = phase1.get("step_analyses", [])
    instance_id = analysis.get("instance_id", "unknown")

    # Per-module error counts
    regex_module_counts = {m: 0 for m in MODULES}
    llm_module_counts = {m: 0 for m in MODULES}

    # Per-error-type counts
    regex_type_counts = defaultdict(int)
    llm_type_counts = defaultdict(int)

    for step in step_analyses:
        for module in MODULES:
            # Regex channel
            regex_key = f"regex_{module}_error"
            regex_val = step.get(regex_key)
            if regex_val is not None:
                regex_module_counts[module] += 1
                etype = regex_val.get("error_type", "unknown")
                regex_type_counts[etype] += 1

            # LLM channel
            llm_key = f"llm_{module}_error"
            llm_val = step.get(llm_key)
            if llm_val is not None and llm_val.get("has_error", True):
                llm_module_counts[module] += 1
                etype = llm_val.get("error_type", "unknown")
                llm_type_counts[etype] += 1

    regex_total = sum(regex_module_counts.values())
    llm_total = sum(llm_module_counts.values())

    return {
        "instance_id": instance_id,
        "total_steps": total_steps,
        "regex_module_counts": regex_module_counts,
        "llm_module_counts": llm_module_counts,
        "regex_total": regex_total,
        "llm_total": llm_total,
        "regex_type_counts": dict(regex_type_counts),
        "llm_type_counts": dict(llm_type_counts),
    }


# ============================================================
# Step 4: Compute statistics
# ============================================================
def safe_mean(values):
    return statistics.mean(values) if values else 0.0

def safe_stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0

def safe_median(values):
    return statistics.median(values) if values else 0.0


def compute_analysis(outcome_map, analyses):
    """Main analysis: split by outcome, compute all metrics."""

    passing = []  # list of error_count dicts for passing trajectories
    failing = []  # list of error_count dicts for failing trajectories
    unmatched = 0

    for analysis in analyses:
        counts = extract_error_counts(analysis)
        iid = counts["instance_id"]
        if iid not in outcome_map:
            unmatched += 1
            continue
        if outcome_map[iid]:
            passing.append(counts)
        else:
            failing.append(counts)

    return passing, failing, unmatched


def print_results(passing, failing, unmatched):
    n_pass = len(passing)
    n_fail = len(failing)

    print("=" * 80)
    print("ERROR-OUTCOME CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: 300 trajectories across 6 batches (batch_01 - batch_06)")
    print(f"  Passing trajectories: {n_pass}")
    print(f"  Failing trajectories: {n_fail}")
    print(f"  Unmatched (no outcome data): {unmatched}")
    print()

    # ------------------------------------------------------------------
    # Table 1: Average Steps
    # ------------------------------------------------------------------
    pass_steps = [c["total_steps"] for c in passing]
    fail_steps = [c["total_steps"] for c in failing]

    print("-" * 80)
    print("TABLE 1: Trajectory Length (Steps)")
    print("-" * 80)
    print(f"{'Metric':<30} {'Passing':>12} {'Failing':>12} {'Ratio F/P':>12}")
    print(f"{'Mean steps':<30} {safe_mean(pass_steps):>12.2f} {safe_mean(fail_steps):>12.2f} {safe_mean(fail_steps)/max(safe_mean(pass_steps),0.001):>12.2f}")
    print(f"{'Median steps':<30} {safe_median(pass_steps):>12.1f} {safe_median(fail_steps):>12.1f} {'':>12}")
    print(f"{'Std dev steps':<30} {safe_stdev(pass_steps):>12.2f} {safe_stdev(fail_steps):>12.2f} {'':>12}")
    print()

    # ------------------------------------------------------------------
    # Table 2: Average Total Errors per Trajectory
    # ------------------------------------------------------------------
    pass_regex_total = [c["regex_total"] for c in passing]
    fail_regex_total = [c["regex_total"] for c in failing]
    pass_llm_total = [c["llm_total"] for c in passing]
    fail_llm_total = [c["llm_total"] for c in failing]

    print("-" * 80)
    print("TABLE 2: Total Errors per Trajectory")
    print("-" * 80)
    print(f"{'Channel':<20} {'Pass Mean':>10} {'Pass SD':>10} {'Fail Mean':>10} {'Fail SD':>10} {'Ratio F/P':>10}")
    for label, pvals, fvals in [
        ("Regex", pass_regex_total, fail_regex_total),
        ("LLM", pass_llm_total, fail_llm_total),
    ]:
        pm = safe_mean(pvals)
        fm = safe_mean(fvals)
        ratio = fm / pm if pm > 0 else float('inf')
        print(f"{label:<20} {pm:>10.2f} {safe_stdev(pvals):>10.2f} {fm:>10.2f} {safe_stdev(fvals):>10.2f} {ratio:>10.2f}")
    print()

    # ------------------------------------------------------------------
    # Table 3: Average Errors per Step (normalized)
    # ------------------------------------------------------------------
    pass_regex_per_step = [c["regex_total"] / max(c["total_steps"], 1) for c in passing]
    fail_regex_per_step = [c["regex_total"] / max(c["total_steps"], 1) for c in failing]
    pass_llm_per_step = [c["llm_total"] / max(c["total_steps"], 1) for c in passing]
    fail_llm_per_step = [c["llm_total"] / max(c["total_steps"], 1) for c in failing]

    print("-" * 80)
    print("TABLE 3: Errors per Step (Normalized Error Rate)")
    print("-" * 80)
    print(f"{'Channel':<20} {'Pass Mean':>10} {'Pass SD':>10} {'Fail Mean':>10} {'Fail SD':>10} {'Ratio F/P':>10}")
    for label, pvals, fvals in [
        ("Regex", pass_regex_per_step, fail_regex_per_step),
        ("LLM", pass_llm_per_step, fail_llm_per_step),
    ]:
        pm = safe_mean(pvals)
        fm = safe_mean(fvals)
        ratio = fm / pm if pm > 0 else float('inf')
        print(f"{label:<20} {pm:>10.4f} {safe_stdev(pvals):>10.4f} {fm:>10.4f} {safe_stdev(fvals):>10.4f} {ratio:>10.2f}")
    print()

    # ------------------------------------------------------------------
    # Table 4: Average Errors per Module for Passing vs Failing
    # ------------------------------------------------------------------
    print("-" * 80)
    print("TABLE 4: Average Errors per Module (Regex Channel)")
    print("-" * 80)
    print(f"{'Module':<15} {'Pass Mean':>10} {'Pass SD':>10} {'Fail Mean':>10} {'Fail SD':>10} {'Ratio F/P':>10} {'Delta':>10}")
    for mod in MODULES:
        pvals = [c["regex_module_counts"][mod] for c in passing]
        fvals = [c["regex_module_counts"][mod] for c in failing]
        pm = safe_mean(pvals)
        fm = safe_mean(fvals)
        ratio = fm / pm if pm > 0 else float('inf')
        delta = fm - pm
        print(f"{mod.capitalize():<15} {pm:>10.3f} {safe_stdev(pvals):>10.3f} {fm:>10.3f} {safe_stdev(fvals):>10.3f} {ratio:>10.2f} {delta:>+10.3f}")
    print()

    print("-" * 80)
    print("TABLE 5: Average Errors per Module (LLM Channel)")
    print("-" * 80)
    print(f"{'Module':<15} {'Pass Mean':>10} {'Pass SD':>10} {'Fail Mean':>10} {'Fail SD':>10} {'Ratio F/P':>10} {'Delta':>10}")
    for mod in MODULES:
        pvals = [c["llm_module_counts"][mod] for c in passing]
        fvals = [c["llm_module_counts"][mod] for c in failing]
        pm = safe_mean(pvals)
        fm = safe_mean(fvals)
        ratio = fm / pm if pm > 0 else float('inf')
        delta = fm - pm
        print(f"{mod.capitalize():<15} {pm:>10.3f} {safe_stdev(pvals):>10.3f} {fm:>10.3f} {safe_stdev(fvals):>10.3f} {ratio:>10.2f} {delta:>+10.3f}")
    print()

    # ------------------------------------------------------------------
    # Table 6: Per-module errors per step (normalized by trajectory length)
    # ------------------------------------------------------------------
    print("-" * 80)
    print("TABLE 6: Per-Module Error Rate (Errors/Step) -- Regex Channel")
    print("-" * 80)
    print(f"{'Module':<15} {'Pass Rate':>10} {'Fail Rate':>10} {'Ratio F/P':>10}")
    for mod in MODULES:
        pvals = [c["regex_module_counts"][mod] / max(c["total_steps"], 1) for c in passing]
        fvals = [c["regex_module_counts"][mod] / max(c["total_steps"], 1) for c in failing]
        pm = safe_mean(pvals)
        fm = safe_mean(fvals)
        ratio = fm / pm if pm > 0 else float('inf')
        print(f"{mod.capitalize():<15} {pm:>10.4f} {fm:>10.4f} {ratio:>10.2f}")
    print()

    print("-" * 80)
    print("TABLE 7: Per-Module Error Rate (Errors/Step) -- LLM Channel")
    print("-" * 80)
    print(f"{'Module':<15} {'Pass Rate':>10} {'Fail Rate':>10} {'Ratio F/P':>10}")
    for mod in MODULES:
        pvals = [c["llm_module_counts"][mod] / max(c["total_steps"], 1) for c in passing]
        fvals = [c["llm_module_counts"][mod] / max(c["total_steps"], 1) for c in failing]
        pm = safe_mean(pvals)
        fm = safe_mean(fvals)
        ratio = fm / pm if pm > 0 else float('inf')
        print(f"{mod.capitalize():<15} {pm:>10.4f} {fm:>10.4f} {ratio:>10.2f}")
    print()

    # ------------------------------------------------------------------
    # Table 8: Error Types Most Over-represented in Failing (Regex)
    # ------------------------------------------------------------------
    pass_type_totals_regex = defaultdict(int)
    fail_type_totals_regex = defaultdict(int)
    pass_type_totals_llm = defaultdict(int)
    fail_type_totals_llm = defaultdict(int)

    for c in passing:
        for etype, cnt in c["regex_type_counts"].items():
            pass_type_totals_regex[etype] += cnt
        for etype, cnt in c["llm_type_counts"].items():
            pass_type_totals_llm[etype] += cnt

    for c in failing:
        for etype, cnt in c["regex_type_counts"].items():
            fail_type_totals_regex[etype] += cnt
        for etype, cnt in c["llm_type_counts"].items():
            fail_type_totals_llm[etype] += cnt

    all_regex_types = set(pass_type_totals_regex.keys()) | set(fail_type_totals_regex.keys())
    all_llm_types = set(pass_type_totals_llm.keys()) | set(fail_type_totals_llm.keys())

    # Compute per-trajectory average for each error type
    def compute_type_overrep(pass_type_totals, fail_type_totals, all_types, n_pass, n_fail):
        rows = []
        for etype in all_types:
            p_avg = pass_type_totals.get(etype, 0) / max(n_pass, 1)
            f_avg = fail_type_totals.get(etype, 0) / max(n_fail, 1)
            ratio = f_avg / p_avg if p_avg > 0 else float('inf')
            delta = f_avg - p_avg
            rows.append((etype, p_avg, f_avg, ratio, delta,
                          pass_type_totals.get(etype, 0),
                          fail_type_totals.get(etype, 0)))
        # Sort by ratio descending, but put inf at end
        rows.sort(key=lambda r: (-r[3] if r[3] != float('inf') else -999999, -r[4]))
        return rows

    print("-" * 80)
    print("TABLE 8: Error Types by Over-representation in Failing (Regex Channel)")
    print("-" * 80)
    print(f"{'Error Type':<30} {'Pass Avg':>9} {'Fail Avg':>9} {'Ratio F/P':>10} {'Pass Tot':>9} {'Fail Tot':>9}")
    regex_rows = compute_type_overrep(pass_type_totals_regex, fail_type_totals_regex, all_regex_types, n_pass, n_fail)
    for etype, p_avg, f_avg, ratio, delta, p_tot, f_tot in regex_rows:
        ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "inf"
        print(f"{etype:<30} {p_avg:>9.3f} {f_avg:>9.3f} {ratio_str:>10} {p_tot:>9} {f_tot:>9}")
    print()

    print("-" * 80)
    print("TABLE 9: Error Types by Over-representation in Failing (LLM Channel)")
    print("-" * 80)
    print(f"{'Error Type':<30} {'Pass Avg':>9} {'Fail Avg':>9} {'Ratio F/P':>10} {'Pass Tot':>9} {'Fail Tot':>9}")
    llm_rows = compute_type_overrep(pass_type_totals_llm, fail_type_totals_llm, all_llm_types, n_pass, n_fail)
    for etype, p_avg, f_avg, ratio, delta, p_tot, f_tot in llm_rows:
        ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "inf"
        print(f"{etype:<30} {p_avg:>9.3f} {f_avg:>9.3f} {ratio_str:>10} {p_tot:>9} {f_tot:>9}")
    print()

    # ------------------------------------------------------------------
    # Table 10: Error-Free Trajectories
    # ------------------------------------------------------------------
    pass_zero_regex = sum(1 for c in passing if c["regex_total"] == 0)
    fail_zero_regex = sum(1 for c in failing if c["regex_total"] == 0)
    pass_zero_llm = sum(1 for c in passing if c["llm_total"] == 0)
    fail_zero_llm = sum(1 for c in failing if c["llm_total"] == 0)

    print("-" * 80)
    print("TABLE 10: Error-Free Trajectories")
    print("-" * 80)
    print(f"{'Channel':<20} {'Pass N':>8} {'Pass %':>8} {'Fail N':>8} {'Fail %':>8}")
    print(f"{'Regex':<20} {pass_zero_regex:>8} {100*pass_zero_regex/max(n_pass,1):>7.1f}% {fail_zero_regex:>8} {100*fail_zero_regex/max(n_fail,1):>7.1f}%")
    print(f"{'LLM':<20} {pass_zero_llm:>8} {100*pass_zero_llm/max(n_pass,1):>7.1f}% {fail_zero_llm:>8} {100*fail_zero_llm/max(n_fail,1):>7.1f}%")
    print()

    # ------------------------------------------------------------------
    # Table 11: Error Count Distribution (Percentiles)
    # ------------------------------------------------------------------
    print("-" * 80)
    print("TABLE 11: Error Count Distribution (Percentiles)")
    print("-" * 80)

    def percentile(values, pct):
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * pct / 100)
        idx = min(idx, len(sorted_v) - 1)
        return sorted_v[idx]

    print(f"{'Metric':<25} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Max':>8}")
    print("  Passing (Regex):")
    vals = [c["regex_total"] for c in passing]
    print(f"  {'':>23} {percentile(vals,25):>8.1f} {percentile(vals,50):>8.1f} {percentile(vals,75):>8.1f} {percentile(vals,90):>8.1f} {max(vals) if vals else 0:>8.1f}")
    print("  Failing (Regex):")
    vals = [c["regex_total"] for c in failing]
    print(f"  {'':>23} {percentile(vals,25):>8.1f} {percentile(vals,50):>8.1f} {percentile(vals,75):>8.1f} {percentile(vals,90):>8.1f} {max(vals) if vals else 0:>8.1f}")
    print("  Passing (LLM):")
    vals = [c["llm_total"] for c in passing]
    print(f"  {'':>23} {percentile(vals,25):>8.1f} {percentile(vals,50):>8.1f} {percentile(vals,75):>8.1f} {percentile(vals,90):>8.1f} {max(vals) if vals else 0:>8.1f}")
    print("  Failing (LLM):")
    vals = [c["llm_total"] for c in failing]
    print(f"  {'':>23} {percentile(vals,25):>8.1f} {percentile(vals,50):>8.1f} {percentile(vals,75):>8.1f} {percentile(vals,90):>8.1f} {max(vals) if vals else 0:>8.1f}")
    print()

    # ------------------------------------------------------------------
    # Summary of Key Findings
    # ------------------------------------------------------------------
    print("=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    # 1. Error rate comparison
    pm_regex = safe_mean(pass_regex_per_step)
    fm_regex = safe_mean(fail_regex_per_step)
    pm_llm = safe_mean(pass_llm_per_step)
    fm_llm = safe_mean(fail_llm_per_step)

    print(f"\n1. Normalized Error Rate (errors/step):")
    print(f"   Regex: Passing={pm_regex:.4f}, Failing={fm_regex:.4f} (Ratio: {fm_regex/max(pm_regex,0.0001):.2f}x)")
    print(f"   LLM:   Passing={pm_llm:.4f}, Failing={fm_llm:.4f} (Ratio: {fm_llm/max(pm_llm,0.0001):.2f}x)")

    # 2. Most discriminative module
    print(f"\n2. Most Discriminative Modules (highest fail/pass ratio by errors/step):")
    for channel_name, channel_key in [("Regex", "regex_module_counts"), ("LLM", "llm_module_counts")]:
        best_mod = None
        best_ratio = 0
        for mod in MODULES:
            pvals = [c[channel_key][mod] / max(c["total_steps"], 1) for c in passing]
            fvals = [c[channel_key][mod] / max(c["total_steps"], 1) for c in failing]
            pm = safe_mean(pvals)
            fm = safe_mean(fvals)
            ratio = fm / pm if pm > 0 else float('inf')
            if ratio != float('inf') and ratio > best_ratio:
                best_ratio = ratio
                best_mod = mod
        if best_mod:
            print(f"   {channel_name}: {best_mod.capitalize()} ({best_ratio:.2f}x higher in failing)")

    # 3. Most over-represented error types in failing
    print(f"\n3. Top 5 Most Over-represented Error Types in Failing:")
    print(f"   Regex Channel:")
    for i, (etype, p_avg, f_avg, ratio, delta, p_tot, f_tot) in enumerate(regex_rows[:5]):
        ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "only in fail"
        print(f"     {i+1}. {etype}: {ratio_str} (pass avg={p_avg:.3f}, fail avg={f_avg:.3f})")

    print(f"   LLM Channel:")
    for i, (etype, p_avg, f_avg, ratio, delta, p_tot, f_tot) in enumerate(llm_rows[:5]):
        ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "only in fail"
        print(f"     {i+1}. {etype}: {ratio_str} (pass avg={p_avg:.3f}, fail avg={f_avg:.3f})")

    # 4. Error-free stats
    print(f"\n4. Error-Free Trajectories:")
    print(f"   Regex: {100*pass_zero_regex/max(n_pass,1):.1f}% of passing vs {100*fail_zero_regex/max(n_fail,1):.1f}% of failing")
    print(f"   LLM:   {100*pass_zero_llm/max(n_pass,1):.1f}% of passing vs {100*fail_zero_llm/max(n_fail,1):.1f}% of failing")

    print()
    print("=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Loading outcome mapping from trajectory files...")
    outcome_map = build_outcome_map()
    n_true = sum(1 for v in outcome_map.values() if v)
    n_false = sum(1 for v in outcome_map.values() if not v)
    print(f"  Loaded {len(outcome_map)} trajectories: {n_true} passing, {n_false} failing")

    print("\nLoading analysis results from 6 batches...")
    analyses = load_all_analyses()
    print(f"  Loaded {len(analyses)} analysis results")

    print("\nPerforming error-outcome correlation analysis...\n")
    passing, failing, unmatched = compute_analysis(outcome_map, analyses)

    print_results(passing, failing, unmatched)
