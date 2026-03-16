"""
Aggregate Analysis for 1000-Trajectory Study
Combines all 20 batches into final comprehensive results
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class StudyAggregator:
    """
    Aggregates results from multiple batches
    """

    def __init__(self, base_dir: str = "results_1000_study"):
        self.base_dir = Path(base_dir)
        self.progress_file = self.base_dir / "progress.json"

    def load_progress(self):
        """Load study progress"""
        with open(self.progress_file, 'r') as f:
            return json.load(f)

    def aggregate_all_batches(self):
        """
        Aggregate results from all completed batches
        """
        print("=" * 80)
        print("AGGREGATING 1000-TRAJECTORY STUDY RESULTS")
        print("=" * 80)

        progress = self.load_progress()
        completed_batches = progress['completed_batches']

        print(f"Total batches completed: {len(completed_batches)}")
        print(f"Total trajectories: {progress['total_trajectories_completed']}")
        print()

        # Initialize aggregate stats
        aggregate = {
            'study_info': {
                'total_batches': len(completed_batches),
                'total_trajectories': 0,
                'total_steps_analyzed': 0,
                'start_date': progress['start_date'],
                'end_date': progress['last_run_date'],
                'total_duration_hours': 0
            },
            'total_errors_detected': 0,
            'automatic_detection_count': 0,
            'llm_detection_count': 0,
            'total_cost': 0.0,
            'errors_by_module': defaultdict(int),
            'errors_by_type': defaultdict(int),
            'critical_errors_by_module': defaultdict(int),
            'critical_errors_by_type': defaultdict(int),
            'batch_summaries': [],
            # Phase 3 aggregate stats
            'phase3': {
                'total_debugged': 0,
                'simulated_success_count': 0,
                'convergence_count': 0,
                'real_verification_count': 0,
                'real_patch_applied_count': 0,
                'real_tests_run_count': 0,
                'real_success_count': 0,
                'gold_verification_count': 0,
                'gold_pass_count': 0,
                'env_compatible_count': 0,
                'fair_comparison_count': 0,
                'quality_distribution': defaultdict(int),
                'gold_failure_categories': defaultdict(int),
                'three_tier': {'total': 0, 'both_pass': 0, 'gold_only': 0, 'corrective_only': 0, 'neither': 0},
                '_sum_specificity': 0.0,
                '_sum_actionability': 0.0,
                '_sum_iterations': 0.0,
                'success_by_module': defaultdict(lambda: {'total': 0, 'success': 0}),
                'real_success_by_module': defaultdict(lambda: {'total': 0, 'success': 0}),
            },
            # Dual-channel aggregate stats
            'dual_channel': {
                'total_module_comparisons': 0,
                'agreement_counts': {'both_error': 0, 'both_clean': 0, 'regex_only': 0, 'llm_only': 0},
                'agreement_rate': 0.0,
                'regex_total_errors': 0,
                'llm_total_errors': 0,
                'llm_only_errors': 0,
                'regex_only_errors': 0,
                'llm_errors_by_module': defaultdict(int),
                'llm_errors_by_type': defaultdict(int),
                'total_llm_duration_seconds': 0.0,
                'llm_timeouts': 0
            }
        }

        # Process each batch
        for batch_info in completed_batches:
            batch_num = batch_info['batch_number']
            print(f"Processing batch {batch_num}...")

            batch_dir = self.base_dir / f"batch_{batch_num:02d}"

            # Find run directory
            run_dirs = list(batch_dir.glob("run_*"))
            if not run_dirs:
                print(f"  ⚠️ Warning: No run directory found for batch {batch_num}")
                continue

            run_dir = sorted(run_dirs)[-1]  # pick latest run directory
            agg_file = run_dir / "experiments" / "aggregate_statistics.json"

            if not agg_file.exists():
                print(f"  ⚠️ Warning: No aggregate stats for batch {batch_num}")
                continue

            # Load batch statistics
            with open(agg_file, 'r') as f:
                batch_stats = json.load(f)

            # Aggregate statistics
            aggregate['study_info']['total_trajectories'] += batch_stats['total_trajectories']
            aggregate['study_info']['total_steps_analyzed'] += batch_stats['total_steps_analyzed']
            aggregate['study_info']['total_duration_hours'] += batch_info['duration_hours']

            aggregate['total_errors_detected'] += batch_stats['total_errors_detected']
            aggregate['automatic_detection_count'] += batch_stats['automatic_detection_count']
            aggregate['llm_detection_count'] += batch_stats['llm_detection_count']
            aggregate['total_cost'] += batch_stats['total_cost']

            # Aggregate error distributions
            for module, count in batch_stats['errors_by_module'].items():
                aggregate['errors_by_module'][module] += count

            for error_type, count in batch_stats['errors_by_type'].items():
                aggregate['errors_by_type'][error_type] += count

            for module, count in batch_stats['critical_errors_by_module'].items():
                aggregate['critical_errors_by_module'][module] += count

            for error_type, count in batch_stats['critical_errors_by_type'].items():
                aggregate['critical_errors_by_type'][error_type] += count

            # Phase 3 stats
            batch_p3 = batch_stats.get('phase3', {})
            if batch_p3:
                p3 = aggregate['phase3']
                p3['total_debugged'] += batch_p3.get('total_debugged', 0)
                p3['simulated_success_count'] += batch_p3.get('simulated_success_count', 0)
                p3['convergence_count'] += batch_p3.get('convergence_count', 0)
                p3['real_verification_count'] += batch_p3.get('real_verification_count', 0)
                p3['real_patch_applied_count'] += batch_p3.get('real_patch_applied_count', 0)
                p3['real_tests_run_count'] += batch_p3.get('real_tests_run_count', 0)
                p3['real_success_count'] += batch_p3.get('real_success_count', 0)
                p3['gold_verification_count'] += batch_p3.get('gold_verification_count', 0)
                p3['gold_pass_count'] += batch_p3.get('gold_pass_count', 0)
                p3['env_compatible_count'] += batch_p3.get('env_compatible_count', 0)
                p3['fair_comparison_count'] += batch_p3.get('fair_comparison_count', 0)
                # Quality distribution
                for qual in ('high', 'medium', 'low'):
                    p3['quality_distribution'][qual] += batch_p3.get('quality_distribution', {}).get(qual, 0)
                # Gold failure categories
                for cat, cnt in batch_p3.get('gold_failure_categories', {}).items():
                    p3['gold_failure_categories'][cat] += cnt
                # Three-tier
                batch_tier = batch_p3.get('three_tier', {})
                if batch_tier:
                    for key in ('total', 'both_pass', 'gold_only', 'corrective_only', 'neither'):
                        p3['three_tier'][key] += batch_tier.get(key, 0)
                # Accumulate for averaging
                p3['_sum_specificity'] += batch_p3.get('avg_specificity', 0) * batch_p3.get('total_debugged', 0)
                p3['_sum_actionability'] += batch_p3.get('avg_actionability', 0) * batch_p3.get('total_debugged', 0)
                p3['_sum_iterations'] += batch_p3.get('avg_iterations_total', 0) * batch_p3.get('total_debugged', 0)
                # Success by module
                for mod, data in batch_p3.get('success_by_module', {}).items():
                    p3['success_by_module'][mod]['total'] += data.get('total', 0)
                    p3['success_by_module'][mod]['success'] += data.get('success', 0)
                for mod, data in batch_p3.get('real_success_by_module', {}).items():
                    p3['real_success_by_module'][mod]['total'] += data.get('total', 0)
                    p3['real_success_by_module'][mod]['success'] += data.get('success', 0)

            # Dual-channel stats
            batch_dc = batch_stats.get('dual_channel', {})
            if batch_dc:
                dc = aggregate['dual_channel']
                dc['total_module_comparisons'] += batch_dc.get('total_module_comparisons', 0)
                dc['regex_total_errors'] += batch_dc.get('regex_total_errors', 0)
                dc['llm_total_errors'] += batch_dc.get('llm_total_errors', 0)
                dc['llm_only_errors'] += batch_dc.get('llm_only_errors', 0)
                dc['regex_only_errors'] += batch_dc.get('regex_only_errors', 0)
                dc['total_llm_duration_seconds'] += batch_dc.get('total_llm_duration_seconds', 0)
                dc['llm_timeouts'] += batch_dc.get('llm_timeouts', 0)
                for ag_type in ['both_error', 'both_clean', 'regex_only', 'llm_only']:
                    dc['agreement_counts'][ag_type] += batch_dc.get('agreement_counts', {}).get(ag_type, 0)
                for m, c in batch_dc.get('llm_errors_by_module', {}).items():
                    dc['llm_errors_by_module'][m] += c
                for t, c in batch_dc.get('llm_errors_by_type', {}).items():
                    dc['llm_errors_by_type'][t] += c

            # Save batch summary
            aggregate['batch_summaries'].append({
                'batch_number': batch_num,
                'trajectories': batch_stats['total_trajectories'],
                'steps': batch_stats['total_steps_analyzed'],
                'errors': batch_stats['total_errors_detected'],
                'automatic_rate': batch_stats['automatic_detection_rate'],
                'cost': batch_stats['total_cost'],
                'duration_hours': batch_info['duration_hours'],
                'agreement_rate': batch_dc.get('agreement_rate', 0) if batch_dc else 0
            })

        # Convert defaultdicts to regular dicts
        aggregate['errors_by_module'] = dict(aggregate['errors_by_module'])
        aggregate['errors_by_type'] = dict(aggregate['errors_by_type'])
        aggregate['critical_errors_by_module'] = dict(aggregate['critical_errors_by_module'])
        aggregate['critical_errors_by_type'] = dict(aggregate['critical_errors_by_type'])
        aggregate['dual_channel']['llm_errors_by_module'] = dict(aggregate['dual_channel']['llm_errors_by_module'])
        aggregate['dual_channel']['llm_errors_by_type'] = dict(aggregate['dual_channel']['llm_errors_by_type'])

        # Compute dual-channel agreement rate
        dc = aggregate['dual_channel']
        total_comp = dc['total_module_comparisons']
        if total_comp > 0:
            agree = dc['agreement_counts']['both_error'] + dc['agreement_counts']['both_clean']
            dc['agreement_rate'] = (agree / total_comp) * 100
        if dc['total_llm_duration_seconds'] > 0 and aggregate['study_info']['total_trajectories'] > 0:
            dc['avg_llm_duration_per_trajectory'] = dc['total_llm_duration_seconds'] / aggregate['study_info']['total_trajectories']

        # Compute Phase 3 derived statistics
        p3 = aggregate['phase3']
        if p3['total_debugged'] > 0:
            p3['simulated_success_rate'] = round(p3['simulated_success_count'] / p3['total_debugged'] * 100, 1)
            p3['convergence_rate'] = round(p3['convergence_count'] / p3['total_debugged'] * 100, 1)
            p3['avg_specificity'] = round(p3['_sum_specificity'] / p3['total_debugged'], 3)
            p3['avg_actionability'] = round(p3['_sum_actionability'] / p3['total_debugged'], 3)
            p3['avg_iterations_total'] = round(p3['_sum_iterations'] / p3['total_debugged'], 2)
        if p3['real_verification_count'] > 0:
            p3['real_success_rate'] = round(p3['real_success_count'] / p3['real_verification_count'] * 100, 1)
        if p3['gold_verification_count'] > 0:
            p3['gold_pass_rate'] = round(p3['gold_pass_count'] / p3['gold_verification_count'] * 100, 1)
        if p3['env_compatible_count'] > 0:
            p3['env_compatible_rate'] = round(p3['env_compatible_count'] / p3['gold_verification_count'] * 100, 1)
        if p3['fair_comparison_count'] > 0:
            fair_n = p3['fair_comparison_count']
            tier = p3['three_tier']
            tier['corrective_success_rate_fair'] = round(tier['both_pass'] / fair_n * 100, 1) if fair_n > 0 else 0.0
            # Simulated success among fair instances (need to compute from batch data)
        # Remove internal accumulator fields
        for key in ('_sum_specificity', '_sum_actionability', '_sum_iterations'):
            del p3[key]
        # Convert defaultdicts
        p3['quality_distribution'] = dict(p3['quality_distribution'])
        p3['gold_failure_categories'] = dict(p3['gold_failure_categories'])
        p3['success_by_module'] = {k: dict(v) for k, v in p3['success_by_module'].items()}
        p3['real_success_by_module'] = {k: dict(v) for k, v in p3['real_success_by_module'].items()}

        # Compute derived statistics
        if aggregate['total_errors_detected'] > 0:
            aggregate['automatic_detection_rate'] = (
                aggregate['automatic_detection_count'] / aggregate['total_errors_detected']
            ) * 100
        else:
            aggregate['automatic_detection_rate'] = 0

        if aggregate['study_info']['total_trajectories'] > 0:
            aggregate['average_steps_per_trajectory'] = (
                aggregate['study_info']['total_steps_analyzed'] /
                aggregate['study_info']['total_trajectories']
            )
            aggregate['average_errors_per_trajectory'] = (
                aggregate['total_errors_detected'] /
                aggregate['study_info']['total_trajectories']
            )
        else:
            aggregate['average_steps_per_trajectory'] = 0
            aggregate['average_errors_per_trajectory'] = 0

        # Save aggregate results
        output_file = self.base_dir / "aggregate_1000_trajectories.json"
        with open(output_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print()
        print("=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_file}")
        print()

        self.print_summary(aggregate)

        return aggregate

    def print_summary(self, aggregate):
        """Print comprehensive summary"""

        print("=" * 80)
        print("1000-TRAJECTORY STUDY SUMMARY")
        print("=" * 80)
        print()

        # Study info
        info = aggregate['study_info']
        print(f"Study Period: {info['start_date']} → {info['end_date']}")
        print(f"Total Batches: {info['total_batches']}")
        print(f"Total Duration: {info['total_duration_hours']:.1f} hours ({info['total_duration_hours']/24:.1f} days)")
        print()

        # Overall statistics
        print(f"Total Trajectories: {info['total_trajectories']}")
        print(f"Total Steps Analyzed: {info['total_steps_analyzed']}")
        print(f"Average Steps per Trajectory: {aggregate['average_steps_per_trajectory']:.1f}")
        print()

        # Detection performance
        print(f"Total Errors Detected: {aggregate['total_errors_detected']}")
        print(f"Average Errors per Trajectory: {aggregate['average_errors_per_trajectory']:.1f}")
        print(f"Automatic Detection Rate: {aggregate['automatic_detection_rate']:.1f}%")
        print(f"Total Cost: ${aggregate['total_cost']:.2f}")
        print()

        # Error distribution by module
        print("Errors by Module:")
        sorted_modules = sorted(aggregate['errors_by_module'].items(),
                               key=lambda x: -x[1])
        for module, count in sorted_modules:
            pct = (count / aggregate['total_errors_detected']) * 100
            print(f"  {module:12s}: {count:4d} ({pct:5.1f}%)")
        print()

        # Error distribution by type (top 10)
        print("Top 10 Error Types:")
        sorted_types = sorted(aggregate['errors_by_type'].items(),
                             key=lambda x: -x[1])
        for error_type, count in sorted_types[:10]:
            pct = (count / aggregate['total_errors_detected']) * 100
            print(f"  {error_type:25s}: {count:4d} ({pct:5.1f}%)")
        print()

        # Critical errors
        total_critical = sum(aggregate['critical_errors_by_module'].values())
        print(f"Critical Errors: {total_critical}")
        print("Critical Errors by Module:")
        sorted_critical = sorted(aggregate['critical_errors_by_module'].items(),
                                key=lambda x: -x[1])
        for module, count in sorted_critical:
            pct = (count / total_critical) * 100 if total_critical > 0 else 0
            print(f"  {module:12s}: {count:4d} ({pct:5.1f}%)")
        print()

        # Dual-channel agreement
        dc = aggregate.get('dual_channel', {})
        if dc.get('total_module_comparisons', 0) > 0:
            ac = dc.get('agreement_counts', {})
            total_comp = dc['total_module_comparisons']
            print(f"Dual-Channel Agreement (Regex vs Local LLM):")
            print(f"  Total module-level comparisons: {total_comp}")
            print(f"  Overall agreement rate: {dc.get('agreement_rate', 0):.1f}%")
            print(f"  Both detected error:    {ac.get('both_error', 0)} ({ac.get('both_error',0)/total_comp*100:.1f}%)")
            print(f"  Both detected clean:    {ac.get('both_clean', 0)} ({ac.get('both_clean',0)/total_comp*100:.1f}%)")
            print(f"  Regex only (LLM missed): {ac.get('regex_only', 0)} ({ac.get('regex_only',0)/total_comp*100:.1f}%)")
            print(f"  LLM only (regex missed): {ac.get('llm_only', 0)} ({ac.get('llm_only',0)/total_comp*100:.1f}%)")
            print(f"  LLM timeouts: {dc.get('llm_timeouts', 0)}")
            print(f"  Total LLM compute time: {dc.get('total_llm_duration_seconds', 0)/3600:.1f} hours")
            print()

            print(f"  LLM Errors by Module:")
            for module, count in sorted(dc.get('llm_errors_by_module', {}).items(), key=lambda x: -x[1]):
                print(f"    {module:12s}: {count}")
            print()

        # Phase 3 stats
        p3 = aggregate.get('phase3', {})
        if p3.get('total_debugged', 0) > 0:
            print(f"Phase 3 - Iterative Debugging:")
            print(f"  Total debugged: {p3['total_debugged']}")
            print(f"  Simulated success: {p3.get('simulated_success_count', 0)} ({p3.get('simulated_success_rate', 0):.1f}%)")
            print(f"  Real success: {p3.get('real_success_count', 0)} ({p3.get('real_success_rate', 0):.1f}%)")
            print(f"  Gold pass: {p3.get('gold_pass_count', 0)} ({p3.get('gold_pass_rate', 0):.1f}%)")
            print(f"  Env compatible: {p3.get('env_compatible_count', 0)} ({p3.get('env_compatible_rate', 0):.1f}%)")
            print(f"  Fair comparison: {p3.get('fair_comparison_count', 0)}")
            print(f"  Quality distribution: {p3.get('quality_distribution', {})}")
            print(f"  Avg specificity: {p3.get('avg_specificity', 0):.3f}")
            print(f"  Avg actionability: {p3.get('avg_actionability', 0):.3f}")
            tier = p3.get('three_tier', {})
            if tier.get('total', 0) > 0:
                print(f"  Three-Tier Comparison (N={tier['total']}):")
                print(f"    both_pass: {tier.get('both_pass', 0)}")
                print(f"    gold_only: {tier.get('gold_only', 0)}")
                print(f"    corrective_only: {tier.get('corrective_only', 0)}")
                print(f"    neither: {tier.get('neither', 0)}")
            print()

        # Batch consistency
        print("Batch Consistency:")
        auto_rates = [b['automatic_rate'] for b in aggregate['batch_summaries']]
        costs = [b['cost'] for b in aggregate['batch_summaries']]
        durations = [b['duration_hours'] for b in aggregate['batch_summaries']]

        print(f"  Automatic detection rate: {min(auto_rates):.1f}% - {max(auto_rates):.1f}%")
        print(f"  Cost per batch: ${min(costs):.2f} - ${max(costs):.2f}")
        print(f"  Duration per batch: {min(durations):.1f}h - {max(durations):.1f}h (avg: {sum(durations)/len(durations):.1f}h)")
        print()

        print("=" * 80)
        print("PUBLICATION READINESS")
        print("=" * 80)

        # Check publication readiness
        ready = True
        reasons = []

        if info['total_trajectories'] >= 1000:
            print("✅ Sample size: 1000+ trajectories")
        else:
            print(f"⚠️ Sample size: {info['total_trajectories']} trajectories (target: 1000)")
            ready = False
            reasons.append("Incomplete sample")

        if aggregate['automatic_detection_rate'] >= 95.0:
            print(f"✅ Automatic detection: {aggregate['automatic_detection_rate']:.1f}%")
        else:
            print(f"⚠️ Automatic detection: {aggregate['automatic_detection_rate']:.1f}% (target: ≥95%)")
            ready = False
            reasons.append("Low automatic detection")

        print(f"✅ Zero cost (local Ollama): ${aggregate['total_cost']:.2f}")

        dc = aggregate.get('dual_channel', {})
        if dc.get('total_module_comparisons', 0) > 0:
            print(f"✅ Dual-channel agreement: {dc.get('agreement_rate', 0):.1f}%")
            print(f"✅ LLM found {dc.get('llm_only_errors', 0)} additional errors regex missed")
        else:
            print(f"⚠️ No dual-channel data available")
            ready = False
            reasons.append("Missing dual-channel data")

        print()

        if ready:
            print("🎉 STUDY IS PUBLICATION READY!")
            print()
            print("Next steps:")
            print("1. Review aggregate_1000_trajectories.json")
            print("2. Generate visualizations")
            print("3. Write methodology and results sections")
            print("4. Prepare for submission")
        else:
            print("⚠️ Study not yet ready for publication")
            print()
            print("Issues to address:")
            for reason in reasons:
                print(f"  - {reason}")

        print("=" * 80)


def generate_interim_report(base_dir: str = "results_1000_study"):
    """
    Generate interim report for current progress
    Useful for checking progress at 100, 200, 500 trajectories
    """
    aggregator = StudyAggregator(base_dir)

    print("=" * 80)
    print("INTERIM PROGRESS REPORT")
    print("=" * 80)

    progress = aggregator.load_progress()
    completed = progress['total_trajectories_completed']

    print(f"Current Progress: {completed}/1000 trajectories ({completed/10:.1f}%)")
    print(f"Batches completed: {len(progress['completed_batches'])}/20")
    print()

    if len(progress['completed_batches']) > 0:
        # Generate aggregate for current batches
        aggregate = aggregator.aggregate_all_batches()
        return aggregate
    else:
        print("No batches completed yet.")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate 1000-trajectory study results")
    parser.add_argument("--base-dir", default="results_1000_study",
                       help="Base directory for study")
    parser.add_argument("--interim", action="store_true",
                       help="Generate interim report (before all batches complete)")

    args = parser.parse_args()

    if args.interim:
        generate_interim_report(args.base_dir)
    else:
        aggregator = StudyAggregator(args.base_dir)
        aggregator.aggregate_all_batches()
