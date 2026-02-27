"""
Experiment Runner for Code Domain Extension
Runs complete AgentDebug pipeline on SWE-bench trajectories
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.swebench_integration import SWEBenchLoader
from detector.code_phase1_detector import CodePhase1Detector
from detector.code_phase2_detector import CodePhase2Detector


class CodeExperimentRunner:
    """
    Experiment runner for code domain extension
    Runs Phase 1 + Phase 2 analysis on SWE-bench trajectories
    """

    def __init__(self, llm, output_dir: str = "results/code_domain"):
        """
        Initialize experiment runner

        Args:
            llm: Language model
            output_dir: Directory to save results
        """
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.swebench_loader = SWEBenchLoader()
        self.phase1_detector = CodePhase1Detector(llm, use_automatic_detection=True)
        self.phase2_detector = CodePhase2Detector(llm)

    async def run_single_trajectory(self, trajectory: Dict) -> Dict[str, Any]:
        """
        Run complete analysis on single trajectory using dual-channel detection

        Args:
            trajectory: Trajectory dictionary

        Returns:
            Complete analysis results with both regex and LLM channels
        """
        instance_id = trajectory.get('instance_id', 'unknown')

        print(f"\n{'='*80}")
        print(f"ANALYZING: {instance_id}")
        print(f"{'='*80}")

        # Phase 1: Dual-channel error analysis (regex + LLM)
        print("\n[Phase 1] Dual-channel error analysis (regex + LLM)...")
        phase1_dual_results = await self.phase1_detector.analyze_trajectory_dual(trajectory)

        # Build regex-only view for Phase 2 compatibility
        phase1_regex_view = self._extract_regex_view(phase1_dual_results)

        # Phase 2: Critical error identification (uses regex view)
        print("\n[Phase 2] Critical error identification...")
        phase2_results = await self.phase2_detector.analyze_with_phase2(
            phase1_regex_view, trajectory
        )

        # Combine results
        complete_results = {
            'instance_id': instance_id,
            'phase1': phase1_dual_results,
            'phase1_regex_view': phase1_regex_view,
            'phase2': phase2_results,
            'timestamp': datetime.now().isoformat()
        }

        return complete_results

    def _extract_regex_view(self, dual_results: Dict) -> Dict:
        """Convert dual-channel results to old regex-only format for Phase 2 compatibility"""
        converted_steps = []
        for step in dual_results['step_analyses']:
            converted_steps.append({
                'step_number': step['step_number'],
                'memory_error': step.get('regex_memory_error'),
                'reflection_error': step.get('regex_reflection_error'),
                'planning_error': step.get('regex_planning_error'),
                'action_error': step.get('regex_action_error'),
                'system_error': step.get('regex_system_error'),
                'detection_method': 'automatic',
                'cost': 0.0
            })

        dual_summary = dual_results.get('summary', {})
        return {
            'instance_id': dual_results['instance_id'],
            'task_description': dual_results.get('task_description', ''),
            'total_steps': dual_results['total_steps'],
            'step_analyses': converted_steps,
            'summary': {
                'total_errors': dual_summary.get('regex_total_errors', 0),
                'automatic_detection_count': dual_summary.get('regex_total_errors', 0),
                'automatic_detection_rate': 100.0,
                'llm_detection_count': 0,
                'errors_by_module': dual_summary.get('regex_errors_by_module', {}),
                'errors_by_type': dual_summary.get('regex_errors_by_type', {}),
                'total_cost': 0.0,
                'cost_per_step': 0.0
            },
            'total_cost': 0.0,
            'timestamp': dual_results.get('timestamp', '')
        }

    async def run_batch_experiments(self, trajectory_dir: str,
                                    max_trajectories: int = 100,
                                    start_index: int = 0) -> Dict[str, Any]:
        """
        Run experiments on batch of trajectories

        Args:
            trajectory_dir: Directory containing trajectory files
            max_trajectories: Maximum number to process
            start_index: Index to start loading from (for batch processing)

        Returns:
            Batch results summary
        """
        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT: Code Domain Extension")
        print(f"{'='*80}")
        print(f"Directory: {trajectory_dir}")
        print(f"Max trajectories: {max_trajectories}")
        print(f"Start index: {start_index}")

        # Load trajectories
        print("\nLoading trajectories...")
        trajectories = self.swebench_loader.load_multiple_trajectories(
            trajectory_dir, max_count=max_trajectories, start_index=start_index
        )

        print(f"Loaded {len(trajectories)} trajectories")

        # Check for already-processed trajectories (resume support)
        individual_dir = self.output_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        failed_count = 0
        skipped_count = 0

        for i, trajectory in enumerate(trajectories):
            instance_id = trajectory.get('instance_id', 'unknown')
            result_file = individual_dir / f"{instance_id}_analysis.json"

            # Skip if already processed (resume support)
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                    all_results.append(existing_result)
                    skipped_count += 1
                    print(f"\n[{i+1}/{len(trajectories)}] ⏭ Skipping {instance_id} (already processed)")
                    continue
                except Exception:
                    pass  # If file is corrupted, re-process it

            print(f"\n[{i+1}/{len(trajectories)}] Processing {instance_id}")

            try:
                result = await self.run_single_trajectory(trajectory)
                all_results.append(result)

                # Save individual result
                self._save_individual_result(result)

            except Exception as e:
                print(f"✗ Error processing trajectory: {e}")
                failed_count += 1
                continue

        if skipped_count > 0:
            print(f"\n✓ Resumed: skipped {skipped_count} already-processed trajectories")

        # Generate aggregate statistics
        print("\n" + "="*80)
        print("GENERATING AGGREGATE STATISTICS")
        print("="*80)

        aggregate_stats = self._compute_aggregate_statistics(all_results)

        # Save aggregate results
        self._save_aggregate_results(aggregate_stats, all_results)

        print(f"\n✓ Batch experiment complete!")
        print(f"  Successfully analyzed: {len(all_results)}/{len(trajectories)}")
        print(f"  Failed: {failed_count}")
        print(f"  Results saved to: {self.output_dir}")

        return aggregate_stats

    def _save_individual_result(self, result: Dict):
        """Save individual trajectory result"""
        instance_id = result['instance_id']
        filename = f"{instance_id}_analysis.json"
        filepath = self.output_dir / "individual" / filename

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

    def _save_aggregate_results(self, stats: Dict, all_results: List[Dict]):
        """Save aggregate results"""
        # Save statistics
        stats_file = self.output_dir / "aggregate_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        # Save all results
        all_results_file = self.output_dir / "all_results.json"
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Saved aggregate statistics to: {stats_file}")
        print(f"✓ Saved all results to: {all_results_file}")

    def _compute_aggregate_statistics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across all trajectories (dual-channel aware)"""
        stats = {
            'total_trajectories': len(all_results),
            'total_steps_analyzed': 0,
            'total_errors_detected': 0,
            'automatic_detection_count': 0,
            'llm_detection_count': 0,
            'total_cost': 0.0,
            'errors_by_module': {},
            'errors_by_type': {},
            'critical_errors_by_module': {},
            'critical_errors_by_type': {},
            'average_steps_per_trajectory': 0.0,
            'average_errors_per_trajectory': 0.0,
            'automatic_detection_rate': 0.0,
            # Dual-channel agreement stats
            'dual_channel': {
                'total_module_comparisons': 0,
                'agreement_counts': {'both_error': 0, 'both_clean': 0, 'regex_only': 0, 'llm_only': 0},
                'agreement_rate': 0.0,
                'error_type_agreement_rate': 0.0,
                'error_type_comparisons': 0,
                'regex_total_errors': 0,
                'llm_total_errors': 0,
                'llm_only_errors': 0,
                'regex_only_errors': 0,
                'llm_errors_by_module': {},
                'llm_errors_by_type': {},
                'total_llm_duration_seconds': 0.0,
                'llm_timeouts': 0
            }
        }

        for result in all_results:
            phase1 = result.get('phase1', {})
            phase2 = result.get('phase2', {})
            p1_summary = phase1.get('summary', {})

            stats['total_steps_analyzed'] += phase1.get('total_steps', 0)

            # Use regex errors as the "detected" count (backward compat)
            regex_errors = p1_summary.get('regex_total_errors', p1_summary.get('total_errors', 0))
            stats['total_errors_detected'] += regex_errors
            stats['automatic_detection_count'] += regex_errors
            stats['total_cost'] += p1_summary.get('total_cost', 0.0)

            # Aggregate regex errors by module/type
            regex_by_module = p1_summary.get('regex_errors_by_module', p1_summary.get('errors_by_module', {}))
            regex_by_type = p1_summary.get('regex_errors_by_type', p1_summary.get('errors_by_type', {}))

            for module, count in regex_by_module.items():
                stats['errors_by_module'][module] = stats['errors_by_module'].get(module, 0) + count
            for error_type, count in regex_by_type.items():
                stats['errors_by_type'][error_type] = stats['errors_by_type'].get(error_type, 0) + count

            # Phase 2 stats
            critical_error = phase2.get('critical_error')
            if critical_error:
                module = critical_error['module']
                error_type = critical_error['error_type']
                stats['critical_errors_by_module'][module] = stats['critical_errors_by_module'].get(module, 0) + 1
                stats['critical_errors_by_type'][error_type] = stats['critical_errors_by_type'].get(error_type, 0) + 1

            # Dual-channel stats
            dc = stats['dual_channel']
            if 'agreement_counts' in p1_summary:
                dc['total_module_comparisons'] += p1_summary.get('total_module_comparisons', 0)
                dc['regex_total_errors'] += p1_summary.get('regex_total_errors', 0)
                dc['llm_total_errors'] += p1_summary.get('llm_total_errors', 0)
                dc['llm_only_errors'] += p1_summary.get('llm_only_errors', 0)
                dc['regex_only_errors'] += p1_summary.get('regex_only_errors', 0)
                dc['total_llm_duration_seconds'] += p1_summary.get('total_llm_duration_seconds', 0)
                dc['llm_timeouts'] += p1_summary.get('llm_timeouts', 0)
                dc['error_type_comparisons'] += p1_summary.get('error_type_comparisons', 0)

                for ag_type in ['both_error', 'both_clean', 'regex_only', 'llm_only']:
                    dc['agreement_counts'][ag_type] += p1_summary.get('agreement_counts', {}).get(ag_type, 0)

                for module, count in p1_summary.get('llm_errors_by_module', {}).items():
                    dc['llm_errors_by_module'][module] = dc['llm_errors_by_module'].get(module, 0) + count
                for etype, count in p1_summary.get('llm_errors_by_type', {}).items():
                    dc['llm_errors_by_type'][etype] = dc['llm_errors_by_type'].get(etype, 0) + count

        # Compute averages
        if stats['total_trajectories'] > 0:
            stats['average_steps_per_trajectory'] = stats['total_steps_analyzed'] / stats['total_trajectories']
            stats['average_errors_per_trajectory'] = stats['total_errors_detected'] / stats['total_trajectories']

        if stats['total_errors_detected'] > 0:
            stats['automatic_detection_rate'] = (stats['automatic_detection_count'] / stats['total_errors_detected']) * 100

        # Compute dual-channel agreement rate
        dc = stats['dual_channel']
        total_comp = dc['total_module_comparisons']
        if total_comp > 0:
            agree = dc['agreement_counts']['both_error'] + dc['agreement_counts']['both_clean']
            dc['agreement_rate'] = (agree / total_comp) * 100

        if dc['error_type_comparisons'] > 0:
            # Recompute from individual summaries isn't exact but close enough
            pass

        if dc['total_llm_duration_seconds'] > 0 and stats['total_trajectories'] > 0:
            dc['avg_llm_duration_per_trajectory'] = dc['total_llm_duration_seconds'] / stats['total_trajectories']

        # Sort by frequency
        stats['errors_by_module'] = dict(sorted(stats['errors_by_module'].items(), key=lambda x: x[1], reverse=True))
        stats['errors_by_type'] = dict(sorted(stats['errors_by_type'].items(), key=lambda x: x[1], reverse=True))
        stats['critical_errors_by_module'] = dict(sorted(stats['critical_errors_by_module'].items(), key=lambda x: x[1], reverse=True))
        stats['critical_errors_by_type'] = dict(sorted(stats['critical_errors_by_type'].items(), key=lambda x: x[1], reverse=True))
        dc['llm_errors_by_module'] = dict(sorted(dc['llm_errors_by_module'].items(), key=lambda x: x[1], reverse=True))
        dc['llm_errors_by_type'] = dict(sorted(dc['llm_errors_by_type'].items(), key=lambda x: x[1], reverse=True))

        return stats

    def print_summary_report(self, stats: Dict):
        """Print human-readable summary report"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY REPORT")
        print("="*80)

        print(f"\n## Overall Statistics")
        print(f"Total trajectories analyzed: {stats['total_trajectories']}")
        print(f"Total steps analyzed: {stats['total_steps_analyzed']}")
        print(f"Total errors detected: {stats['total_errors_detected']}")
        print(f"Average errors per trajectory: {stats['average_errors_per_trajectory']:.1f}")
        print(f"Average steps per trajectory: {stats['average_steps_per_trajectory']:.1f}")

        print(f"\n## Detection Method Statistics")
        print(f"Automatic detection: {stats['automatic_detection_count']} ({stats['automatic_detection_rate']:.1f}%)")
        print(f"LLM detection: {stats['llm_detection_count']} ({100-stats['automatic_detection_rate']:.1f}%)")
        print(f"Total cost: ${stats['total_cost']:.2f}")

        print(f"\n## Errors by Module (All Steps)")
        for module, count in list(stats['errors_by_module'].items())[:5]:
            percentage = (count / stats['total_errors_detected']) * 100 if stats['total_errors_detected'] > 0 else 0
            print(f"  {module}: {count} ({percentage:.1f}%)")

        print(f"\n## Top Error Types (All Steps)")
        for error_type, count in list(stats['errors_by_type'].items())[:10]:
            percentage = (count / stats['total_errors_detected']) * 100 if stats['total_errors_detected'] > 0 else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")

        print(f"\n## Critical Errors by Module")
        for module, count in stats['critical_errors_by_module'].items():
            percentage = (count / stats['total_trajectories']) * 100 if stats['total_trajectories'] > 0 else 0
            print(f"  {module}: {count} ({percentage:.1f}%)")

        print(f"\n## Critical Error Types")
        for error_type, count in list(stats['critical_errors_by_type'].items())[:10]:
            percentage = (count / stats['total_trajectories']) * 100 if stats['total_trajectories'] > 0 else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")

        # Dual-channel agreement report
        dc = stats.get('dual_channel', {})
        if dc.get('total_module_comparisons', 0) > 0:
            ac = dc.get('agreement_counts', {})
            total_comp = dc['total_module_comparisons']
            print(f"\n## Dual-Channel Agreement (Regex vs LLM)")
            print(f"  Total module-level comparisons: {total_comp}")
            print(f"  Overall agreement rate: {dc.get('agreement_rate', 0):.1f}%")
            print(f"  Both detected error:    {ac.get('both_error', 0)} ({ac.get('both_error',0)/total_comp*100:.1f}%)")
            print(f"  Both detected clean:    {ac.get('both_clean', 0)} ({ac.get('both_clean',0)/total_comp*100:.1f}%)")
            print(f"  Regex only (LLM missed): {ac.get('regex_only', 0)} ({ac.get('regex_only',0)/total_comp*100:.1f}%)")
            print(f"  LLM only (regex missed): {ac.get('llm_only', 0)} ({ac.get('llm_only',0)/total_comp*100:.1f}%)")
            print(f"  Avg LLM time/trajectory: {dc.get('avg_llm_duration_per_trajectory', 0):.1f}s")
            print(f"  LLM timeouts: {dc.get('llm_timeouts', 0)}")

        print("\n" + "="*80)


async def main_demo():
    """Demo run of experiment runner"""
    from detector.swebench_integration import create_sample_trajectory

    # Create mock LLM
    class MockLLM:
        def invoke(self, prompt):
            if "Phase 1" in prompt or "detect_module_errors" in str(prompt):
                return "ERROR_TYPE: api_hallucination\nCONFIDENCE: 0.9\nEXPLANATION: Agent used non-existent attribute\nEVIDENCE: token.is_expired"
            else:
                return """STEP_NUMBER: 4
MODULE: planning
ERROR_TYPE: api_hallucination
CONFIDENCE: 0.95
EXPLANATION: Critical error that caused failure
COUNTERFACTUAL: Would have succeeded if API was verified first
PROPAGATION: api_hallucination -> AttributeError -> wrong fix"""

    llm = MockLLM()

    # Create runner
    runner = CodeExperimentRunner(llm, output_dir="results/demo")

    # Create sample trajectory
    trajectory = create_sample_trajectory()

    # Run single analysis
    print("Running demo analysis...")
    result = await runner.run_single_trajectory(trajectory)

    print("\n✓ Demo complete!")
    print(f"Phase 1 errors detected: {result['phase1']['summary']['total_errors']}")
    print(f"Automatic detection rate: {result['phase1']['summary']['automatic_detection_rate']:.1f}%")
    print(f"Critical error: {result['phase2']['critical_error']['error_type']} at step {result['phase2']['critical_error']['step_number']}")


if __name__ == "__main__":
    asyncio.run(main_demo())
