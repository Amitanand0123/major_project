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
from detector.code_phase3_debugger import CodePhase3Debugger
from detector.patch_verifier import DockerPatchVerifier


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

        # Initialize Docker patch verifier (optional — graceful fallback)
        self.verifier = None
        try:
            verifier = DockerPatchVerifier()
            if verifier.is_available():
                self.verifier = verifier
                print("  Docker patch verifier: ENABLED")
            else:
                print("  Docker patch verifier: DISABLED (Docker or metadata not available)")
        except Exception as e:
            print(f"  Docker patch verifier: DISABLED ({e})")

        self.phase3_debugger = CodePhase3Debugger(llm, verifier=self.verifier)

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

        # Build regex-only view (kept for backward compat in saved results)
        phase1_regex_view = self._extract_regex_view(phase1_dual_results)

        # Build merged dual-channel view for Phase 2 (uses both regex + LLM errors)
        phase1_merged_view = self._extract_merged_view(phase1_dual_results)

        # Phase 2: Critical error identification (uses merged dual-channel view)
        print("\n[Phase 2] Critical error identification...")
        phase2_results = await self.phase2_detector.analyze_with_phase2(
            phase1_merged_view, trajectory
        )

        # Phase 3: Simulated iterative debugging (only for failed trajectories with critical errors)
        phase3_results = None
        if phase2_results.get('critical_error') is not None:
            print("\n[Phase 3] Simulated iterative debugging...")
            try:
                phase3_results = await self.phase3_debugger.run_phase3(
                    phase2_results, phase1_dual_results, trajectory
                )
            except Exception as e:
                print(f"  Phase 3 error: {e}")

        # Combine results
        complete_results = {
            'instance_id': instance_id,
            'phase1': phase1_dual_results,
            'phase1_regex_view': phase1_regex_view,
            'phase2': phase2_results,
            'phase3': phase3_results,
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

    def _extract_merged_view(self, dual_results: Dict) -> Dict:
        """Merge regex + LLM errors into a single view for Phase 2.

        For each step-module, prefer the higher-confidence detection.
        This gives Phase 2 a richer set of candidate errors to reason about,
        rather than being limited to regex-only detections.
        """
        converted_steps = []
        for step in dual_results['step_analyses']:
            step_entry = {
                'step_number': step['step_number'],
                'detection_method': 'dual_channel',
                'cost': 0.0,
            }

            for module in ['memory', 'reflection', 'planning', 'action', 'system']:
                regex_err = step.get(f'regex_{module}_error')
                llm_err = step.get(f'llm_{module}_error')

                # Pick the best available error for this module
                merged = None
                if regex_err and llm_err:
                    # Both detected — use whichever has higher confidence
                    r_conf = regex_err.get('confidence', 0)
                    l_conf = llm_err.get('confidence', 0)
                    if r_conf >= l_conf:
                        merged = regex_err
                    else:
                        merged = {
                            'error_type': llm_err.get('error_type', 'unknown'),
                            'confidence': llm_err.get('confidence', 0),
                            'explanation': llm_err.get('explanation', ''),
                        }
                elif regex_err:
                    merged = regex_err
                elif llm_err:
                    merged = {
                        'error_type': llm_err.get('error_type', 'unknown'),
                        'confidence': llm_err.get('confidence', 0),
                        'explanation': llm_err.get('explanation', ''),
                    }

                step_entry[f'{module}_error'] = merged

            converted_steps.append(step_entry)

        dual_summary = dual_results.get('summary', {})
        total_regex = dual_summary.get('regex_total_errors', 0)
        total_llm = dual_summary.get('llm_total_errors', 0)

        return {
            'instance_id': dual_results['instance_id'],
            'task_description': dual_results.get('task_description', ''),
            'total_steps': dual_results['total_steps'],
            'step_analyses': converted_steps,
            'summary': {
                'total_errors': total_regex + total_llm,
                'automatic_detection_count': total_regex,
                'automatic_detection_rate': 100.0,
                'llm_detection_count': total_llm,
                'errors_by_module': {
                    **dual_summary.get('regex_errors_by_module', {}),
                    **{k: dual_summary.get('llm_errors_by_module', {}).get(k, 0)
                       for k in dual_summary.get('llm_errors_by_module', {})},
                },
                'errors_by_type': {
                    **dual_summary.get('regex_errors_by_type', {}),
                    **{k: dual_summary.get('llm_errors_by_type', {}).get(k, 0)
                       for k in dual_summary.get('llm_errors_by_type', {})},
                },
                'total_cost': 0.0,
                'cost_per_step': 0.0,
            },
            'total_cost': 0.0,
            'timestamp': dual_results.get('timestamp', ''),
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
            # Phase 3: Simulated debugging stats
            'phase3': {
                'total_debugged': 0,
                'simulated_success_count': 0,
                'simulated_success_rate': 0.0,
                'avg_iterations_to_success': 0.0,
                'avg_iterations_total': 0.0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'avg_specificity': 0.0,
                'avg_actionability': 0.0,
                'convergence_rate': 0.0,
                'convergence_count': 0,
                'success_by_agreement': {},
                'success_by_module': {},
                'iteration_counts': [],
                'iterations_to_success': [],
                'all_specificity': [],
                'all_actionability': [],
                # Real verification stats
                'real_verification_count': 0,
                'real_patch_applied_count': 0,
                'real_tests_run_count': 0,
                'real_success_count': 0,
                'real_success_rate': 0.0,
                'real_success_by_agreement': {},
                'real_success_by_module': {},
                'simulated_vs_real_match_count': 0,
                'simulated_vs_real_total': 0,
                'prediction_accuracy': 0.0,
                'avg_verification_duration': 0.0,
                'all_verification_durations': [],
                # Gold patch verification stats
                'gold_verification_count': 0,
                'gold_pass_count': 0,
                'gold_pass_rate': 0.0,
                'gold_failure_categories': {},
                'env_compatible_count': 0,
                'env_compatible_rate': 0.0,
                'fair_comparison_count': 0,
                # Three-tier comparison (only for fair_comparison_eligible instances)
                'three_tier': {
                    'total': 0,
                    'both_pass': 0,
                    'gold_only': 0,
                    'corrective_only': 0,
                    'neither': 0,
                    'corrective_success_rate_fair': 0.0,
                    'simulated_success_rate_fair': 0.0,
                    'overconfidence_gap_fair': 0.0,
                },
                # Temp lists for fair comparison computation
                '_fair_simulated_successes': 0,
                '_fair_total': 0,
            },
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

            # Phase 3 stats
            phase3 = result.get('phase3')
            if phase3:
                p3 = stats['phase3']
                p3['total_debugged'] += 1
                p3['iteration_counts'].append(phase3['total_iterations'])
                if phase3['final_success']:
                    p3['simulated_success_count'] += 1
                    if phase3['successful_iteration']:
                        p3['iterations_to_success'].append(phase3['successful_iteration'])
                quality = phase3.get('final_feedback_quality', 'low')
                p3['quality_distribution'][quality] = p3['quality_distribution'].get(quality, 0) + 1
                p3['all_specificity'].append(phase3.get('avg_specificity', 0))
                p3['all_actionability'].append(phase3.get('avg_actionability', 0))
                if phase3.get('convergence'):
                    p3['convergence_count'] += 1
                # Track success by agreement type
                ag = phase3.get('dual_channel_agreement', 'unknown')
                if ag not in p3['success_by_agreement']:
                    p3['success_by_agreement'][ag] = {'total': 0, 'success': 0}
                p3['success_by_agreement'][ag]['total'] += 1
                if phase3['final_success']:
                    p3['success_by_agreement'][ag]['success'] += 1
                # Track success by error module
                em = phase3.get('critical_error_module', 'unknown')
                if em not in p3['success_by_module']:
                    p3['success_by_module'][em] = {'total': 0, 'success': 0}
                p3['success_by_module'][em]['total'] += 1
                if phase3['final_success']:
                    p3['success_by_module'][em]['success'] += 1

                # Real verification stats
                rv = phase3.get('real_verification')
                if rv and isinstance(rv, dict) and 'tests_passed' in rv:
                    p3['real_verification_count'] += 1
                    if rv.get('patch_applied'):
                        p3['real_patch_applied_count'] += 1
                    if rv.get('tests_run'):
                        p3['real_tests_run_count'] += 1
                    if rv.get('tests_passed'):
                        p3['real_success_count'] += 1
                    if rv.get('duration_seconds'):
                        p3['all_verification_durations'].append(rv['duration_seconds'])
                    # Track simulated vs real match
                    sim_vs_real = phase3.get('simulated_vs_real_match')
                    if sim_vs_real is not None:
                        p3['simulated_vs_real_total'] += 1
                        if sim_vs_real:
                            p3['simulated_vs_real_match_count'] += 1
                    # Real success by agreement type
                    if ag not in p3['real_success_by_agreement']:
                        p3['real_success_by_agreement'][ag] = {'total': 0, 'success': 0}
                    p3['real_success_by_agreement'][ag]['total'] += 1
                    if rv.get('tests_passed'):
                        p3['real_success_by_agreement'][ag]['success'] += 1
                    # Real success by module
                    if em not in p3['real_success_by_module']:
                        p3['real_success_by_module'][em] = {'total': 0, 'success': 0}
                    p3['real_success_by_module'][em]['total'] += 1
                    if rv.get('tests_passed'):
                        p3['real_success_by_module'][em]['success'] += 1

                # Gold verification stats
                gv = phase3.get('gold_verification')
                if gv and isinstance(gv, dict) and 'tests_passed' in gv:
                    p3['gold_verification_count'] += 1
                    cat = gv.get('failure_category', 'unknown_error')
                    p3['gold_failure_categories'][cat] = p3['gold_failure_categories'].get(cat, 0) + 1

                    if gv.get('tests_passed'):
                        p3['gold_pass_count'] += 1

                    if phase3.get('env_compatible'):
                        p3['env_compatible_count'] += 1

                    if phase3.get('fair_comparison_eligible'):
                        p3['fair_comparison_count'] += 1
                        p3['_fair_total'] += 1
                        if phase3.get('final_success'):
                            p3['_fair_simulated_successes'] += 1

                        # Three-tier comparison
                        tier = phase3.get('corrective_vs_gold')
                        if tier:
                            p3['three_tier']['total'] += 1
                            if tier in p3['three_tier']:
                                p3['three_tier'][tier] += 1

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

        # Compute Phase 3 final stats
        p3 = stats['phase3']
        if p3['total_debugged'] > 0:
            p3['simulated_success_rate'] = round(p3['simulated_success_count'] / p3['total_debugged'] * 100, 1)
            p3['avg_iterations_total'] = round(sum(p3['iteration_counts']) / len(p3['iteration_counts']), 2)
            if p3['iterations_to_success']:
                p3['avg_iterations_to_success'] = round(sum(p3['iterations_to_success']) / len(p3['iterations_to_success']), 2)
            if p3['all_specificity']:
                p3['avg_specificity'] = round(sum(p3['all_specificity']) / len(p3['all_specificity']), 3)
            if p3['all_actionability']:
                p3['avg_actionability'] = round(sum(p3['all_actionability']) / len(p3['all_actionability']), 3)
            p3['convergence_rate'] = round(p3['convergence_count'] / p3['total_debugged'] * 100, 1)
        # Real verification final computation
        if p3['real_verification_count'] > 0:
            p3['real_success_rate'] = round(p3['real_success_count'] / p3['real_verification_count'] * 100, 1)
        if p3['simulated_vs_real_total'] > 0:
            p3['prediction_accuracy'] = round(p3['simulated_vs_real_match_count'] / p3['simulated_vs_real_total'] * 100, 1)
        if p3['all_verification_durations']:
            p3['avg_verification_duration'] = round(sum(p3['all_verification_durations']) / len(p3['all_verification_durations']), 2)

        # Gold verification final computation
        if p3['gold_verification_count'] > 0:
            p3['gold_pass_rate'] = round(p3['gold_pass_count'] / p3['gold_verification_count'] * 100, 1)
            p3['env_compatible_rate'] = round(p3['env_compatible_count'] / p3['gold_verification_count'] * 100, 1)

        # Three-tier final computation
        tt = p3['three_tier']
        if tt['total'] > 0:
            corrective_pass = tt['both_pass'] + tt.get('corrective_only', 0)
            tt['corrective_success_rate_fair'] = round(corrective_pass / tt['total'] * 100, 1)
        if p3['_fair_total'] > 0:
            tt['simulated_success_rate_fair'] = round(p3['_fair_simulated_successes'] / p3['_fair_total'] * 100, 1)
            fair_real = tt.get('corrective_success_rate_fair', 0.0)
            tt['overconfidence_gap_fair'] = round(tt['simulated_success_rate_fair'] - fair_real, 1)

        # Clean up temp lists from serialization
        del p3['iteration_counts']
        del p3['iterations_to_success']
        del p3['all_specificity']
        del p3['all_actionability']
        del p3['all_verification_durations']
        del p3['_fair_simulated_successes']
        del p3['_fair_total']

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

        # Phase 3 report
        p3 = stats.get('phase3', {})
        if p3.get('total_debugged', 0) > 0:
            print(f"\n## Phase 3: Simulated Iterative Debugging")
            print(f"  Trajectories debugged: {p3['total_debugged']}")
            print(f"  Simulated success rate: {p3['simulated_success_rate']:.1f}%")
            print(f"  Avg iterations (total): {p3['avg_iterations_total']:.1f}")
            print(f"  Avg iterations to success: {p3['avg_iterations_to_success']:.1f}")
            print(f"  Feedback quality: high={p3['quality_distribution'].get('high',0)}, "
                  f"medium={p3['quality_distribution'].get('medium',0)}, "
                  f"low={p3['quality_distribution'].get('low',0)}")
            print(f"  Avg specificity: {p3['avg_specificity']:.3f}")
            print(f"  Avg actionability: {p3['avg_actionability']:.3f}")
            print(f"  Convergence rate: {p3['convergence_rate']:.1f}%")
            if p3.get('success_by_agreement'):
                print(f"  Success by agreement type:")
                for ag_type, counts in p3['success_by_agreement'].items():
                    rate = counts['success'] / counts['total'] * 100 if counts['total'] > 0 else 0
                    print(f"    {ag_type}: {counts['success']}/{counts['total']} ({rate:.1f}%)")

            # Real Docker verification report
            if p3.get('real_verification_count', 0) > 0:
                print(f"\n## Phase 3: Real Docker Verification")
                print(f"  Patches verified: {p3['real_verification_count']}")
                print(f"  Patches applied: {p3['real_patch_applied_count']}")
                print(f"  Tests run: {p3['real_tests_run_count']}")
                print(f"  Tests PASSED (real): {p3['real_success_count']} ({p3['real_success_rate']:.1f}%)")
                print(f"  Simulated success rate: {p3['simulated_success_rate']:.1f}%")
                print(f"  Prediction accuracy: {p3['prediction_accuracy']:.1f}%")
                gap = p3['simulated_success_rate'] - p3['real_success_rate']
                print(f"  Overconfidence gap: {gap:.1f}% (simulated - real)")
                print(f"  Avg verification time: {p3['avg_verification_duration']:.1f}s")
                if p3.get('real_success_by_agreement'):
                    print(f"  Real success by agreement type:")
                    for ag_type, counts in p3['real_success_by_agreement'].items():
                        rate = counts['success'] / counts['total'] * 100 if counts['total'] > 0 else 0
                        print(f"    {ag_type}: {counts['success']}/{counts['total']} ({rate:.1f}%)")

            # Gold patch verification report
            if p3.get('gold_verification_count', 0) > 0:
                print(f"\n## Phase 3: Gold Patch Verification (Ceiling Baseline)")
                print(f"  Gold patches verified: {p3['gold_verification_count']}")
                print(f"  Gold patches PASSED: {p3['gold_pass_count']} ({p3['gold_pass_rate']:.1f}%)")
                print(f"  Environment compatible: {p3['env_compatible_count']} ({p3['env_compatible_rate']:.1f}%)")
                print(f"  Fair comparison eligible: {p3['fair_comparison_count']}")

                if p3.get('gold_failure_categories'):
                    print(f"  Gold failure breakdown:")
                    for cat, count in sorted(p3['gold_failure_categories'].items(), key=lambda x: -x[1]):
                        pct = count / p3['gold_verification_count'] * 100
                        print(f"    {cat}: {count} ({pct:.1f}%)")

                tt = p3.get('three_tier', {})
                if tt.get('total', 0) > 0:
                    print(f"\n## Three-Tier Comparison (N={tt['total']} fair instances)")
                    print(f"  Gold + Corrective both pass: {tt['both_pass']}")
                    print(f"  Gold only (corrective failed): {tt['gold_only']}")
                    print(f"  Corrective only (anomaly): {tt['corrective_only']}")
                    print(f"  Neither passed: {tt['neither']}")
                    print(f"  Corrective success rate (fair): {tt['corrective_success_rate_fair']:.1f}%")
                    print(f"  Simulated success rate (fair): {tt['simulated_success_rate_fair']:.1f}%")
                    print(f"  OVERCONFIDENCE GAP (fair): {tt['overconfidence_gap_fair']:.1f}%")

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
