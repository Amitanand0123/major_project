"""
Code Domain Phase 2 Detector - Critical Error Identification
Identifies THE critical error that caused failure in code repair tasks
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CriticalError:
    """The critical error that caused failure"""
    step_number: int
    module: str
    error_type: str
    confidence: float
    explanation: str
    impact_analysis: str
    counterfactual_reasoning: str
    propagation_chain: List[str]


class CodePhase2Detector:
    """
    Phase 2: Critical Error Identification for Code Domain
    Uses counterfactual reasoning to identify THE error that caused failure
    """

    def __init__(self, llm):
        """
        Initialize Phase 2 detector

        Args:
            llm: Language model for critical error identification
        """
        self.llm = llm

    async def identify_critical_error(self, phase1_results: Dict[str, Any],
                                     original_trajectory: Dict[str, Any],
                                     retry_count: int = 0) -> Optional[CriticalError]:
        """
        Identify the critical error that led to failure

        Args:
            phase1_results: Results from Phase 1 analysis
            original_trajectory: Original trajectory data
            retry_count: Number of retries (for Step 1 validation)

        Returns:
            CriticalError or None
        """
        step_analyses = phase1_results.get('step_analyses', [])
        task_description = phase1_results.get('task_description', '')
        final_result = original_trajectory.get('final_result', {})

        # Build comprehensive context
        error_summary = self._build_error_summary(step_analyses)
        trajectory_summary = self._build_trajectory_summary(original_trajectory)

        prompt = f"""You are an expert at debugging code repair agents using counterfactual reasoning.

# Task
{task_description}

# Final Result
Success: {final_result.get('success', False)}
Tests Passed: {final_result.get('tests_passed', 0)} / {final_result.get('tests_total', 0)}
Error: {final_result.get('error_message', 'Unknown')}

# All Detected Errors (Phase 1)
{error_summary}

# Full Trajectory
{trajectory_summary}

# Your Task: Identify THE Critical Error

Use counterfactual reasoning: "If this error had been avoided, would the agent have succeeded?"

Consider:
1. **Error Propagation**: Does this error cause subsequent errors?
2. **Point of No Return**: After this error, was recovery impossible?
3. **Direct Impact**: Does this error directly prevent task completion?

Analyze each error and identify THE SINGLE MOST CRITICAL ERROR that caused failure.

CRITICAL RULES:
- Step 1 CANNOT have memory or reflection errors (no prior context exists)
- If you select Step 1, it MUST be planning, action, or system error
- Focus on the error with the HIGHEST IMPACT, not necessarily the earliest one
- An error at step 1 is only critical if it directly causes downstream failures; do not select step 1 errors if later errors are more impactful
- Prefer high-confidence errors (confidence >= 0.8) over low-confidence ones

Respond in this EXACT format:
STEP_NUMBER: <number>
MODULE: <MUST be one of: memory, reflection, planning, action, system>
ERROR_TYPE: <MUST be one of the 17 error types below>
CONFIDENCE: <0.0 to 1.0>
EXPLANATION: <why this is the critical error>
COUNTERFACTUAL: <what would have happened if this error was avoided>
PROPAGATION: <how this error led to subsequent failures, separated by ' -> '>

IMPORTANT CONSTRAINTS:
1. MODULE must be one of the 5 cognitive modules (memory, reflection, planning, action, system), NOT a file name or package name!
2. ERROR_TYPE must be one of these valid types from the AgentErrorTaxonomy:

   Memory errors: dependency_omission, file_location_forgetting, over_simplification, hallucination, retrieval_failure
   Reflection errors: progress_misjudge, outcome_misinterpretation, error_dismissal, repetition_blindness
   Planning errors: constraint_ignorance, impossible_action, inefficient_plan, redundant_plan, api_hallucination, scope_violation, test_interpretation_error
   Action errors: format_error, parameter_error, misalignment, syntax_error, indentation_error, logic_error, wrong_file_edit
   System errors: step_limit_exhaustion, tool_execution_error, environment_error, compilation_timeout, test_timeout

   DO NOT invent new error types like "user_input_error", "FileNotFoundError", "missing_test_case", etc.

Example PROPAGATION: "api_hallucination in step 3 -> AttributeError in step 4 -> wrong fix in step 5 -> test failure"

Response:"""

        # Call LLM
        if hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                response = response.content
        else:
            response = self.llm(prompt)

        response_text = str(response).strip()

        # Parse response
        critical_error = self._parse_critical_error_response(response_text)

        # Fallback when LLM response can't be parsed (common with high-error trajectories)
        if critical_error is None:
            print("⚠️ LLM response could not be parsed. Selecting highest-confidence Phase 1 error as fallback.")
            fallback = self._select_fallback_critical_error(step_analyses)
            if fallback:
                print(f"   Fallback selected: {fallback.error_type} at step {fallback.step_number} (conf: {fallback.confidence:.2f})")
                return fallback
            else:
                print("   No fallback available — no Phase 1 errors found.")
                return None

        # Validate Step 1 errors
        if critical_error and critical_error.step_number == 1:
            reject = False
            reason = ""

            if critical_error.module in ['memory', 'reflection']:
                reject = True
                reason = f"Step 1 cannot have {critical_error.module} error"
            elif critical_error.confidence < 0.7:
                reject = True
                reason = f"Step 1 error has low confidence ({critical_error.confidence:.2f})"
            elif critical_error.error_type == 'scope_violation' and critical_error.confidence < 0.8:
                reject = True
                reason = f"Step 1 scope_violation has low confidence ({critical_error.confidence:.2f}), likely boilerplate false positive"

            if reject:
                print(f"⚠️ WARNING: {reason}!")
                # Try to find a better error from later steps first
                fallback = self._select_fallback_critical_error(step_analyses, skip_step1=True)
                if fallback:
                    print(f"   Using fallback: {fallback.error_type} at step {fallback.step_number}")
                    return fallback
                elif retry_count < 3:
                    print(f"   No later-step fallback. Retrying LLM... (attempt {retry_count + 1}/3)")
                    return await self.identify_critical_error(
                        phase1_results, original_trajectory, retry_count + 1
                    )
                else:
                    print("   Max retries reached. Selecting next best error.")
                    return self._select_fallback_critical_error(step_analyses)

        return critical_error

    def _build_error_summary(self, step_analyses: List[Dict], max_chars: int = 3500) -> str:
        """Build summary of all errors from Phase 1, truncated to fit prompt limits.

        For trajectories with many errors, we prioritize high-confidence errors
        and include a compact summary of the rest.
        """
        # Collect all errors with metadata
        all_errors = []
        for analysis in step_analyses:
            step_num = analysis['step_number']
            for module in ['memory', 'reflection', 'planning', 'action', 'system']:
                error_key = f'{module}_error'
                if analysis.get(error_key):
                    error = analysis[error_key]
                    all_errors.append({
                        'step': step_num,
                        'module': module,
                        'type': error['error_type'],
                        'confidence': error.get('confidence', 0.5),
                        'explanation': error.get('explanation', ''),
                    })

        if not all_errors:
            return "No errors detected."

        # If few errors, include all with full detail
        if len(all_errors) <= 20:
            lines = []
            for analysis in step_analyses:
                step_num = analysis['step_number']
                errors = []
                for module in ['memory', 'reflection', 'planning', 'action', 'system']:
                    error_key = f'{module}_error'
                    if analysis.get(error_key):
                        error = analysis[error_key]
                        errors.append(f"  - {module.upper()}: {error['error_type']} (conf: {error['confidence']:.2f})")
                        errors.append(f"    Explanation: {error['explanation']}")
                if errors:
                    lines.append(f"\n## Step {step_num}")
                    lines.extend(errors)
            return "\n".join(lines)

        # Many errors: show top errors by confidence, ensuring all modules represented
        sorted_errors = sorted(all_errors, key=lambda e: e['confidence'], reverse=True)
        # Ensure at least top 2 errors per module are included
        top_errors = []
        module_counts = {}
        for e in sorted_errors:
            m = e['module']
            module_counts.setdefault(m, 0)
            if module_counts[m] < 2 or len(top_errors) < 30:
                top_errors.append(e)
                module_counts[m] += 1
            if len(top_errors) >= 30:
                break

        lines = [f"(Total: {len(all_errors)} errors across {len(step_analyses)} steps. Showing top {len(top_errors)} by confidence.)\n"]
        for e in top_errors:
            lines.append(f"Step {e['step']} | {e['module'].upper()} | {e['type']} | conf={e['confidence']:.2f}")
            expl = e['explanation'][:120] + "..." if len(e['explanation']) > 120 else e['explanation']
            lines.append(f"  {expl}")

        # Add type frequency summary
        from collections import Counter
        type_counts = Counter(e['type'] for e in all_errors)
        lines.append(f"\nError type frequency: {dict(type_counts.most_common(8))}")

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        return result

    def _build_trajectory_summary(self, trajectory: Dict) -> str:
        """Build summary of trajectory for context.

        Shows first 5 + last 5 steps to capture both early context and
        late-trajectory errors that often matter most.
        """
        steps = trajectory.get('steps', [])
        lines = []

        if len(steps) <= 12:
            # Short trajectory: show all steps
            show_steps = steps
        else:
            # Long trajectory: first 5 + last 5
            show_steps = steps[:5]
            lines_before_gap = len(show_steps)

        for i, step in enumerate(steps[:5] if len(steps) > 12 else steps):
            step_num = step.get('step_number', '?')
            action = step.get('modules', {}).get('action', 'N/A')[:100]
            obs = step.get('observation', 'N/A')[:150]

            lines.append(f"\nStep {step_num}:")
            lines.append(f"  Action: {action}")
            lines.append(f"  Result: {obs}...")

        if len(steps) > 12:
            lines.append(f"\n... ({len(steps) - 10} middle steps omitted) ...")

            for step in steps[-5:]:
                step_num = step.get('step_number', '?')
                action = step.get('modules', {}).get('action', 'N/A')[:100]
                obs = step.get('observation', 'N/A')[:150]

                lines.append(f"\nStep {step_num}:")
                lines.append(f"  Action: {action}")
                lines.append(f"  Result: {obs}...")

        return "\n".join(lines)

    def _parse_critical_error_response(self, response: str) -> Optional[CriticalError]:
        """Parse LLM response into CriticalError"""
        lines = response.split('\n')

        step_number = None
        module = None
        error_type = None
        confidence = 0.0
        explanation = ""
        counterfactual = ""
        propagation = ""

        for line in lines:
            line = line.strip()
            if line.startswith('STEP_NUMBER:'):
                try:
                    step_number = int(line.replace('STEP_NUMBER:', '').strip())
                except (ValueError, TypeError):
                    pass
            elif line.startswith('MODULE:'):
                module = line.replace('MODULE:', '').strip().lower()
            elif line.startswith('ERROR_TYPE:'):
                error_type = line.replace('ERROR_TYPE:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except (ValueError, TypeError):
                    confidence = 0.5
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('COUNTERFACTUAL:'):
                counterfactual = line.replace('COUNTERFACTUAL:', '').strip()
            elif line.startswith('PROPAGATION:'):
                propagation = line.replace('PROPAGATION:', '').strip()

        # Validate module is one of the 5 cognitive modules
        valid_modules = {'memory', 'reflection', 'planning', 'action', 'system'}

        if module and module not in valid_modules:
            print(f"⚠️ WARNING: LLM returned invalid module '{module}'. Inferring correct module from error type...")
            module = self._infer_module_from_error_type(error_type)
            if module:
                print(f"   Corrected to: '{module}'")
            else:
                print(f"   Could not infer module. Returning None.")
                return None

        # Validate error_type is one of the 17 taxonomy types
        valid_error_types = {
            # Memory module errors
            'dependency_omission', 'file_location_forgetting', 'over_simplification',
            'hallucination', 'retrieval_failure',
            # Reflection module errors
            'progress_misjudge', 'outcome_misinterpretation', 'error_dismissal',
            'repetition_blindness',
            # Planning module errors
            'constraint_ignorance', 'impossible_action', 'inefficient_plan',
            'redundant_plan', 'api_hallucination', 'scope_violation',
            'test_interpretation_error',
            # Action module errors
            'format_error', 'parameter_error', 'misalignment', 'syntax_error',
            'indentation_error', 'logic_error', 'wrong_file_edit',
            # System module errors
            'step_limit_exhaustion', 'tool_execution_error', 'environment_error',
            'compilation_timeout', 'test_timeout'
        }

        if error_type and error_type not in valid_error_types:
            print(f"⚠️ WARNING: LLM returned invalid error_type '{error_type}'. Attempting to correct...")
            corrected_error_type = self._correct_invalid_error_type(error_type, module)
            if corrected_error_type:
                print(f"   Corrected '{error_type}' -> '{corrected_error_type}'")
                error_type = corrected_error_type
            else:
                print(f"   Could not correct error_type '{error_type}'. Returning None.")
                return None

        if step_number and module and error_type:
            # Parse propagation chain
            propagation_chain = [p.strip() for p in propagation.split('->')]

            return CriticalError(
                step_number=step_number,
                module=module,
                error_type=error_type,
                confidence=confidence,
                explanation=explanation,
                impact_analysis=explanation,  # Use explanation as impact
                counterfactual_reasoning=counterfactual,
                propagation_chain=propagation_chain
            )

        return None

    def _infer_module_from_error_type(self, error_type: str) -> Optional[str]:
        """Infer cognitive module from error type"""
        # Map error types to their modules
        error_to_module = {
            # Memory module errors
            'dependency_omission': 'memory',
            'file_location_forgetting': 'memory',
            'over_simplification': 'memory',
            'hallucination': 'memory',
            'retrieval_failure': 'memory',

            # Reflection module errors
            'progress_misjudge': 'reflection',
            'outcome_misinterpretation': 'reflection',
            'error_dismissal': 'reflection',
            'repetition_blindness': 'reflection',

            # Planning module errors
            'constraint_ignorance': 'planning',
            'impossible_action': 'planning',
            'inefficient_plan': 'planning',
            'redundant_plan': 'planning',
            'api_hallucination': 'planning',
            'scope_violation': 'planning',
            'test_interpretation_error': 'planning',

            # Action module errors
            'format_error': 'action',
            'parameter_error': 'action',
            'misalignment': 'action',
            'syntax_error': 'action',
            'indentation_error': 'action',
            'logic_error': 'action',
            'wrong_file_edit': 'action',

            # System module errors
            'step_limit_exhaustion': 'system',
            'tool_execution_error': 'system',
            'environment_error': 'system',
            'compilation_timeout': 'system',
            'test_timeout': 'system'
        }

        return error_to_module.get(error_type)

    def _correct_invalid_error_type(self, invalid_error_type: str, module: str) -> Optional[str]:
        """Correct common invalid error types to valid taxonomy types"""
        # Common mapping of invalid error types to valid ones
        invalid_to_valid = {
            # System/environment related
            'user_input_error': 'environment_error',
            'filenotfounderror': 'environment_error',
            'file_not_found_error': 'environment_error',
            'file_not_found': 'environment_error',
            'path_error': 'environment_error',
            'import_error': 'dependency_omission',
            'module_not_found': 'dependency_omission',

            # Planning related
            'missing_test_case': 'test_interpretation_error',
            'missing_action': 'inefficient_plan',
            'incomplete_plan': 'inefficient_plan',
            'wrong_approach': 'inefficient_plan',

            # Syntax/action related
            'parse_error': 'syntax_error',
            'compilation_error': 'syntax_error',
            'type_error': 'logic_error',

            # Memory related
            'forgotten_context': 'file_location_forgetting',
            'missing_import': 'dependency_omission',
            'missing_dependency': 'dependency_omission',

            # Reflection related
            'missed_error': 'error_dismissal',
            'ignored_error': 'error_dismissal'
        }

        # Try exact match (case-insensitive)
        corrected = invalid_to_valid.get(invalid_error_type.lower())

        if corrected:
            return corrected

        # If no mapping found, try to infer from module
        # Use the most common error type for each module
        module_defaults = {
            'memory': 'dependency_omission',
            'reflection': 'error_dismissal',
            'planning': 'inefficient_plan',
            'action': 'syntax_error',
            'system': 'environment_error'
        }

        if module in module_defaults:
            print(f"   No mapping for '{invalid_error_type}', using module default: {module_defaults[module]}")
            return module_defaults[module]

        return None

    def _select_fallback_critical_error(self, step_analyses: List[Dict],
                                        skip_step1: bool = False) -> Optional[CriticalError]:
        """Select fallback critical error if LLM fails validation.

        Args:
            step_analyses: Phase 1 step analysis data
            skip_step1: If True, skip all step 1 errors entirely (used when
                        step 1 was already rejected as likely false positive)
        """
        best_error = None
        best_confidence = 0.0

        for analysis in step_analyses:
            step_num = analysis['step_number']

            # Skip step 1 entirely if requested
            if skip_step1 and step_num == 1:
                continue

            for module in ['planning', 'action', 'system', 'memory', 'reflection']:
                error_key = f'{module}_error'
                if analysis.get(error_key):
                    error = analysis[error_key]

                    # Skip invalid Step 1 errors
                    if step_num == 1 and module in ['memory', 'reflection']:
                        continue

                    if error['confidence'] > best_confidence:
                        best_confidence = error['confidence']
                        best_error = CriticalError(
                            step_number=step_num,
                            module=module,
                            error_type=error['error_type'],
                            confidence=error['confidence'],
                            explanation=error.get('explanation', ''),
                            impact_analysis="Fallback selection: highest confidence valid error",
                            counterfactual_reasoning="Unable to determine via LLM",
                            propagation_chain=[]
                        )

        return best_error

    async def analyze_with_phase2(self, phase1_results: Dict[str, Any],
                                  original_trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete Phase 2 analysis

        Args:
            phase1_results: Phase 1 results
            original_trajectory: Original trajectory

        Returns:
            Phase 2 analysis with critical error
        """
        instance_id = phase1_results.get('instance_id', 'unknown')

        print(f"\n=== Phase 2: Identifying Critical Error for {instance_id} ===")

        critical_error = await self.identify_critical_error(phase1_results, original_trajectory)

        if not critical_error:
            print("⚠️ No critical error identified")
            return {
                'instance_id': instance_id,
                'critical_error': None,
                'phase1_summary': phase1_results.get('summary', {}),
                'timestamp': datetime.now().isoformat()
            }

        print(f"\n✓ Critical Error Identified:")
        print(f"  Step: {critical_error.step_number}")
        print(f"  Module: {critical_error.module}")
        print(f"  Type: {critical_error.error_type}")
        print(f"  Confidence: {critical_error.confidence:.2f}")

        return {
            'instance_id': instance_id,
            'critical_error': asdict(critical_error),
            'phase1_summary': phase1_results.get('summary', {}),
            'timestamp': datetime.now().isoformat()
        }


async def main_demo():
    """Demo of Phase 2 detector"""
    from swebench_integration import create_sample_trajectory

    # Create mock LLM
    class MockLLM:
        def invoke(self, prompt):
            return """STEP_NUMBER: 4
MODULE: planning
ERROR_TYPE: api_hallucination
CONFIDENCE: 0.95
EXPLANATION: Agent planned to use token.is_expired attribute that doesn't exist. This is THE critical error because it caused AttributeError which led to wrong fix.
COUNTERFACTUAL: If agent had first read tokens.py to check API, it would have used correct method is_expired_token() and fix would succeed.
PROPAGATION: api_hallucination in step 4 -> AttributeError in observation -> wrong assumption -> incorrect fix attempt -> continued errors"""

    llm = MockLLM()

    # Create mock Phase 1 results
    phase1_results = {
        'instance_id': 'test-001',
        'task_description': 'Fix auth bug',
        'step_analyses': [
            {
                'step_number': 4,
                'planning_error': {
                    'error_type': 'api_hallucination',
                    'confidence': 0.95,
                    'explanation': 'Used non-existent attribute'
                }
            }
        ],
        'summary': {'total_errors': 3}
    }

    trajectory = create_sample_trajectory()

    detector = CodePhase2Detector(llm)
    result = await detector.analyze_with_phase2(phase1_results, trajectory)

    print("\n=== Phase 2 Complete ===")
    if result['critical_error']:
        print(f"Critical Error: {result['critical_error']['error_type']} at Step {result['critical_error']['step_number']}")


if __name__ == "__main__":
    asyncio.run(main_demo())
