"""
Code Domain Phase 3 Debugger - Simulated Iterative Debugging with Targeted Feedback
Generates corrective feedback from Phase 2 critical errors and simulates
what the agent should have done differently, following Zhu et al.'s Stage 3.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from detector.code_error_taxonomy import CodeErrorTaxonomy


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class CorrectiveFeedback:
    """Targeted feedback generated from the critical error diagnosis."""
    iteration: int
    target_step: int
    target_module: str
    error_type: str
    feedback_text: str
    grounding_source: str       # 'both_error' | 'llm_only' | 'regex_only' | 'unknown'
    addresses_propagation: bool

@dataclass
class SimulatedCorrection:
    """LLM-simulated corrected action at the critical step."""
    iteration: int
    original_action: str
    corrected_action: str
    corrected_planning: str
    corrected_memory: str
    reasoning: str

@dataclass
class DownstreamPrediction:
    """LLM prediction of whether the correction would lead to success."""
    iteration: int
    predicted_success: bool
    predicted_remaining_steps: int
    confidence: float
    reasoning: str

@dataclass
class CorrectionEvaluation:
    """LLM-as-judge evaluation of the correction quality."""
    iteration: int
    addresses_root_cause: bool
    breaks_propagation_chain: bool
    introduces_new_errors: bool
    specificity_score: float
    actionability_score: float
    overall_quality: str        # 'high' | 'medium' | 'low'
    reasoning: str
    should_iterate: bool

@dataclass
class DebugIteration:
    """One complete iteration of the debug loop."""
    iteration: int
    feedback: CorrectiveFeedback
    correction: SimulatedCorrection
    downstream: DownstreamPrediction
    evaluation: CorrectionEvaluation

@dataclass
class Phase3Result:
    """Complete Phase 3 output for one trajectory."""
    instance_id: str
    critical_error_step: int
    critical_error_module: str
    critical_error_type: str
    dual_channel_agreement: str
    total_iterations: int
    iterations: List[Dict]       # List of asdict(DebugIteration)
    final_success: bool
    successful_iteration: Optional[int]
    final_feedback_quality: str
    avg_specificity: float
    avg_actionability: float
    convergence: bool
    # Real verification fields
    real_verification: Optional[Dict] = None       # VerificationResult as dict
    simulated_vs_real_match: Optional[bool] = None # simulated prediction == real outcome?
    generated_patch: Optional[str] = None          # the unified diff patch sent to verifier
    # Gold patch verification fields
    gold_verification: Optional[Dict] = None       # gold VerificationResult as dict
    gold_patch_passed: Optional[bool] = None       # did gold patch pass tests?
    gold_failure_category: Optional[str] = None    # why gold patch failed (if it did)
    env_compatible: Optional[bool] = None          # could clone+install+run tests?
    corrective_vs_gold: Optional[str] = None       # "both_pass"|"gold_only"|"neither"|"corrective_only"
    fair_comparison_eligible: Optional[bool] = None # gold passed → fair comparison possible
    timestamp: str = ""


# ============================================================
# Phase 3 Debugger
# ============================================================

class CodePhase3Debugger:
    """
    Phase 3: Simulated Iterative Debugging with Targeted Feedback

    Since we cannot re-execute SWE-agent in a real environment, we simulate
    the debugging loop via LLM:
      1. Generate corrective feedback from critical error diagnosis
      2. Simulate what the agent SHOULD have done at the critical step
      3. Predict whether the correction would lead to task success
      4. Evaluate correction quality (LLM-as-judge)
      5. If insufficient, refine feedback and repeat (up to max_iterations)
    """

    MAX_ITERATIONS = 3

    def __init__(self, llm, taxonomy=None, verifier=None):
        """
        Initialize Phase 3 debugger

        Args:
            llm: Language model with .invoke(prompt) -> str interface
            taxonomy: CodeErrorTaxonomy instance (optional)
            verifier: DockerPatchVerifier instance (optional, for real verification)
        """
        self.llm = llm
        self.taxonomy = taxonomy or CodeErrorTaxonomy()
        self.verifier = verifier  # None = simulated only

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with standard invocation pattern matching Phase 1/2."""
        if hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                response = response.content
        else:
            response = self.llm(prompt)
        return str(response).strip()

    async def run_phase3(self, phase2_results: Dict, phase1_results: Dict,
                         original_trajectory: Dict) -> Optional[Phase3Result]:
        """
        Run the full Phase 3 debugging simulation.

        Args:
            phase2_results: Output from Phase 2 (contains 'critical_error' dict)
            phase1_results: Output from Phase 1 dual-channel analysis
            original_trajectory: The raw trajectory dict

        Returns:
            Phase3Result or None if no critical error exists
        """
        critical_error = phase2_results.get('critical_error')
        if not critical_error:
            return None

        instance_id = phase2_results.get('instance_id', 'unknown')
        task_description = original_trajectory.get('task_description', '')
        steps = original_trajectory.get('steps', [])

        # Get dual-channel agreement at the critical step
        agreement = self._get_dual_channel_agreement(
            phase1_results, critical_error['step_number'], critical_error['module']
        )

        print(f"\n=== Phase 3: Simulated Iterative Debugging for {instance_id} ===")
        print(f"  Critical error: {critical_error['error_type']} at Step {critical_error['step_number']} ({critical_error['module']})")
        print(f"  Dual-channel agreement: {agreement}")

        iterations = []
        previous_eval = None

        for k in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n  [Iteration {k}/{self.MAX_ITERATIONS}]")

            # Step 1: Generate feedback
            feedback = self._generate_feedback(
                critical_error, steps, task_description, k, agreement, previous_eval
            )
            print(f"    Feedback: {feedback.feedback_text[:80]}...")

            # Step 2: Simulate correction
            correction = self._simulate_correction(
                feedback, critical_error, steps, task_description, k
            )
            print(f"    Corrected action: {correction.corrected_action[:80]}...")

            # Step 3: Predict downstream
            downstream = self._predict_downstream(
                correction, critical_error, steps, task_description, k
            )
            print(f"    Predicted success: {downstream.predicted_success} (conf: {downstream.confidence:.2f})")

            # Step 4: Evaluate
            evaluation = self._evaluate_correction(
                feedback, correction, downstream, critical_error, k
            )
            print(f"    Quality: {evaluation.overall_quality} (spec: {evaluation.specificity_score:.1f}, act: {evaluation.actionability_score:.1f})")

            debug_iter = DebugIteration(
                iteration=k,
                feedback=feedback,
                correction=correction,
                downstream=downstream,
                evaluation=evaluation
            )
            iterations.append(debug_iter)

            # Early exit conditions
            if downstream.predicted_success and evaluation.overall_quality == 'high':
                print(f"    Success predicted with high quality - stopping.")
                break
            if not evaluation.should_iterate:
                print(f"    No further iteration needed - stopping.")
                break

            previous_eval = evaluation

        # Build final result
        result = self._build_result(instance_id, critical_error, agreement, iterations)

        # Real verification (if verifier is available)
        if self.verifier and self.verifier.is_available():
            from dataclasses import asdict as _asdict

            # Step A: Gold patch verification (run first — warms repo cache)
            print(f"\n  [Gold Verification] Testing gold (correct) patch...")
            try:
                gold_result = self.verifier.verify_gold_patch_cached(instance_id)
                result.gold_verification = _asdict(gold_result)
                result.gold_patch_passed = gold_result.tests_passed
                result.gold_failure_category = gold_result.failure_category
                result.env_compatible = gold_result.tests_run
                result.fair_comparison_eligible = gold_result.tests_passed

                print(f"    Gold applied: {gold_result.patch_applied}")
                print(f"    Gold tests passed: {gold_result.tests_passed}")
                print(f"    Failure category: {gold_result.failure_category}")
                print(f"    Env compatible: {result.env_compatible}")
                print(f"    Duration: {gold_result.duration_seconds:.1f}s")
            except Exception as e:
                print(f"    Gold verification error: {e}")
                result.gold_verification = {"error": str(e)}
                result.gold_patch_passed = False
                result.env_compatible = False
                result.fair_comparison_eligible = False

            # Step B: Corrective patch verification
            best_patch = self._extract_best_patch(iterations)
            if best_patch:
                print(f"\n  [Corrective Verification] Verifying valid diff patch ({len(best_patch)} chars)...")
                try:
                    verification = self.verifier.verify_patch(instance_id, best_patch)
                    result.real_verification = _asdict(verification)
                    result.generated_patch = best_patch[:2000]  # truncate for storage
                    result.simulated_vs_real_match = (
                        result.final_success == verification.tests_passed
                    )
                    print(f"    Patch applied: {verification.patch_applied}")
                    print(f"    Tests passed: {verification.tests_passed}")
                    print(f"    Simulated predicted: {result.final_success} | Real: {verification.tests_passed} | Match: {result.simulated_vs_real_match}")
                    print(f"    Duration: {verification.duration_seconds:.1f}s")

                    # Step C: Three-tier comparison
                    if result.gold_patch_passed is not None:
                        gp = result.gold_patch_passed
                        cp = verification.tests_passed
                        if gp and cp:
                            result.corrective_vs_gold = "both_pass"
                        elif gp and not cp:
                            result.corrective_vs_gold = "gold_only"
                        elif not gp and cp:
                            result.corrective_vs_gold = "corrective_only"
                        else:
                            result.corrective_vs_gold = "neither"
                        print(f"    Three-tier: {result.corrective_vs_gold}")
                except Exception as e:
                    print(f"    Corrective verification error: {e}")
                    result.real_verification = {"error": str(e)}
            else:
                print(f"\n  [Corrective Verification] No valid unified diff extracted — LLM likely output natural language or pseudo-code")
                result.real_verification = {"error": "no_valid_diff_extracted", "details": "LLM output did not contain a valid unified diff"}
                result.generated_patch = None
        else:
            if self.verifier:
                print(f"\n  [Verification] Skipped (verifier not available)")

        print(f"\n  Phase 3 complete: {result.total_iterations} iterations, "
              f"success={result.final_success}, quality={result.final_feedback_quality}")

        return asdict(result)

    def _get_dual_channel_agreement(self, phase1_results: Dict,
                                     step_number: int, module: str) -> str:
        """Look up dual-channel agreement label for the critical error's step+module."""
        step_analyses = phase1_results.get('step_analyses', [])
        for step in step_analyses:
            if step.get('step_number') == step_number:
                agreement = step.get('agreement', {})
                return agreement.get(module, 'unknown')
        return 'unknown'

    def _get_step_context(self, steps: List[Dict], step_number: int) -> Dict:
        """Extract context for a specific step."""
        for step in steps:
            if step.get('step_number') == step_number:
                return step
        return {}

    def _get_previous_steps_summary(self, steps: List[Dict], up_to_step: int, max_steps: int = 3) -> str:
        """Build summary of steps preceding the critical step."""
        lines = []
        relevant = [s for s in steps if s.get('step_number', 0) < up_to_step]
        for step in relevant[-max_steps:]:
            sn = step.get('step_number', '?')
            action = step.get('modules', {}).get('action', 'N/A')[:100]
            obs = step.get('observation', 'N/A')[:100]
            lines.append(f"Step {sn}: Action={action} | Result={obs}")
        return "\n".join(lines) if lines else "No previous steps."

    # ============================================================
    # Step 1: Generate Feedback
    # ============================================================

    def _generate_feedback(self, critical_error: Dict, steps: List[Dict],
                           task_description: str, iteration: int,
                           agreement: str,
                           previous_eval: Optional[CorrectionEvaluation] = None) -> CorrectiveFeedback:
        """Generate targeted corrective feedback from critical error diagnosis."""
        step_ctx = self._get_step_context(steps, critical_error['step_number'])
        modules = step_ctx.get('modules', {})
        prev_summary = self._get_previous_steps_summary(steps, critical_error['step_number'])

        # Get taxonomy definition for this error type
        error_def_dict = self.taxonomy.get_error_definition(
            critical_error['module'], critical_error['error_type']
        )
        error_def = error_def_dict.get('description', '') if error_def_dict else ''

        refinement = ""
        if previous_eval and iteration > 1:
            refinement = f"""
# Previous Feedback Was Insufficient (Iteration {iteration - 1})
Previous quality: {previous_eval.overall_quality}
Specificity: {previous_eval.specificity_score}/1.0
Actionability: {previous_eval.actionability_score}/1.0
Critique: {previous_eval.reasoning}

Make the NEW feedback MORE SPECIFIC and MORE ACTIONABLE than before."""

        prompt = f"""You are an expert at debugging code repair agents. Generate targeted corrective feedback.

# Task
{task_description[:500]}

# Critical Error at Step {critical_error['step_number']}
Module: {critical_error['module']}
Error Type: {critical_error['error_type']}
Definition: {error_def}
Explanation: {critical_error.get('explanation', '')}
Counterfactual: {critical_error.get('counterfactual_reasoning', '')}
Propagation: {' -> '.join(critical_error.get('propagation_chain', []))}

# What the agent did at Step {critical_error['step_number']}
Action: {modules.get('action', 'N/A')[:300]}
Planning: {modules.get('planning', 'N/A')[:200]}
Observation: {step_ctx.get('observation', 'N/A')[:300]}

# Previous Steps
{prev_summary}
{refinement}

Generate TARGETED corrective feedback that:
1. Names the specific error ({critical_error['error_type']}) and why it occurred
2. Explains what the agent should have done differently
3. Provides forward-looking guidance to prevent downstream failures
4. Is specific enough for the agent to act on

Respond in this EXACT format:
FEEDBACK: <corrective feedback, 2-4 sentences, specific and actionable>
ADDRESSES_PROPAGATION: <YES or NO>

Response:"""

        response = self._call_llm(prompt)
        return self._parse_feedback_response(
            response, iteration, critical_error, agreement
        )

    def _parse_feedback_response(self, response: str, iteration: int,
                                  critical_error: Dict, agreement: str) -> CorrectiveFeedback:
        """Parse feedback generation response."""
        feedback_text = ""
        addresses_prop = False

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('FEEDBACK:'):
                feedback_text = line.replace('FEEDBACK:', '').strip()
            elif line.startswith('ADDRESSES_PROPAGATION:'):
                addresses_prop = 'YES' in line.upper()

        if not feedback_text:
            feedback_text = response[:300]

        return CorrectiveFeedback(
            iteration=iteration,
            target_step=critical_error['step_number'],
            target_module=critical_error['module'],
            error_type=critical_error['error_type'],
            feedback_text=feedback_text,
            grounding_source=agreement,
            addresses_propagation=addresses_prop
        )

    # ============================================================
    # Step 2: Simulate Correction
    # ============================================================

    def _simulate_correction(self, feedback: CorrectiveFeedback,
                              critical_error: Dict, steps: List[Dict],
                              task_description: str, iteration: int) -> SimulatedCorrection:
        """Simulate what the agent SHOULD have done given corrective feedback."""
        step_ctx = self._get_step_context(steps, critical_error['step_number'])
        modules = step_ctx.get('modules', {})
        prev_summary = self._get_previous_steps_summary(steps, critical_error['step_number'])

        # Get context from surrounding steps for better patch generation
        obs_before = step_ctx.get('observation', 'N/A')[:500]

        prompt = f"""You are simulating what a code repair agent SHOULD have done if given corrective feedback.

# Task
{task_description[:500]}

# Corrective Feedback (given to agent before Step {critical_error['step_number']})
{feedback.feedback_text}

# What the agent actually did at Step {critical_error['step_number']}
Memory: {modules.get('memory', 'N/A')[:200]}
Planning: {modules.get('planning', 'N/A')[:200]}
Action: {modules.get('action', 'N/A')[:300]}
Observation: {obs_before[:300]}

# Previous Steps
{prev_summary}

Simulate what the agent SHOULD have done at Step {critical_error['step_number']} with the feedback.
IMPORTANT: For the CORRECTED_ACTION, provide the actual corrected code edit as a unified diff patch.
Use the format: --- a/file.py +++ b/file.py @@ -line,count +line,count @@ context

Respond in this EXACT format:
CORRECTED_MEMORY: <what memory should recall given the feedback>
CORRECTED_PLANNING: <what the corrected plan should be>
CORRECTED_ACTION: <the specific corrected action/command, preferably as a unified diff>
REASONING: <why this correction addresses the root cause>

Response:"""

        response = self._call_llm(prompt)
        return self._parse_correction_response(
            response, iteration, modules.get('action', 'N/A')
        )

    def _parse_correction_response(self, response: str, iteration: int,
                                    original_action: str) -> SimulatedCorrection:
        """Parse correction simulation response, supporting multi-line fields."""
        fields = {
            'CORRECTED_ACTION': '',
            'CORRECTED_PLANNING': '',
            'CORRECTED_MEMORY': '',
            'REASONING': '',
        }
        field_keys = list(fields.keys())
        current_field = None

        for line in response.split('\n'):
            stripped = line.strip()
            # Check if this line starts a new field
            matched_field = None
            for key in field_keys:
                if stripped.startswith(f'{key}:'):
                    matched_field = key
                    break
            if matched_field:
                current_field = matched_field
                fields[current_field] = stripped.replace(f'{matched_field}:', '').strip()
            elif current_field:
                # Continuation of the current multi-line field
                fields[current_field] += '\n' + line.rstrip()

        corrected_action = fields['CORRECTED_ACTION'].strip()
        corrected_planning = fields['CORRECTED_PLANNING'].strip()
        corrected_memory = fields['CORRECTED_MEMORY'].strip()
        reasoning = fields['REASONING'].strip()

        if not corrected_action:
            corrected_action = response[:200]

        return SimulatedCorrection(
            iteration=iteration,
            original_action=original_action[:300],
            corrected_action=corrected_action,
            corrected_planning=corrected_planning,
            corrected_memory=corrected_memory,
            reasoning=reasoning
        )

    # ============================================================
    # Step 3: Predict Downstream
    # ============================================================

    def _predict_downstream(self, correction: SimulatedCorrection,
                             critical_error: Dict, steps: List[Dict],
                             task_description: str, iteration: int) -> DownstreamPrediction:
        """Predict whether the corrected action would lead to task success."""
        total_steps = len(steps)
        crit_step = critical_error['step_number']

        # Summarize remaining steps after the critical step
        remaining = [s for s in steps if s.get('step_number', 0) > crit_step]
        remaining_summary = ""
        for s in remaining[:5]:
            sn = s.get('step_number', '?')
            act = s.get('modules', {}).get('action', 'N/A')[:80]
            remaining_summary += f"Step {sn}: {act}\n"
        if len(remaining) > 5:
            remaining_summary += f"... ({len(remaining) - 5} more steps)\n"

        prompt = f"""You are predicting whether a corrected agent action would lead to task success.

# Task
{task_description[:500]}

# Original trajectory: {total_steps} steps, FAILED

# Critical Error at Step {crit_step}: {critical_error['error_type']} in {critical_error['module']}

# Corrected Action at Step {crit_step}
{correction.corrected_action}
(Original was: {correction.original_action[:200]})

# Remaining Steps After Step {crit_step}
{remaining_summary if remaining_summary else "No remaining steps."}

Predict: If the agent had taken the corrected action, would the task SUCCEED?

Respond in this EXACT format:
PREDICTED_SUCCESS: <TRUE or FALSE>
REMAINING_STEPS: <estimated steps still needed, as integer>
CONFIDENCE: <0.0 to 1.0>
REASONING: <2-3 sentences>

Response:"""

        response = self._call_llm(prompt)
        return self._parse_downstream_response(response, iteration)

    def _parse_downstream_response(self, response: str, iteration: int) -> DownstreamPrediction:
        """Parse downstream prediction response. Defaults to predicted_success=False
        so parse failures don't inflate simulated success rate."""
        predicted_success = False  # conservative: assume failure on parse failure
        remaining_steps = 0
        confidence = 0.5           # neutral default on parse failure
        reasoning = ""
        _parsed_success = False    # track if PREDICTED_SUCCESS was actually parsed

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('PREDICTED_SUCCESS:'):
                predicted_success = 'TRUE' in line.upper()
                _parsed_success = True
            elif line.startswith('REMAINING_STEPS:'):
                try:
                    remaining_steps = int(line.replace('REMAINING_STEPS:', '').strip())
                except (ValueError, TypeError):
                    remaining_steps = 0
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    pass  # keep conservative default
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()

        if not _parsed_success:
            print(f"    [Prediction Parse] Failed to parse PREDICTED_SUCCESS — defaulting to False")

        return DownstreamPrediction(
            iteration=iteration,
            predicted_success=predicted_success,
            predicted_remaining_steps=remaining_steps,
            confidence=confidence,
            reasoning=reasoning
        )

    # ============================================================
    # Step 4: Evaluate Correction (LLM-as-Judge)
    # ============================================================

    def _evaluate_correction(self, feedback: CorrectiveFeedback,
                              correction: SimulatedCorrection,
                              downstream: DownstreamPrediction,
                              critical_error: Dict, iteration: int) -> CorrectionEvaluation:
        """LLM-as-judge evaluation of the correction quality."""
        prompt = f"""You are an expert evaluator assessing corrective feedback quality for a code repair agent.

# Critical Error
Step {critical_error['step_number']}, Module: {critical_error['module']}, Type: {critical_error['error_type']}
Propagation: {' -> '.join(critical_error.get('propagation_chain', []))}

# Corrective Feedback
{feedback.feedback_text}

# Simulated Correction
Original: {correction.original_action[:200]}
Corrected: {correction.corrected_action[:200]}
Reasoning: {correction.reasoning}

# Downstream Prediction
Success: {downstream.predicted_success}, Confidence: {downstream.confidence:.2f}

Evaluate the correction:
1. Does feedback address the root cause ({critical_error['error_type']})?
2. Would it break the propagation chain?
3. Does the correction risk new errors?
4. How specific is the feedback (0.0=generic, 1.0=highly specific)?
5. How actionable is it (0.0=vague, 1.0=immediately actionable)?

Respond in this EXACT format:
ROOT_CAUSE: <YES or NO>
PROPAGATION: <YES or NO>
NEW_ERRORS: <YES or NO>
SPECIFICITY: <0.0 to 1.0>
ACTIONABILITY: <0.0 to 1.0>
OVERALL: <HIGH or MEDIUM or LOW>
ITERATE: <YES or NO>
REASONING: <2-3 sentences>

Response:"""

        response = self._call_llm(prompt)
        return self._parse_evaluation_response(response, iteration)

    def _parse_evaluation_response(self, response: str, iteration: int) -> CorrectionEvaluation:
        """Parse evaluation response. Uses conservative defaults so parse failures
        don't inflate quality metrics (previously 97/100 rated 'high')."""
        addresses_root = False   # conservative: assume not addressed on parse failure
        breaks_prop = False
        new_errors = False
        specificity = 0.5        # neutral default on parse failure
        actionability = 0.5      # neutral default on parse failure
        overall = 'medium'       # neutral default on parse failure
        reasoning = ""
        should_iterate = True    # encourage retry on parse failure
        _parsed_any = False      # track if we parsed any field

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('ROOT_CAUSE:'):
                addresses_root = 'YES' in line.upper()
                _parsed_any = True
            elif line.startswith('PROPAGATION:'):
                breaks_prop = 'YES' in line.upper()
                _parsed_any = True
            elif line.startswith('NEW_ERRORS:'):
                new_errors = 'YES' in line.upper()
                _parsed_any = True
            elif line.startswith('SPECIFICITY:'):
                try:
                    specificity = float(line.replace('SPECIFICITY:', '').strip())
                    specificity = max(0.0, min(1.0, specificity))
                    _parsed_any = True
                except (ValueError, TypeError):
                    pass  # keep conservative default
            elif line.startswith('ACTIONABILITY:'):
                try:
                    actionability = float(line.replace('ACTIONABILITY:', '').strip())
                    actionability = max(0.0, min(1.0, actionability))
                    _parsed_any = True
                except (ValueError, TypeError):
                    pass  # keep conservative default
            elif line.startswith('OVERALL:'):
                val = line.replace('OVERALL:', '').strip().lower()
                if val in ('high', 'medium', 'low'):
                    overall = val
                    _parsed_any = True
            elif line.startswith('ITERATE:'):
                should_iterate = 'YES' in line.upper()
                _parsed_any = True
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()

        if not _parsed_any:
            print(f"    [Eval Parse] Failed to parse evaluation response — using conservative defaults (low quality)")

        return CorrectionEvaluation(
            iteration=iteration,
            addresses_root_cause=addresses_root,
            breaks_propagation_chain=breaks_prop,
            introduces_new_errors=new_errors,
            specificity_score=specificity,
            actionability_score=actionability,
            overall_quality=overall,
            reasoning=reasoning,
            should_iterate=should_iterate
        )

    # ============================================================
    # Patch Extraction
    # ============================================================

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences from LLM output."""
        import re
        # Remove opening fences like ```diff, ```python, ```, ```patch, etc.
        text = re.sub(r'^```\w*\s*\n?', '', text.strip(), flags=re.MULTILINE)
        # Remove closing fences
        text = re.sub(r'\n?```\s*$', '', text.strip(), flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def _is_valid_unified_diff(text: str) -> bool:
        """Check if text looks like a valid unified diff."""
        lines = text.strip().split('\n')
        has_minus_header = any(l.strip().startswith('--- ') for l in lines)
        has_plus_header = any(l.strip().startswith('+++ ') for l in lines)
        has_hunk = any(l.strip().startswith('@@') for l in lines)
        # Also accept diff --git format
        has_git_diff = any(l.strip().startswith('diff --git') for l in lines)
        return (has_minus_header and has_plus_header and has_hunk) or has_git_diff

    def _extract_best_patch(self, iterations: List[DebugIteration]) -> Optional[str]:
        """
        Extract the best unified diff patch from the correction iterations.
        Prefers the iteration with highest quality evaluation.
        Returns None if no valid unified diff can be extracted (prevents
        sending natural language text to git apply).
        """
        # Find best iteration (highest quality, prefer successful predictions)
        best = None
        for it in iterations:
            if best is None:
                best = it
            elif it.evaluation.overall_quality == 'high' and best.evaluation.overall_quality != 'high':
                best = it
            elif it.downstream.predicted_success and not best.downstream.predicted_success:
                best = it

        if not best:
            return None

        corrected = best.correction.corrected_action
        if not corrected or len(corrected.strip()) < 5:
            return None

        # Step 1: Strip markdown code fences (common in 7B model output)
        corrected = self._strip_markdown_fences(corrected)

        # Step 2: Check if the whole text is a valid unified diff
        if self._is_valid_unified_diff(corrected):
            return corrected.strip()

        # Step 3: Try to extract a diff embedded in surrounding text
        lines = corrected.split('\n')
        diff_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for diff --git header
            if stripped.startswith('diff --git'):
                diff_start = i
                break
            # Look for --- / +++ pair
            if stripped.startswith('---') and i + 1 < len(lines) and lines[i + 1].strip().startswith('+++'):
                diff_start = i
                break
        if diff_start is not None:
            diff_lines = []
            for line in lines[diff_start:]:
                stripped = line.strip()
                if diff_lines and not stripped and not any(
                    stripped.startswith(p) for p in ['---', '+++', '@@', '+', '-', ' ', 'diff ']
                ):
                    break
                diff_lines.append(line)
            candidate = '\n'.join(diff_lines).strip()
            if self._is_valid_unified_diff(candidate):
                return candidate

        # No valid diff found — return None instead of raw text
        # This prevents meaningless patch_apply_failure from natural language
        print(f"    [Patch Extraction] No valid unified diff found in LLM output ({len(corrected)} chars)")
        return None

    # ============================================================
    # Build Final Result
    # ============================================================

    def _build_result(self, instance_id: str, critical_error: Dict,
                       agreement: str,
                       iterations: List[DebugIteration]) -> Phase3Result:
        """Assemble the final Phase3Result from iterations."""
        # Determine final success
        final_success = any(it.downstream.predicted_success for it in iterations)
        successful_iter = None
        for it in iterations:
            if it.downstream.predicted_success:
                successful_iter = it.iteration
                break

        # Quality of final iteration
        final_quality = iterations[-1].evaluation.overall_quality if iterations else 'low'

        # Compute average scores
        specs = [it.evaluation.specificity_score for it in iterations]
        acts = [it.evaluation.actionability_score for it in iterations]
        avg_spec = sum(specs) / len(specs) if specs else 0.0
        avg_act = sum(acts) / len(acts) if acts else 0.0

        # Convergence: did quality improve from first to last iteration?
        convergence = False
        if len(iterations) > 1:
            first_spec = iterations[0].evaluation.specificity_score
            last_spec = iterations[-1].evaluation.specificity_score
            first_act = iterations[0].evaluation.actionability_score
            last_act = iterations[-1].evaluation.actionability_score
            convergence = (last_spec > first_spec) or (last_act > first_act)

        return Phase3Result(
            instance_id=instance_id,
            critical_error_step=critical_error['step_number'],
            critical_error_module=critical_error['module'],
            critical_error_type=critical_error['error_type'],
            dual_channel_agreement=agreement,
            total_iterations=len(iterations),
            iterations=[asdict(it) for it in iterations],
            final_success=final_success,
            successful_iteration=successful_iter,
            final_feedback_quality=final_quality,
            avg_specificity=round(avg_spec, 3),
            avg_actionability=round(avg_act, 3),
            convergence=convergence,
            timestamp=datetime.now().isoformat()
        )


# ============================================================
# Demo
# ============================================================

async def main_demo():
    """Demo of Phase 3 debugger"""

    class MockLLM:
        def invoke(self, prompt):
            if 'Generate TARGETED corrective feedback' in prompt:
                return """FEEDBACK: The agent committed an api_hallucination error by attempting to use token.is_expired attribute which does not exist in the Token class. Instead, the agent should have first run 'grep -r "is_expired" .' to find the correct method name (is_expired_token()) before planning the fix. This would have prevented the cascade of AttributeErrors in subsequent steps.
ADDRESSES_PROPAGATION: YES"""
            elif 'SHOULD have done' in prompt:
                return """CORRECTED_MEMORY: The Token class has methods is_expired_token() and refresh_token(), not is_expired.
CORRECTED_PLANNING: First search for the correct method name using grep, then apply the fix using the verified API.
CORRECTED_ACTION: grep -r "is_expired" src/auth/tokens.py
REASONING: By verifying the actual API before using it, the agent avoids the AttributeError that cascaded into all subsequent steps."""
            elif 'would the task SUCCEED' in prompt:
                return """PREDICTED_SUCCESS: TRUE
REMAINING_STEPS: 4
CONFIDENCE: 0.75
REASONING: Finding the correct method name would let the agent write a valid fix. The remaining test execution steps would likely pass since the underlying logic is sound."""
            else:
                return """ROOT_CAUSE: YES
PROPAGATION: YES
NEW_ERRORS: NO
SPECIFICITY: 0.85
ACTIONABILITY: 0.90
OVERALL: HIGH
ITERATE: NO
REASONING: The feedback correctly identifies the api_hallucination and provides a concrete corrective action (grep for the method name). The simulated correction is specific and actionable."""

    llm = MockLLM()
    debugger = CodePhase3Debugger(llm)

    # Mock Phase 2 results
    phase2_results = {
        'instance_id': 'test-demo-001',
        'critical_error': {
            'step_number': 4,
            'module': 'planning',
            'error_type': 'api_hallucination',
            'confidence': 0.95,
            'explanation': 'Agent used non-existent token.is_expired attribute',
            'counterfactual_reasoning': 'If agent had checked API first, fix would succeed',
            'propagation_chain': ['api_hallucination step 4', 'AttributeError step 5', 'wrong fix step 6']
        }
    }

    # Mock Phase 1 results
    phase1_results = {
        'step_analyses': [
            {
                'step_number': 4,
                'agreement': {'planning': 'both_error', 'action': 'both_clean',
                              'memory': 'both_clean', 'reflection': 'llm_only',
                              'system': 'both_clean'}
            }
        ]
    }

    # Mock trajectory
    trajectory = {
        'instance_id': 'test-demo-001',
        'task_description': 'Fix authentication token expiry check in auth module',
        'steps': [
            {'step_number': 1, 'modules': {'memory': '', 'reflection': '', 'planning': 'Read auth code', 'action': 'cat src/auth/tokens.py'}, 'observation': 'class Token: ...'},
            {'step_number': 2, 'modules': {'memory': 'Token class found', 'reflection': 'Need to fix expiry', 'planning': 'Find expiry method', 'action': 'grep -r "expir" src/'}, 'observation': 'src/auth/tokens.py: is_expired_token()'},
            {'step_number': 3, 'modules': {'memory': 'Found is_expired_token', 'reflection': 'Ready to fix', 'planning': 'Edit the check', 'action': 'edit src/auth/views.py'}, 'observation': 'File opened'},
            {'step_number': 4, 'modules': {'memory': 'Editing views.py', 'reflection': 'Applying fix', 'planning': 'Use token.is_expired', 'action': 'replace token.check() with token.is_expired'}, 'observation': 'AttributeError: Token has no attribute is_expired'},
            {'step_number': 5, 'modules': {'memory': 'Error occurred', 'reflection': 'Something wrong', 'planning': 'Try again', 'action': 'replace with token.expired'}, 'observation': 'AttributeError again'},
        ],
        'final_result': {'success': False, 'tests_passed': 1, 'tests_total': 5}
    }

    result = await debugger.run_phase3(phase2_results, phase1_results, trajectory)

    print("\n=== Phase 3 Demo Complete ===")
    if result:
        print(f"  Success: {result['final_success']}")
        print(f"  Iterations: {result['total_iterations']}")
        print(f"  Quality: {result['final_feedback_quality']}")
        print(f"  Avg Specificity: {result['avg_specificity']}")
        print(f"  Avg Actionability: {result['avg_actionability']}")


if __name__ == "__main__":
    asyncio.run(main_demo())
