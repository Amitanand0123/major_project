"""
Code Domain Phase 1 Detector
Adapts fine-grained error analysis for code repair tasks (SWE-bench)
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

# Import from existing detector
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code_error_taxonomy import CodeErrorTaxonomy
from automatic_error_detection import AutomaticErrorDetector, HybridErrorDetector


@dataclass
class ModuleError:
    """Error detected in a specific module"""
    module: str
    error_type: str
    confidence: float
    explanation: str
    evidence: str
    automatic: bool = False


@dataclass
class StepAnalysis:
    """Analysis of errors in a single step"""
    step_number: int
    memory_error: Optional[ModuleError]
    reflection_error: Optional[ModuleError]
    planning_error: Optional[ModuleError]
    action_error: Optional[ModuleError]
    system_error: Optional[ModuleError]
    detection_method: str  # 'automatic' or 'llm'
    cost: float


@dataclass
class LLMModuleClassification:
    """LLM's independent classification for a single module within a step"""
    module: str
    has_error: bool
    error_type: Optional[str]
    confidence: float
    explanation: str


@dataclass
class DualStepAnalysis:
    """Analysis of a single step with both regex and LLM results"""
    step_number: int
    # Regex channel results
    regex_memory_error: Optional[ModuleError]
    regex_reflection_error: Optional[ModuleError]
    regex_planning_error: Optional[ModuleError]
    regex_action_error: Optional[ModuleError]
    regex_system_error: Optional[ModuleError]
    # LLM channel results
    llm_memory_error: Optional[LLMModuleClassification]
    llm_reflection_error: Optional[LLMModuleClassification]
    llm_planning_error: Optional[LLMModuleClassification]
    llm_action_error: Optional[LLMModuleClassification]
    llm_system_error: Optional[LLMModuleClassification]
    # Agreement metrics
    agreement: Dict[str, str] = field(default_factory=dict)
    llm_call_duration_seconds: float = 0.0
    llm_timeout: bool = False


class CodePhase1Detector:
    """
    Phase 1 Error Detection for Code Domain
    Extends fine_grained_analysis.py with code-specific error types
    """

    def __init__(self, llm, use_automatic_detection: bool = True):
        """
        Initialize code phase 1 detector

        Args:
            llm: Language model for LLM-based detection
            use_automatic_detection: Whether to use automatic detection first
        """
        self.llm = llm
        self.taxonomy = CodeErrorTaxonomy()
        self.auto_detector = AutomaticErrorDetector() if use_automatic_detection else None
        self.use_automatic = use_automatic_detection
        self.current_instance_id = None  # Track current instance being analyzed

    async def detect_module_errors(self, module_name: str, module_content: str,
                                   step_num: int, step_data: Dict,
                                   task_description: str,
                                   previous_steps: List[Dict]) -> ModuleError:
        """
        Detect errors in a specific module

        Args:
            module_name: Module name (memory, reflection, planning, action, system)
            module_content: Content from the module
            step_num: Step number
            step_data: Full step data
            task_description: Original task description
            previous_steps: Previous step data

        Returns:
            ModuleError or None if no error detected
        """
        # Step 1 cannot have memory or reflection errors
        if step_num == 1 and module_name in ['memory', 'reflection']:
            return None

        # Try automatic detection first for certain modules
        if self.use_automatic and module_name in ['action', 'memory', 'system', 'planning', 'reflection']:
            auto_error = await self._try_automatic_detection(
                module_name, step_data, task_description, previous_steps
            )
            if auto_error:
                return auto_error

        # Fall back to LLM-based detection
        return await self._llm_based_detection(
            module_name, module_content, step_num, step_data, task_description, previous_steps
        )

    async def _try_automatic_detection(self, module_name: str, step_data: Dict,
                                       task_description: str,
                                       previous_steps: List[Dict]) -> Optional[ModuleError]:
        """Try automatic error detection"""
        if not self.auto_detector:
            return None

        observation = step_data.get('observation', '')
        step_number = step_data.get('step_number', 0)

        # Detect from compiler/test output
        auto_errors = self.auto_detector.detect_from_output(observation, step_number)

        # Filter by module
        module_errors = [e for e in auto_errors if e.module == module_name]

        if module_errors:
            # Use the highest confidence error
            best_error = max(module_errors, key=lambda e: e.confidence)

            return ModuleError(
                module=best_error.module,
                error_type=best_error.error_type,
                confidence=best_error.confidence,
                explanation=f"Automatically detected from output: {best_error.error_type}",
                evidence=best_error.evidence,
                automatic=True
            )

        # Try specialized detections
        if module_name == 'memory':
            dep_error = self.auto_detector.detect_dependency_issues(step_data, previous_steps)
            if dep_error:
                return ModuleError(
                    module=dep_error.module,
                    error_type=dep_error.error_type,
                    confidence=dep_error.confidence,
                    explanation="Missing required imports or dependencies",
                    evidence=dep_error.evidence,
                    automatic=True
                )

        if module_name == 'planning':
            scope_error = self.auto_detector.detect_scope_violation(step_data, task_description, self.current_instance_id)
            if scope_error:
                return ModuleError(
                    module=scope_error.module,
                    error_type=scope_error.error_type,
                    confidence=scope_error.confidence,
                    explanation="Plan modifies files outside task scope",
                    evidence=scope_error.evidence,
                    automatic=True
                )

        if module_name == 'reflection':
            reflection_error = self.auto_detector.detect_reflection_errors(step_data, previous_steps)
            if reflection_error:
                return ModuleError(
                    module=reflection_error.module,
                    error_type=reflection_error.error_type,
                    confidence=reflection_error.confidence,
                    explanation=reflection_error.evidence,
                    evidence=reflection_error.evidence,
                    automatic=True
                )

        return None

    async def _llm_based_detection(self, module_name: str, module_content: str,
                                   step_num: int, step_data: Dict,
                                   task_description: str,
                                   previous_steps: List[Dict]) -> Optional[ModuleError]:
        """LLM-based error detection"""
        # Build error taxonomy prompt
        taxonomy_prompt = self.taxonomy.format_for_phase1_prompt(module_name)

        # Build context
        observation = step_data.get('observation', '')
        action = step_data.get('modules', {}).get('action', '')

        # Format previous steps
        prev_context = self._format_previous_steps(previous_steps, limit=3)

        prompt = f"""You are an expert at debugging code repair agents.

Task: {task_description}

{taxonomy_prompt}

# Current Step (Step {step_num})

## {module_name.upper()} Module Output:
{module_content}

## Action Taken:
{action}

## Observation/Result:
{observation}

## Previous Steps:
{prev_context}

# Your Task

Analyze the {module_name} module for errors. Consider:

1. Does this module have any of the error types listed above?
2. What is the evidence for this error?
3. How confident are you (0.0 to 1.0)?

If you detect an error, respond in this EXACT format:
ERROR_TYPE: <error_type>
CONFIDENCE: <0.0 to 1.0>
EXPLANATION: <brief explanation>
EVIDENCE: <specific evidence from the module output>

If NO error, respond with:
NO_ERROR

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
        if 'NO_ERROR' in response_text.upper():
            return None

        return self._parse_llm_response(response_text, module_name)

    def _parse_llm_response(self, response: str, module_name: str) -> Optional[ModuleError]:
        """Parse LLM response into ModuleError"""
        lines = response.split('\n')

        error_type = None
        confidence = 0.0
        explanation = ""
        evidence = ""

        for line in lines:
            line = line.strip()
            if line.startswith('ERROR_TYPE:'):
                error_type = line.replace('ERROR_TYPE:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                conf_str = line.replace('CONFIDENCE:', '').strip()
                try:
                    confidence = float(conf_str)
                except:
                    confidence = 0.5
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('EVIDENCE:'):
                evidence = line.replace('EVIDENCE:', '').strip()

        if error_type:
            return ModuleError(
                module=module_name,
                error_type=error_type,
                confidence=confidence,
                explanation=explanation,
                evidence=evidence,
                automatic=False
            )

        return None

    def _format_previous_steps(self, previous_steps: List[Dict], limit: int = 3) -> str:
        """Format previous steps for context"""
        if not previous_steps:
            return "No previous steps."

        recent = previous_steps[-limit:]
        formatted = []

        for step in recent:
            step_num = step.get('step_number', '?')
            action = step.get('modules', {}).get('action', 'N/A')
            obs = step.get('observation', 'N/A')[:150]

            formatted.append(f"Step {step_num}:")
            formatted.append(f"  Action: {action}")
            formatted.append(f"  Result: {obs}...")

        return "\n".join(formatted)

    async def analyze_step(self, step_data: Dict, task_description: str,
                          previous_steps: List[Dict]) -> StepAnalysis:
        """
        Analyze all modules in a single step

        Args:
            step_data: Step data dictionary
            task_description: Task description
            previous_steps: Previous steps

        Returns:
            StepAnalysis with errors from all modules
        """
        step_num = step_data.get('step_number', 0)
        modules = step_data.get('modules', {})

        total_cost = 0.0
        detection_methods = []

        # Analyze each module
        memory_error = None
        reflection_error = None
        planning_error = None
        action_error = None
        system_error = None

        if step_num > 1:  # Skip memory/reflection for step 1
            if 'memory' in modules:
                memory_error = await self.detect_module_errors(
                    'memory', modules['memory'], step_num, step_data, task_description, previous_steps
                )
                if memory_error and not memory_error.automatic:
                    total_cost += 0.01
                detection_methods.append('automatic' if (memory_error and memory_error.automatic) else 'llm')

            if 'reflection' in modules:
                reflection_error = await self.detect_module_errors(
                    'reflection', modules['reflection'], step_num, step_data, task_description, previous_steps
                )
                if reflection_error and not reflection_error.automatic:
                    total_cost += 0.01
                detection_methods.append('automatic' if (reflection_error and reflection_error.automatic) else 'llm')

        if 'planning' in modules:
            planning_error = await self.detect_module_errors(
                'planning', modules['planning'], step_num, step_data, task_description, previous_steps
            )
            if planning_error and not planning_error.automatic:
                total_cost += 0.01
            detection_methods.append('automatic' if (planning_error and planning_error.automatic) else 'llm')

        if 'action' in modules:
            action_error = await self.detect_module_errors(
                'action', modules['action'], step_num, step_data, task_description, previous_steps
            )
            if action_error and not action_error.automatic:
                total_cost += 0.01
            detection_methods.append('automatic' if (action_error and action_error.automatic) else 'llm')

        # Check for system errors in observation
        observation = step_data.get('observation', '')
        if 'timeout' in observation.lower() or 'error' in observation.lower():
            system_error = await self.detect_module_errors(
                'system', '', step_num, step_data, task_description, previous_steps
            )
            if system_error and not system_error.automatic:
                total_cost += 0.01
            detection_methods.append('automatic' if (system_error and system_error.automatic) else 'llm')

        # Determine overall detection method
        if any(m == 'automatic' for m in detection_methods):
            overall_method = 'hybrid' if any(m == 'llm' for m in detection_methods) else 'automatic'
        else:
            overall_method = 'llm'

        return StepAnalysis(
            step_number=step_num,
            memory_error=memory_error,
            reflection_error=reflection_error,
            planning_error=planning_error,
            action_error=action_error,
            system_error=system_error,
            detection_method=overall_method,
            cost=total_cost
        )

    async def analyze_trajectory(self, trajectory: Dict) -> Dict[str, Any]:
        """
        Analyze entire trajectory

        Args:
            trajectory: Trajectory dictionary from SWE-bench

        Returns:
            Complete phase 1 analysis
        """
        instance_id = trajectory.get('instance_id', 'unknown')
        task_description = trajectory.get('task_description', '')
        steps = trajectory.get('steps', [])

        # Store instance_id for use in automatic detection
        self.current_instance_id = instance_id

        print(f"\n=== Analyzing Trajectory: {instance_id} ===")
        print(f"Total steps: {len(steps)}")

        step_analyses = []
        previous_steps = []
        total_cost = 0.0

        for step_data in steps:
            step_num = step_data.get('step_number', 0)
            print(f"\nAnalyzing Step {step_num}...")

            analysis = await self.analyze_step(step_data, task_description, previous_steps)
            step_analyses.append(analysis)

            total_cost += analysis.cost
            previous_steps.append(step_data)

        # Generate summary
        summary = self._generate_summary(step_analyses, total_cost)

        result = {
            'instance_id': instance_id,
            'task_description': task_description,
            'total_steps': len(steps),
            'step_analyses': [asdict(analysis) for analysis in step_analyses],
            'summary': summary,
            'total_cost': total_cost,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def _generate_summary(self, step_analyses: List[StepAnalysis], total_cost: float) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_errors = 0
        automatic_errors = 0
        errors_by_module = {}
        errors_by_type = {}

        for analysis in step_analyses:
            for error_attr in ['memory_error', 'reflection_error', 'planning_error', 'action_error', 'system_error']:
                error = getattr(analysis, error_attr)
                if error:
                    total_errors += 1
                    if error.automatic:
                        automatic_errors += 1

                    errors_by_module[error.module] = errors_by_module.get(error.module, 0) + 1
                    errors_by_type[error.error_type] = errors_by_type.get(error.error_type, 0) + 1

        automatic_rate = (automatic_errors / total_errors * 100) if total_errors > 0 else 0

        return {
            'total_errors': total_errors,
            'automatic_detection_count': automatic_errors,
            'automatic_detection_rate': automatic_rate,
            'llm_detection_count': total_errors - automatic_errors,
            'errors_by_module': errors_by_module,
            'errors_by_type': errors_by_type,
            'total_cost': total_cost,
            'cost_per_step': total_cost / len(step_analyses) if step_analyses else 0
        }


    # ================================================================
    # DUAL-CHANNEL METHODS (Regex + LLM on every step)
    # ================================================================

    async def _run_regex_channel(self, step_data: Dict, task_description: str,
                                  previous_steps: List[Dict]) -> Dict[str, Optional[ModuleError]]:
        """Run regex/automatic detection on all modules for a step"""
        step_num = step_data.get('step_number', 0)
        results = {}

        for module_name in ['memory', 'reflection', 'planning', 'action', 'system']:
            if step_num == 1 and module_name in ['memory', 'reflection']:
                results[module_name] = None
                continue

            # For system, only check if observation suggests issues
            if module_name == 'system':
                observation = step_data.get('observation', '')
                if 'timeout' not in observation.lower() and 'error' not in observation.lower():
                    results[module_name] = None
                    continue

            auto_error = await self._try_automatic_detection(
                module_name, step_data, task_description, previous_steps
            )
            results[module_name] = auto_error

        return results

    def _build_dual_detection_prompt(self, step_num: int, modules: Dict,
                                      observation: str, task_description: str,
                                      prev_context: str) -> str:
        """Build prompt for LLM to classify all 5 modules in one call"""
        # Get full taxonomy (all modules)
        full_taxonomy = self.taxonomy.format_for_prompt()

        # Build module content sections
        module_sections = []
        for mod_name in ['memory', 'reflection', 'planning', 'action']:
            content = modules.get(mod_name, '')
            if content:
                module_sections.append(f"## {mod_name.upper()} Module:\n{str(content)[:500]}")

        module_text = "\n\n".join(module_sections) if module_sections else "No module data available."

        prompt = f"""You are an expert at debugging code repair agents. Analyze this agent step for errors in ALL cognitive modules.

# Task Description
{str(task_description)[:500]}

# Error Taxonomy
{full_taxonomy}

# Step {step_num}

{module_text}

## Observation/Result:
{str(observation)[:1000]}

## Previous Context:
{prev_context}

# Instructions
For EACH of the 5 modules (memory, reflection, planning, action, system), determine if there is an error.
- Step 1 CANNOT have memory or reflection errors.
- System errors come from the observation (timeouts, environment issues).
- Only report an error if you have clear evidence.

Respond in this EXACT format (one section per module):

MEMORY: [ERROR|NO_ERROR]
MEMORY_TYPE: [error_type or NONE]
MEMORY_CONFIDENCE: [0.0-1.0]
MEMORY_EXPLANATION: [brief explanation or NONE]

REFLECTION: [ERROR|NO_ERROR]
REFLECTION_TYPE: [error_type or NONE]
REFLECTION_CONFIDENCE: [0.0-1.0]
REFLECTION_EXPLANATION: [brief explanation or NONE]

PLANNING: [ERROR|NO_ERROR]
PLANNING_TYPE: [error_type or NONE]
PLANNING_CONFIDENCE: [0.0-1.0]
PLANNING_EXPLANATION: [brief explanation or NONE]

ACTION: [ERROR|NO_ERROR]
ACTION_TYPE: [error_type or NONE]
ACTION_CONFIDENCE: [0.0-1.0]
ACTION_EXPLANATION: [brief explanation or NONE]

SYSTEM: [ERROR|NO_ERROR]
SYSTEM_TYPE: [error_type or NONE]
SYSTEM_CONFIDENCE: [0.0-1.0]
SYSTEM_EXPLANATION: [brief explanation or NONE]

Response:"""
        return prompt

    def _parse_multi_module_llm_response(self, response_text: str) -> Dict[str, Optional[LLMModuleClassification]]:
        """Parse structured LLM response into per-module classifications"""
        results = {}
        for module in ['memory', 'reflection', 'planning', 'action', 'system']:
            prefix = module.upper()
            has_error = False
            error_type = None
            confidence = 0.0
            explanation = ""

            for line in response_text.split('\n'):
                line = line.strip()
                # Match "MEMORY: ERROR" but not "MEMORY_TYPE:" etc.
                if line.startswith(f'{prefix}:') and '_' not in line.split(':')[0]:
                    has_error = 'ERROR' in line.upper() and 'NO_ERROR' not in line.upper()
                elif line.startswith(f'{prefix}_TYPE:'):
                    val = line.split(':', 1)[1].strip()
                    error_type = val if val.upper() != 'NONE' else None
                elif line.startswith(f'{prefix}_CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        confidence = 0.5
                elif line.startswith(f'{prefix}_EXPLANATION:'):
                    val = line.split(':', 1)[1].strip()
                    explanation = val if val.upper() != 'NONE' else ""

            if has_error and error_type:
                results[module] = LLMModuleClassification(
                    module=module,
                    has_error=True,
                    error_type=error_type,
                    confidence=confidence,
                    explanation=explanation
                )
            else:
                results[module] = None

        return results

    async def _run_llm_channel(self, step_data: Dict, step_num: int,
                                task_description: str,
                                previous_steps: List[Dict]) -> Tuple[Dict[str, Optional[LLMModuleClassification]], float, bool]:
        """Run LLM analysis on the full step, classifying all modules at once"""
        modules = step_data.get('modules', {})
        observation = step_data.get('observation', '')
        prev_context = self._format_previous_steps(previous_steps, limit=3)

        prompt = self._build_dual_detection_prompt(
            step_num, modules, observation, task_description, prev_context
        )

        start = time.time()
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    response = response.content
            else:
                response = self.llm(prompt)

            duration = time.time() - start
            response_text = str(response).strip()

            parsed = self._parse_multi_module_llm_response(response_text)

            # Enforce step 1 constraint
            if step_num == 1:
                parsed['memory'] = None
                parsed['reflection'] = None

            return parsed, duration, False

        except Exception as e:
            duration = time.time() - start
            print(f"  LLM call failed for step {step_num}: {e}")
            empty = {m: None for m in ['memory', 'reflection', 'planning', 'action', 'system']}
            return empty, duration, True

    async def analyze_step_dual(self, step_data: Dict, task_description: str,
                                 previous_steps: List[Dict]) -> DualStepAnalysis:
        """Analyze a step with both regex and LLM channels independently"""
        step_num = step_data.get('step_number', 0)

        # Channel 1: Regex (fast, free)
        regex_results = await self._run_regex_channel(step_data, task_description, previous_steps)

        # Channel 2: LLM (one call for all modules)
        llm_results, llm_duration, llm_timeout = await self._run_llm_channel(
            step_data, step_num, task_description, previous_steps
        )

        # Compute per-module agreement
        agreement = {}
        for module in ['memory', 'reflection', 'planning', 'action', 'system']:
            regex_found = regex_results.get(module) is not None
            llm_found = llm_results.get(module) is not None

            if regex_found and llm_found:
                agreement[module] = 'both_error'
            elif not regex_found and not llm_found:
                agreement[module] = 'both_clean'
            elif regex_found and not llm_found:
                agreement[module] = 'regex_only'
            else:
                agreement[module] = 'llm_only'

        return DualStepAnalysis(
            step_number=step_num,
            regex_memory_error=regex_results.get('memory'),
            regex_reflection_error=regex_results.get('reflection'),
            regex_planning_error=regex_results.get('planning'),
            regex_action_error=regex_results.get('action'),
            regex_system_error=regex_results.get('system'),
            llm_memory_error=llm_results.get('memory'),
            llm_reflection_error=llm_results.get('reflection'),
            llm_planning_error=llm_results.get('planning'),
            llm_action_error=llm_results.get('action'),
            llm_system_error=llm_results.get('system'),
            agreement=agreement,
            llm_call_duration_seconds=llm_duration,
            llm_timeout=llm_timeout
        )

    async def analyze_trajectory_dual(self, trajectory: Dict) -> Dict[str, Any]:
        """Analyze entire trajectory with dual-channel detection"""
        instance_id = trajectory.get('instance_id', 'unknown')
        task_description = trajectory.get('task_description', '')
        steps = trajectory.get('steps', [])

        self.current_instance_id = instance_id

        print(f"\n=== Dual-Channel Analysis: {instance_id} ===")
        print(f"Total steps: {len(steps)}")

        step_analyses = []
        previous_steps = []

        for step_data in steps:
            step_num = step_data.get('step_number', 0)
            print(f"\n  Step {step_num}/{len(steps)}...", end='', flush=True)

            dual_analysis = await self.analyze_step_dual(step_data, task_description, previous_steps)
            step_analyses.append(dual_analysis)
            previous_steps.append(step_data)

            # Print agreement summary for this step
            agree_counts = {}
            for v in dual_analysis.agreement.values():
                agree_counts[v] = agree_counts.get(v, 0) + 1
            print(f" [LLM:{dual_analysis.llm_call_duration_seconds:.1f}s | {agree_counts}]")

        summary = self._generate_dual_summary(step_analyses)

        return {
            'instance_id': instance_id,
            'task_description': task_description,
            'total_steps': len(steps),
            'step_analyses': [asdict(a) for a in step_analyses],
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_dual_summary(self, step_analyses: List[DualStepAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics for dual-channel analysis"""
        regex_errors = 0
        llm_errors = 0
        agreements = {'both_error': 0, 'both_clean': 0, 'regex_only': 0, 'llm_only': 0}
        regex_errors_by_module = {}
        llm_errors_by_module = {}
        regex_errors_by_type = {}
        llm_errors_by_type = {}
        total_llm_duration = 0.0
        llm_timeouts = 0
        total_module_comparisons = 0
        error_type_agreements = 0
        error_type_comparisons = 0

        for analysis in step_analyses:
            total_llm_duration += analysis.llm_call_duration_seconds
            if analysis.llm_timeout:
                llm_timeouts += 1

            for module in ['memory', 'reflection', 'planning', 'action', 'system']:
                ag = analysis.agreement.get(module, 'both_clean')
                agreements[ag] += 1
                total_module_comparisons += 1

                # Count regex errors
                regex_err = getattr(analysis, f'regex_{module}_error')
                if regex_err:
                    regex_errors += 1
                    regex_errors_by_module[module] = regex_errors_by_module.get(module, 0) + 1
                    regex_errors_by_type[regex_err.error_type] = regex_errors_by_type.get(regex_err.error_type, 0) + 1

                # Count LLM errors
                llm_err = getattr(analysis, f'llm_{module}_error')
                if llm_err:
                    llm_errors += 1
                    llm_errors_by_module[module] = llm_errors_by_module.get(module, 0) + 1
                    llm_errors_by_type[llm_err.error_type] = llm_errors_by_type.get(llm_err.error_type, 0) + 1

                # Error type agreement (when both detect error)
                if regex_err and llm_err:
                    error_type_comparisons += 1
                    if regex_err.error_type == llm_err.error_type:
                        error_type_agreements += 1

        agreement_rate = ((agreements['both_error'] + agreements['both_clean']) / total_module_comparisons * 100) if total_module_comparisons > 0 else 0
        error_type_agreement_rate = (error_type_agreements / error_type_comparisons * 100) if error_type_comparisons > 0 else 0

        return {
            'regex_total_errors': regex_errors,
            'llm_total_errors': llm_errors,
            'regex_errors_by_module': regex_errors_by_module,
            'llm_errors_by_module': llm_errors_by_module,
            'regex_errors_by_type': regex_errors_by_type,
            'llm_errors_by_type': llm_errors_by_type,
            'agreement_counts': agreements,
            'agreement_rate': agreement_rate,
            'error_type_agreement_rate': error_type_agreement_rate,
            'error_type_comparisons': error_type_comparisons,
            'total_module_comparisons': total_module_comparisons,
            'llm_only_errors': agreements['llm_only'],
            'regex_only_errors': agreements['regex_only'],
            'total_llm_duration_seconds': total_llm_duration,
            'avg_llm_duration_seconds': total_llm_duration / len(step_analyses) if step_analyses else 0,
            'llm_timeouts': llm_timeouts
        }


async def main_demo():
    """Demo of code phase 1 detector"""
    from swebench_integration import create_sample_trajectory

    # Create mock LLM for demo
    class MockLLM:
        def invoke(self, prompt):
            return "ERROR_TYPE: api_hallucination\nCONFIDENCE: 0.9\nEXPLANATION: Agent used non-existent attribute\nEVIDENCE: token.is_expired does not exist"

    llm = MockLLM()
    detector = CodePhase1Detector(llm, use_automatic_detection=True)

    # Create sample trajectory
    trajectory = create_sample_trajectory()

    # Analyze
    result = await detector.analyze_trajectory(trajectory)

    print("\n=== Phase 1 Analysis Complete ===")
    print(f"Total errors detected: {result['summary']['total_errors']}")
    print(f"Automatic detection rate: {result['summary']['automatic_detection_rate']:.1f}%")
    print(f"Total cost: ${result['total_cost']:.3f}")
    print(f"\nErrors by module: {result['summary']['errors_by_module']}")
    print(f"Errors by type: {result['summary']['errors_by_type']}")


if __name__ == "__main__":
    asyncio.run(main_demo())
